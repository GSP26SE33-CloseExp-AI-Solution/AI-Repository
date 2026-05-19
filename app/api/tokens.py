"""
Token Management API endpoints.

Provides endpoints to view monthly AI token usage and budgets per user.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_api_key, get_user_id
from app.services.token_service import MONTHLY_BUDGET, TOKEN_COST, token_service

router = APIRouter()


@router.get(
    "/status",
    summary="Get current month token usage for all features",
    description=(
        "Returns current month token budget and usage for OCR and Pricing AI features "
        "for the authenticated user (X-User-Id header)."
    ),
)
async def get_token_status(
    month: Optional[str] = Query(
        default=None,
        description="Month in YYYY-MM format. Defaults to current month.",
        example="2026-05",
    ),
    user_id: str = Depends(get_user_id),
    _: str = Depends(get_api_key),
):
    return {
        "success": True,
        "data": token_service.get_all_usage(user_id, month),
    }


@router.get(
    "/status/{feature}",
    summary="Get token usage for a specific feature",
    description="Returns budget / usage for one feature: `ocr` or `pricing`.",
)
async def get_feature_token_status(
    feature: str,
    month: Optional[str] = Query(
        default=None,
        description="Month in YYYY-MM format. Defaults to current month.",
    ),
    user_id: str = Depends(get_user_id),
    _: str = Depends(get_api_key),
):
    if feature not in MONTHLY_BUDGET:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown feature '{feature}'. Valid values: {list(MONTHLY_BUDGET.keys())}",
        )

    return {
        "success": True,
        "data": token_service.get_usage(feature, user_id, month),
    }


@router.get(
    "/history",
    summary="Get full token usage history across all months",
    description="Returns a month-by-month breakdown of AI token consumption for the user.",
)
async def get_token_history(
    user_id: str = Depends(get_user_id),
    _: str = Depends(get_api_key),
):
    return {
        "success": True,
        "data": token_service.get_history(user_id),
    }


@router.get(
    "/config",
    summary="Get token budget configuration",
    description="Returns the monthly budget and per-call token costs for each feature.",
)
async def get_token_config(
    _: str = Depends(get_api_key),
):
    return {
        "success": True,
        "data": {
            "monthly_budgets": MONTHLY_BUDGET,
            "token_costs": TOKEN_COST,
            "description": {
                "ocr_1_image": "OCR with 1 image costs 1 token",
                "ocr_2_images": "OCR with 2 images costs 2 tokens",
                "ocr_3_images": "OCR with 3 images costs 3 tokens",
                "pricing": "Pricing suggestion costs 1 token per request",
            },
        },
    }
