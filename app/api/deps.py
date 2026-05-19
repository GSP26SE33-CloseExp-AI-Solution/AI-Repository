from typing import Optional

from fastapi import Depends, Header, HTTPException

from app.core.security import verify_api_key
from app.services.ocr import OCRService, ocr_service
from app.services.pricing import PricingService, pricing_service


async def get_api_key(
    api_key: Optional[str] = Depends(verify_api_key),
) -> Optional[str]:
    """Dependency for API key verification."""
    return api_key


async def get_user_id(
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
) -> str:
    """Resolve the authenticated backend user id for per-user token accounting."""
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(
            status_code=400,
            detail="X-User-Id header is required for AI token tracking",
        )
    return x_user_id.strip()


def get_ocr_service() -> OCRService:
    """Dependency for OCR service."""
    return ocr_service


def get_pricing_service() -> PricingService:
    """Dependency for pricing service."""
    return pricing_service
