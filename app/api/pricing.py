from fastapi import APIRouter

from app.models.pricing import PricingRequest, PricingResponse
from app.services.pricing import suggest_price

router = APIRouter(tags=["pricing"])


@router.post("/suggest", response_model=PricingResponse)
async def suggest(payload: PricingRequest) -> PricingResponse:
    return suggest_price(payload)
