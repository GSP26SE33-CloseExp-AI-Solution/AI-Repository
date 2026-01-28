from fastapi import APIRouter, Depends

from app.api.deps import get_api_key
from app.models.pricing import PricingRequest, PricingResponse
from app.services.pricing import suggest_price

router = APIRouter()


@router.post(
    "/suggest",
    response_model=PricingResponse,
    summary="Suggest price for near-expiry product",
    description="Calculate optimal price based on expiry date, product type and market factors",
)
async def suggest(
    payload: PricingRequest,
    _: str = Depends(get_api_key),
) -> PricingResponse:
    """
    Suggest optimal price for a near-expiry product.
    
    Takes into account:
    - Days until expiration
    - Product category
    - Demand index
    - Competitor pricing
    - Pricing strategy
    
    Returns suggested price with confidence score and rationale.
    """
    return suggest_price(payload)
