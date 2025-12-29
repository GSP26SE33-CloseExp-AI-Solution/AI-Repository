from app.core.config import settings
from app.models.pricing import PricingRequest, PricingResponse


def _decay_factor(days_to_expire: int) -> float:
    if days_to_expire <= 1:
        return 0.45
    if days_to_expire <= 3:
        return 0.55
    if days_to_expire <= 7:
        return 0.70
    return 0.80


def suggest_price(payload: PricingRequest) -> PricingResponse:
    decay = _decay_factor(payload.days_to_expire)
    suggested = payload.base_price * decay
    rationale = {
        "decay_factor": decay,
        "days_to_expire": payload.days_to_expire,
        "region": payload.region or "n/a",
        "product_type": payload.product_type,
        "brand": payload.brand or "n/a",
    }
    confidence = min(1.0, settings.pricing_default_confidence + 0.05)
    return PricingResponse(
        suggested_price=round(suggested, 2),
        confidence=confidence,
        rationale=rationale,
    )
