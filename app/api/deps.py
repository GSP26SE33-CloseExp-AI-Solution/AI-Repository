from typing import Optional

from fastapi import Depends

from app.core.security import verify_api_key
from app.services.ocr import OCRService, ocr_service
from app.services.pricing import PricingService, pricing_service


async def get_api_key(
    api_key: Optional[str] = Depends(verify_api_key),
) -> Optional[str]:
    """Dependency for API key verification."""
    return api_key


def get_ocr_service() -> OCRService:
    """Dependency for OCR service."""
    return ocr_service


def get_pricing_service() -> PricingService:
    """Dependency for pricing service."""
    return pricing_service
