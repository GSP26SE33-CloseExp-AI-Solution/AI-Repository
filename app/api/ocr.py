from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_api_key
from app.core.exceptions import ImageProcessingError, OCRExtractionError
from app.models.ocr import OcrRequest, OcrResponse
from app.services.ocr import extract_product_fields

router = APIRouter()


@router.post(
    "/extract",
    response_model=OcrResponse,
    summary="Extract product information from image",
    description="Extract expiry date, manufacturing date, barcode and other product info using OCR",
)
async def extract(
    request: OcrRequest,
    _: str = Depends(get_api_key),
) -> OcrResponse:
    """
    Extract product information from image.
    
    - Extracts expiry date and manufacturing date
    - Extracts barcode/QR code
    - Returns confidence scores for each extracted field
    """
    if not request.has_image():
        raise HTTPException(
            status_code=400,
            detail="Either image_url or image_b64 is required",
        )

    try:
        return extract_product_fields(request)
    except ImageProcessingError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except OCRExtractionError as e:
        raise HTTPException(status_code=422, detail=e.message) from e
