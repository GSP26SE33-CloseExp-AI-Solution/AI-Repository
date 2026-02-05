from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_api_key
from app.core.config import settings
from app.core.exceptions import ImageProcessingError, OCRExtractionError
from app.models.ocr import OcrRequest, OcrResponse
from app.services.ocr import extract_product_fields
from app.utils.validators import validate_image_content_type, extract_content_type_from_data_url

router = APIRouter()

# Max image size in bytes (from settings, default 10MB)
MAX_IMAGE_SIZE_BYTES = int(settings.max_image_size_mb * 1024 * 1024)


def validate_image_request(request: OcrRequest) -> None:
    """Validate image request before processing."""
    # Check that at least one image source is provided
    if not request.has_image():
        raise HTTPException(
            status_code=400,
            detail="Either image_url or image_b64 is required",
        )
    
    # Validate base64 image if provided
    if request.image_b64:
        # Check for data URL format and validate content type
        if request.image_b64.startswith("data:"):
            content_type = extract_content_type_from_data_url(request.image_b64)
            if content_type and not validate_image_content_type(content_type):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported image type: {content_type}. Supported: jpeg, png, webp",
                )
        
        # Estimate base64 size (base64 is ~33% larger than binary)
        # Remove data URL prefix if present
        b64_data = request.image_b64
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        
        estimated_size = len(b64_data) * 3 // 4
        if estimated_size > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Maximum size: {settings.max_image_size_mb}MB",
            )


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
    
    - Extracts product name and brand
    - Extracts expiry date and manufacturing date
    - Extracts barcode/QR code
    - Returns confidence scores for each extracted field
    """
    # Validate request
    validate_image_request(request)

    try:
        return extract_product_fields(request)
    except ImageProcessingError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except OCRExtractionError as e:
        raise HTTPException(status_code=422, detail=e.message) from e
