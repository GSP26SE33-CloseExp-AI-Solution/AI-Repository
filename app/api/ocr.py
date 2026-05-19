from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field

from app.api.deps import get_api_key, get_user_id
from app.core.config import settings
from app.core.exceptions import ImageProcessingError, OCRExtractionError
from app.models.ocr import OcrRequest, OcrResponse
from app.services.ocr import extract_product_fields
from app.services.token_service import token_service
from app.utils.validators import validate_image_content_type, extract_content_type_from_data_url

router = APIRouter()

# Max image size in bytes (from settings, default 10MB)
MAX_IMAGE_SIZE_BYTES = int(settings.max_image_size_mb * 1024 * 1024)
MAX_IMAGES_PER_BATCH = 3


# ─── Single image schemas (unchanged) ────────────────────────────────────────

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
        b64_data = request.image_b64
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        
        estimated_size = len(b64_data) * 3 // 4
        if estimated_size > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Maximum size: {settings.max_image_size_mb}MB",
            )


# ─── Multi-image schemas ──────────────────────────────────────────────────────

class ImageItem(BaseModel):
    """Single image in a batch request."""
    image_url: Optional[str] = Field(
        default=None,
        description="Public URL of the image",
    )
    image_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded image data (with or without data URL prefix)",
    )
    label: Optional[str] = Field(
        default=None,
        description="Optional label for this image (e.g. 'front', 'back', 'barcode')",
    )


class MultiOcrRequest(BaseModel):
    """Batch OCR request – up to 3 images analyzed in parallel."""
    images: List[ImageItem] = Field(
        ...,
        min_length=1,
        max_length=MAX_IMAGES_PER_BATCH,
        description=f"List of images to analyze (max {MAX_IMAGES_PER_BATCH})",
    )
    extract_dates: bool = Field(default=True, description="Extract expiry and manufacturing dates")
    extract_barcode: bool = Field(default=True, description="Extract barcode/QR code")
    return_regions: bool = Field(default=False, description="Return individual text regions")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence")
    languages: List[str] = Field(default=["vi", "en"], description="Languages to detect")


class SingleOcrResultItem(BaseModel):
    """OCR result for a single image in a batch."""
    index: int
    label: Optional[str] = None
    success: bool
    error: Optional[str] = None
    result: Optional[OcrResponse] = None
    token_cost: int = 1


class MultiOcrResponse(BaseModel):
    """Response for batch OCR request."""
    total_images: int
    successful: int
    failed: int
    total_token_cost: int
    token_usage: dict
    results: List[SingleOcrResultItem]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/extract",
    response_model=OcrResponse,
    summary="Extract product information from a single image",
    description=(
        "Extract expiry date, manufacturing date, barcode and other product info using OCR. "
        "Consumes **1 token** per request. Monthly budget: **100 tokens**."
    ),
)
async def extract(
    request: OcrRequest,
    user_id: str = Depends(get_user_id),
    _: str = Depends(get_api_key),
) -> OcrResponse:
    """
    Extract product information from a single image.
    
    - Extracts product name and brand
    - Extracts expiry date and manufacturing date
    - Extracts barcode/QR code
    - Returns confidence scores for each extracted field
    
    **Token cost**: 1 token per call
    """
    # Validate request
    validate_image_request(request)

    # Check token budget
    cost = 1
    if not token_service.check_budget("ocr", user_id, cost):
        usage = token_service.get_usage("ocr", user_id)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "TOKEN_BUDGET_EXCEEDED",
                "message": (
                    f"Monthly OCR token budget exceeded. "
                    f"Budget: {usage['budget']}, Used: {usage['used']}, "
                    f"Remaining: {usage['remaining']}. Resets next month."
                ),
                "token_usage": usage,
            },
        )

    try:
        result = await extract_product_fields(request)
        # Consume token after successful extraction
        token_service.consume("ocr", user_id, cost)
        return result
    except ImageProcessingError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except OCRExtractionError as e:
        raise HTTPException(status_code=422, detail=e.message) from e


@router.post(
    "/extract-batch",
    response_model=MultiOcrResponse,
    summary="Extract product information from up to 3 images simultaneously",
    description=(
        "Analyze multiple images in one request (max 3). "
        "Each image consumes **1 token**, so a batch of 3 images costs **3 tokens**. "
        "Monthly OCR budget: **100 tokens**."
    ),
)
async def extract_batch(
    request: MultiOcrRequest,
    user_id: str = Depends(get_user_id),
    _: str = Depends(get_api_key),
) -> MultiOcrResponse:
    """
    Extract product information from multiple images (up to 3) in one call.

    **Token cost**: 1 token per image (e.g. 2 images = 2 tokens)
    
    Images are processed sequentially. If budget runs out mid-batch,
    remaining images will fail with budget exceeded error.
    """
    import asyncio

    image_count = len(request.images)
    if image_count < 1:
        raise HTTPException(status_code=400, detail="At least 1 image is required")
    if image_count > MAX_IMAGES_PER_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_IMAGES_PER_BATCH} images allowed per batch request",
        )

    total_cost = image_count

    # Check if we have enough budget for the entire batch
    if not token_service.check_budget("ocr", user_id, total_cost):
        usage = token_service.get_usage("ocr", user_id)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "TOKEN_BUDGET_EXCEEDED",
                "message": (
                    f"Insufficient OCR token budget for batch of {image_count} image(s). "
                    f"Budget: {usage['budget']}, Used: {usage['used']}, "
                    f"Remaining: {usage['remaining']} (need {total_cost}). Resets next month."
                ),
                "token_usage": usage,
                "images_requested": image_count,
                "tokens_needed": total_cost,
            },
        )

    results: List[SingleOcrResultItem] = []
    successful = 0
    failed = 0

    for idx, img_item in enumerate(request.images):
        # Build individual OcrRequest for each image
        ocr_req = OcrRequest(
            image_url=img_item.image_url,
            image_b64=img_item.image_b64,
            extract_dates=request.extract_dates,
            extract_barcode=request.extract_barcode,
            return_regions=request.return_regions,
            min_confidence=request.min_confidence,
        )

        # Basic validation
        if not ocr_req.has_image():
            results.append(SingleOcrResultItem(
                index=idx,
                label=img_item.label,
                success=False,
                error="Either image_url or image_b64 is required",
                token_cost=0,
            ))
            failed += 1
            continue

        try:
            result = await extract_product_fields(ocr_req)
            results.append(SingleOcrResultItem(
                index=idx,
                label=img_item.label,
                success=True,
                result=result,
                token_cost=1,
            ))
            successful += 1
        except (ImageProcessingError, OCRExtractionError) as e:
            results.append(SingleOcrResultItem(
                index=idx,
                label=img_item.label,
                success=False,
                error=e.message,
                token_cost=0,
            ))
            failed += 1
        except Exception as e:
            results.append(SingleOcrResultItem(
                index=idx,
                label=img_item.label,
                success=False,
                error=str(e),
                token_cost=0,
            ))
            failed += 1

    # Consume tokens only for successfully processed images
    actual_cost = successful
    if actual_cost > 0:
        token_service.consume("ocr", user_id, actual_cost)

    token_usage = token_service.get_usage("ocr", user_id)

    return MultiOcrResponse(
        total_images=image_count,
        successful=successful,
        failed=failed,
        total_token_cost=actual_cost,
        token_usage=token_usage,
        results=results,
    )


@router.get(
    "/token-status",
    summary="Get OCR token usage for current month",
    description="Quick access to OCR token budget and remaining usage.",
)
async def get_ocr_token_status(
    user_id: str = Depends(get_user_id),
    _: str = Depends(get_api_key),
):
    """Get current month OCR token usage."""
    return {
        "success": True,
        "data": token_service.get_usage("ocr", user_id),
    }
