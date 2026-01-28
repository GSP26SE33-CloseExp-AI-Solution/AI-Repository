from fastapi import APIRouter, Depends, HTTPException, Response

from app.api.deps import get_api_key
from app.core.exceptions import ImageProcessingError, ModelNotLoadedError
from app.models.vision import VisionAnalyzeRequest, VisionAnalyzeResponse
from app.services.vision import analyze_product_image, analyze_product_image_png

router = APIRouter()


@router.post(
    "/analyze",
    response_model=VisionAnalyzeResponse,
    summary="Analyze product image",
    description="Detect products in image using YOLO and assess quality",
)
async def analyze(
    request: VisionAnalyzeRequest,
    _: str = Depends(get_api_key),
) -> VisionAnalyzeResponse:
    """
    Analyze a product image for object detection and quality assessment.
    
    - Detects objects using YOLO model
    - Classifies products into categories
    - Assesses image quality
    - Optionally returns annotated image and crops
    """
    if not request.has_image():
        raise HTTPException(
            status_code=400,
            detail="Either image_url or image_b64 is required",
        )

    try:
        return analyze_product_image(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ImageProcessingError as e:
        raise HTTPException(status_code=422, detail=e.message) from e
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=e.message) from e


@router.post(
    "/analyze/annotated",
    summary="Get annotated image",
    description="Return annotated PNG image with bounding boxes",
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Annotated PNG image",
        }
    },
)
async def analyze_annotated(
    request: VisionAnalyzeRequest,
    _: str = Depends(get_api_key),
) -> Response:
    """
    Return annotated image with bounding boxes and labels.
    
    Useful for quick visualization without parsing JSON response.
    """
    if not request.has_image():
        raise HTTPException(
            status_code=400,
            detail="Either image_url or image_b64 is required",
        )

    try:
        png_bytes = analyze_product_image_png(request)
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ImageProcessingError as e:
        raise HTTPException(status_code=422, detail=e.message) from e
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=e.message) from e
