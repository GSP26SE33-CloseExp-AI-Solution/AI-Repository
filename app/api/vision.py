from fastapi import APIRouter, HTTPException, Response

from app.models.vision import VisionAnalyzeRequest, VisionAnalyzeResponse
from app.services.vision import analyze_product_image, analyze_product_image_png

router = APIRouter(tags=["vision"])


@router.post("/analyze", response_model=VisionAnalyzeResponse)
async def analyze(payload: VisionAnalyzeRequest) -> VisionAnalyzeResponse:
    try:
        return analyze_product_image(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/analyze.png")
async def analyze_png(payload: VisionAnalyzeRequest) -> Response:
    try:
        png_bytes = analyze_product_image_png(payload)
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
