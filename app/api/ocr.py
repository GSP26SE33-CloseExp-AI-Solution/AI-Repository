from fastapi import APIRouter, HTTPException

from app.models.ocr import OcrRequest, OcrResponse
from app.services.ocr import extract_product_fields

router = APIRouter(tags=["ocr"])


@router.post("/extract", response_model=OcrResponse)
async def extract(request: OcrRequest) -> OcrResponse:
    if not request.image_url and not request.image_b64:
        raise HTTPException(status_code=400, detail="image_url or image_b64 is required")
    return extract_product_fields(request)
