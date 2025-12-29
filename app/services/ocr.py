from datetime import date

from app.models.ocr import OcrRequest, OcrResponse


def extract_product_fields(request: OcrRequest) -> OcrResponse:
    # Placeholder OCR pipeline: replace with model inference and parsing
    return OcrResponse(
        expiry_date=date(2025, 3, 1),
        manufactured_date=date(2024, 3, 1),
        name="Sample Product",
        brand="Sample Brand",
        barcode="0000000000000",
        confidence=0.50,
    )
