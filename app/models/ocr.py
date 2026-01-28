from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.common import BoundingBox, ImageInput


class OCRLanguage(str, Enum):
    """Supported OCR languages."""

    VIETNAMESE = "vi"
    ENGLISH = "en"


class TextRegion(BaseModel):
    """Detected text region with bounding box."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox
    language: Optional[OCRLanguage] = None


class DateInfo(BaseModel):
    """Extracted date information."""

    value: Optional[date] = None
    raw_text: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    format_detected: Optional[str] = None


class ProductInfo(BaseModel):
    """Extracted product information from OCR."""

    name: Optional[str] = None
    brand: Optional[str] = None
    barcode: Optional[str] = None
    weight: Optional[str] = None
    ingredients: Optional[List[str]] = None
    nutrition_facts: Optional[Dict[str, str]] = None


class OcrRequest(ImageInput):
    """OCR extraction request schema."""

    languages: List[OCRLanguage] = Field(
        default=[OCRLanguage.VIETNAMESE, OCRLanguage.ENGLISH],
        description="Languages to detect",
    )
    extract_dates: bool = Field(
        default=True,
        description="Extract expiry and manufacturing dates",
    )
    extract_barcode: bool = Field(
        default=True,
        description="Extract barcode/QR code",
    )
    return_regions: bool = Field(
        default=False,
        description="Return individual text regions",
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )


class OcrResponse(BaseModel):
    """OCR extraction response schema."""

    # Core extracted fields
    expiry_date: Optional[DateInfo] = None
    manufactured_date: Optional[DateInfo] = None
    product_info: Optional[ProductInfo] = None

    # Legacy fields for backward compatibility
    name: Optional[str] = None
    brand: Optional[str] = None
    barcode: Optional[str] = None

    # Raw OCR output
    raw_text: Optional[str] = None
    text_regions: Optional[List[TextRegion]] = None

    # Metadata
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time_ms: Optional[float] = None
    warnings: Optional[List[str]] = None

    @field_validator("confidence", mode="before")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)
