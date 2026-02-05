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


class WeightInfo(BaseModel):
    """Product weight/volume information."""
    
    value: float
    unit: str
    raw: Optional[str] = None


class ManufacturerInfo(BaseModel):
    """Manufacturer and distributor information."""
    
    name: Optional[str] = None
    distributor: Optional[str] = None
    address: Optional[str] = None
    contact: Optional[List[str]] = None


class BarcodeInfo(BaseModel):
    """Barcode lookup information.
    
    Note: company and category fields will be populated by the Backend
    service using external APIs (Open Food Facts, UPCitemdb).
    The AI service only provides barcode origin detection via GS1 prefix.
    """
    
    barcode: str
    is_vietnamese: bool = False
    company: Optional[str] = None
    category: Optional[str] = None
    prefix: Optional[str] = None
    note: Optional[str] = None
    country: Optional[str] = None  # Country of origin from GS1 prefix


class CategoryInfo(BaseModel):
    """Detected product category."""
    
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    keywords_vi: Optional[List[str]] = None


class ProductCodesInfo(BaseModel):
    """Product identification codes."""
    
    sku: Optional[str] = None
    batch: Optional[str] = None
    msktvsty: Optional[str] = None


class ProductInfo(BaseModel):
    """Extracted product information from OCR - Enhanced for Vietnamese products."""

    # Basic info
    name: Optional[str] = None
    brand: Optional[str] = None
    barcode: Optional[str] = None
    barcode_info: Optional[BarcodeInfo] = None
    
    # Weight/Volume
    weight: Optional[str] = None
    weight_info: Optional[WeightInfo] = None
    
    # Ingredients and composition
    ingredients: Optional[List[str]] = None
    nutrition_facts: Optional[Dict[str, Any]] = None
    
    # Instructions
    storage_instructions: Optional[str] = None
    usage_instructions: Optional[str] = None
    
    # Manufacturer/Distributor
    manufacturer: Optional[ManufacturerInfo] = None
    origin: Optional[str] = None
    
    # Certifications and quality
    certifications: Optional[List[str]] = None
    quality_standards: Optional[List[str]] = None
    
    # Warnings and notes
    warnings: Optional[List[str]] = None
    
    # Product codes
    product_codes: Optional[ProductCodesInfo] = None
    
    # Shelf life
    shelf_life_days: Optional[int] = None
    
    # Category detection
    detected_category: Optional[CategoryInfo] = None


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
