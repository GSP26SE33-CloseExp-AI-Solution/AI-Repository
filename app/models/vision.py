from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.common import BoundingBox, ImageInput


class ProductType(str, Enum):
    """Product type classification."""

    FRUIT = "fruit"
    VEGETABLE = "vegetable"
    MEAT = "meat"
    FISH = "fish"
    DAIRY = "dairy"
    BAKERY = "bakery"
    BEVERAGE = "beverage"
    PACKAGED = "packaged"
    UNKNOWN = "unknown"


class QualityLabel(str, Enum):
    """Quality assessment label."""

    GOOD = "good"
    OK = "ok"
    POOR = "poor"


class FreshnessLevel(str, Enum):
    """Freshness level for produce."""

    FRESH = "fresh"
    ACCEPTABLE = "acceptable"
    DECLINING = "declining"
    SPOILED = "spoiled"


class QualityAssessment(BaseModel):
    """Image quality assessment result."""

    label: QualityLabel
    score: float = Field(ge=0.0, le=1.0)
    metrics: Dict[str, float] = Field(
        description="Quality metrics (blur, brightness, contrast)",
    )
    reasons: List[str] = Field(
        default_factory=list,
        description="Reasons for quality assessment",
    )


class FreshnessAssessment(BaseModel):
    """Product freshness assessment (for produce)."""

    level: FreshnessLevel
    score: float = Field(ge=0.0, le=1.0)
    indicators: List[str] = Field(
        default_factory=list,
        description="Visual freshness indicators",
    )
    estimated_days_remaining: Optional[int] = None


class Detection(BaseModel):
    """Single object detection result."""

    index: int = Field(description="Detection index")
    class_name: str = Field(description="Detected class name")
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox

    # Enriched fields
    product_type: ProductType = ProductType.UNKNOWN
    quality: Optional[QualityAssessment] = None
    freshness: Optional[FreshnessAssessment] = None

    # Optional crop data
    crop_image_b64: Optional[str] = None
    crop_image_content_type: Optional[str] = None

    # Legacy field for backward compatibility
    @property
    def xyxy(self) -> List[float]:
        return self.bounding_box.to_xyxy()


class VisionAnalyzeRequest(ImageInput):
    """Vision analysis request schema."""

    model: Optional[str] = Field(
        default=None,
        description="YOLO model to use (e.g., 'yolo11n.pt')",
    )
    min_confidence: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence",
    )
    max_detections: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of detections to return",
    )
    return_crops: bool = Field(
        default=False,
        description="Return cropped images for each detection",
    )
    return_annotated_image: bool = Field(
        default=True,
        description="Return annotated image with bounding boxes",
    )
    assess_quality: bool = Field(
        default=True,
        description="Assess image quality",
    )
    assess_freshness: bool = Field(
        default=False,
        description="Assess product freshness (for produce)",
    )


class VisionAnalyzeResponse(BaseModel):
    """Vision analysis response schema."""

    # Detection results
    detections: List[Detection] = Field(default_factory=list)
    detection_count: int = Field(description="Number of detections")

    # Quality assessment
    image_quality: Optional[QualityAssessment] = None

    # Summary statistics
    class_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of each detected class",
    )
    product_type_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of each product type",
    )

    # Annotated image
    annotated_image_b64: Optional[str] = None
    annotated_image_content_type: Optional[str] = None

    # Metadata
    model: str = Field(description="Model used for detection")
    inference_time_ms: float = Field(description="Inference time in milliseconds")
    image_dimensions: Optional[Dict[str, int]] = None
