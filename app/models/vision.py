from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


ProductType = Literal["vegetable", "fruit", "meat", "fish", "unknown"]


class VisionAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_url: Optional[HttpUrl] = Field(default=None, description="Public image URL")
    image_b64: Optional[str] = Field(default=None, description="Base64 encoded image data")

    model: Optional[str] = Field(
        default=None,
        description="Ultralytics YOLO weights name/path. If omitted, server picks a default (tries newest then fallbacks).",
    )

    # Demo knobs: quality scoring only (no OCR/pricing here)
    min_confidence: float = Field(default=0.25, ge=0.0, le=1.0)

    return_crops: bool = Field(
        default=True,
        description="If true, response includes per-object cropped images (PNG base64).",
    )
    max_crops: int = Field(default=10, ge=0, le=50)

    return_annotated_image: bool = Field(
        default=True,
        description="If true, response includes annotated_image_b64 (PNG).",
    )


class QualityAssessment(BaseModel):
    label: Literal["good", "ok", "poor", "unknown"]
    score: float = Field(ge=0.0, le=1.0)
    metrics: Dict[str, float] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)


class Detection(BaseModel):
    index: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    xyxy: List[float] = Field(description="[x1, y1, x2, y2] in pixels")
    product_type: ProductType = "unknown"
    quality: Optional[QualityAssessment] = None

    crop_image_content_type: Optional[str] = None
    crop_image_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded cropped image bytes (PNG) for this object.",
    )


class VisionAnalyzeResponse(BaseModel):
    model: str
    inference_ms: float = Field(ge=0.0)

    detections: List[Detection] = Field(default_factory=list)
    quality: QualityAssessment

    annotated_image_content_type: Optional[str] = None
    annotated_image_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded annotated image bytes (no data: prefix).",
    )
