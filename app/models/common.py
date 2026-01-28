from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ErrorDetail(BaseModel):
    """Standard error detail schema."""

    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""

    success: bool
    data: Optional[T] = None
    error: Optional[ErrorDetail] = None
    meta: Optional[Dict[str, Any]] = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response schema."""

    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int


class ImageInput(BaseModel):
    """Common image input schema."""

    model_config = ConfigDict(extra="forbid")

    image_url: Optional[str] = Field(
        default=None,
        description="Public URL of the image",
    )
    image_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded image data (with or without data URL prefix)",
    )

    def has_image(self) -> bool:
        """Check if either image source is provided."""
        return bool(self.image_url or self.image_b64)


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    x1: float = Field(description="Left coordinate")
    y1: float = Field(description="Top coordinate")
    x2: float = Field(description="Right coordinate")
    y2: float = Field(description="Bottom coordinate")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_xyxy(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]
