from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class OcrRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_url: Optional[HttpUrl] = Field(default=None, description="Public image URL")
    image_b64: Optional[str] = Field(default=None, description="Base64 encoded image data")


class OcrResponse(BaseModel):
    expiry_date: Optional[date] = None
    manufactured_date: Optional[date] = None
    name: Optional[str] = None
    brand: Optional[str] = None
    barcode: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
