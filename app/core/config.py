import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    # App info
    app_name: str = Field(default="CloseExp AI Service")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)

    # API Security
    api_key: Optional[str] = Field(default=None)
    api_key_header: str = Field(default="X-API-Key")
    allowed_origins: List[str] = Field(default=["*"])

    # Model paths
    yolo_model_path: str = Field(default="yolo11n.pt")
    ocr_model_path: Optional[str] = Field(default=None)

    # OCR settings
    ocr_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    ocr_supported_languages: List[str] = Field(default=["vi", "en"])

    # Vision settings
    vision_min_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    vision_max_detections: int = Field(default=100, ge=1)

    # Pricing settings
    pricing_default_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    pricing_min_decay_factor: float = Field(default=0.3, ge=0.0, le=1.0)
    pricing_max_decay_factor: float = Field(default=0.9, ge=0.0, le=1.0)

    # Image constraints
    max_image_size_mb: float = Field(default=10.0, gt=0)
    allowed_image_types: List[str] = Field(
        default=["image/jpeg", "image/png", "image/webp"]
    )

    # Cache
    cache_ttl_seconds: int = Field(default=3600)
    enable_cache: bool = Field(default=True)

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()

    class Config:
        env_prefix = "AI_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings(
        app_name=os.getenv("AI_APP_NAME", "CloseExp AI Service"),
        version=os.getenv("AI_APP_VERSION", "1.0.0"),
        environment=os.getenv("AI_ENVIRONMENT", "development"),
        debug=os.getenv("AI_DEBUG", "false").lower() == "true",
        host=os.getenv("AI_HOST", "0.0.0.0"),
        port=int(os.getenv("AI_PORT", "8000")),
        api_key=os.getenv("AI_API_KEY"),
        yolo_model_path=os.getenv("AI_YOLO_MODEL_PATH", "yolo11n.pt"),
        ocr_model_path=os.getenv("AI_OCR_MODEL_PATH"),
        log_level=os.getenv("AI_LOG_LEVEL", "INFO"),
    )


settings = get_settings()
