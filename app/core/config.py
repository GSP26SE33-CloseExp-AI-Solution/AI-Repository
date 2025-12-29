import os
from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = Field(default=os.getenv("AI_APP_NAME", "AI Service - Pricing & OCR"))
    version: str = Field(default=os.getenv("AI_APP_VERSION", "0.1.0"))
    environment: str = Field(default=os.getenv("AI_ENVIRONMENT", "local"))
    pricing_default_confidence: float = Field(
        default=float(os.getenv("AI_PRICING_DEFAULT_CONFIDENCE", "0.75")),
        ge=0.0,
        le=1.0,
    )


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
