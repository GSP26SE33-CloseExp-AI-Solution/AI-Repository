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
    allowed_origins: List[str] = Field(default_factory=list)

    # LLM provider: auto (GGUF if configured, else Gemini) | gemini | gguf
    llm_provider: str = Field(default="auto")

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
    pricing_auto_crawl_enabled: bool = Field(default=False)
    pricing_auto_crawl_deep: bool = Field(default=False)

    # Image constraints
    max_image_size_mb: float = Field(default=10.0, gt=0)
    allowed_image_types: List[str] = Field(
        default=["image/jpeg", "image/png", "image/webp"]
    )

    # Gemini LLM (API fallback after local GGUF)
    gemini_api_key: Optional[str] = Field(default=None)
    gemini_model: str = Field(default="gemini-3.1-flash-lite-preview")

    # Local GGUF (llama.cpp via llama-cpp-python) — Path A: Q4_0 / Gemma-class
    llm_gguf_path: Optional[str] = Field(
        default=None,
        description="Absolute or relative path to .gguf (e.g. gemma-3-4b-it-Q4_0.gguf)",
    )
    llm_ctx_size: int = Field(default=8192, ge=512)
    llm_gpu_layers: int = Field(default=0, ge=0)
    llm_cpu_threads: Optional[int] = Field(default=None, ge=1)
    llm_max_tokens: int = Field(default=2048, ge=64)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_chat_format: Optional[str] = Field(
        default=None,
        description="Optional llama-cpp chat_format, e.g. gemma, gemma-2",
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

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        allowed = {"auto", "gemini", "gguf"}
        normalized = v.lower().strip()
        if normalized not in allowed:
            raise ValueError(f"llm_provider must be one of {allowed}")
        return normalized

    class Config:
        env_prefix = "AI_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    environment = os.getenv("AI_ENVIRONMENT", "development").lower()

    def _parse_origins(raw: Optional[str]) -> List[str]:
        if raw is None:
            return [] if environment == "production" else ["*"]
        stripped = raw.strip()
        if not stripped:
            return []
        return [part.strip() for part in stripped.split(",") if part.strip()]

    return Settings(
        app_name=os.getenv("AI_APP_NAME", "CloseExp AI Service"),
        version=os.getenv("AI_APP_VERSION", "1.0.0"),
        environment=environment,
        debug=os.getenv("AI_DEBUG", "false").lower() == "true",
        host=os.getenv("AI_HOST", "0.0.0.0"),
        port=int(os.getenv("AI_PORT", "8000")),
        workers=int(os.getenv("AI_WORKERS", "1")),
        api_key=os.getenv("AI_API_KEY"),
        api_key_header=os.getenv("AI_API_KEY_HEADER", "X-API-Key"),
        allowed_origins=_parse_origins(os.getenv("AI_ALLOWED_ORIGINS")),
        llm_provider=os.getenv("AI_LLM_PROVIDER", "auto"),
        yolo_model_path=os.getenv(
            "AI_YOLO_MODEL_PATH",
            "/app/models/yolo11n.pt" if environment == "production" else "yolo11n.pt",
        ),
        ocr_model_path=os.getenv("AI_OCR_MODEL_PATH"),
        gemini_api_key=os.getenv("AI_GEMINI_API_KEY"),
        gemini_model=os.getenv("AI_GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
        llm_gguf_path=os.getenv("AI_LLM_GGUF_PATH") or None,
        llm_ctx_size=int(os.getenv("AI_LLM_CTX_SIZE", "8192")),
        llm_gpu_layers=int(os.getenv("AI_LLM_GPU_LAYERS", "0")),
        llm_cpu_threads=(
            int(t)
            if (t := os.getenv("AI_LLM_CPU_THREADS", "").strip())
            else None
        ),
        llm_max_tokens=int(os.getenv("AI_LLM_MAX_TOKENS", "2048")),
        llm_temperature=float(os.getenv("AI_LLM_TEMPERATURE", "0.1")),
        llm_chat_format=os.getenv("AI_LLM_CHAT_FORMAT") or None,
        pricing_auto_crawl_enabled=os.getenv("AI_PRICING_AUTO_CRAWL_ENABLED", "false").lower() == "true",
        pricing_auto_crawl_deep=os.getenv("AI_PRICING_AUTO_CRAWL_DEEP", "false").lower() == "true",
        log_level=os.getenv("AI_LOG_LEVEL", "INFO"),
        log_format=os.getenv("AI_LOG_FORMAT", "json"),
    )


def validate_production_settings() -> None:
    """Fail fast when production is misconfigured."""
    if settings.environment != "production":
        return

    if not settings.api_key or len(settings.api_key) < 32:
        raise RuntimeError(
            "AI_API_KEY must be set to a random string of at least 32 characters in production."
        )

    if settings.llm_provider in {"auto", "gemini"} and not settings.gemini_api_key:
        raise RuntimeError(
            "AI_GEMINI_API_KEY is required in production when AI_LLM_PROVIDER is gemini or auto."
        )

    if settings.allowed_origins and "*" in settings.allowed_origins:
        raise RuntimeError(
            "AI_ALLOWED_ORIGINS must not include '*' in production (server-to-server only)."
        )


settings = get_settings()
