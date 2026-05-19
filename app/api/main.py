from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import health, ocr, pricing, recommendation, tokens
from app.core.config import settings
from app.core.exceptions import AIServiceError
from app.core.api_key_middleware import ApiKeyMiddleware
from app.core.config import validate_production_settings
from app.core.logging import get_logger, setup_logging
from app.infra.model_store import model_store

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    validate_production_settings()

    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    logger.info(f"Environment: {settings.environment}")

    # Pre-load models
    try:
        model_store.load_yolo()
        logger.info("YOLO model pre-loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to pre-load YOLO model: {e}")

    yield

    # Shutdown
    logger.info("Shutting down application")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    setup_logging()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="AI Service for OCR extraction and price suggestion for near-expiry products",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        openapi_url="/openapi.json" if settings.environment != "production" else None,
        lifespan=lifespan,
    )

    # CORS: disabled in production (BE calls server-to-server; browsers must not access AI)
    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    app.add_middleware(ApiKeyMiddleware)

    # Exception handlers
    @app.exception_handler(AIServiceError)
    async def ai_service_error_handler(
        request: Request,
        exc: AIServiceError,
    ) -> JSONResponse:
        logger.error(
            f"AI Service Error: {exc.error_code} - {exc.message}",
            extra={"details": exc.details},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                },
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                },
            },
        )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(ocr.router, prefix="/v1/ocr", tags=["OCR"])
    app.include_router(pricing.router, prefix="/v1/pricing", tags=["Pricing"])
    app.include_router(recommendation.router, prefix="/v1/recommendation", tags=["Recommendation"])
    app.include_router(tokens.router, prefix="/v1/tokens", tags=["Tokens"])
    return app


app = create_app()
