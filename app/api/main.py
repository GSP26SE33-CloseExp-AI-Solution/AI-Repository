from fastapi import FastAPI

from app.api import health, ocr, pricing
from app.core.config import settings
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    setup_logging()
    application = FastAPI(
        title=settings.app_name,
        version=settings.version,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    application.include_router(health.router)
    application.include_router(ocr.router, prefix="/v1/ocr")
    application.include_router(pricing.router, prefix="/v1/pricing")
    return application


app = create_app()
