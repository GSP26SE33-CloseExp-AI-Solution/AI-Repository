from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter

from app.core.config import settings
from app.infra.model_store import model_store

router = APIRouter()


@router.get(
    "/health",
    summary="Health check",
    description="Basic health check endpoint",
)
async def health() -> Dict[str, Any]:
    """Basic health check - always returns ok if service is running."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/ready",
    summary="Readiness check",
    description="Check if service is ready to handle requests",
)
async def ready() -> Dict[str, Any]:
    """
    Readiness check - verifies models are loaded.
    
    Used by Kubernetes/Docker health checks.
    """
    checks: Dict[str, bool] = {}

    # Check YOLO model
    try:
        model_store.load_yolo()
        checks["yolo_model"] = True
    except Exception:
        checks["yolo_model"] = False

    all_ready = all(checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/info",
    summary="Service information",
    description="Get service metadata and configuration",
)
async def info() -> Dict[str, Any]:
    """Return service information and configuration."""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "endpoints": {
            "ocr": "/v1/ocr/extract",
            "pricing": "/v1/pricing/suggest",
        },
        "docs": "/docs" if settings.environment != "production" else None,
    }
