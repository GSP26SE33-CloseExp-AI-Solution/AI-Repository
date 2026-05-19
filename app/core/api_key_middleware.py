"""Enforce API key on all routes except public health checks (production only)."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from app.core.config import settings

# Only /health stays public for load balancers & platform health checks.
PUBLIC_PATHS = frozenset({"/health"})


class ApiKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        if settings.environment != "production":
            return await call_next(request)

        path = request.url.path.rstrip("/") or "/"
        if path in PUBLIC_PATHS:
            return await call_next(request)

        header_name = settings.api_key_header.lower()
        provided = request.headers.get(header_name) or request.headers.get(
            settings.api_key_header
        )

        if not provided:
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "error": {
                        "code": "MISSING_API_KEY",
                        "message": "API key is required",
                    },
                },
            )

        if provided != settings.api_key:
            return JSONResponse(
                status_code=403,
                content={
                    "success": False,
                    "error": {
                        "code": "INVALID_API_KEY",
                        "message": "Invalid API key",
                    },
                },
            )

        return await call_next(request)
