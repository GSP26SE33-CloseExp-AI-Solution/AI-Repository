from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.core.config import settings

api_key_header = APIKeyHeader(
    name=settings.api_key_header,
    auto_error=False,
)


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """
    Verify API key from request header.
    
    Returns None if API key is not configured (dev mode).
    Raises HTTPException if key is invalid.
    """
    # Skip validation if no API key is configured
    if not settings.api_key:
        return None

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error_code": "MISSING_API_KEY",
                "message": "API key is required",
            },
        )

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": "INVALID_API_KEY",
                "message": "Invalid API key",
            },
        )

    return api_key
