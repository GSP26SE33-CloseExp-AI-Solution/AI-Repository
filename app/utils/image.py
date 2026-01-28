"""Utility functions for image processing."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Optional, Tuple

from app.core.exceptions import ImageProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)


def load_image_from_url(url: str, timeout: int = 30) -> Tuple[Any, bytes]:
    """
    Load image from URL.
    
    Args:
        url: Image URL
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (PIL Image, raw bytes)
    """
    try:
        import requests
        from PIL import Image
        from urllib.parse import urlparse
    except ImportError as e:
        raise ImageProcessingError("Required packages not installed") from e

    # Extract domain for Referer header
    parsed_url = urlparse(url)
    referer = f"{parsed_url.scheme}://{parsed_url.netloc}/"

    # Headers to mimic a browser request (many CDNs block requests without proper headers)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": referer,
        "Origin": referer.rstrip("/"),
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
    }

    try:
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        image_bytes = resp.content
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image, image_bytes
    except requests.RequestException as e:
        raise ImageProcessingError(f"Failed to fetch image: {e}") from e
    except Exception as e:
        raise ImageProcessingError(f"Failed to decode image: {e}") from e


def load_image_from_base64(b64_data: str) -> Tuple[Any, bytes]:
    """
    Load image from base64 string.
    
    Args:
        b64_data: Base64 encoded image (with or without data URL prefix)
        
    Returns:
        Tuple of (PIL Image, raw bytes)
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImageProcessingError("Pillow is required") from e

    try:
        # Remove data URL prefix if present
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]

        image_bytes = base64.b64decode(b64_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image, image_bytes
    except Exception as e:
        raise ImageProcessingError(f"Invalid base64 image: {e}") from e


def image_to_base64(image: Any, format: str = "PNG") -> str:
    """
    Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image
        format: Output format (PNG, JPEG, etc.)
        
    Returns:
        Base64 encoded string
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def resize_image(
    image: Any,
    max_size: int = 1024,
    maintain_aspect: bool = True,
) -> Any:
    """
    Resize image to fit within max_size.
    
    Args:
        image: PIL Image
        max_size: Maximum dimension (width or height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    width, height = image.size

    if width <= max_size and height <= max_size:
        return image

    if maintain_aspect:
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
    else:
        new_width = new_height = max_size

    return image.resize((new_width, new_height))


def validate_image_size(
    image_bytes: bytes,
    max_size_mb: float = 10.0,
) -> bool:
    """
    Validate image size.
    
    Args:
        image_bytes: Raw image bytes
        max_size_mb: Maximum size in megabytes
        
    Returns:
        True if valid, raises exception otherwise
    """
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ImageProcessingError(
            f"Image size ({size_mb:.2f}MB) exceeds maximum ({max_size_mb}MB)",
            details={"size_mb": size_mb, "max_size_mb": max_size_mb},
        )
    return True


def get_image_dimensions(image: Any) -> dict[str, int]:
    """Get image dimensions."""
    width, height = image.size
    return {"width": width, "height": height}
