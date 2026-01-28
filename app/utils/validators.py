"""Custom validators for input validation."""

from __future__ import annotations

import re
from datetime import date
from typing import List, Optional


def validate_date_string(date_str: str) -> Optional[date]:
    """
    Validate and parse date string in various formats.
    
    Supported formats:
    - DD/MM/YYYY
    - DD-MM-YYYY
    - DD.MM.YYYY
    - YYYY/MM/DD
    - YYYY-MM-DD
    """
    patterns = [
        (r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$", "DMY"),
        (r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$", "YMD"),
    ]

    for pattern, format_type in patterns:
        match = re.match(pattern, date_str.strip())
        if match:
            groups = match.groups()
            try:
                if format_type == "DMY":
                    day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                else:
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                return date(year, month, day)
            except ValueError:
                continue

    return None


def validate_barcode(barcode: str) -> bool:
    """
    Validate barcode format.
    
    Supports:
    - EAN-13 (13 digits)
    - EAN-8 (8 digits)
    - UPC-A (12 digits)
    """
    if not barcode:
        return False

    # Remove any whitespace
    barcode = barcode.strip()

    # Check if numeric
    if not barcode.isdigit():
        return False

    # Check length
    valid_lengths = [8, 12, 13]
    return len(barcode) in valid_lengths


def validate_price(price: float, min_price: float = 0, max_price: float = 1e9) -> bool:
    """Validate price is within acceptable range."""
    return min_price < price <= max_price


def validate_confidence(confidence: float) -> bool:
    """Validate confidence score is between 0 and 1."""
    return 0.0 <= confidence <= 1.0


def sanitize_text(text: str, max_length: int = 1000) -> str:
    """
    Sanitize text input.
    
    - Remove control characters
    - Limit length
    - Strip whitespace
    """
    if not text:
        return ""

    # Remove control characters except newlines
    text = "".join(char for char in text if char.isprintable() or char == "\n")

    # Limit length
    if len(text) > max_length:
        text = text[:max_length]

    return text.strip()


def validate_image_content_type(content_type: str) -> bool:
    """Validate image content type."""
    valid_types = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/bmp",
    ]
    return content_type.lower() in valid_types


def extract_content_type_from_data_url(data_url: str) -> Optional[str]:
    """Extract content type from data URL."""
    if not data_url.startswith("data:"):
        return None

    try:
        # Format: data:image/png;base64,....
        header = data_url.split(",")[0]
        content_type = header.split(":")[1].split(";")[0]
        return content_type
    except (IndexError, ValueError):
        return None
