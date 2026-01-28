from __future__ import annotations

import base64
import re
from datetime import date
from io import BytesIO
from typing import Any, List, Optional, Tuple

from app.core.config import settings
from app.core.exceptions import ImageProcessingError, OCRExtractionError
from app.core.logging import get_logger
from app.models.ocr import (
    DateInfo,
    OcrRequest,
    OcrResponse,
    OCRLanguage,
    ProductInfo,
    TextRegion,
)
from app.models.common import BoundingBox

logger = get_logger(__name__)


class OCRService:
    """Service for OCR extraction from product images."""

    # Vietnamese date patterns
    DATE_PATTERNS = [
        # DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
        (r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", "DMY"),
        # YYYY/MM/DD, YYYY-MM-DD
        (r"(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})", "YMD"),
        # Vietnamese: ngày DD tháng MM năm YYYY
        (r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})", "DMY_VI"),
        # NSX/HSD prefixes
        (r"(?:NSX|HSD|EXP|MFG)[:\s]*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", "DMY_PREFIX"),
    ]

    # Keywords for date type detection
    EXPIRY_KEYWORDS = ["hsd", "exp", "het han", "hết hạn", "best before", "use by"]
    MFG_KEYWORDS = ["nsx", "mfg", "san xuat", "sản xuất", "production"]

    def __init__(self) -> None:
        self._ocr_engine: Optional[Any] = None
        self._barcode_reader: Optional[Any] = None

    def _get_ocr_engine(self) -> Any:
        """Lazy load OCR engine (PaddleOCR or EasyOCR)."""
        if self._ocr_engine is None:
            try:
                from paddleocr import PaddleOCR  # type: ignore

                self._ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang="vi",
                    show_log=False,
                )
                logger.info("Loaded PaddleOCR engine")
            except ImportError:
                try:
                    import easyocr  # type: ignore

                    self._ocr_engine = easyocr.Reader(
                        ["vi", "en"],
                        gpu=False,
                    )
                    logger.info("Loaded EasyOCR engine")
                except ImportError:
                    logger.warning("No OCR engine available, using placeholder")
                    self._ocr_engine = "placeholder"
        return self._ocr_engine

    def _get_barcode_reader(self) -> Any:
        """Lazy load barcode reader."""
        if self._barcode_reader is None:
            try:
                from pyzbar import pyzbar  # type: ignore

                self._barcode_reader = pyzbar
                logger.info("Loaded pyzbar barcode reader")
            except ImportError:
                logger.warning("pyzbar not available for barcode reading")
                self._barcode_reader = "placeholder"
        return self._barcode_reader

    def _load_image(self, request: OcrRequest) -> Tuple[Any, bytes]:
        """Load image from URL or base64."""
        try:
            from PIL import Image  # type: ignore
        except ImportError as e:
            raise ImageProcessingError("Pillow is required for image processing") from e

        if request.image_url:
            try:
                import requests

                resp = requests.get(str(request.image_url), timeout=30)
                resp.raise_for_status()
                image_bytes = resp.content
            except Exception as e:
                raise ImageProcessingError(f"Failed to fetch image: {e}") from e
        elif request.image_b64:
            try:
                b64_data = request.image_b64
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]
                image_bytes = base64.b64decode(b64_data)
            except Exception as e:
                raise ImageProcessingError(f"Invalid base64 image: {e}") from e
        else:
            raise ImageProcessingError("No image provided")

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            return image, image_bytes
        except Exception as e:
            raise ImageProcessingError(f"Failed to decode image: {e}") from e

    def _extract_text(
        self,
        image: Any,
        languages: List[OCRLanguage],
    ) -> Tuple[str, List[TextRegion]]:
        """Extract text using OCR engine."""
        engine = self._get_ocr_engine()

        if engine == "placeholder":
            # Return placeholder for development
            return "Sample Product\nHSD: 01/03/2025\nNSX: 01/03/2024", []

        try:
            import numpy as np

            image_np = np.array(image)

            # PaddleOCR
            if hasattr(engine, "ocr"):
                result = engine.ocr(image_np, cls=True)
                if not result or not result[0]:
                    return "", []

                regions: List[TextRegion] = []
                texts: List[str] = []

                for line in result[0]:
                    if len(line) < 2:
                        continue
                    box, (text, conf) = line[0], line[1]
                    texts.append(text)

                    # Convert box points to bounding box
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]
                    regions.append(
                        TextRegion(
                            text=text,
                            confidence=float(conf),
                            bounding_box=BoundingBox(
                                x1=min(x_coords),
                                y1=min(y_coords),
                                x2=max(x_coords),
                                y2=max(y_coords),
                            ),
                        )
                    )

                return "\n".join(texts), regions

            # EasyOCR
            elif hasattr(engine, "readtext"):
                result = engine.readtext(image_np)
                regions = []
                texts = []

                for box, text, conf in result:
                    texts.append(text)
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]
                    regions.append(
                        TextRegion(
                            text=text,
                            confidence=float(conf),
                            bounding_box=BoundingBox(
                                x1=min(x_coords),
                                y1=min(y_coords),
                                x2=max(x_coords),
                                y2=max(y_coords),
                            ),
                        )
                    )

                return "\n".join(texts), regions

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise OCRExtractionError(f"OCR extraction failed: {e}") from e

        return "", []

    def _parse_date(self, text: str) -> Optional[date]:
        """Parse date from text using multiple patterns."""
        for pattern, format_type in self.DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                try:
                    if format_type in ("DMY", "DMY_VI", "DMY_PREFIX"):
                        day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                    else:  # YMD
                        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])

                    return date(year, month, day)
                except ValueError:
                    continue
        return None

    def _extract_dates(self, text: str) -> Tuple[Optional[DateInfo], Optional[DateInfo]]:
        """Extract expiry and manufacturing dates from text."""
        lines = text.lower().split("\n")
        expiry_date: Optional[DateInfo] = None
        mfg_date: Optional[DateInfo] = None

        for line in lines:
            parsed_date = self._parse_date(line)
            if parsed_date is None:
                continue

            # Determine date type based on keywords
            is_expiry = any(kw in line for kw in self.EXPIRY_KEYWORDS)
            is_mfg = any(kw in line for kw in self.MFG_KEYWORDS)

            if is_expiry and expiry_date is None:
                expiry_date = DateInfo(
                    value=parsed_date,
                    raw_text=line.strip(),
                    confidence=0.85,
                )
            elif is_mfg and mfg_date is None:
                mfg_date = DateInfo(
                    value=parsed_date,
                    raw_text=line.strip(),
                    confidence=0.85,
                )
            elif expiry_date is None:
                # Assume first unclassified date is expiry
                expiry_date = DateInfo(
                    value=parsed_date,
                    raw_text=line.strip(),
                    confidence=0.6,
                )

        return expiry_date, mfg_date

    def _extract_barcode(self, image_bytes: bytes) -> Optional[str]:
        """Extract barcode from image."""
        reader = self._get_barcode_reader()
        if reader == "placeholder":
            return None

        try:
            from PIL import Image

            image = Image.open(BytesIO(image_bytes))
            barcodes = reader.decode(image)
            if barcodes:
                return barcodes[0].data.decode("utf-8")
        except Exception as e:
            logger.warning(f"Barcode extraction failed: {e}")

        return None

    def extract(self, request: OcrRequest) -> OcrResponse:
        """
        Extract product information from image.
        
        Args:
            request: OCR request with image data
            
        Returns:
            Extracted product information
        """
        import time

        start_time = time.perf_counter()
        warnings: List[str] = []

        # Load image
        image, image_bytes = self._load_image(request)

        # Extract text
        raw_text, text_regions = self._extract_text(image, request.languages)

        if not raw_text.strip():
            warnings.append("No text detected in image")

        # Extract dates
        expiry_date, mfg_date = None, None
        if request.extract_dates:
            expiry_date, mfg_date = self._extract_dates(raw_text)

        # Extract barcode
        barcode = None
        if request.extract_barcode:
            barcode = self._extract_barcode(image_bytes)

        # Build product info
        product_info = ProductInfo(
            barcode=barcode,
        )

        # Calculate overall confidence
        confidences = []
        if expiry_date and expiry_date.confidence:
            confidences.append(expiry_date.confidence)
        if mfg_date and mfg_date.confidence:
            confidences.append(mfg_date.confidence)
        if text_regions:
            confidences.extend([r.confidence for r in text_regions])

        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        processing_time = (time.perf_counter() - start_time) * 1000

        return OcrResponse(
            expiry_date=expiry_date,
            manufactured_date=mfg_date,
            product_info=product_info,
            barcode=barcode,
            raw_text=raw_text if request.return_regions else None,
            text_regions=text_regions if request.return_regions else None,
            confidence=overall_confidence,
            processing_time_ms=round(processing_time, 2),
            warnings=warnings if warnings else None,
        )


# Singleton instance
ocr_service = OCRService()


def extract_product_fields(request: OcrRequest) -> OcrResponse:
    """Extract product fields from image (backward compatible function)."""
    return ocr_service.extract(request)
