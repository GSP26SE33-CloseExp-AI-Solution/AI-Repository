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
    BarcodeInfo,
    CategoryInfo,
    DateInfo,
    ManufacturerInfo,
    OcrRequest,
    OcrResponse,
    OCRLanguage,
    ProductInfo,
    TextRegion,
    WeightInfo,
)
from app.models.common import BoundingBox
from app.services.vietnamese_product import vn_product_service
from app.services.text_postprocessor import text_postprocessor
from app.services.region_based_extractor import region_extractor
from app.services.llm_postprocessor import llm_postprocessor
from app.infra.model_store import model_store

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
        self._barcode_reader: Optional[Any] = None

    def _get_ocr_engine(self) -> Any:
        """Return shared OCR engine from model_store (pre-loaded at startup)."""
        engine = model_store.load_ocr()
        if engine is None:
            logger.warning("No OCR engine available, using placeholder")
            return "placeholder"
        return engine

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

    def _extract_product_name_and_brand(self, raw_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract product name and brand from OCR text.
        
        Uses heuristics to identify brand and product name from text lines.
        Typically brand appears first or is all uppercase, product name follows.
        """
        if not raw_text.strip():
            return None, None

        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        
        # Filter out date-related lines
        non_date_lines = []
        date_keywords = ["hsd", "nsx", "exp", "mfg", "ngày", "tháng", "năm"]
        for line in lines:
            line_lower = line.lower()
            if not any(kw in line_lower for kw in date_keywords):
                # Also filter lines that are just numbers (likely barcodes or dates)
                if not line.replace("/", "").replace("-", "").replace(".", "").isdigit():
                    non_date_lines.append(line)

        if not non_date_lines:
            return None, None

        # Common Vietnamese brand patterns (often all caps or well-known names)
        known_brands = [
            # Dairy
            "th true milk", "vinamilk", "dutch lady", "nutifood", "mộc châu", "ba vì",
            # Nestle products
            "nestle", "nestlé", "milo", "nescafe", "nescafé", "maggi", "la vie",
            # Beverages
            "coca-cola", "coca cola", "pepsi", "7up", "fanta", "sprite", 
            "aquafina", "dasani", "lavie", "trà xanh 0 độ", "c2", "sting",
            "number one", "dr thanh", "tân hiệp phát", "trà thảo mộc",
            # Masan Group
            "chinsu", "nam ngư", "omachi", "kokomi", "tam thái tử", "heo cao bồi",
            # Instant noodles
            "acecook", "hảo hảo", "vifon", "miliket", "gấu đỏ",
            # Food brands
            "kinh đô", "bibica", "hữu nghị", "vissan", "cầu tre", "cholimex",
            "aji-no-moto", "ajinomoto", "knorr", "bột canh", "hạt nêm",
            # Snacks
            "orion", "bánh pía", "bánh tráng", "oishi", "poca",
            # Others
            "unilever", "p&g", "colgate", "omo", "comfort", "sunlight",
        ]

        brand: Optional[str] = None
        name: Optional[str] = None

        # Try to find brand
        for line in non_date_lines:
            line_lower = line.lower()
            for known_brand in known_brands:
                if known_brand in line_lower:
                    brand = line
                    break
            if brand:
                break

        # If brand not found in known list, check for all-caps line (often brand)
        if not brand:
            for line in non_date_lines:
                # All caps and reasonable length suggests brand name
                if line.isupper() and 3 <= len(line) <= 30:
                    brand = line.title()  # Convert to title case
                    break

        # Product name is typically the longest descriptive line
        name_candidates = [
            line for line in non_date_lines 
            if line != brand and len(line) > 5
        ]
        
        if name_candidates:
            # Prefer line with Vietnamese descriptive words
            descriptive_keywords = [
                # Dairy
                "sữa", "sữa tươi", "sữa chua", "yaourt", "phô mai", "bơ", "kem",
                # Beverages
                "nước", "nước ngọt", "nước suối", "nước khoáng", "trà", "cà phê", 
                "nước ép", "sinh tố", "nước tăng lực", "nước yến",
                # Food
                "bánh", "mì", "mì ăn liền", "phở", "bún", "miến", "gạo", "cháo",
                # Meat & Seafood
                "thịt", "thịt heo", "thịt bò", "thịt gà", "giò", "chả", "xúc xích",
                "cá", "tôm", "mực", "cá viên", "chả cá",
                # Vegetables & Fruits
                "rau", "củ", "quả", "trái cây", "cam", "táo", "chuối", "xoài",
                # Condiments
                "nước mắm", "nước tương", "tương ớt", "tương cà", "dầu ăn",
                "muối", "đường", "bột ngọt", "hạt nêm", "gia vị",
                # Snacks
                "snack", "kẹo", "bánh quy", "khô", "mứt", "hạt"
            ]
            
            for candidate in name_candidates:
                if any(kw in candidate.lower() for kw in descriptive_keywords):
                    name = candidate
                    break
            
            # Fallback to first non-brand line
            if not name and name_candidates:
                name = name_candidates[0]

        return name, brand

    async def extract(self, request: OcrRequest) -> OcrResponse:
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
            
        # Apply region-based filtering and extraction for better accuracy
        region_based_info = None
        if text_regions:
            try:
                # Convert text_regions to dict format for region_extractor
                regions_data = [
                    {
                        "text": r.text,
                        "confidence": r.confidence,
                        "bounding_box": r.bounding_box.model_dump() if r.bounding_box else None,
                    }
                    for r in text_regions
                ]
                region_based_info = region_extractor.extract_from_regions(regions_data, raw_text)
                logger.debug(f"Region-based extraction: {region_based_info}")
            except Exception as e:
                logger.warning(f"Region-based extraction failed: {e}")

        # Extract dates
        expiry_date, mfg_date = None, None
        if request.extract_dates:
            expiry_date, mfg_date = self._extract_dates(raw_text)

        # Extract barcode
        barcode = None
        barcode_info = None
        if request.extract_barcode:
            barcode = self._extract_barcode(image_bytes)
            if barcode:
                # Get barcode origin information (country detection)
                # Note: Full product details are handled by Backend API
                barcode_lookup = vn_product_service.lookup_barcode(barcode)
                if barcode_lookup:
                    barcode_info = BarcodeInfo(
                        barcode=barcode,
                        is_vietnamese=barcode_lookup.get("is_vietnamese", False),
                        company=None,  # Will be populated by Backend API
                        category=None,  # Will be populated by Backend API  
                        prefix=barcode_lookup.get("gs1_prefix"),
                        note=barcode_lookup.get("note"),
                        country=barcode_lookup.get("country"),
                    )

        # Extract product name and brand from text
        extracted_name, extracted_brand = self._extract_product_name_and_brand(raw_text)
        
        # Enhance with region-based extraction results (higher confidence)
        if region_based_info:
            # Use region-based name if available and better
            if region_based_info.name and (
                not extracted_name or 
                len(region_based_info.name) > len(extracted_name) or
                region_based_info.name_confidence > 0.7
            ):
                extracted_name = region_based_info.name
                logger.debug(f"Using region-based name: {extracted_name}")
            
            # Use region-based brand if available
            if region_based_info.brand and (
                not extracted_brand or 
                region_based_info.brand_confidence > 0.8
            ):
                extracted_brand = region_based_info.brand
                logger.debug(f"Using region-based brand: {extracted_brand}")
        
        # If barcode lookup found company, use it as brand if not extracted
        if barcode_info and barcode_info.company and not extracted_brand:
            extracted_brand = barcode_info.company

        # Extract detailed packaging information using Vietnamese Product Service
        packaging_info = vn_product_service.extract_all_packaging_info(raw_text)
        
        # Build weight info
        weight_info = None
        weight_str = None
        if packaging_info.get("weight"):
            w = packaging_info["weight"]
            weight_info = WeightInfo(
                value=w["value"],
                unit=w["unit"],
                raw=w.get("raw"),
            )
            weight_str = f"{w['value']} {w['unit']}"
        
        # Build manufacturer info (enhanced with distributor and contact)
        manufacturer_info = None
        if packaging_info.get("manufacturer"):
            m = packaging_info["manufacturer"]
            manufacturer_info = ManufacturerInfo(
                name=m.get("name"),
                distributor=m.get("distributor"),
                address=m.get("address"),
                contact=m.get("contact"),
            )
        
        # Build category info
        category_info = None
        if packaging_info.get("detected_category"):
            c = packaging_info["detected_category"]
            category_info = CategoryInfo(
                name=c["name"],
                confidence=c["confidence"],
                keywords_vi=c.get("keywords_vi"),
            )
        
        # Override with region-based category if higher confidence
        if region_based_info and region_based_info.detected_category:
            region_category = region_based_info.detected_category
            if not category_info or region_category.get("confidence", 0) > category_info.confidence:
                category_info = CategoryInfo(
                    name=region_category.get("name", ""),
                    confidence=region_category.get("confidence", 0.0),
                    keywords_vi=region_category.get("keywords_vi"),
                )
                logger.debug(f"Using region-based category: {category_info.name}")
        
        # Merge ingredients from region-based extraction
        merged_ingredients = packaging_info.get("ingredients")
        if region_based_info and region_based_info.ingredients:
            if not merged_ingredients or len(region_based_info.ingredients) > len(merged_ingredients):
                merged_ingredients = region_based_info.ingredients
                logger.debug(f"Using region-based ingredients")
        
        # Merge storage instructions
        merged_storage = packaging_info.get("storage")
        if region_based_info and region_based_info.storage_instructions:
            if not merged_storage or len(region_based_info.storage_instructions) > len(merged_storage):
                merged_storage = region_based_info.storage_instructions
                logger.debug(f"Using region-based storage instructions")
        
        # Merge usage instructions
        merged_usage = packaging_info.get("usage")
        if region_based_info and region_based_info.usage_instructions:
            if not merged_usage:
                merged_usage = region_based_info.usage_instructions
                logger.debug(f"Using region-based usage instructions")
        
        # Merge warnings
        merged_warnings = packaging_info.get("warnings")
        if region_based_info and region_based_info.warnings:
            if not merged_warnings:
                merged_warnings = region_based_info.warnings
                logger.debug(f"Using region-based warnings")
        
        # Build product codes info
        from app.models.ocr import ProductCodesInfo
        product_codes_info = None
        if packaging_info.get("product_codes"):
            pc = packaging_info["product_codes"]
            product_codes_info = ProductCodesInfo(
                sku=pc.get("sku"),
                batch=pc.get("batch"),
                msktvsty=pc.get("msktvsty"),
            )

        # Build product info with all enhanced fields
        product_info = ProductInfo(
            # Basic info
            name=extracted_name,
            brand=extracted_brand,
            barcode=barcode,
            barcode_info=barcode_info,
            # Weight
            weight=weight_str,
            weight_info=weight_info,
            # Ingredients and nutrition (use merged from region-based extraction)
            ingredients=self._normalize_string_list(merged_ingredients),
            nutrition_facts=packaging_info.get("nutrition") or None,
            # Instructions (use merged from region-based extraction)
            storage_instructions=merged_storage,
            usage_instructions=merged_usage,
            # Manufacturer/Distributor
            manufacturer=manufacturer_info,
            origin=packaging_info.get("origin"),
            # Certifications and quality
            certifications=self._normalize_string_list(packaging_info.get("certifications")),
            quality_standards=self._normalize_string_list(packaging_info.get("quality_standards")),
            # Warnings (use merged from region-based extraction)
            warnings=self._normalize_string_list(merged_warnings),
            # Product codes
            product_codes=product_codes_info,
            # Shelf life
            shelf_life_days=packaging_info.get("shelf_life_days"),
            # Category
            detected_category=category_info,
        )

        # Calculate overall confidence
        confidences = []
        if expiry_date and expiry_date.confidence:
            confidences.append(expiry_date.confidence)
        if mfg_date and mfg_date.confidence:
            confidences.append(mfg_date.confidence)
        if text_regions:
            confidences.extend([r.confidence for r in text_regions])
        if category_info:
            confidences.append(category_info.confidence)

        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        processing_time = (time.perf_counter() - start_time) * 1000

        # Build initial response
        initial_response = OcrResponse(
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

        # Post-processing: local GGUF → Gemini → rule-based (inside llm_postprocessor)
        try:
            response_dict = initial_response.dict()
            # Important: LLM post-processor (and JSONL data collection) needs raw OCR text
            # and detected text regions. Backend callers may set `return_regions=false`
            # to avoid returning these fields in the API response, but we still want
            # them available for prompt building + training pair export.
            response_dict["raw_text"] = raw_text
            response_dict["text_regions"] = text_regions
            try:
                processed_dict = await llm_postprocessor.process_ocr_response(response_dict)
            except Exception as llm_err:
                logger.warning(
                    "Post-processor raised, falling back to rule-based: %s", llm_err
                )
                processed_dict = text_postprocessor.process_ocr_response(response_dict)
            
            # Rebuild response with processed data
            processed_product_info = processed_dict.get("product_info", {})
            
            # Rebuild ProductInfo with corrected data
            corrected_product_info = ProductInfo(
                name=processed_product_info.get("name"),
                brand=processed_product_info.get("brand"),
                barcode=processed_product_info.get("barcode"),
                barcode_info=product_info.barcode_info,  # Keep original barcode info object
                weight=processed_product_info.get("weight"),
                weight_info=product_info.weight_info,  # Keep original weight info object
                ingredients=self._normalize_string_list(processed_product_info.get("ingredients")),
                nutrition_facts=processed_product_info.get("nutrition_facts"),
                storage_instructions=processed_product_info.get("storage_instructions"),
                usage_instructions=processed_product_info.get("usage_instructions"),
                manufacturer=product_info.manufacturer,  # Keep original manufacturer object
                origin=processed_product_info.get("origin"),
                certifications=self._normalize_string_list(processed_product_info.get("certifications")),
                quality_standards=self._normalize_string_list(processed_product_info.get("quality_standards")),
                warnings=self._normalize_string_list(processed_product_info.get("warnings")),
                product_codes=product_info.product_codes,  # Keep original product codes object
                shelf_life_days=processed_product_info.get("shelf_life_days"),
                detected_category=self._rebuild_category_info(processed_product_info.get("detected_category")),
            )
            
            # Return processed response
            return OcrResponse(
                expiry_date=expiry_date,
                manufactured_date=mfg_date,
                product_info=corrected_product_info,
                name=corrected_product_info.name,  # Also set top-level name
                brand=corrected_product_info.brand,  # Also set top-level brand
                barcode=barcode,
                raw_text=processed_dict.get("raw_text") if request.return_regions else None,
                text_regions=text_regions if request.return_regions else None,
                confidence=overall_confidence,
                processing_time_ms=round(processing_time, 2),
                warnings=warnings if warnings else None,
            )
        except Exception as e:
            logger.warning(f"Post-processing failed, returning original response: {e}")
            return initial_response

    @staticmethod
    def _normalize_string_list(value: Any) -> Optional[List[str]]:
        """Normalize string/list fields into clean list[str] for ProductInfo."""
        if value is None:
            return None

        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            return cleaned or None

        if isinstance(value, str):
            text = value.strip()
            return [text] if text else None

        text = str(value).strip()
        return [text] if text else None

    def _rebuild_category_info(self, category_dict: Optional[dict]) -> Optional[CategoryInfo]:
        """Rebuild CategoryInfo from dictionary."""
        if not category_dict:
            return None
        return CategoryInfo(
            name=category_dict.get("name", ""),
            confidence=category_dict.get("confidence", 0.0),
            keywords_vi=category_dict.get("keywords_vi"),
        )


# Singleton instance
ocr_service = OCRService()


async def extract_product_fields(request: OcrRequest) -> OcrResponse:
    """Extract product fields from image (backward compatible function)."""
    return await ocr_service.extract(request)
