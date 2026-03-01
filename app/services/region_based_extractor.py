"""
Region-based OCR Text Extractor.

This module extracts and classifies text from OCR text_regions
using confidence-based filtering and pattern matching.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from app.core.logging import get_logger
from app.services.text_postprocessor import text_postprocessor

logger = get_logger(__name__)


class FieldType(Enum):
    """Field types for product information."""
    NAME = "name"
    BRAND = "brand"
    INGREDIENTS = "ingredients"
    STORAGE = "storage_instructions"
    USAGE = "usage_instructions"
    WARNINGS = "warnings"
    WEIGHT = "weight"
    QUALITY = "quality_standards"
    MANUFACTURER = "manufacturer"
    NOISE = "noise"  # Text to be discarded
    UNKNOWN = "unknown"


@dataclass
class TextRegion:
    """Represents a text region from OCR."""
    text: str
    confidence: float
    x1: float = 0
    y1: float = 0
    x2: float = 0
    y2: float = 0
    
    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    @property
    def center_y(self) -> float:
        """Get vertical center position."""
        return (self.y1 + self.y2) / 2


@dataclass
class ClassifiedRegion:
    """A text region with its classified field type."""
    region: TextRegion
    field_type: FieldType
    classification_confidence: float
    corrected_text: str


@dataclass
class ExtractedProductInfo:
    """Extracted and validated product information."""
    name: Optional[str] = None
    brand: Optional[str] = None
    ingredients: Optional[str] = None
    storage_instructions: Optional[str] = None
    usage_instructions: Optional[str] = None
    warnings: Optional[str] = None
    weight: Optional[str] = None
    weight_info: Optional[Dict[str, Any]] = None
    net_weight: Optional[str] = None  # Net weight string like "150g"
    quality_standards: Optional[str] = None
    manufacturer: Optional[str] = None
    detected_category: Optional[Dict[str, Any]] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def name_confidence(self) -> float:
        """Get confidence for name extraction."""
        return self.confidence_scores.get("name", 0.0)
    
    @property
    def brand_confidence(self) -> float:
        """Get confidence for brand extraction."""
        return self.confidence_scores.get("brand", 0.0)


class RegionBasedExtractor:
    """
    Extract product information from OCR text regions.
    
    Uses confidence-based filtering and pattern matching to:
    1. Filter out low-confidence noise
    2. Classify each region into appropriate fields
    3. Merge and clean up related regions
    4. Build structured product information
    """
    
    # Minimum confidence thresholds for different purposes
    MIN_CONFIDENCE_CRITICAL = 0.7  # Brand, product name
    MIN_CONFIDENCE_NORMAL = 0.4    # Ingredients, instructions
    MIN_CONFIDENCE_NOISE = 0.15   # Below this is definitely noise
    
    # Field identification patterns
    FIELD_PATTERNS = {
        FieldType.INGREDIENTS: {
            "prefixes": ["thành phần", "thành phẩn", "nguyên liệu", "ingredients"],
            "keywords": ["nước", "muối", "đường", "bột", "dầu", "gia vị", "chất", "hành", "tỏi", "gừng", "sả", "tiêu", "ớt"],
            "patterns": [r"\(\d+[i,]*\)", r"\d+\s*%", r"\(\d{3}\)"],  # E-numbers, percentages
        },
        FieldType.STORAGE: {
            "prefixes": ["bảo quản", "hướng dẫn bảo quản", "storage"],
            "keywords": ["thoáng mát", "nhiệt độ", "tủ lạnh", "đông lạnh", "nơi khô"],
        },
        FieldType.USAGE: {
            "prefixes": ["hướng dẫn sử dụng", "cách dùng", "cách sử dụng", "hướng dẫn sử dung", "hdsd"],
            "keywords": ["làm nóng", "dùng ngay", "chế biến", "pha chế", "dùng trực tiếp", "hâm nóng", "rã đông"],
        },
        FieldType.MANUFACTURER: {
            "prefixes": ["sx bởi", "sản xuất bởi", "nsx bởi", "công ty", "manufacturer"],
            "keywords": ["ctcp", "tnhh", "công ty"],
        },
        FieldType.WARNINGS: {
            "prefixes": ["cảnh báo", "thông tin cảnh báo", "lưu ý", "warning"],
            "keywords": ["không làm", "tránh", "cẩn thận", "không dùng cho", "lò vi sóng", "vi song", "trong lò"],
        },
        FieldType.QUALITY: {
            "prefixes": ["chỉ tiêu", "chất lượng", "tiêu chuẩn"],
            "keywords": ["mg/100", "kcal", "protein", "chất béo"],
        },
        FieldType.WEIGHT: {
            "patterns": [r"(\d+)\s*(g|kg|ml|l)\b", r"tịnh[:\s]*\d+", r"net\s*w"],
        },
    }
    
    # Date patterns to identify date-related regions (NOT part of product name)
    DATE_PATTERNS = [
        r"nsx[:\s]",                    # NSX: (ngày sản xuất)
        r"hsd[:\s]",                    # HSD: (hạn sử dụng)
        r"ngày\s*(sản\s*xuất|sx)",      # Ngày sản xuất
        r"hạn\s*(sử\s*dụng|sd)",        # Hạn sử dụng
        r"exp[:\s]",                    # EXP:
        r"mfg[:\s]",                    # MFG:
        r"date[:\s]",                   # Date:
        r"best\s*before",               # Best before
        r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}",  # Date formats
    ]
    
    # Known brand patterns
    BRAND_PATTERNS = [
        r"^VISSAN$", r"^Vinamilk$", r"^TH True Milk$", r"^Masan$",
        r"^Acecook$", r"^Kinh Do$", r"^Bibica$", r"^Orion$",
    ]
    
    # Noise patterns (text to discard)
    NOISE_PATTERNS = [
        r"^[0-9\s\-/.,:;]+$",  # Only numbers/punctuation
        r"^[a-z]{1,3}$",       # Very short lowercase
        r"^\d+eq$",            # OCR errors like "8eq"
        r"^uhcil",             # Common OCR noise
        r"^canned=",           # OCR noise
        r"^mol\s*luong",       # OCR noise
        r"^hluong$",           # OCR noise
        r"^[a-z]+=$",          # Pattern like "xxx="
    ]
    
    def __init__(self, min_confidence: float = 0.3):
        """
        Initialize the extractor.
        
        Args:
            min_confidence: Minimum confidence threshold for including regions
        """
        self.min_confidence = min_confidence
        self.text_postprocessor = text_postprocessor
    
    def extract_from_regions(
        self, 
        text_regions: List[Dict[str, Any]],
        raw_text: Optional[str] = None,
    ) -> ExtractedProductInfo:
        """
        Extract product information from OCR text regions.
        
        Args:
            text_regions: List of text region dictionaries from OCR response
            raw_text: Optional full raw text for context
            
        Returns:
            ExtractedProductInfo with classified and cleaned data
        """
        # Step 1: Parse and filter regions
        parsed_regions = self._parse_regions(text_regions)
        filtered_regions = self._filter_noise(parsed_regions)
        
        logger.debug(f"Parsed {len(parsed_regions)} regions, {len(filtered_regions)} after filtering")
        
        # Step 2: Classify each region
        classified_regions = self._classify_regions(filtered_regions)
        
        # Step 3: Group regions by field type
        grouped = self._group_by_field(classified_regions)
        
        # Step 4: Build product info from grouped regions
        product_info = self._build_product_info(grouped, raw_text)
        
        return product_info
    
    def _parse_regions(self, text_regions: List[Dict[str, Any]]) -> List[TextRegion]:
        """Parse raw region dictionaries into TextRegion objects."""
        regions = []
        for region_dict in text_regions:
            # Support both "bbox" and "bounding_box" keys
            bbox = region_dict.get("bbox") or region_dict.get("bounding_box") or {}
            
            # If bbox is a dict with x1/y1/x2/y2
            if isinstance(bbox, dict):
                x1 = bbox.get("x1", 0)
                y1 = bbox.get("y1", 0)
                x2 = bbox.get("x2", 0)
                y2 = bbox.get("y2", 0)
            else:
                x1, y1, x2, y2 = 0, 0, 0, 0
            
            regions.append(TextRegion(
                text=region_dict.get("text", "").strip(),
                confidence=region_dict.get("confidence", 0),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            ))
        return regions
    
    def _filter_noise(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Filter out noise regions based on confidence and patterns."""
        filtered = []
        
        for region in regions:
            # Skip very low confidence
            if region.confidence < self.MIN_CONFIDENCE_NOISE:
                logger.debug(f"Filtered (low confidence {region.confidence:.2f}): '{region.text}'")
                continue
            
            # Skip empty or very short text
            if not region.text or len(region.text.strip()) < 2:
                continue
            
            # Check noise patterns
            text_lower = region.text.lower().strip()
            is_noise = False
            for pattern in self.NOISE_PATTERNS:
                if re.match(pattern, text_lower, re.IGNORECASE):
                    is_noise = True
                    logger.debug(f"Filtered (noise pattern): '{region.text}'")
                    break
            
            if not is_noise:
                filtered.append(region)
        
        return filtered
    
    def _classify_regions(self, regions: List[TextRegion]) -> List[ClassifiedRegion]:
        """Classify each region into a field type."""
        classified = []
        
        for region in regions:
            field_type, class_confidence = self._identify_field_type(region)
            corrected_text = self.text_postprocessor.correct_vietnamese_text(region.text)
            
            classified.append(ClassifiedRegion(
                region=region,
                field_type=field_type,
                classification_confidence=class_confidence,
                corrected_text=corrected_text,
            ))
        
        return classified
    
    def _identify_field_type(self, region: TextRegion) -> Tuple[FieldType, float]:
        """
        Identify which field type a region belongs to.
        
        Returns:
            Tuple of (FieldType, confidence_score)
        """
        text_lower = region.text.lower().strip()
        
        # Check if this is a date-related region (exclude from name detection)
        for pattern in self.DATE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.debug(f"Detected date region: '{region.text}'")
                return FieldType.UNKNOWN, 0.0  # Mark as unknown, not name
        
        # Check for brand patterns first (high confidence, short text)
        if region.confidence > self.MIN_CONFIDENCE_CRITICAL:
            for pattern in self.BRAND_PATTERNS:
                if re.match(pattern, region.text, re.IGNORECASE):
                    return FieldType.BRAND, 0.95
        
        # Check field-specific patterns
        scores = {}
        
        for field_type, patterns in self.FIELD_PATTERNS.items():
            score = 0.0
            
            # Check prefixes (strong indicator)
            for prefix in patterns.get("prefixes", []):
                if text_lower.startswith(prefix):
                    score += 0.7
                    break
            
            # Check keywords
            for keyword in patterns.get("keywords", []):
                if keyword in text_lower:
                    score += 0.15
            
            # Check regex patterns
            for pattern in patterns.get("patterns", []):
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score += 0.2
            
            if score > 0:
                scores[field_type] = min(score, 1.0)
        
        if scores:
            best_type = max(scores.keys(), key=lambda k: scores[k])
            return best_type, scores[best_type]
        
        # Default: could be product name or unknown
        # High confidence, position near top, not too long
        # Exclude technical text patterns that are NOT product names
        not_name_patterns = [
            "hướng dẫn", "bảo quản", "thành phần", "cảnh báo",
            "oxy hoa", "chất", "(\\d+)", "mg/", "451", "452", "621", "316",  # E-numbers, additives
            "nước", "muối", "đường",  # Common ingredients
            "làm nóng", "dùng ngay", "chế biến",  # Usage instructions
        ]
        if (region.confidence > 0.8 and 
            region.y1 < 250 and  # Very top of image only
            5 < len(region.text) < 30 and  # Shorter text for name
            not any(p in text_lower for p in not_name_patterns) and
            not re.search(r"\(\d+[i,]*\)", text_lower)):  # No E-numbers
            return FieldType.NAME, 0.5
        
        return FieldType.UNKNOWN, 0.0
    
    def _group_by_field(
        self, 
        classified_regions: List[ClassifiedRegion]
    ) -> Dict[FieldType, List[ClassifiedRegion]]:
        """
        Group classified regions by field type.
        
        Also merges UNKNOWN regions into the preceding field if they are
        vertically adjacent (continuation of content).
        """
        # First, sort all regions by vertical position
        sorted_regions = sorted(classified_regions, key=lambda cr: cr.region.y1)
        
        # Merge UNKNOWN regions into previous field if adjacent
        merged_regions = []
        current_field_type = None
        y_threshold = 40  # Max vertical gap to consider regions as continuation
        
        for cr in sorted_regions:
            if cr.field_type == FieldType.UNKNOWN:
                # Check if this UNKNOWN region continues a previous field
                if current_field_type and current_field_type not in [FieldType.BRAND, FieldType.NAME]:
                    # Check vertical proximity with last region
                    if merged_regions:
                        last_region = merged_regions[-1]
                        y_gap = cr.region.y1 - last_region.region.y2
                        if y_gap < y_threshold and last_region.field_type == current_field_type:
                            # Re-classify this region as continuation of current field
                            merged_regions.append(ClassifiedRegion(
                                region=cr.region,
                                field_type=current_field_type,
                                classification_confidence=0.3,  # Lower confidence for merged
                                corrected_text=cr.corrected_text
                            ))
                            continue
                merged_regions.append(cr)
            else:
                current_field_type = cr.field_type
                merged_regions.append(cr)
        
        # Group by field type
        grouped: Dict[FieldType, List[ClassifiedRegion]] = {}
        
        for cr in merged_regions:
            if cr.field_type not in grouped:
                grouped[cr.field_type] = []
            grouped[cr.field_type].append(cr)
        
        # Sort each group by vertical position (y1)
        for field_type in grouped:
            grouped[field_type].sort(key=lambda cr: cr.region.y1)
        
        return grouped
    
    def _build_product_info(
        self,
        grouped: Dict[FieldType, List[ClassifiedRegion]],
        raw_text: Optional[str] = None,
    ) -> ExtractedProductInfo:
        """Build ExtractedProductInfo from grouped regions."""
        
        info = ExtractedProductInfo()
        confidence_scores = {}
        
        # Extract brand
        brand_regions = grouped.get(FieldType.BRAND, [])
        if brand_regions:
            # Take the highest confidence brand
            best_brand = max(brand_regions, key=lambda cr: cr.region.confidence)
            info.brand = best_brand.corrected_text
            confidence_scores["brand"] = best_brand.region.confidence
        
        # Extract name - find product name from raw_text or high-confidence regions
        name = self._extract_product_name(grouped, raw_text, info.brand)
        if name:
            info.name = name
            confidence_scores["name"] = 0.8  # Estimated
        
        # Extract ingredients
        ingredients_regions = grouped.get(FieldType.INGREDIENTS, [])
        if ingredients_regions:
            # Filter out non-ingredient regions that were misclassified
            valid_ingredient_regions = []
            
            # Track the vertical range of ingredient content
            # Ingredients typically start after THÀNH PHẦN: and end before HƯỚNG DẪN
            first_ingredient_y = None
            last_ingredient_y = None
            
            for cr in ingredients_regions:
                text_lower = cr.corrected_text.lower()
                
                # Skip regions that are clearly NOT ingredients
                skip_patterns = [
                    "chỉ tiêu", "chất lượng", "chủ yếu",  # Quality standards
                    "hướng dẫn", "cảnh báo", "bảo quản",  # Other field labels
                    "món ăn", "biên thành", "chế biến",   # Cooking instructions
                    "trong lò", "vi sóng", "vi song",     # Microwave instructions
                    "làm nóng", "dùng ngay",              # Usage instructions
                ]
                if any(p in text_lower for p in skip_patterns):
                    continue
                    
                # Check if this looks like actual ingredient content
                # Should contain food items or additives
                ingredient_indicators = [
                    "%", "(", ")", "muối", "muoi", "nước", "bò", "heo", "gà",
                    "hành", "tỏi", "gừng", "sả", "đường", "dầu", "bột",
                    "451", "452", "621", "316", "i-ot", "iốt"
                ]
                if any(ind in text_lower for ind in ingredient_indicators):
                    if first_ingredient_y is None:
                        first_ingredient_y = cr.region.y1
                    last_ingredient_y = cr.region.y2
                    valid_ingredient_regions.append(cr)
            
            if valid_ingredient_regions:
                # Merge consecutive ingredient regions
                ingredients_text = self._merge_regions_text(valid_ingredient_regions)
                # Remove prefix if present
                ingredients_text = self._remove_field_prefix(
                    ingredients_text, 
                    ["thành phần:", "thành phẩn:", "nguyên liệu:"]
                )
                # Apply Vietnamese text correction
                ingredients_text = self.text_postprocessor.correct_vietnamese_text(ingredients_text)
                info.ingredients = ingredients_text
                confidence_scores["ingredients"] = sum(
                    cr.region.confidence for cr in valid_ingredient_regions
                ) / len(valid_ingredient_regions)
        
        # Extract storage instructions
        storage_regions = grouped.get(FieldType.STORAGE, [])
        if storage_regions:
            storage_text = self._merge_regions_text(storage_regions)
            storage_text = self._remove_field_prefix(
                storage_text,
                ["hướng dẫn bảo quản:", "bảo quản:"]
            )
            # Clean storage text
            is_valid, cleaned, _ = self.text_postprocessor.validate_field_content(
                "storage_instructions", storage_text
            )
            info.storage_instructions = cleaned if is_valid else storage_text
            confidence_scores["storage_instructions"] = sum(
                cr.region.confidence for cr in storage_regions
            ) / len(storage_regions)
        
        # Extract usage instructions
        usage_regions = grouped.get(FieldType.USAGE, [])
        if usage_regions:
            usage_text = self._merge_regions_text(usage_regions)
            usage_text = self._remove_field_prefix(
                usage_text,
                ["hướng dẫn sử dụng:", "hướng dẫn sử dung:", "cách dùng:", "hdsd:"]
            )
            # Apply Vietnamese text correction
            usage_text = self.text_postprocessor.correct_vietnamese_text(usage_text)
            info.usage_instructions = usage_text
            confidence_scores["usage_instructions"] = sum(
                cr.region.confidence for cr in usage_regions
            ) / len(usage_regions)
        
        # Extract warnings
        warning_regions = grouped.get(FieldType.WARNINGS, [])
        if warning_regions:
            warnings_parts = []
            for cr in warning_regions:
                warning_text = self._remove_field_prefix(
                    cr.corrected_text,
                    ["cảnh báo:", "thông tin cảnh báo:", "lưu ý:"]
                )
                # Skip very short or noisy text
                if not warning_text or len(warning_text) < 5:
                    continue
                # Skip noise patterns
                if re.match(r'^[a-z\s\|]+$', warning_text.lower()) and len(warning_text) < 10:
                    continue
                warnings_parts.append(warning_text.strip().rstrip(';'))
            
            if warnings_parts:
                # Join warnings and merge consecutive ones
                full_warning = " ".join(warnings_parts)
                # Clean up: apply Vietnamese correction
                full_warning = self.text_postprocessor.correct_vietnamese_text(full_warning)
                info.warnings = full_warning
                confidence_scores["warnings"] = sum(
                    cr.region.confidence for cr in warning_regions
                ) / len(warning_regions)
        
        # Extract weight
        weight_regions = grouped.get(FieldType.WEIGHT, [])
        if weight_regions:
            for cr in weight_regions:
                weight_match = re.search(
                    r"(\d+(?:\.\d+)?)\s*(g|kg|ml|l)\b",
                    cr.corrected_text,
                    re.IGNORECASE
                )
                if weight_match:
                    value = float(weight_match.group(1))
                    unit = weight_match.group(2).lower()
                    info.weight = f"{value} {unit}"
                    info.net_weight = f"{int(value) if value.is_integer() else value}{unit}"
                    info.weight_info = {
                        "value": value,
                        "unit": unit,
                        "raw": weight_match.group(0)
                    }
                    confidence_scores["weight"] = cr.region.confidence
                    break
        
        # Extract manufacturer
        manufacturer_regions = grouped.get(FieldType.MANUFACTURER, [])
        if manufacturer_regions:
            mfr_parts = []
            for cr in manufacturer_regions:
                text = self._remove_field_prefix(
                    cr.corrected_text,
                    ["sx bởi:", "sản xuất bởi:", "nsx bởi:", "manufacturer:"]
                )
                # Skip date info that got included
                if not re.search(r"(nsx|hsd|exp|mfg)[:\s]", text.lower()):
                    mfr_parts.append(text.strip())
            
            if mfr_parts:
                info.manufacturer = " ".join(mfr_parts)
                confidence_scores["manufacturer"] = sum(
                    cr.region.confidence for cr in manufacturer_regions
                ) / len(manufacturer_regions)
        
        # Detect category based on extracted info
        info.detected_category = self._detect_category(info)
        
        # Store confidence scores
        info.confidence_scores = confidence_scores
        
        # Metadata
        info.extraction_metadata = {
            "total_regions_processed": sum(len(regions) for regions in grouped.values()),
            "fields_extracted": [k.value for k in grouped.keys() if k != FieldType.UNKNOWN],
        }
        
        return info
    
    def _extract_product_name(
        self,
        grouped: Dict[FieldType, List[ClassifiedRegion]],
        raw_text: Optional[str],
        brand: Optional[str],
    ) -> Optional[str]:
        """Extract product name from grouped regions or raw text."""
        
        # Try to find name in explicitly classified NAME regions
        name_regions = grouped.get(FieldType.NAME, [])
        if name_regions:
            # Filter out invalid name candidates
            valid_names = []
            for cr in name_regions:
                text_lower = cr.corrected_text.lower()
                # Skip if it looks like ingredients or other content
                invalid_patterns = [
                    "thành phần", "thành phẩn", "hướng dẫn", "bảo quản", "cảnh báo",
                    "oxy hoa", "chất", "(", "%", "muối", "đường", "nước",
                    "451", "452", "621", "316",  # E-numbers
                ]
                if not any(p in text_lower for p in invalid_patterns):
                    valid_names.append(cr)
            
            if valid_names:
                # Prefer high confidence regions
                best = max(valid_names, key=lambda cr: cr.region.confidence)
                if best.region.confidence > 0.7:
                    return best.corrected_text
        
        # Try to extract from raw_text using postprocessor
        if raw_text:
            name = self.text_postprocessor.extract_product_name_from_text(raw_text, brand)
            if name:
                # Validate the extracted name
                name_lower = name.lower()
                invalid_patterns = [
                    "thành phần", "thành phẩn", "hướng dẫn", "bảo quản",
                    "oxy hoa", "muối", "đường", "%", "(",
                ]
                if not any(p in name_lower for p in invalid_patterns):
                    return name
        
        # Look for potential name in UNKNOWN regions near the top
        unknown_regions = grouped.get(FieldType.UNKNOWN, [])
        for cr in unknown_regions:
            # Name candidates: moderate length, high confidence, near top
            if (cr.region.confidence > 0.85 and
                cr.region.y1 < 200 and
                5 < len(cr.corrected_text) < 40):
                text_lower = cr.corrected_text.lower()
                # Skip if it looks like a field label or content
                invalid_patterns = [
                    "thành phần", "hướng dẫn", "bảo quản", "cảnh báo",
                    "muối", "đường", "nước", "oxy", "(", "%"
                ]
                if not any(p in text_lower for p in invalid_patterns):
                    return cr.corrected_text
        
        # Return None if no valid name found
        return None
    
    def _merge_regions_text(self, regions: List[ClassifiedRegion]) -> str:
        """Merge text from multiple regions into a single string."""
        texts = [cr.corrected_text for cr in regions]
        return " ".join(texts)
    
    def _remove_field_prefix(self, text: str, prefixes: List[str]) -> str:
        """Remove field label prefixes from text."""
        text_lower = text.lower()
        for prefix in prefixes:
            if text_lower.startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                break
        return text
    
    def _detect_category(self, info: ExtractedProductInfo) -> Dict[str, Any]:
        """Detect product category based on extracted information."""
        
        # Combine all text for analysis
        all_text = " ".join(filter(None, [
            info.name,
            info.ingredients,
            info.brand,
        ])).lower()
        
        # Category detection rules
        categories = {
            "meat": {
                "keywords": ["thịt", "nạc", "heo", "bò", "gà", "vịt", "giò", "chả", "xúc xích", "pate"],
                "confidence": 0.0,
            },
            "seafood": {
                "keywords": ["cá", "tôm", "mực", "cua", "hải sản", "ghẹ"],
                "confidence": 0.0,
            },
            "dairy": {
                "keywords": ["sữa", "phô mai", "bơ", "yaourt", "kem"],
                "confidence": 0.0,
            },
            "vegetable": {
                "keywords": ["rau", "củ", "quả", "cà chua", "dưa", "cải"],
                "confidence": 0.0,
            },
            "beverage": {
                "keywords": ["nước", "trà", "cà phê", "sinh tố", "nước ép"],
                "confidence": 0.0,
            },
            "snack": {
                "keywords": ["bánh", "kẹo", "snack", "chip"],
                "confidence": 0.0,
            },
        }
        
        # Calculate scores
        for category, data in categories.items():
            matches = sum(1 for kw in data["keywords"] if kw in all_text)
            if matches > 0:
                data["confidence"] = min(0.3 + (matches * 0.2), 0.95)
        
        # Find best category
        best_category = max(categories.keys(), key=lambda c: categories[c]["confidence"])
        best_confidence = categories[best_category]["confidence"]
        
        if best_confidence > 0.3:
            return {
                "name": best_category,
                "confidence": best_confidence,
                "keywords_vi": categories[best_category]["keywords"][:5],
            }
        
        return {
            "name": "unknown",
            "confidence": 0.0,
            "keywords_vi": [],
        }


# Singleton instance
region_extractor = RegionBasedExtractor()
