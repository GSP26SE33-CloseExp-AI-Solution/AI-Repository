"""
Vietnamese OCR Text Post-processor.

This module provides post-processing for OCR extracted text:
1. Vietnamese text correction (fix common OCR errors)
2. Field validation (check if text matches the expected field type)
3. Smart field re-mapping (classify text into correct fields)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# COMMON OCR ERRORS IN VIETNAMESE TEXT
# ============================================================================

# Character-level corrections for common OCR misreadings
CHAR_CORRECTIONS: Dict[str, str] = {
    # Vowels with diacritics
    "ẩ": "ẩ",  # Already correct, placeholder
    "ã": "ã",
    # Common misreadings
    "i-ot": "iốt",
    "i-ốt": "iốt",
    "l-ot": "iốt",
    # Punctuation
    " ,": ",",
    "  ": " ",
}

# Word-level corrections for Vietnamese
WORD_CORRECTIONS: Dict[str, str] = {
    # ============================================================================
    # Common OCR character misreadings
    # ============================================================================
    "nac": "nạc",
    "nuoc": "nước",
    "duong": "đường",
    "sua": "sữa",
    "thit": "thịt",
    "banh": "bánh",
    "rau": "rau",
    "ca": "cá",
    "tom": "tôm",
    "muoi": "muối",
    "bot": "bột",
    "dau": "dầu",
    "gia vi": "gia vị",
    "hanh": "hành",
    "toi": "tỏi",
    "gung": "gừng",
    "sa": "sả",
    "ot": "ớt",
    "tieu": "tiêu",
    "ngot": "ngọt",
    "man": "mặn",
    "chua": "chua",
    "nong": "nóng",
    "lanh": "lạnh",
    "mat": "mát",
    "am": "ẩm",
    "kho": "khô",
    "tuoi": "tươi",
    "dong": "đông",
    "dong lanh": "đông lạnh",
    "bao quan": "bảo quản",
    "su dung": "sử dụng",
    "san xuat": "sản xuất",
    "thanh phan": "thành phần",
    "huong dan": "hướng dẫn",
    "canh bao": "cảnh báo",
    "luu y": "lưu ý",
    "chat luong": "chất lượng",
    "khoi luong": "khối lượng",
    "trong luong": "trọng lượng",
    "han su dung": "hạn sử dụng",
    "ngay san xuat": "ngày sản xuất",
    "noi": "nơi",
    "khong": "không",
    "thoang": "thoáng",
    "noi thoang mat": "nơi thoáng mát",
    
    # ============================================================================
    # OCR-specific misreadings (common in Vietnamese product labels)
    # ============================================================================
    # T/J confusion
    "tjnh": "tịnh",
    "TJNH": "Tịnh",
    # A/ẩ confusion  
    "chẩt": "chất",
    "mat": "mát",
    "mảt": "mát",
    # D/Đ confusion
    "đong": "đông",
    # S/Ś confusion
    "sả": "sả",
    # Numeric confusions
    "8eq": "",
    # Common multi-word corrections
    "noi thoang": "nơi thoáng",
    "de noi": "để nơi",
    
    # ============================================================================
    # Product terms
    # ============================================================================
    "chat giu am": "chất giữ ẩm",
    "chat dieu vi": "chất điều vị",
    "chat bao quan": "chất bảo quản",
    "chat tao mau": "chất tạo màu",
    "chat chong oxy hoa": "chất chống oxy hóa",
    "oxy hoa": "oxy hóa",
    "vi song": "vi sóng",
    "lo vi song": "lò vi sóng",
    
    # ============================================================================
    # Product type keywords
    # ============================================================================
    "pate": "pate",
    "pa te": "pate",
    "xuc xich": "xúc xích",
    "gio": "giò",
    "cha": "chả",
    "nem": "nem",
    "ga": "gà",
    "heo": "heo",
    "bo": "bò",
    "lon": "lợn",
    "vit": "vịt",
    "ca": "cá",
    
    # ============================================================================
    # Company names
    # ============================================================================
    "vissan": "VISSAN",
    "vinamilk": "Vinamilk",
    "th true milk": "TH True Milk",
    
    # ============================================================================
    # Common OCR noise (remove)
    # ============================================================================
    "jrao": "ráo",
    "hluong": "lượng",
    "mol luong": "khối lượng",
    "canned": "",  # Remove English text that doesn't belong
    
    # ============================================================================
    # Usage instruction corrections
    # ============================================================================
    "trươc": "trước",
    "truoc": "trước",
    "hoàc": "hoặc",
    "hoac": "hoặc",
    "chể": "chế",
    "che": "chế",
    "biên": "biến",
    "bien": "biến",
    "cảc": "các",
    "cac": "các",
    "khac": "khác",
    "co the": "có thể",
    "lam nong": "làm nóng",
    "dung ngay": "dùng ngay",
    "che bien": "chế biến",
    "mon an": "món ăn",
}

# Phrase-level corrections
PHRASE_CORRECTIONS: Dict[str, str] = {
    "noi thoang mat, khong de noi nong, am": "Nơi thoáng mát, không để nơi nóng, ẩm",
    "khong lam truc tiep san pham trong lo vi song": "Không làm nóng trực tiếp sản phẩm trong lò vi sóng",
    "dung nong ngay": "Dùng nóng ngay",
    "co the lam nong truoc khi dung": "Có thể làm nóng trước khi dùng",
    "hoac che bien thanh cac mon an khac": "hoặc chế biến thành các món ăn khác",
    "chat giu am (451i, 452i)": "chất giữ ẩm (E451i, E452i)",
    "chat dieu vi (621)": "chất điều vị (E621 - bột ngọt)",
    "chat oxy hoa (316)": "chất chống oxy hóa (E316)",
}


# ============================================================================
# FIELD IDENTIFICATION PATTERNS
# ============================================================================

@dataclass
class FieldPattern:
    """Pattern to identify which field a text belongs to."""
    keywords: List[str]  # Keywords that indicate this field
    anti_keywords: List[str]  # Keywords that indicate this is NOT this field
    min_length: int = 0
    max_length: int = 500


FIELD_PATTERNS: Dict[str, FieldPattern] = {
    "name": FieldPattern(
        keywords=[],  # Name is determined by exclusion and position
        anti_keywords=[
            "thành phần", "nguyên liệu", "ingredients",
            "hướng dẫn", "cách dùng", "cách sử dụng",
            "bảo quản", "storage",
            "cảnh báo", "lưu ý", "chú ý", "warning",
            "sản xuất bởi", "phân phối", "công ty",
            "khối lượng", "trọng lượng", "net weight",
            "hsd", "nsx", "exp", "mfg",
            "chỉ tiêu", "chất lượng",
        ],
        min_length=3,
        max_length=100,
    ),
    "ingredients": FieldPattern(
        keywords=[
            "thành phần", "nguyên liệu", "ingredients",
            "gồm có", "bao gồm", "chứa",
        ],
        anti_keywords=[],
        min_length=10,
        max_length=1000,
    ),
    "storage_instructions": FieldPattern(
        keywords=[
            "bảo quản", "hướng dẫn bảo quản", "cách bảo quản",
            "storage", "điều kiện bảo quản",
            "nơi thoáng mát", "nhiệt độ", "tủ lạnh", "đông lạnh",
        ],
        anti_keywords=[],
        min_length=10,
        max_length=500,
    ),
    "usage_instructions": FieldPattern(
        keywords=[
            "hướng dẫn sử dụng", "cách dùng", "cách sử dụng",
            "cách chế biến", "directions", "how to use",
            "dùng nóng", "làm nóng", "pha chế",
        ],
        anti_keywords=[],
        min_length=10,
        max_length=500,
    ),
    "warnings": FieldPattern(
        keywords=[
            "cảnh báo", "lưu ý", "chú ý", "warning", "caution",
            "không dùng cho", "tránh", "cẩn thận",
            "thông tin cảnh báo",
        ],
        anti_keywords=[],
        min_length=5,
        max_length=500,
    ),
    "manufacturer": FieldPattern(
        keywords=[
            "sản xuất bởi", "nhà sản xuất", "công ty",
            "manufactured by", "producer",
            "chịu trách nhiệm", "đơn vị chịu trách nhiệm",
        ],
        anti_keywords=[],
        min_length=5,
        max_length=200,
    ),
    "quality_standards": FieldPattern(
        keywords=[
            "chỉ tiêu", "chất lượng", "tiêu chuẩn",
            "quality", "standard",
        ],
        anti_keywords=[],
        min_length=5,
        max_length=300,
    ),
}


class TextPostProcessor:
    """Post-processor for OCR extracted Vietnamese text."""

    def __init__(self):
        self.char_corrections = CHAR_CORRECTIONS
        self.word_corrections = WORD_CORRECTIONS
        self.phrase_corrections = PHRASE_CORRECTIONS
        self.field_patterns = FIELD_PATTERNS

    def correct_vietnamese_text(self, text: str) -> str:
        """
        Apply Vietnamese text corrections for common OCR errors.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Corrected text
        """
        if not text:
            return text
            
        result = text
        
        # Step 1: Character-level corrections
        for wrong, correct in self.char_corrections.items():
            result = result.replace(wrong, correct)
        
        # Step 2: Word-level corrections (case-insensitive)
        # Use regex to split but preserve punctuation context
        words = result.split()
        corrected_words = []
        for word in words:
            # Strip punctuation for matching
            word_stripped = word.strip('.,;:!?()[]{}')
            prefix = word[:len(word) - len(word.lstrip('.,;:!?()[]{}'))] if word != word.lstrip('.,;:!?()[]{}') else ''
            suffix = word[len(word_stripped) + len(prefix):] if len(word) > len(word_stripped) + len(prefix) else ''
            
            word_lower = word_stripped.lower()
            
            # Check single word corrections
            if word_lower in self.word_corrections:
                correction = self.word_corrections[word_lower]
                if correction:  # Only add if not empty (some are removed)
                    # Preserve original case for proper nouns
                    if word_stripped.isupper():
                        # For uppercase words, use title case or uppercase correction
                        if correction[0].isupper():
                            corrected_words.append(prefix + correction + suffix)
                        else:
                            corrected_words.append(prefix + correction.title() + suffix)
                    else:
                        corrected_words.append(prefix + correction + suffix)
                # If correction is empty, skip the word (noise removal)
            else:
                corrected_words.append(word)
        result = " ".join(corrected_words)
        
        # Step 3: Phrase-level corrections
        result_lower = result.lower()
        for wrong_phrase, correct_phrase in self.phrase_corrections.items():
            if wrong_phrase in result_lower:
                # Find and replace preserving some case
                pattern = re.compile(re.escape(wrong_phrase), re.IGNORECASE)
                result = pattern.sub(correct_phrase, result)
        
        # Step 4: Clean up
        result = re.sub(r'\s+', ' ', result)  # Normalize whitespace
        result = re.sub(r'\s*,\s*', ', ', result)  # Fix comma spacing
        result = result.strip()
        
        return result

    def identify_field_type(self, text: str) -> Tuple[str, float]:
        """
        Identify which field type a text belongs to.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (field_name, confidence)
        """
        if not text:
            return "unknown", 0.0
            
        text_lower = text.lower()
        scores: Dict[str, float] = {}
        
        for field_name, pattern in self.field_patterns.items():
            score = 0.0
            
            # Check length constraints
            if len(text) < pattern.min_length or len(text) > pattern.max_length:
                continue
            
            # Check for field keywords
            keyword_count = sum(1 for kw in pattern.keywords if kw in text_lower)
            if keyword_count > 0:
                score += 0.5 + (keyword_count * 0.1)
            
            # Check for anti-keywords (disqualifiers)
            anti_keyword_count = sum(1 for kw in pattern.anti_keywords if kw in text_lower)
            if anti_keyword_count > 0:
                score -= 0.3 * anti_keyword_count
            
            if score > 0:
                scores[field_name] = min(1.0, score)
        
        if not scores:
            return "unknown", 0.0
            
        best_field = max(scores, key=scores.get)
        return best_field, scores[best_field]

    def validate_field_content(self, field_name: str, content: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate if content is appropriate for the given field.
        
        Args:
            field_name: Name of the field (e.g., "name", "ingredients")
            content: Content to validate
            
        Returns:
            Tuple of (is_valid, cleaned_content, suggested_field_if_invalid)
        """
        if not content:
            return False, "", None
            
        content_lower = content.lower()
        
        # Check if content belongs to a different field
        # Skip this check for storage_instructions since they often contain mixed content
        # that will be cleaned up in the field-specific logic
        if field_name not in ["storage_instructions"]:
            detected_field, confidence = self.identify_field_type(content)
            
            if detected_field != field_name and detected_field != "unknown" and confidence > 0.5:
                logger.debug(f"Content '{content[:50]}...' detected as '{detected_field}' instead of '{field_name}'")
                return False, content, detected_field
        
        # Field-specific validation
        if field_name == "name":
            # Name should not start with field labels
            invalid_prefixes = [
                "thành phần", "thành phẩn", "nguyên liệu", "ingredients",
                "hướng dẫn", "bảo quản", "cảnh báo",
                "sản xuất", "công ty",
            ]
            for prefix in invalid_prefixes:
                if content_lower.startswith(prefix):
                    # This is actually ingredients or other field
                    return False, content, "ingredients"
            
            # Clean up name - remove trailing field labels
            name_cleaned = content
            for sep in ["thành phần", "thành phẩn", "nguyên liệu", "hướng dẫn"]:
                if sep in name_cleaned.lower():
                    idx = name_cleaned.lower().index(sep)
                    name_cleaned = name_cleaned[:idx].strip()
            
            # Remove noise characters
            name_cleaned = re.sub(r'[^\w\s\-\(\)%,.]', '', name_cleaned)
            name_cleaned = name_cleaned.strip(' ,:;')
            
            # If name is empty after cleanup, it's invalid
            if not name_cleaned or len(name_cleaned) < 3:
                return False, "", "ingredients"
            
            return True, name_cleaned, None
        
        elif field_name == "ingredients":
            # Ingredients should contain ingredient-like content
            if ":" in content:
                # Remove the prefix (e.g., "THÀNH PHẦN:")
                parts = content.split(":", 1)
                if len(parts) > 1:
                    content = parts[1].strip()
            
            # Correct the text
            content = self.correct_vietnamese_text(content)
            return True, content, None
        
        elif field_name == "storage_instructions":
            # First correct the text
            content = self.correct_vietnamese_text(content)
            
            # Split out other field content (remove everything after these patterns)
            stop_patterns = [
                r"thông tin cảnh báo",
                r"hướng dẫn sử dụng",
                r"chỉ tiêu chất lượng",
                r"tịnh:",  # Weight info (Tịnh: 150g)
                r"không làm",  # Instructions often embedded
            ]
            
            # Find earliest stop pattern
            earliest_pos = len(content)
            for pattern in stop_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match and match.start() < earliest_pos:
                    earliest_pos = match.start()
            
            if earliest_pos < len(content):
                content = content[:earliest_pos].strip()
            
            # Remove trailing noise and punctuation
            content = re.sub(r'[\s.,:;]+$', '', content)
            
            # Clean up any remaining noise
            noise_patterns = [
                r'\b8eq\b',
                r'\buhcil\b',
            ]
            for noise in noise_patterns:
                content = re.sub(noise, '', content, flags=re.IGNORECASE)
            
            # Normalize and clean
            content = re.sub(r'\s+', ' ', content).strip()
            
            return True, content, None
        
        # Default: apply corrections
        content = self.correct_vietnamese_text(content)
        return True, content, None

    def extract_product_name_from_text(self, raw_text: str, brand: Optional[str] = None) -> Optional[str]:
        """
        Extract clean product name from raw OCR text.
        
        The product name should be a short, descriptive name without
        ingredients, instructions, or other metadata.
        
        Args:
            raw_text: Full OCR text
            brand: Detected brand name (to exclude from name search)
            
        Returns:
            Clean product name or None
        """
        if not raw_text:
            return None
            
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        
        # Filter out lines that are clearly not product names
        candidate_lines = []
        for line in lines:
            line_lower = line.lower()
            
            # Skip lines that start with field labels
            skip_prefixes = [
                "thành phần", "nguyên liệu", "ingredients",
                "hướng dẫn", "cách dùng", "cách sử dụng",
                "bảo quản", "storage",
                "cảnh báo", "lưu ý", "chú ý",
                "sản xuất", "công ty", "địa chỉ",
                "khối lượng", "trọng lượng", "net",
                "hsd", "nsx", "exp", "mfg",
                "chỉ tiêu", "chất lượng",
                "uhcil", "8eq",  # Common OCR noise
            ]
            if any(line_lower.startswith(prefix) for prefix in skip_prefixes):
                continue
            
            # Skip lines that are too short or too long for product names
            if len(line) < 3 or len(line) > 100:
                continue
            
            # Skip lines that are just numbers or special characters
            if re.match(r'^[\d\s\-/.:]+$', line):
                continue
            
            # Skip brand line (we want product name, not brand)
            if brand and line.lower() == brand.lower():
                continue
            
            candidate_lines.append(line)
        
        if not candidate_lines:
            return None
        
        # Prefer lines with product-descriptive words
        product_keywords = [
            "đóng hộp", "hộp", "lon", "chai", "gói", "túi",
            "heo", "bò", "gà", "cá", "tôm",
            "sữa", "nước", "trà", "cà phê",
            "bánh", "mì", "phở", "bún",
            "rau", "củ", "quả",
        ]
        
        for line in candidate_lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in product_keywords):
                # Clean up the line
                name = self.correct_vietnamese_text(line)
                # Remove any trailing field indicators
                name = re.split(r'(?:thành phần|nguyên liệu|hướng dẫn)', name, flags=re.IGNORECASE)[0]
                return name.strip()
        
        # Fallback: return first candidate that's not all caps (likely not a brand)
        for line in candidate_lines:
            if not line.isupper():
                return self.correct_vietnamese_text(line)
        
        return self.correct_vietnamese_text(candidate_lines[0]) if candidate_lines else None

    def process_ocr_response(self, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process entire OCR response to fix and validate all fields.
        
        Args:
            ocr_data: Original OCR response dictionary
            
        Returns:
            Cleaned and validated OCR response
        """
        result = ocr_data.copy()
        raw_text = ocr_data.get("raw_text", "")
        product_info = ocr_data.get("product_info", {})
        
        if not product_info:
            return result
        
        # Process each field
        new_product_info = product_info.copy()
        
        # 1. Fix name field
        current_name = product_info.get("name", "")
        if current_name:
            is_valid, cleaned, suggested = self.validate_field_content("name", current_name)
            if not is_valid and suggested:
                # Move content to the correct field if needed
                if suggested == "ingredients" and not new_product_info.get("ingredients"):
                    # Extract ingredients from misplaced name
                    _, ingredients_content, _ = self.validate_field_content("ingredients", current_name)
                    new_product_info["ingredients"] = ingredients_content
                # Try to find actual product name from raw text
                actual_name = self.extract_product_name_from_text(raw_text, product_info.get("brand"))
                new_product_info["name"] = actual_name
            else:
                new_product_info["name"] = cleaned
        
        # 2. Fix ingredients field
        current_ingredients = product_info.get("ingredients")
        if current_ingredients:
            if isinstance(current_ingredients, list):
                current_ingredients = ", ".join(current_ingredients)
            _, cleaned, _ = self.validate_field_content("ingredients", current_ingredients)
            new_product_info["ingredients"] = cleaned
        
        # 3. Fix storage instructions
        current_storage = product_info.get("storage_instructions")
        if current_storage:
            _, cleaned, _ = self.validate_field_content("storage_instructions", current_storage)
            new_product_info["storage_instructions"] = cleaned
        
        # 4. Fix usage instructions  
        current_usage = product_info.get("usage_instructions")
        if current_usage:
            cleaned = self.correct_vietnamese_text(current_usage)
            new_product_info["usage_instructions"] = cleaned
        
        # 5. Fix warnings - filter out weight info and noise
        current_warnings = product_info.get("warnings")
        if current_warnings:
            if isinstance(current_warnings, list):
                cleaned_warnings = []
                for w in current_warnings:
                    cleaned_w = self.correct_vietnamese_text(w)
                    
                    # Check if this is actually weight info (e.g., "TJNH: 150g" -> "Tịnh: 150g")
                    weight_match = re.match(r'(tịnh|tjnh|net\s*w?):?\s*(\d+)\s*(g|kg|ml|l)', cleaned_w, re.IGNORECASE)
                    if weight_match:
                        # This is weight info, extract and store separately
                        weight_value = weight_match.group(2)
                        weight_unit = weight_match.group(3).lower()
                        if not new_product_info.get("net_weight"):
                            new_product_info["net_weight"] = f"{weight_value}{weight_unit}"
                        continue  # Don't add to warnings
                    
                    # Remove warnings that are just noise
                    if len(cleaned_w) > 5 and not re.match(r'^[\d\s\-/:]+$', cleaned_w):
                        cleaned_warnings.append(cleaned_w)
                new_product_info["warnings"] = cleaned_warnings if cleaned_warnings else None
            else:
                new_product_info["warnings"] = [self.correct_vietnamese_text(current_warnings)]
        
        # 6. Try to extract missing product name from raw text
        if not new_product_info.get("name") or new_product_info.get("name") == "":
            name_from_text = self.extract_product_name_from_text(
                raw_text, 
                product_info.get("brand")
            )
            if name_from_text:
                new_product_info["name"] = name_from_text
        
        # 7. Correct brand name casing
        brand = new_product_info.get("brand")
        if brand:
            brand_corrections = {
                "VISSAN": "VISSAN",
                "vissan": "VISSAN",
                "Vissan": "VISSAN",
            }
            new_product_info["brand"] = brand_corrections.get(brand, brand)
        
        # 8. Clean up detected category based on corrected name
        if new_product_info.get("name"):
            name_lower = new_product_info["name"].lower()
            category_info = new_product_info.get("detected_category", {})
            
            # Override category detection based on product keywords
            if any(kw in name_lower for kw in ["thịt", "nạc", "heo", "bò", "gà", "giò", "chả"]):
                category_info = {
                    "name": "meat",
                    "confidence": 0.9,
                    "keywords_vi": ["thịt", "thịt heo", "thịt bò", "giò", "chả", "xúc xích"]
                }
            elif any(kw in name_lower for kw in ["cá", "tôm", "mực", "cua"]):
                category_info = {
                    "name": "seafood",
                    "confidence": 0.9,
                    "keywords_vi": ["cá", "tôm", "mực", "cua", "hải sản"]
                }
            elif any(kw in name_lower for kw in ["sữa", "yaourt", "phô mai"]):
                category_info = {
                    "name": "dairy",
                    "confidence": 0.9,
                    "keywords_vi": ["sữa", "sữa tươi", "sữa chua", "phô mai"]
                }
            
            new_product_info["detected_category"] = category_info
        
        result["product_info"] = new_product_info
        
        # Correct raw_text if present
        if raw_text:
            result["raw_text"] = self.correct_vietnamese_text(raw_text)
        
        return result


# Singleton instance
text_postprocessor = TextPostProcessor()
