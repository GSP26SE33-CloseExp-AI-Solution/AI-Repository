"""
Vietnamese Product Recognition Service.

Enhanced service for recognizing Vietnamese products, fresh produce,
and extracting detailed packaging information.

Note: Barcode lookup is handled by the Backend service using external APIs:
- Open Food Facts (vn.openfoodfacts.org)
- UPCitemdb
This service focuses on:
- Detecting Vietnamese barcodes (GS1 prefix 893)
- Extracting packaging information from OCR text
- Categorizing products based on Vietnamese keywords
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# GS1 COUNTRY PREFIXES
# ============================================================================

# GS1 Country prefixes for barcode origin detection
GS1_COUNTRY_PREFIXES: Dict[str, str] = {
    # Vietnam
    "893": "Vietnam",
    # Southeast Asia
    "885": "Thailand",
    "888": "Singapore", 
    "890": "India",
    "899": "Indonesia",
    "955": "Malaysia",
    # East Asia
    "471": "Taiwan",
    "489": "Hong Kong",
    "880": "South Korea",
    # China (690-699)
    "690": "China", "691": "China", "692": "China", "693": "China",
    "694": "China", "695": "China", "696": "China", "697": "China",
    "698": "China", "699": "China",
    # Japan (450-459, 490-499)
    "450": "Japan", "451": "Japan", "452": "Japan", "453": "Japan",
    "454": "Japan", "455": "Japan", "456": "Japan", "457": "Japan",
    "458": "Japan", "459": "Japan", "490": "Japan", "491": "Japan",
    "492": "Japan", "493": "Japan", "494": "Japan", "495": "Japan",
    "496": "Japan", "497": "Japan", "498": "Japan", "499": "Japan",
}


# ============================================================================
# VIETNAMESE PRODUCT CATEGORIES & KEYWORDS
# ============================================================================

VN_PRODUCT_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "dairy": {
        "keywords_vi": ["sữa", "sữa tươi", "sữa chua", "phô mai", "bơ", "kem", "yaourt"],
        "keywords_en": ["milk", "yogurt", "cheese", "butter", "cream"],
        "shelf_life_days": {"refrigerated": 14, "uht": 180, "powder": 365},
    },
    "meat": {
        "keywords_vi": ["thịt", "thịt heo", "thịt bò", "thịt gà", "giò", "chả", "xúc xích", "lạp xưởng", "nem", "patê"],
        "keywords_en": ["pork", "beef", "chicken", "sausage", "ham", "bacon"],
        "shelf_life_days": {"fresh": 3, "frozen": 90, "processed": 30},
    },
    "seafood": {
        "keywords_vi": ["cá", "tôm", "mực", "cua", "nghêu", "sò", "hàu", "chả cá", "cá viên"],
        "keywords_en": ["fish", "shrimp", "squid", "crab", "oyster"],
        "shelf_life_days": {"fresh": 2, "frozen": 90, "dried": 180},
    },
    "vegetable": {
        "keywords_vi": ["rau", "rau xanh", "rau muống", "cải", "bắp cải", "xà lách", "cà chua", "khoai", "củ", "nấm", "hành", "tỏi", "ớt"],
        "keywords_en": ["vegetable", "lettuce", "cabbage", "tomato", "potato", "mushroom"],
        "shelf_life_days": {"fresh": 7, "frozen": 180, "canned": 730},
    },
    "fruit": {
        "keywords_vi": ["trái cây", "quả", "cam", "táo", "chuối", "xoài", "dưa", "nho", "ổi", "mít", "sầu riêng", "thanh long", "bưởi"],
        "keywords_en": ["fruit", "orange", "apple", "banana", "mango", "grape", "watermelon"],
        "shelf_life_days": {"fresh": 7, "dried": 180, "canned": 730},
    },
    "instant_noodle": {
        "keywords_vi": ["mì", "mì ăn liền", "mì gói", "phở", "bún", "miến", "hủ tiếu", "cháo"],
        "keywords_en": ["noodle", "instant noodle", "pho", "vermicelli"],
        "shelf_life_days": {"dry": 180, "fresh": 3},
    },
    "beverage": {
        "keywords_vi": ["nước", "nước ngọt", "nước suối", "trà", "cà phê", "nước ép", "sinh tố", "nước tăng lực"],
        "keywords_en": ["water", "soft drink", "juice", "tea", "coffee", "energy drink"],
        "shelf_life_days": {"bottled": 365, "fresh": 3, "powder": 730},
    },
    "seasoning": {
        "keywords_vi": ["gia vị", "nước mắm", "nước tương", "tương ớt", "tương cà", "muối", "đường", "bột ngọt", "hạt nêm", "dầu ăn"],
        "keywords_en": ["fish sauce", "soy sauce", "salt", "sugar", "msg", "cooking oil"],
        "shelf_life_days": {"liquid": 365, "powder": 730},
    },
    "snack": {
        "keywords_vi": ["bánh", "snack", "kẹo", "bánh quy", "bánh tráng", "khô", "mứt", "hạt"],
        "keywords_en": ["snack", "candy", "cookie", "biscuit", "chips", "nuts"],
        "shelf_life_days": {"dry": 180, "fresh": 7},
    },
    "bakery": {
        "keywords_vi": ["bánh mì", "bánh ngọt", "bánh kem", "bánh bông lan", "bánh croissant"],
        "keywords_en": ["bread", "cake", "pastry", "croissant"],
        "shelf_life_days": {"fresh": 3, "packaged": 14},
    },
}


# ============================================================================
# PACKAGING TEXT PATTERNS (Vietnamese) - Enhanced for detailed extraction
# ============================================================================

VN_PACKAGING_PATTERNS = {
    # Date patterns - Enhanced
    "expiry": [
        r"(?:HSD|Hạn sử dụng|Hạn dùng|Sử dụng trước|Thời hạn sử dụng)[:\s]*(.+?)(?:\n|$)",
        r"(?:Best before|Use by|Expiry|Exp\.?)[:\s]*(.+?)(?:\n|$)",
        r"(?:EXP)[:\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
        r"(?:Thời hạn sử dụng)[:\s]*(\d+\s*(?:ngày|tháng|năm))",
    ],
    "manufacturing": [
        r"(?:NSX|Ngày sản xuất|Sản xuất ngày|Ngày đóng gói)[:\s]*(.+?)(?:\n|$)",
        r"(?:MFG|Manufacturing date|Produced|Production date)[:\s]*(.+?)(?:\n|$)",
    ],
    
    # Product info patterns - Enhanced
    "weight": [
        r"(?:Khối lượng tịnh|KL tịnh|Khối lượng|Net weight|Net wt|Trọng lượng)[:\s]*(\d+(?:[.,]\d+)?)\s*(g|kg|ml|l|gram|kilogram|lít)",
        r"(\d+(?:[.,]\d+)?)\s*(g|kg|ml|l|G|Kg|ML|L)\b",
    ],
    
    # Ingredients - Enhanced with Vietnamese food additives
    "ingredients": [
        r"(?:Thành phần|Nguyên liệu|Ingredients)[:\s]*(.+?)(?:(?:Hướng dẫn|Cách dùng|Bảo quản|Thông tin dinh dưỡng|Chỉ tiêu|Công ty|Sản xuất|$))",
        r"(?:THÀNH PHẦN)[:\s]*(.+?)(?:(?:HƯỚNG DẪN|BẢO QUẢN|$))",
    ],
    
    # Storage instructions - Enhanced
    "storage": [
        r"(?:Hướng dẫn bảo quản|Bảo quản|Cách bảo quản|Điều kiện bảo quản|Storage)[:\s]*(.+?)(?:(?:Chú ý|Lưu ý|Hướng dẫn sử dụng|Thành phần|\n\n|$))",
        r"(?:HƯỚNG DẪN BẢO QUẢN)[:\s]*(.+?)(?:(?:CHÚ Ý|HƯỚNG DẪN SỬ DỤNG|\n\n|$))",
        r"(?:bảo quản ở nhiệt độ|bảo quản nơi)[^\.]+\.",
    ],
    
    # Usage instructions - Enhanced
    "usage": [
        r"(?:Hướng dẫn sử dụng|Cách dùng|Cách sử dụng|Directions|How to use|Cách chế biến)[:\s]*(.+?)(?:(?:Bảo quản|Hướng dẫn bảo quản|Thành phần|Lưu ý|\n\n|$))",
        r"(?:HƯỚNG DẪN SỬ DỤNG)[:\s]*(.+?)(?:(?:BẢO QUẢN|LƯU Ý|\n\n|$))",
    ],
    
    # Manufacturer & Distributor - Enhanced
    "manufacturer": [
        r"(?:Sản xuất bởi|Nhà sản xuất|Công ty sản xuất|Manufactured by|Producer|Sản xuất tại)[:\s]*(.+?)(?:\n|$)",
        r"(?:CÔNG TY)[^:]*[:\s]*(.+?)(?:\n|$)",
    ],
    "distributor": [
        r"(?:Nhà phân phối|Phân phối bởi|Distributed by|Distributor)[:\s]*(.+?)(?:\n|$)",
        r"(?:Sản xuất và phân phối bởi)[:\s]*(.+?)(?:\n|$)",
    ],
    "address": [
        r"(?:Địa chỉ|Address)[:\s]*(.+?)(?:(?:Điện thoại|Tel|Fax|Email|Website|\n\n|$))",
        r"(?:Chi nhánh)[^:]*[:\s]*(.+?)(?:\n|$)",
    ],
    "contact": [
        r"(?:Điện thoại|Tel|Hotline|Tư vấn khách hàng)[:\s]*(.+?)(?:\n|$)",
        r"(?:Website)[:\s]*(www\.[^\s]+|https?://[^\s]+)",
        r"(?:Email)[:\s]*([^\s]+@[^\s]+)",
    ],
    
    # Origin - Enhanced
    "origin": [
        r"(?:Xuất xứ|Nguồn gốc|Origin|Made in|Country of origin|Sản xuất tại)[:\s]*(.+?)(?:\n|$)",
        r"(?:Nguyên liệu nhập khẩu từ)[:\s]*(.+?)(?:\n|$)",
    ],
    
    # Certifications - Enhanced with Vietnamese certifications
    "certification": [
        r"(HACCP|ISO \d+(?::\d+)?|VietGAP|GlobalGAP|USDA Organic|Organic|Hữu cơ|BRC|FSSC|GMP|Halal)",
        r"(Chứng nhận số|Mã số chứng nhận|CU \d+)",
        r"(VN-BIO-\d+)",
    ],
    
    # Quality standards - New
    "quality_standards": [
        r"(?:Chỉ tiêu chất lượng|Tiêu chuẩn chất lượng)[:\s]*(.+?)(?:(?:Hướng dẫn|\n\n|$))",
        r"(?:Hàm lượng \w+)[:\s]*([^,\n]+)",
    ],
    
    # Warnings/Notes - New
    "warnings": [
        r"(?:Chú ý|Lưu ý|Cảnh báo|Warning|Note)[:\s]*(.+?)(?:\n|$)",
        r"(?:CHÚ Ý|LƯU Ý)[:\s]*(.+?)(?:\n|$)",
    ],
    
    # Product code/batch - New
    "product_code": [
        r"(?:Mã sản phẩm|SKU|Product code)[:\s]*([A-Z0-9\-]+)",
        r"(?:Số lô|Batch|Lot)[:\s]*([A-Z0-9\-]+)",
        r"(?:MSKTVSTY)[:\s]*([A-Z0-9\.\-]+)",
    ],
}


# ============================================================================
# NUTRITION PATTERNS (Vietnamese) - Enhanced
# ============================================================================

VN_NUTRITION_PATTERNS = {
    "calories": r"(?:Năng lượng|Calories|Energy)[:\s]*(\d+(?:[.,]\d+)?)\s*(?:kcal|kJ)?",
    "protein": r"(?:Đạm|Chất đạm|Protein|Protid|Hàm lượng Protid)[:\s]*[≥]?(\d+(?:[.,]\d+)?)\s*[%g]?",
    "fat": r"(?:Chất béo|Fat|Lipid|Béo)[:\s]*(\d+(?:[.,]\d+)?)\s*g?",
    "carbs": r"(?:Carbohydrate|Tinh bột|Carbonhydrat|Carbs)[:\s]*(\d+(?:[.,]\d+)?)\s*g?",
    "fiber": r"(?:Chất xơ|Fiber)[:\s]*(\d+(?:[.,]\d+)?)\s*g?",
    "sodium": r"(?:Natri|Sodium|Na)[:\s]*(\d+(?:[.,]\d+)?)\s*(?:mg|g)?",
    "sugar": r"(?:Đường|Sugar)[:\s]*(\d+(?:[.,]\d+)?)\s*g?",
    "calcium": r"(?:Canxi|Calcium|Ca)[:\s]*(\d+(?:[.,]\d+)?)\s*(?:mg|g)?",
    "iron": r"(?:Sắt|Iron|Fe)[:\s]*(\d+(?:[.,]\d+)?)\s*(?:mg|g)?",
    "cholesterol": r"(?:Cholesterol)[:\s]*(\d+(?:[.,]\d+)?)\s*(?:mg|g)?",
    "saturated_fat": r"(?:Chất béo bão hòa|Saturated fat)[:\s]*(\d+(?:[.,]\d+)?)\s*g?",
    "trans_fat": r"(?:Chất béo trans|Trans fat)[:\s]*(\d+(?:[.,]\d+)?)\s*g?",
    "vitamin_a": r"(?:Vitamin A)[:\s]*(\d+(?:[.,]\d+)?)\s*(?:IU|mcg|µg)?",
    "vitamin_c": r"(?:Vitamin C)[:\s]*(\d+(?:[.,]\d+)?)\s*mg?",
    "vitamin_d": r"(?:Vitamin D)[:\s]*(\d+(?:[.,]\d+)?)\s*(?:IU|mcg|µg)?",
    # Vietnamese specific: NH3 content for meat products
    "nh3": r"(?:Hàm lượng NH3|NH3)[:\s]*[<]?(\d+(?:[.,]\d+)?)\s*(?:mg)?",
}


class VietnameseProductService:
    """Service for Vietnamese product recognition and information extraction.
    
    Note: Product lookup from barcode is handled by the Backend service
    using external APIs (Open Food Facts, UPCitemdb). This service focuses on:
    - Detecting barcode country of origin
    - Extracting packaging information from OCR text
    - Categorizing products based on Vietnamese keywords
    """

    def __init__(self):
        self.country_prefixes = GS1_COUNTRY_PREFIXES
        self.categories = VN_PRODUCT_CATEGORIES
        self.packaging_patterns = VN_PACKAGING_PATTERNS
        self.nutrition_patterns = VN_NUTRITION_PATTERNS

    def is_vietnamese_barcode(self, barcode: str) -> bool:
        """Check if barcode is Vietnamese (GS1 prefix 893)."""
        if not barcode:
            return False
        cleaned = barcode.replace(" ", "").replace("-", "")
        return cleaned.startswith("893")

    def get_barcode_origin(self, barcode: str) -> Optional[Dict[str, Any]]:
        """
        Detect country of origin from barcode GS1 prefix.
        
        Args:
            barcode: EAN-13 barcode string
            
        Returns:
            Dictionary with country origin info
        """
        if not barcode:
            return None
            
        cleaned = barcode.replace(" ", "").replace("-", "")
        
        # Check if valid EAN barcode (8 or 13 digits)
        if len(cleaned) not in [8, 13]:
            return {
                "valid": False,
                "error": "Mã vạch không hợp lệ (cần 8 hoặc 13 chữ số)"
            }
        
        # Try to match country prefix (first 3 digits)
        prefix_3 = cleaned[:3]
        if prefix_3 in self.country_prefixes:
            country = self.country_prefixes[prefix_3]
            return {
                "valid": True,
                "barcode": cleaned,
                "gs1_prefix": prefix_3,
                "country": country,
                "is_vietnamese": country == "Vietnam",
                "note": f"Sản phẩm từ {country}" if country != "Vietnam" else "Sản phẩm Việt Nam"
            }
        
        # Unknown prefix - could be other countries
        return {
            "valid": True,
            "barcode": cleaned,
            "gs1_prefix": prefix_3,
            "country": None,
            "is_vietnamese": False,
            "note": f"Không xác định được quốc gia (prefix: {prefix_3})"
        }

    def lookup_barcode(self, barcode: str) -> Optional[Dict[str, Any]]:
        """
        Get barcode origin information.
        
        Note: For full product details, use the Backend API which calls
        external services (Open Food Facts, UPCitemdb).
        
        Args:
            barcode: EAN-13 barcode string
            
        Returns:
            Dictionary with barcode origin info
        """
        return self.get_barcode_origin(barcode)

    def detect_category(self, text: str) -> Tuple[Optional[str], float]:
        """
        Detect product category from OCR text.
        
        Args:
            text: OCR extracted text
            
        Returns:
            Tuple of (category_name, confidence)
        """
        if not text:
            return None, 0.0
            
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        
        for category, info in self.categories.items():
            score = 0
            # Check Vietnamese keywords (higher weight)
            for kw in info["keywords_vi"]:
                if kw in text_lower:
                    score += 2
            # Check English keywords
            for kw in info["keywords_en"]:
                if kw in text_lower:
                    score += 1
            if score > 0:
                scores[category] = score
        
        if not scores:
            return None, 0.0
            
        best_category = max(scores, key=scores.get)
        max_score = scores[best_category]
        confidence = min(1.0, max_score / 5.0)  # Normalize to 0-1
        
        return best_category, confidence

    def extract_weight(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract product weight/volume from text."""
        for pattern in self.packaging_patterns["weight"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).replace(",", ".")
                unit = match.group(2).lower()
                return {
                    "value": float(value),
                    "unit": unit,
                    "raw": match.group(0),
                }
        return None

    def extract_ingredients(self, text: str) -> Optional[List[str]]:
        """Extract ingredients list from text."""
        for pattern in self.packaging_patterns["ingredients"]:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                ingredients_text = match.group(1).strip()
                # Split by common separators
                ingredients = re.split(r'[,;،]', ingredients_text)
                # Clean up
                ingredients = [ing.strip() for ing in ingredients if ing.strip()]
                if ingredients:
                    return ingredients
        return None

    def extract_storage_instructions(self, text: str) -> Optional[str]:
        """Extract storage instructions from text."""
        for pattern in self.packaging_patterns["storage"]:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result = match.group(1).strip() if match.lastindex else match.group(0).strip()
                # Clean up the result
                result = re.sub(r'\s+', ' ', result)
                return result[:500]  # Limit length
        return None

    def extract_usage_instructions(self, text: str) -> Optional[str]:
        """Extract usage/cooking instructions from text."""
        for pattern in self.packaging_patterns["usage"]:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result = match.group(1).strip() if match.lastindex else match.group(0).strip()
                result = re.sub(r'\s+', ' ', result)
                return result[:500]
        return None

    def extract_manufacturer(self, text: str) -> Optional[Dict[str, str]]:
        """Extract manufacturer information from text."""
        result = {}
        
        # Extract manufacturer name
        for pattern in self.packaging_patterns["manufacturer"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["name"] = match.group(1).strip()[:200]
                break
        
        # Extract distributor
        for pattern in self.packaging_patterns.get("distributor", []):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["distributor"] = match.group(1).strip()[:200]
                break
        
        # Extract address
        for pattern in self.packaging_patterns.get("address", []):
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                address = match.group(1).strip()
                address = re.sub(r'\s+', ' ', address)
                result["address"] = address[:300]
                break
        
        # Extract contact info
        contacts = []
        for pattern in self.packaging_patterns.get("contact", []):
            matches = re.findall(pattern, text, re.IGNORECASE)
            contacts.extend(matches)
        if contacts:
            result["contact"] = contacts[:3]  # Max 3 contact items
        
        return result if result else None

    def extract_warnings(self, text: str) -> Optional[List[str]]:
        """Extract warnings and notes from text."""
        warnings = []
        for pattern in self.packaging_patterns.get("warnings", []):
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                warning = match.group(1).strip() if match.lastindex else match.group(0).strip()
                warning = re.sub(r'\s+', ' ', warning)
                if warning and len(warning) > 5:
                    warnings.append(warning[:200])
        return warnings if warnings else None

    def extract_product_codes(self, text: str) -> Optional[Dict[str, str]]:
        """Extract product codes, batch numbers, etc."""
        codes = {}
        for pattern in self.packaging_patterns.get("product_code", []):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                code = match.group(1).strip()
                if "MSKTVSTY" in pattern:
                    codes["msktvsty"] = code
                elif "lô" in pattern.lower() or "batch" in pattern.lower():
                    codes["batch"] = code
                else:
                    codes["sku"] = code
        return codes if codes else None

    def extract_quality_standards(self, text: str) -> Optional[List[str]]:
        """Extract quality standards and specifications."""
        standards = []
        for pattern in self.packaging_patterns.get("quality_standards", []):
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                standard = match.strip()
                standard = re.sub(r'\s+', ' ', standard)
                if standard and len(standard) > 3:
                    standards.append(standard[:150])
        return standards[:5] if standards else None  # Max 5 standards

    def extract_origin(self, text: str) -> Optional[str]:
        """Extract country/region of origin from text."""
        for pattern in self.packaging_patterns["origin"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def extract_certifications(self, text: str) -> List[str]:
        """Extract quality certifications from text."""
        certifications = []
        for pattern in self.packaging_patterns["certification"]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certifications.extend(matches)
        return list(set(certifications))

    def extract_nutrition_facts(self, text: str) -> Dict[str, Any]:
        """Extract nutrition facts from text."""
        nutrition = {}
        for nutrient, pattern in self.nutrition_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).replace(",", ".")
                try:
                    nutrition[nutrient] = float(value)
                except ValueError:
                    nutrition[nutrient] = value
        return nutrition

    def estimate_shelf_life(
        self, 
        category: Optional[str], 
        storage_type: str = "fresh"
    ) -> Optional[int]:
        """
        Estimate typical shelf life for a product category.
        
        Args:
            category: Product category
            storage_type: Type of storage (fresh, frozen, canned, etc.)
            
        Returns:
            Estimated shelf life in days
        """
        if not category or category not in self.categories:
            return None
            
        shelf_life = self.categories[category].get("shelf_life_days", {})
        return shelf_life.get(storage_type)

    def extract_all_packaging_info(self, text: str) -> Dict[str, Any]:
        """
        Extract all available information from packaging text.
        
        Args:
            text: Full OCR text from product packaging
            
        Returns:
            Dictionary with all extracted information
        """
        result = {
            "weight": self.extract_weight(text),
            "ingredients": self.extract_ingredients(text),
            "storage": self.extract_storage_instructions(text),
            "usage": self.extract_usage_instructions(text),
            "manufacturer": self.extract_manufacturer(text),
            "origin": self.extract_origin(text),
            "certifications": self.extract_certifications(text),
            "nutrition": self.extract_nutrition_facts(text),
            "warnings": self.extract_warnings(text),
            "product_codes": self.extract_product_codes(text),
            "quality_standards": self.extract_quality_standards(text),
        }
        
        # Detect category
        category, cat_confidence = self.detect_category(text)
        if category:
            result["detected_category"] = {
                "name": category,
                "confidence": cat_confidence,
                "keywords_vi": self.categories[category]["keywords_vi"][:5],
            }
        
        # Extract shelf life info from storage instructions
        if result["storage"]:
            shelf_life_match = re.search(
                r"(\d+)\s*(ngày|tháng|năm|days?|months?|years?)",
                result["storage"],
                re.IGNORECASE
            )
            if shelf_life_match:
                value = int(shelf_life_match.group(1))
                unit = shelf_life_match.group(2).lower()
                if "tháng" in unit or "month" in unit:
                    value *= 30
                elif "năm" in unit or "year" in unit:
                    value *= 365
                result["shelf_life_days"] = value
        
        return result


# Singleton instance
vn_product_service = VietnameseProductService()
