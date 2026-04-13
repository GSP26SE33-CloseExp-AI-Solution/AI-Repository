"""
Test with real OCR data from API response.
"""

import sys
sys.path.insert(0, "d:/study/SP26/Capstone/AI-Repository")

from app.services.region_based_extractor import region_extractor


# Real text_regions from OCR API response
REAL_REGIONS = [
    {
        "text": "uhcil mang tinh chẩt minh họa",
        "confidence": 0.11229020971493926,
        "bounding_box": {"x1": 83, "y1": 216, "x2": 288, "y2": 242}
    },
    {
        "text": "THÀNH PHẨN: Nac bò (55 %), nước, sả,",
        "confidence": 0.8369331742069233,
        "bounding_box": {"x1": 304, "y1": 226, "x2": 616, "y2": 258}
    },
    {
        "text": "muoi i-ot, hành, tỏi, gừng; chẩt giữ ẩm (451i, 452i) ,",
        "confidence": 0.42408543011919175,
        "bounding_box": {"x1": 304, "y1": 254, "x2": 706, "y2": 286}
    },
    {
        "text": "chát điêu vị (621), chat",
        "confidence": 0.47622848253433464,
        "bounding_box": {"x1": 304, "y1": 284, "x2": 486, "y2": 312}
    },
    {
        "text": "oxy hoa (316).",
        "confidence": 0.9589977056853242,
        "bounding_box": {"x1": 536, "y1": 284, "x2": 654, "y2": 312}
    },
    {
        "text": "VISSAN",
        "confidence": 0.9997550523033388,
        "bounding_box": {"x1": 736, "y1": 262, "x2": 917, "y2": 324}
    },
    {
        "text": "HƯỚNG DẪN SỬ DUNG:",
        "confidence": 0.8987618944017636,
        "bounding_box": {"x1": 306, "y1": 312, "x2": 504, "y2": 344}
    },
    {
        "text": "ngay; co thể làm",
        "confidence": 0.6177753016384817,
        "bounding_box": {"x1": 348, "y1": 344, "x2": 480, "y2": 374}
    },
    {
        "text": "trươc khi dùng hoàc chể",
        "confidence": 0.41883857410603875,
        "bounding_box": {"x1": 522, "y1": 342, "x2": 714, "y2": 372}
    },
    {
        "text": "biên thành cảc món ăn khac",
        "confidence": 0.6407639527619462,
        "bounding_box": {"x1": 304, "y1": 372, "x2": 530, "y2": 400}
    },
    {
        "text": "HƯỚNG DẪN BẢO QUẢN:",
        "confidence": 0.7290759518058132,
        "bounding_box": {"x1": 305, "y1": 397, "x2": 517, "y2": 435}
    },
    {
        "text": "Noi thoang mảt, khong để nơi nóng, ẩm.",
        "confidence": 0.44888079522825414,
        "bounding_box": {"x1": 306, "y1": 430, "x2": 624, "y2": 462}
    },
    {
        "text": "8eq",
        "confidence": 0.10716270917656122,
        "bounding_box": {"x1": 882, "y1": 384, "x2": 984, "y2": 486}
    },
    {
        "text": "THÔNG TIN CẢNH BÁO:",
        "confidence": 0.804760325855619,
        "bounding_box": {"x1": 306, "y1": 460, "x2": 502, "y2": 492}
    },
    {
        "text": "TJNH: 150g",
        "confidence": 0.34411524458815346,
        "bounding_box": {"x1": 155, "y1": 476, "x2": 267, "y2": 521}
    },
    {
        "text": "Không làm",
        "confidence": 0.855090755757626,
        "bounding_box": {"x1": 306, "y1": 490, "x2": 392, "y2": 522}
    },
    {
        "text": "trực tiếp sản phẩm trong lò vi song;",
        "confidence": 0.7435878479141101,
        "bounding_box": {"x1": 434, "y1": 490, "x2": 710, "y2": 522}
    },
    {
        "text": "Khoi |",
        "confidence": 0.4067736013850422,
        "bounding_box": {"x1": 104, "y1": 510, "x2": 134, "y2": 536}
    },
    {
        "text": "Jrao: 75 g",
        "confidence": 0.23546994706126725,
        "bounding_box": {"x1": 158, "y1": 518, "x2": 218, "y2": 546}
    },
    {
        "text": "(HỈ TIÊU CHẤT LƯỢNG CHỦ YẾU:",
        "confidence": 0.6346746044827832,
        "bounding_box": {"x1": 305, "y1": 519, "x2": 581, "y2": 555}
    },
    {
        "text": "KSX và KSD; Kem ở đáy lon:",
        "confidence": 0.09237559503020609,
        "bounding_box": {"x1": 87, "y1": 537, "x2": 241, "y2": 575}
    },
    {
        "text": "NH;",
        "confidence": 0.7082642423220332,
        "bounding_box": {"x1": 306, "y1": 550, "x2": 340, "y2": 578}
    },
    {
        "text": "40 mg/100 g",
        "confidence": 0.7028919661636237,
        "bounding_box": {"x1": 354, "y1": 552, "x2": 460, "y2": 582}
    },
]

RAW_TEXT = """uhcil mang tinh chẩt minh họa
THÀNH PHẨN: Nac bò (55 %), nước, sả,
muoi i-ot, hành, tỏi, gừng; chẩt giữ ẩm (451i, 452i) ,
chát điêu vị (621), chat
oxy hoa (316).
VISSAN
HƯỚNG DẪN SỬ DUNG:
ngay; co thể làm
trươc khi dùng hoàc chể
biên thành cảc món ăn khac
HƯỚNG DẪN BẢO QUẢN:
Noi thoang mảt, khong để nơi nóng, ẩm.
8eq
THÔNG TIN CẢNH BÁO:
TJNH: 150g
Không làm
trực tiếp sản phẩm trong lò vi song;
Khoi |
Jrao: 75 g
(HỈ TIÊU CHẤT LƯỢNG CHỦ YẾU:
KSX và KSD; Kem ở đáy lon:
NH;
40 mg/100 g"""


def test_real_extraction():
    """Test region-based extraction with real OCR data."""
    print("=" * 70)
    print("TEST: Real OCR Data Extraction")
    print("=" * 70)
    
    result = region_extractor.extract_from_regions(REAL_REGIONS, RAW_TEXT)
    
    print("\n📦 EXTRACTION RESULTS:")
    print("-" * 50)
    
    print(f"\n🏷️ Name: {result.name}")
    print(f"   Confidence: {result.name_confidence:.2f}")
    
    print(f"\n🏢 Brand: {result.brand}")
    print(f"   Confidence: {result.brand_confidence:.2f}")
    
    print(f"\n🥩 Ingredients: {result.ingredients}")
    
    print(f"\n❄️ Storage: {result.storage_instructions}")
    
    print(f"\n📋 Usage: {result.usage_instructions}")
    
    print(f"\n⚠️ Warnings: {result.warnings}")
    
    print(f"\n⚖️ Weight: {result.weight}")
    print(f"   Net weight: {result.net_weight}")
    
    if result.detected_category:
        print(f"\n📂 Category: {result.detected_category.get('name')}")
        print(f"   Confidence: {result.detected_category.get('confidence', 0):.2f}")
    
    print(f"\n🏭 Manufacturer: {result.manufacturer}")
    
    print("\n" + "=" * 70)
    print("EXPECTED vs ACTUAL:")
    print("=" * 70)
    
    expected = {
        "name": "Bò hầm" or None,  # From product image context
        "brand": "VISSAN",
        "ingredients_contains": ["bò", "nước", "sả", "muối", "hành", "tỏi", "gừng"],
        "category": "meat",
    }
    
    print(f"\n✅ Brand correct: {result.brand == 'VISSAN'}")
    print(f"✅ Category is meat: {result.detected_category.get('name') == 'meat' if result.detected_category else False}")
    
    ingredients = result.ingredients or ""
    ingredients_check = any(k in ingredients.lower() for k in expected["ingredients_contains"])
    print(f"✅ Ingredients contains meat keywords: {ingredients_check}")


def debug_classification():
    """Debug how each region is classified."""
    print("\n" + "=" * 70)
    print("DEBUG: Region Classification")
    print("=" * 70)
    
    from app.services.region_based_extractor import RegionBasedExtractor
    
    extractor = RegionBasedExtractor()
    
    # Parse regions
    parsed = extractor._parse_regions(REAL_REGIONS)
    print(f"\nParsed {len(parsed)} regions")
    
    # Filter noise
    filtered = extractor._filter_noise(parsed)
    print(f"After noise filter: {len(filtered)} regions")
    
    # Classify each
    print("\nClassification results:")
    for region in filtered:
        field_type, conf = extractor._identify_field_type(region)
        print(f"  [{region.confidence:.2f}] {field_type.value:20} | {region.text[:50]}")


if __name__ == "__main__":
    debug_classification()
    print("\n")
    test_real_extraction()
