"""
Test script for region-based extractor.
Tests extraction from text_regions with confidence filtering.
"""

import sys
sys.path.insert(0, "d:/study/SP26/Capstone/AI-Repository")

from app.services.region_based_extractor import region_extractor


# Sample text_regions from OCR API response
SAMPLE_REGIONS = [
    {"text": "VISSAN", "confidence": 0.999, "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 80}},
    {"text": "BÒ HẦM", "confidence": 0.95, "bbox": {"x1": 120, "y1": 85, "x2": 220, "y2": 120}},
    {"text": "THÀNH PHẨN: Nac bò, mỡ, muối, đường, tiêu, ớt", "confidence": 0.837, "bbox": {"x1": 50, "y1": 200, "x2": 300, "y2": 250}},
    {"text": "Noi thoang mảt, nhiệt độ dưới 25°C", "confidence": 0.449, "bbox": {"x1": 50, "y1": 300, "x2": 300, "y2": 330}},
    {"text": "HDSD: Dùng trực tiếp hoặc hâm nóng", "confidence": 0.721, "bbox": {"x1": 50, "y1": 340, "x2": 300, "y2": 370}},
    {"text": "T.LƯỢNG TỊNH: 150g", "confidence": 0.88, "bbox": {"x1": 50, "y1": 400, "x2": 200, "y2": 430}},
    {"text": "8eq", "confidence": 0.107, "bbox": {"x1": 10, "y1": 450, "x2": 30, "y2": 470}},
    {"text": "uhcil mang tinh", "confidence": 0.112, "bbox": {"x1": 300, "y1": 460, "x2": 380, "y2": 480}},
    {"text": "SX bởi: CÔNG TY VISSAN", "confidence": 0.85, "bbox": {"x1": 50, "y1": 480, "x2": 250, "y2": 510}},
    {"text": "NSX: 01/01/2024 HSD: 01/07/2024", "confidence": 0.92, "bbox": {"x1": 50, "y1": 520, "x2": 280, "y2": 550}},
]

RAW_TEXT = """
VISSAN
BÒ HẦM
THÀNH PHẨN: Nac bò, mỡ, muối, đường, tiêu, ớt
Noi thoang mảt, nhiệt độ dưới 25°C
HDSD: Dùng trực tiếp hoặc hâm nóng
T.LƯỢNG TỊNH: 150g
8eq
uhcil mang tinh
SX bởi: CÔNG TY VISSAN
NSX: 01/01/2024 HSD: 01/07/2024
"""


def test_region_extraction():
    """Test region-based extraction."""
    print("=" * 60)
    print("TEST: Region-Based Extraction")
    print("=" * 60)
    
    result = region_extractor.extract_from_regions(SAMPLE_REGIONS, RAW_TEXT)
    
    print("\n📦 EXTRACTION RESULTS:")
    print("-" * 40)
    
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
        print(f"   Keywords: {result.detected_category.get('keywords_vi')}")
    
    print(f"\n🏭 Manufacturer: {result.manufacturer}")
    
    print("\n" + "=" * 60)
    print("FILTERED REGIONS (noise removed):")
    print("=" * 60)
    
    # Show what regions were kept vs filtered
    print("\n✅ Kept regions:")
    for r in SAMPLE_REGIONS:
        if r["confidence"] >= 0.15:
            print(f"   [{r['confidence']:.3f}] {r['text'][:50]}")
    
    print("\n❌ Filtered (noise):")
    for r in SAMPLE_REGIONS:
        if r["confidence"] < 0.15:
            print(f"   [{r['confidence']:.3f}] {r['text'][:50]}")


def test_noise_filtering():
    """Test that low-confidence regions are filtered."""
    print("\n" + "=" * 60)
    print("TEST: Noise Filtering")
    print("=" * 60)
    
    noise_regions = [
        {"text": "8eq", "confidence": 0.107, "bbox": None},
        {"text": "uhcil mang tinh", "confidence": 0.112, "bbox": None},
        {"text": "...", "confidence": 0.05, "bbox": None},
        {"text": "VISSAN", "confidence": 0.999, "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 80}},
    ]
    
    result = region_extractor.extract_from_regions(noise_regions, "VISSAN")
    
    print(f"\nInput: 4 regions (3 noise, 1 good)")
    print(f"Brand extracted: {result.brand}")
    
    # Check that noise didn't pollute results
    assert result.brand == "VISSAN", "Should extract VISSAN as brand"
    assert "8eq" not in (result.warnings or ""), "Noise should be filtered"
    print("✅ Noise filtering working correctly")


def test_category_detection():
    """Test Vietnamese category detection from ingredients."""
    print("\n" + "=" * 60)
    print("TEST: Category Detection")
    print("=" * 60)
    
    # Test meat detection
    meat_regions = [
        {"text": "THÀNH PHẦN: Thịt heo, mỡ, muối", "confidence": 0.9, "bbox": None},
    ]
    result = region_extractor.extract_from_regions(meat_regions, "Thịt heo xay")
    print(f"\nMeat product category: {result.detected_category}")
    
    # Test vegetable detection
    veg_regions = [
        {"text": "THÀNH PHẦN: Rau xà lách, cà chua", "confidence": 0.9, "bbox": None},
    ]
    result = region_extractor.extract_from_regions(veg_regions, "Rau tươi")
    print(f"Vegetable product category: {result.detected_category}")
    
    # Test dairy detection
    dairy_regions = [
        {"text": "THÀNH PHẦN: Sữa tươi, đường", "confidence": 0.9, "bbox": None},
    ]
    result = region_extractor.extract_from_regions(dairy_regions, "Sữa tươi")
    print(f"Dairy product category: {result.detected_category}")
    
    print("\n✅ Category detection tests completed")


def test_field_classification():
    """Test that regions are classified to correct fields."""
    print("\n" + "=" * 60)
    print("TEST: Field Classification")
    print("=" * 60)
    
    regions = [
        {"text": "HƯỚNG DẪN SỬ DỤNG: Rã đông trước khi dùng", "confidence": 0.8, "bbox": None},
        {"text": "BẢO QUẢN: Nơi khô ráo, thoáng mát", "confidence": 0.85, "bbox": None},
        {"text": "CẢNH BÁO: Không dành cho trẻ dưới 3 tuổi", "confidence": 0.75, "bbox": None},
        {"text": "THÀNH PHẦN: Bột mì, đường, trứng", "confidence": 0.9, "bbox": None},
    ]
    
    result = region_extractor.extract_from_regions(regions, "Sample text")
    
    print(f"\n📋 Usage: {result.usage_instructions}")
    print(f"❄️ Storage: {result.storage_instructions}")
    print(f"⚠️ Warnings: {result.warnings}")
    print(f"🥄 Ingredients: {result.ingredients}")
    
    assert "Rã đông" in (result.usage_instructions or ""), "Usage should be extracted"
    assert "khô ráo" in (result.storage_instructions or ""), "Storage should be extracted"
    assert "trẻ dưới 3 tuổi" in (result.warnings or ""), "Warning should be extracted"
    assert "Bột mì" in (result.ingredients or ""), "Ingredients should be extracted"
    
    print("\n✅ Field classification working correctly")


if __name__ == "__main__":
    test_region_extraction()
    test_noise_filtering()
    test_category_detection()
    test_field_classification()
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS COMPLETED!")
    print("=" * 60)
