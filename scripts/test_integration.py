#!/usr/bin/env python3
"""
Integration Test Script for CloseExp AI Service + Backend
Run this script to verify the integration is working correctly.

Prerequisites:
1. AI Service running on http://localhost:8000
2. Backend running on http://localhost:5000

Usage:
    python scripts/test_integration.py
"""

import requests
import json
import time
from datetime import datetime, timedelta

# Configuration
AI_SERVICE_URL = "http://localhost:8000"
BACKEND_URL = "http://localhost:5000"
API_KEY = "dev-api-key-for-testing"

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}


def test_ai_service_health():
    """Test AI Service health check directly"""
    print("\n" + "="*50)
    print("Testing AI Service Health...")
    print("="*50)
    
    try:
        response = requests.get(f"{AI_SERVICE_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_backend_ai_health():
    """Test Backend AI health endpoint"""
    print("\n" + "="*50)
    print("Testing Backend AI Health Endpoint...")
    print("="*50)
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/ai/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_ocr_direct():
    """Test OCR extraction directly on AI Service"""
    print("\n" + "="*50)
    print("Testing OCR Extraction (Direct to AI Service)...")
    print("="*50)
    
    payload = {
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
        "extract_dates": True,
        "extract_barcode": True
    }
    
    try:
        response = requests.post(
            f"{AI_SERVICE_URL}/api/v1/ocr/extract",
            json=payload,
            headers=HEADERS,
            timeout=30
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, default=str)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_pricing_direct():
    """Test pricing suggestion directly on AI Service"""
    print("\n" + "="*50)
    print("Testing Pricing Suggestion (Direct to AI Service)...")
    print("="*50)
    
    payload = {
        "product_type": "food",
        "days_to_expire": 7,
        "base_price": 100000,
        "strategy": "balanced"
    }
    
    try:
        response = requests.post(
            f"{AI_SERVICE_URL}/api/v1/pricing/suggest",
            json=payload,
            headers=HEADERS,
            timeout=30
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, default=str)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_backend_pricing():
    """Test pricing through Backend"""
    print("\n" + "="*50)
    print("Testing Pricing (Through Backend)...")
    print("="*50)
    
    expiry_date = (datetime.utcnow() + timedelta(days=10)).isoformat()
    
    payload = {
        "category": "food",
        "expiryDate": expiry_date,
        "originalPrice": 150000,
        "brand": "Test Brand"
    }
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/ai/pricing",
            json=payload,
            timeout=30
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, default=str)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_vision_direct():
    """Test vision detection directly on AI Service"""
    print("\n" + "="*50)
    print("Testing Vision Detection (Direct to AI Service)...")
    print("="*50)
    
    payload = {
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
        "min_confidence": 0.25,
        "return_annotated_image": False
    }
    
    try:
        response = requests.post(
            f"{AI_SERVICE_URL}/api/v1/vision/detect",
            json=payload,
            headers=HEADERS,
            timeout=30
        )
        print(f"Status Code: {response.status_code}")
        # Don't print full response as it may contain base64 image
        data = response.json()
        if "annotated_image_b64" in data:
            data["annotated_image_b64"] = "[BASE64_IMAGE_TRUNCATED]"
        print(f"Response: {json.dumps(data, indent=2, default=str)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "#"*60)
    print("CloseExp AI Integration Tests")
    print("#"*60)
    print(f"AI Service URL: {AI_SERVICE_URL}")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Started at: {datetime.now().isoformat()}")
    
    results = {}
    
    # Test 1: AI Service Health
    results["AI Service Health"] = test_ai_service_health()
    
    # Test 2: Backend AI Health
    results["Backend AI Health"] = test_backend_ai_health()
    
    # Test 3: OCR Direct
    results["OCR Direct"] = test_ocr_direct()
    
    # Test 4: Pricing Direct
    results["Pricing Direct"] = test_pricing_direct()
    
    # Test 5: Backend Pricing
    results["Backend Pricing"] = test_backend_pricing()
    
    # Test 6: Vision Direct
    results["Vision Direct"] = test_vision_direct()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-"*60)
    print(f"Total: {passed}/{len(results)} tests passed")
    
    if failed == 0:
        print("\n🎉 All tests passed! Integration is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the logs above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
