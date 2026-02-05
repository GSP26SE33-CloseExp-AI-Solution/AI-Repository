"""
API endpoints for fresh produce recognition (vegetables, fruits, meat, seafood).
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from app.api.deps import get_api_key
from app.core.exceptions import ImageProcessingError
from app.models.common import ImageInput


router = APIRouter()


class FreshProduceInfo(BaseModel):
    """Information about fresh produce."""
    
    category: str = Field(description="Product category (vegetable, fruit, meat, seafood)")
    name_vi: Optional[str] = Field(None, description="Vietnamese name")
    name_en: Optional[str] = Field(None, description="English name")
    typical_shelf_life_days: Optional[int] = Field(None, description="Typical shelf life in days")
    storage_recommendation: Optional[str] = Field(None, description="Storage recommendation")
    freshness_indicators: Optional[List[str]] = Field(None, description="How to identify freshness")
    confidence: float = Field(ge=0.0, le=1.0)


class FreshProduceResponse(BaseModel):
    """Response for fresh produce recognition."""
    
    detected_items: List[FreshProduceInfo]
    image_quality: Optional[Dict[str, Any]] = None
    processing_time_ms: float
    warnings: Optional[List[str]] = None


# Vietnamese fresh produce database
FRESH_PRODUCE_DB: Dict[str, Dict[str, Any]] = {
    # === VEGETABLES ===
    "rau_muong": {
        "category": "vegetable",
        "name_vi": "Rau muống",
        "name_en": "Water spinach",
        "typical_shelf_life_days": 3,
        "storage": "Bảo quản trong ngăn mát tủ lạnh, bọc khăn ẩm",
        "freshness": ["Lá xanh tươi, không héo", "Thân giòn, không mềm nhũn", "Không có đốm vàng hoặc nâu"],
    },
    "cai_ngot": {
        "category": "vegetable",
        "name_vi": "Cải ngọt",
        "name_en": "Choy sum",
        "typical_shelf_life_days": 5,
        "storage": "Bảo quản trong ngăn mát tủ lạnh",
        "freshness": ["Lá xanh tươi", "Thân còn giòn", "Không có vết sâu bệnh"],
    },
    "ca_chua": {
        "category": "vegetable",
        "name_vi": "Cà chua",
        "name_en": "Tomato",
        "typical_shelf_life_days": 7,
        "storage": "Để nơi thoáng mát, tránh ánh nắng trực tiếp",
        "freshness": ["Màu đỏ đều", "Vỏ căng mọng", "Không có vết thâm"],
    },
    "khoai_tay": {
        "category": "vegetable",
        "name_vi": "Khoai tây",
        "name_en": "Potato",
        "typical_shelf_life_days": 21,
        "storage": "Bảo quản nơi khô ráo, thoáng mát, tránh ánh sáng",
        "freshness": ["Vỏ không có vết xanh", "Không mọc mầm", "Không bị mềm"],
    },
    "hanh_la": {
        "category": "vegetable",
        "name_vi": "Hành lá",
        "name_en": "Green onion",
        "typical_shelf_life_days": 7,
        "storage": "Bọc giấy ẩm, để ngăn mát tủ lạnh",
        "freshness": ["Lá xanh tươi", "Củ trắng sạch", "Không héo"],
    },
    "toi": {
        "category": "vegetable",
        "name_vi": "Tỏi",
        "name_en": "Garlic",
        "typical_shelf_life_days": 60,
        "storage": "Để nơi khô ráo, thoáng mát",
        "freshness": ["Tép tỏi cứng chắc", "Vỏ khô", "Không mọc mầm"],
    },
    "bi_do": {
        "category": "vegetable",
        "name_vi": "Bí đỏ",
        "name_en": "Pumpkin",
        "typical_shelf_life_days": 30,
        "storage": "Để nơi khô ráo, thoáng mát",
        "freshness": ["Vỏ cứng, không trầy xước", "Cuống khô", "Không có vết mốc"],
    },
    
    # === FRUITS ===
    "chuoi": {
        "category": "fruit",
        "name_vi": "Chuối",
        "name_en": "Banana",
        "typical_shelf_life_days": 5,
        "storage": "Để nhiệt độ phòng, tránh tủ lạnh",
        "freshness": ["Vỏ vàng đều (khi chín)", "Không có vết thâm lớn", "Cuống còn xanh"],
    },
    "xoai": {
        "category": "fruit",
        "name_vi": "Xoài",
        "name_en": "Mango",
        "typical_shelf_life_days": 7,
        "storage": "Chín: tủ lạnh. Xanh: để ngoài cho chín",
        "freshness": ["Vỏ căng mịn", "Có mùi thơm đặc trưng", "Không có vết thâm"],
    },
    "cam": {
        "category": "fruit",
        "name_vi": "Cam",
        "name_en": "Orange",
        "typical_shelf_life_days": 14,
        "storage": "Bảo quản trong ngăn mát tủ lạnh",
        "freshness": ["Vỏ căng bóng", "Nặng tay (nhiều nước)", "Không héo, không mốc"],
    },
    "thanh_long": {
        "category": "fruit",
        "name_vi": "Thanh long",
        "name_en": "Dragon fruit",
        "typical_shelf_life_days": 7,
        "storage": "Bảo quản trong ngăn mát tủ lạnh",
        "freshness": ["Vỏ đỏ/hồng tươi", "Tai xanh", "Thân cứng"],
    },
    "buoi": {
        "category": "fruit",
        "name_vi": "Bưởi",
        "name_en": "Pomelo",
        "typical_shelf_life_days": 21,
        "storage": "Để nơi thoáng mát hoặc tủ lạnh",
        "freshness": ["Vỏ căng", "Nặng tay", "Không héo"],
    },
    
    # === MEAT ===
    "thit_heo": {
        "category": "meat",
        "name_vi": "Thịt heo",
        "name_en": "Pork",
        "typical_shelf_life_days": 3,
        "storage": "Ngăn mát: 2-3 ngày. Ngăn đông: 3-6 tháng",
        "freshness": ["Màu hồng tươi", "Không có mùi hôi", "Thịt đàn hồi khi ấn"],
    },
    "thit_bo": {
        "category": "meat",
        "name_vi": "Thịt bò",
        "name_en": "Beef",
        "typical_shelf_life_days": 3,
        "storage": "Ngăn mát: 2-3 ngày. Ngăn đông: 4-6 tháng",
        "freshness": ["Màu đỏ tươi", "Vân mỡ phân bố đều", "Không có mùi lạ"],
    },
    "thit_ga": {
        "category": "meat",
        "name_vi": "Thịt gà",
        "name_en": "Chicken",
        "typical_shelf_life_days": 2,
        "storage": "Ngăn mát: 1-2 ngày. Ngăn đông: 3-4 tháng",
        "freshness": ["Da vàng nhạt hoặc hồng", "Thịt đàn hồi", "Không có mùi hôi"],
    },
    
    # === SEAFOOD ===
    "ca": {
        "category": "seafood",
        "name_vi": "Cá",
        "name_en": "Fish",
        "typical_shelf_life_days": 2,
        "storage": "Ngăn mát: 1-2 ngày. Ngăn đông: 2-3 tháng",
        "freshness": ["Mắt trong, lồi", "Mang đỏ tươi", "Thịt đàn hồi", "Không có mùi tanh nặng"],
    },
    "tom": {
        "category": "seafood",
        "name_vi": "Tôm",
        "name_en": "Shrimp",
        "typical_shelf_life_days": 2,
        "storage": "Ngăn mát: 1-2 ngày. Ngăn đông: 3-6 tháng",
        "freshness": ["Vỏ sáng bóng", "Thân cứng, không mềm", "Không có mùi hôi"],
    },
    "muc": {
        "category": "seafood",
        "name_vi": "Mực",
        "name_en": "Squid",
        "typical_shelf_life_days": 2,
        "storage": "Ngăn mát: 1-2 ngày. Ngăn đông: 2-3 tháng",
        "freshness": ["Màu trắng trong", "Thịt đàn hồi", "Không có mùi hôi"],
    },
}


# YOLO class to Vietnamese produce mapping
YOLO_TO_VN_PRODUCE: Dict[str, Optional[str]] = {
    # Fruits
    "banana": "chuoi",
    "apple": "tao",
    "orange": "cam",
    
    # Vegetables
    "carrot": "ca_rot",
    "broccoli": "bong_cai_xanh",
    
    # Generic mappings (will be refined based on context)
    "fruit": None,  # Use vision service to identify specific fruit
    "vegetable": None,  # Use vision service to identify specific vegetable
}


@router.post(
    "/identify",
    response_model=FreshProduceResponse,
    summary="Identify fresh produce from image",
    description="Identify vegetables, fruits, meat, or seafood and provide freshness guidelines",
)
async def identify_fresh_produce(
    request: ImageInput,
    _: str = Depends(get_api_key),
) -> FreshProduceResponse:
    """
    Identify fresh produce from image.
    
    - Identifies vegetables, fruits, meat, and seafood
    - Provides Vietnamese and English names
    - Returns storage recommendations and shelf life
    - Includes freshness indicators
    """
    import time
    from app.services.vision import analyze_product_image
    from app.models.vision import VisionAnalyzeRequest
    
    start_time = time.perf_counter()
    warnings: List[str] = []
    detected_items: List[FreshProduceInfo] = []
    
    try:
        # Use vision service to detect objects
        vision_request = VisionAnalyzeRequest(
            image_url=request.image_url,
            image_b64=request.image_b64,
            assess_quality=True,
        )
        vision_result = analyze_product_image(vision_request)
        
        # Process detections
        seen_categories = set()
        for detection in vision_result.detections or []:
            class_name = detection.class_name.lower()
            
            # Map YOLO class to Vietnamese produce
            vn_key = YOLO_TO_VN_PRODUCE.get(class_name)
            
            if vn_key and vn_key in FRESH_PRODUCE_DB:
                produce_info = FRESH_PRODUCE_DB[vn_key]
                
                if vn_key not in seen_categories:
                    seen_categories.add(vn_key)
                    detected_items.append(FreshProduceInfo(
                        category=produce_info["category"],
                        name_vi=produce_info["name_vi"],
                        name_en=produce_info["name_en"],
                        typical_shelf_life_days=produce_info["typical_shelf_life_days"],
                        storage_recommendation=produce_info["storage"],
                        freshness_indicators=produce_info["freshness"],
                        confidence=detection.confidence,
                    ))
            else:
                # Try to match by product type
                product_type = detection.product_type
                if product_type in ["fruit", "vegetable", "meat"]:
                    # Add generic info based on type
                    if product_type not in seen_categories:
                        seen_categories.add(product_type)
                        detected_items.append(FreshProduceInfo(
                            category=product_type,
                            name_vi=None,
                            name_en=class_name.title(),
                            typical_shelf_life_days=None,
                            storage_recommendation=None,
                            freshness_indicators=None,
                            confidence=detection.confidence,
                        ))
        
        if not detected_items:
            warnings.append("Không phát hiện sản phẩm tươi sống trong hình ảnh")
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return FreshProduceResponse(
            detected_items=detected_items,
            image_quality=vision_result.image_quality.model_dump() if vision_result.image_quality else None,
            processing_time_ms=round(processing_time, 2),
            warnings=warnings if warnings else None,
        )
        
    except ImageProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}") from e


@router.get(
    "/categories",
    summary="Get fresh produce categories",
    description="Get list of supported fresh produce categories with Vietnamese info",
)
async def get_categories(_: str = Depends(get_api_key)) -> Dict[str, List[Dict[str, Any]]]:
    """Get all supported fresh produce categories."""
    result: Dict[str, List[Dict[str, Any]]] = {
        "vegetable": [],
        "fruit": [],
        "meat": [],
        "seafood": [],
    }
    
    for key, info in FRESH_PRODUCE_DB.items():
        category = info["category"]
        if category in result:
            result[category].append({
                "key": key,
                "name_vi": info["name_vi"],
                "name_en": info["name_en"],
                "shelf_life_days": info["typical_shelf_life_days"],
            })
    
    return result


@router.get(
    "/info/{produce_key}",
    summary="Get fresh produce info by key",
    description="Get detailed information about a specific fresh produce item",
)
async def get_produce_info(
    produce_key: str,
    _: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """Get detailed info for a specific produce item."""
    if produce_key not in FRESH_PRODUCE_DB:
        raise HTTPException(
            status_code=404,
            detail=f"Produce '{produce_key}' not found. Use /categories to see available items.",
        )
    
    return FRESH_PRODUCE_DB[produce_key]
