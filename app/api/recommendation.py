import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Any
from app.models.recommendation import (
    RecommendationRequest, 
    StructuredSearchCriteria,
    RankStockLotsRequest,
    RankStockLotsResponse,
    RankedStockLotDto
)
from app.services.recommendation import recommendation_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/parse", response_model=StructuredSearchCriteria)
async def parse_recommendation_query(request: RecommendationRequest) -> Any:
    """
    Phân tích yêu cầu tìm kiếm tự nhiên của người dùng để trả về tiêu chí StructuredSearchCriteria.
    """
    logger.info(f"Parsing natural language recommendation request: {request.query_text}")
    
    result = await recommendation_service.parse_search_query(request.query_text)
    
    if not result:
        raise HTTPException(status_code=500, detail="Không thể phân tích yêu cầu từ người dùng.")
        
    return result


@router.post("/rank-stocklots", response_model=RankStockLotsResponse)
async def rank_stocklots_endpoint(request: RankStockLotsRequest) -> Any:
    """
    Xếp hạng danh sách stocklots dựa trên mức độ phù hợp với yêu cầu tìm kiếm tự nhiên.
    
    - Sử dụng Gemini AI để đánh giá từng stocklot
    - Trả về danh sách được sắp xếp theo điểm phù hợp (từ cao nhất đến thấp nhất)
    - Mỗi kết quả có kèm lý do (reason) giải thích tại sao sản phẩm phù hợp
    """
    logger.info(f"Ranking {len(request.stocklots)} stocklots for query: {request.query_text}")
    
    if not request.stocklots:
        return RankStockLotsResponse(ranked_stocklots=[], total_ranked=0)
    
    # Convert to dict format for ranking service
    stocklots_data = [sl.model_dump() for sl in request.stocklots]
    
    ranked_items = await recommendation_service.rank_stocklots_by_query(
        request.query_text, 
        stocklots_data
    )
    
    if not ranked_items:
        logger.warning(f"No stocklots ranked for query: {request.query_text}")
        return RankStockLotsResponse(ranked_stocklots=[], total_ranked=0)
    
    # Convert to response DTOs
    ranked_dtos = [
        RankedStockLotDto(
            lot_id=item.get("lot_id"),
            relevance_score=item.get("relevance_score", 0.0),
            reason=item.get("reason")
        )
        for item in ranked_items
    ]
    
    return RankStockLotsResponse(
        ranked_stocklots=ranked_dtos,
        total_ranked=len(ranked_dtos)
    )
