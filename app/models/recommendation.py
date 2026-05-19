from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class RecommendationRequest(BaseModel):
    query_text: str = Field(..., description="Yêu cầu tìm kiếm sản phẩm bằng ngôn ngữ tự nhiên")

class StructuredSearchCriteria(BaseModel):
    category: Optional[str] = Field(None, description="Danh mục sản phẩm (vd: dairy, meat, seafood, bakery, produce, frozen, beverage, snack, condiment, other)")
    keyword: Optional[str] = Field(None, description="Từ khóa tìm kiếm (tên sản phẩm, thương hiệu)")
    max_price: Optional[float] = Field(None, description="Giá tối đa")
    min_price: Optional[float] = Field(None, description="Giá tối thiểu")
    max_days_to_expire: Optional[int] = Field(None, description="Số ngày tối đa còn lại đến hạn sử dụng")


# ============ StockLot Ranking Models ============

class StockLotInputDto(BaseModel):
    """Input DTO for a single stocklot to be ranked"""
    lot_id: str
    product_id: str
    product_name: str
    barcode: Optional[str] = None
    category_name: Optional[str] = None
    brand: Optional[str] = None
    quantity: float
    unit_name: str
    price: float
    expiry_date: datetime
    manufacture_date: Optional[datetime] = None


class RankStockLotsRequest(BaseModel):
    """Request to rank stocklots based on natural language query"""
    query_text: str = Field(..., description="Yêu cầu tìm kiếm sản phẩm bằng ngôn ngữ tự nhiên")
    stocklots: List[StockLotInputDto] = Field(..., description="Danh sách stocklots cần xếp hạng")


class RankedStockLotDto(BaseModel):
    """Ranked stocklot with relevance score"""
    lot_id: str
    relevance_score: float = Field(ge=0.0, le=1.0, description="Điểm tương thích từ 0-1")
    reason: Optional[str] = Field(None, description="Lý do tại sao sản phẩm này phù hợp")


class RankStockLotsResponse(BaseModel):
    """Response with ranked stocklots"""
    ranked_stocklots: List[RankedStockLotDto] = Field(..., description="Danh sách stocklots đã xếp hạng")
    total_ranked: int = Field(..., description="Tổng số stocklots được xếp hạng")
