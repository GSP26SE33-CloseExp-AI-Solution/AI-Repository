from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from app.api.deps import get_api_key, get_user_id
from app.models.pricing import PricingRequest, PricingResponse
from app.services.pricing import suggest_price
from app.services.market_price_crawler import MarketPriceCrawlerService
from app.services.pricing_market_enrichment import enrich_pricing_request_with_market_crawl
from app.services.token_service import token_service

router = APIRouter()


# ============= Pricing Suggestion =============

@router.post(
    "/suggest",
    response_model=PricingResponse,
    summary="Suggest price for near-expiry product",
    description=(
        "Calculate optimal price based on expiry date, product type and market factors. "
        "Consumes **1 token** per request. Monthly budget: **150 tokens**."
    ),
)
async def suggest(
    payload: PricingRequest,
    user_id: str = Depends(get_user_id),
    _: str = Depends(get_api_key),
) -> PricingResponse:
    """
    Suggest optimal price for a near-expiry product.
    
    Takes into account:
    - Days until expiration
    - Product category
    - Demand index
    - Competitor pricing
    - Market price benchmarking (optionally auto-filled via crawler when min market price is missing)
    - Pricing strategy
    
    Returns suggested price with confidence score and rationale.
    
    **Token cost**: 1 token per call
    """
    # Check token budget before calling AI
    if not token_service.check_budget("pricing", user_id, 1):
        usage = token_service.get_usage("pricing", user_id)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "TOKEN_BUDGET_EXCEEDED",
                "message": (
                    f"Monthly Pricing token budget exceeded. "
                    f"Budget: {usage['budget']}, Used: {usage['used']}, "
                    f"Remaining: {usage['remaining']}. Resets next month."
                ),
                "token_usage": usage,
            },
        )

    enriched = await enrich_pricing_request_with_market_crawl(payload)
    result = suggest_price(enriched)

    # Consume token after successful result
    token_service.consume("pricing", user_id, 1)
    return result


@router.get(
    "/token-status",
    summary="Get Pricing token usage for current month",
    description="Quick access to Pricing token budget and remaining usage.",
)
async def get_pricing_token_status(
    user_id: str = Depends(get_user_id),
    _: str = Depends(get_api_key),
):
    """Get current month Pricing token usage."""
    return {
        "success": True,
        "data": token_service.get_usage("pricing", user_id),
    }



# ============= Market Price Crawling =============

class MarketPriceCrawlRequest(BaseModel):
    """Request to crawl market prices"""
    barcode: str = Field(..., description="Product barcode")
    product_name: Optional[str] = Field(None, description="Product name for search")
    deep_crawl: bool = Field(
        False, 
        description="Whether to crawl into each page for more accurate prices (slower but more accurate)"
    )


class CrawledPriceItem(BaseModel):
    """Individual price from a source"""
    source: str
    store_name: Optional[str] = None
    product_name: Optional[str] = None
    price: float
    original_price: Optional[float] = None
    url: Optional[str] = None
    unit: Optional[str] = None
    weight: Optional[str] = None
    is_in_stock: bool = True
    confidence: float = 0.5


class MarketPriceStats(BaseModel):
    """Statistics about market prices"""
    min_price: float
    max_price: float
    avg_price: float
    source_count: int
    sources: List[str]


class MarketPriceCrawlResponse(BaseModel):
    """Response from market price crawler"""
    success: bool
    barcode: str
    prices: List[CrawledPriceItem] = []
    stats: Optional[MarketPriceStats] = None
    processing_time_ms: float = 0.0
    error: Optional[str] = None


@router.post(
    "/crawl",
    response_model=MarketPriceCrawlResponse,
    summary="Crawl market prices from Google Search",
    description="Search Google for product prices and extract from various Vietnamese e-commerce sites",
)
async def crawl_market_prices(
    payload: MarketPriceCrawlRequest,
    _: str = Depends(get_api_key),
) -> MarketPriceCrawlResponse:
    """
    Crawl market prices using Google Search.
    
    Workflow:
    1. Search Google with barcode (or product name)
    2. Parse prices directly from Google search results (fast)
    3. Optionally deep crawl each page for more accurate prices (slow)
    
    Sources discovered dynamically from Google results:
    - LOTTE Mart, Vissan Mart, 7-Eleven
    - Bách Hóa Xanh, WinMart, Co.op
    - Shopee, Tiki, Lazada, etc.
    
    Returns list of prices with statistics.
    """
    import time
    start_time = time.perf_counter()
    
    try:
        crawler = MarketPriceCrawlerService()
        
        # Get prices from all sources
        prices = await crawler.get_market_prices(
            barcode=payload.barcode,
            product_name=payload.product_name,
            deep_crawl=payload.deep_crawl,
        )
        
        if not prices:
            return MarketPriceCrawlResponse(
                success=True,
                barcode=payload.barcode,
                prices=[],
                error="No prices found from any source",
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Convert to response format
        price_items = [
            CrawledPriceItem(
                source=p.source,
                store_name=p.store_name,
                product_name=p.product_name,
                price=p.price,
                original_price=p.original_price,
                url=p.source_url,
                unit=p.unit,
                weight=p.weight,
                is_in_stock=p.is_in_stock,
                confidence=p.confidence,
            )
            for p in prices
        ]
        
        # Calculate stats
        stats = crawler.get_price_stats(prices)
        
        return MarketPriceCrawlResponse(
            success=True,
            barcode=payload.barcode,
            prices=price_items,
            stats=MarketPriceStats(
                min_price=stats["min_price"],
                max_price=stats["max_price"],
                avg_price=stats["avg_price"],
                source_count=stats["source_count"],
                sources=stats["sources"],
            ),
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )
        
    except Exception as e:
        return MarketPriceCrawlResponse(
            success=False,
            barcode=payload.barcode,
            error=str(e),
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )


@router.get(
    "/market/{barcode}",
    response_model=MarketPriceCrawlResponse,
    summary="Get cached market prices for a barcode",
    description="Retrieve previously crawled market prices from cache",
)
async def get_market_prices(
    barcode: str,
    _: str = Depends(get_api_key),
) -> MarketPriceCrawlResponse:
    """
    Get cached market prices for a product.
    
    If no cached prices exist, triggers a fresh crawl.
    """
    # For now, just do a fresh crawl
    # In production, this would check cache first
    request = MarketPriceCrawlRequest(barcode=barcode)
    return await crawl_market_prices(request, _)
