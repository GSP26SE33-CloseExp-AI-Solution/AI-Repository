"""
Enrich pricing requests with market prices via crawler when the client did not supply them.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from app.models.pricing import PricingRequest
from app.services.market_price_crawler import market_price_crawler

logger = get_logger(__name__)


async def enrich_pricing_request_with_market_crawl(request: PricingRequest) -> PricingRequest:
    """
    If min_market_price is missing and barcode or product_name is present, run the
    market price crawler and fill min/avg/source for benchmarking inside suggest_price.
    """
    if not settings.pricing_auto_crawl_enabled:
        return request

    if request.min_market_price is not None and request.min_market_price > 0:
        return request

    barcode = (request.barcode or "").strip()
    product_name = (request.product_name or "").strip()
    if not barcode and not product_name:
        return request

    try:
        prices = await market_price_crawler.get_market_prices(
            barcode=barcode or None,
            product_name=product_name or None,
            deep_crawl=settings.pricing_auto_crawl_deep,
        )
    except Exception as exc:
        logger.warning("Auto market crawl failed: %s", exc)
        return request

    if not prices:
        logger.info("Auto market crawl returned no prices for barcode=%s", barcode or "(none)")
        return request

    stats = market_price_crawler.get_price_stats(prices)
    min_p = float(stats.get("min_price") or 0)
    if min_p <= 0:
        return request

    avg_p = stats.get("avg_price")
    avg_f = float(avg_p) if avg_p and float(avg_p) > 0 else None

    sources = stats.get("sources") or []
    source_hint = "ai_auto_crawl"
    if sources:
        source_hint = f"ai_auto_crawl:{','.join(sources[:5])}"[:100]

    logger.info(
        "Auto market crawl filled min=%s avg=%s (sources=%s)",
        min_p,
        avg_f,
        len(sources),
    )

    updates = {
        "min_market_price": min_p,
        "avg_market_price": avg_f,
        "market_price_source": source_hint,
    }
    if hasattr(request, "model_copy"):
        return request.model_copy(update=updates)
    return request.copy(update=updates)
