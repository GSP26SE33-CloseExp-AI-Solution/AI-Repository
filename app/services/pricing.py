from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.exceptions import PricingCalculationError
from app.core.logging import get_logger
from app.models.pricing import (
    DemandLevel,
    PriceBreakdown,
    PricingRequest,
    PricingResponse,
    PricingStrategy,
    ProductCategory,
)

logger = get_logger(__name__)


class PricingService:
    """Service for calculating suggested prices for near-expiry products."""

    # Category-specific decay rates (faster decay for perishables)
    CATEGORY_DECAY_MULTIPLIERS = {
        ProductCategory.DAIRY: 1.2,
        ProductCategory.MEAT: 1.3,
        ProductCategory.SEAFOOD: 1.4,
        ProductCategory.BAKERY: 1.3,
        ProductCategory.PRODUCE: 1.2,
        ProductCategory.FROZEN: 0.9,
        ProductCategory.BEVERAGE: 0.8,
        ProductCategory.SNACK: 0.7,
        ProductCategory.CONDIMENT: 0.6,
        ProductCategory.OTHER: 1.0,
    }

    # Strategy adjustments (additional discount %)
    STRATEGY_ADJUSTMENTS = {
        PricingStrategy.AGGRESSIVE: 0.15,
        PricingStrategy.BALANCED: 0.0,
        PricingStrategy.CONSERVATIVE: -0.10,
    }

    def __init__(self) -> None:
        self.min_decay = settings.pricing_min_decay_factor
        self.max_decay = settings.pricing_max_decay_factor
        self.default_confidence = settings.pricing_default_confidence
        
        # Decay function theo % thời gian còn lại của HSD
        # Key: % còn lại, Value: % giảm giá
        self.DECAY_SCHEDULE = {
            100: 0,    # Còn 100% hạn -> Không giảm
            80: 10,    # Còn 80% hạn -> Giảm 10%
            60: 20,    # Còn 60% hạn -> Giảm 20%
            40: 35,    # Còn 40% hạn -> Giảm 35%
            20: 50,    # Còn 20% hạn -> Giảm 50%
            10: 65,    # Còn 10% hạn -> Giảm 65%
            5: 75,     # Còn 5% hạn -> Giảm 75%
            0: 85,     # Hết hạn -> Giảm 85%
        }

    def _calculate_decay_from_shelf_life(
        self,
        days_to_expire: int,
        total_shelf_life_days: int = 30,
    ) -> float:
        """
        Tính % giảm giá dựa trên % thời gian còn lại của HSD.
        
        Args:
            days_to_expire: Số ngày còn lại đến HSD
            total_shelf_life_days: Tổng thời gian bảo quản (mặc định 30 ngày)
        
        Returns:
            Discount percent (0-100)
        """
        if days_to_expire <= 0:
            return 85  # Hết hạn
        
        if total_shelf_life_days <= 0:
            total_shelf_life_days = 30
        
        # Tính % còn lại
        percent_remaining = min(100, (days_to_expire / total_shelf_life_days) * 100)
        
        # Tìm mức giảm phù hợp từ schedule
        discount = 0
        for threshold, disc in sorted(self.DECAY_SCHEDULE.items(), reverse=True):
            if percent_remaining >= threshold:
                discount = disc
                break
        
        return discount

    def _calculate_base_decay(self, days_to_expire: int) -> float:
        """
        Calculate base decay factor based on days to expiry.
        
        Uses exponential decay with soft caps.
        """
        if days_to_expire <= 0:
            return self.min_decay
        if days_to_expire >= 30:
            return self.max_decay

        # Exponential decay: faster decrease as expiry approaches
        # decay = max_decay * (1 - e^(-k * days))
        k = 0.15  # Decay rate constant
        decay = self.max_decay * (1 - math.exp(-k * days_to_expire))

        return max(self.min_decay, min(self.max_decay, decay))

    def _apply_category_multiplier(
        self,
        decay: float,
        category: ProductCategory,
    ) -> float:
        """Apply category-specific multiplier to decay factor."""
        multiplier = self.CATEGORY_DECAY_MULTIPLIERS.get(category, 1.0)
        adjusted = decay * multiplier

        # More aggressive discount for perishables
        if multiplier > 1.0:
            adjusted = min(adjusted, decay - 0.05)

        return max(self.min_decay, min(self.max_decay, adjusted))

    def _apply_demand_adjustment(
        self,
        decay: float,
        demand_index: Optional[float],
    ) -> tuple[float, float]:
        """
        Adjust decay based on demand.
        
        Returns (adjusted_decay, adjustment_factor)
        """
        if demand_index is None:
            return decay, 0.0

        # High demand = less discount needed
        # Low demand = more discount to move inventory
        if demand_index > 1.2:  # High demand
            adjustment = 0.05
        elif demand_index < 0.8:  # Low demand
            adjustment = -0.10
        else:
            adjustment = 0.0

        adjusted = decay + adjustment
        return max(self.min_decay, min(self.max_decay, adjusted)), adjustment

    def _apply_competitor_adjustment(
        self,
        suggested_price: float,
        base_price: float,
        competitor_price: Optional[float],
    ) -> tuple[float, float]:
        """
        Adjust price based on competitor pricing.
        
        Returns (adjusted_price, adjustment_amount)
        """
        if competitor_price is None:
            return suggested_price, 0.0

        # If competitor is cheaper, consider matching or beating
        if competitor_price < suggested_price:
            # Match competitor with small undercut
            adjusted = competitor_price * 0.98
            adjustment = adjusted - suggested_price
            return max(base_price * self.min_decay, adjusted), adjustment

        return suggested_price, 0.0

    def _apply_market_price_benchmark(
        self,
        suggested_price: float,
        base_price: float,
        min_market_price: Optional[float],
        avg_market_price: Optional[float],
        days_to_expire: int,
    ) -> tuple[float, Dict[str, Any]]:
        """
        Điều chỉnh giá dựa trên giá thị trường (Price Benchmarking).
        
        Chiến lược:
        - Giá đề xuất phải thấp hơn min_market_price để cạnh tranh
        - Mức thấp hơn phụ thuộc vào số ngày còn lại
        
        Returns:
            (adjusted_price, market_comparison_info)
        """
        comparison_info = {
            "has_market_data": False,
            "min_market_price": min_market_price,
            "avg_market_price": avg_market_price,
            "price_vs_market": None,
            "adjustment_reason": None,
        }
        
        if not min_market_price:
            return suggested_price, comparison_info
        
        comparison_info["has_market_data"] = True
        
        # Tính % thấp hơn market cần thiết dựa trên HSD
        if days_to_expire <= 1:
            target_below_market = 0.25  # Thấp hơn 25%
        elif days_to_expire <= 3:
            target_below_market = 0.15  # Thấp hơn 15%
        elif days_to_expire <= 7:
            target_below_market = 0.10  # Thấp hơn 10%
        else:
            target_below_market = 0.05  # Thấp hơn 5%
        
        target_price = min_market_price * (1 - target_below_market)
        
        # Chọn giá thấp hơn giữa suggested và target
        if suggested_price > target_price:
            adjusted_price = target_price
            comparison_info["adjustment_reason"] = f"Điều chỉnh để thấp hơn giá thị trường {target_below_market*100:.0f}%"
        else:
            adjusted_price = suggested_price
            comparison_info["adjustment_reason"] = "Giá đã cạnh tranh với thị trường"
        
        # Tính % so với market
        comparison_info["price_vs_market"] = round(
            ((min_market_price - adjusted_price) / min_market_price) * 100, 1
        )
        
        # Đảm bảo không dưới giá sàn
        min_allowed = base_price * self.min_decay
        adjusted_price = max(adjusted_price, min_allowed)
        
        return adjusted_price, comparison_info

    def _get_urgency_level(self, days_to_expire: int) -> str:
        """Determine urgency level based on days to expiry."""
        if days_to_expire <= 1:
            return "critical"
        if days_to_expire <= 3:
            return "high"
        if days_to_expire <= 7:
            return "medium"
        return "low"

    def _calculate_expected_sell_rate(
        self,
        discount_percent: float,
        days_to_expire: int,
        category: ProductCategory,
        demand_index: Optional[float],
    ) -> float:
        """
        Calculate expected sell rate based on discount and other factors.
        
        Returns percentage (0-100) of likelihood to sell at this price.
        """
        # Base sell rate from discount (higher discount = higher sell rate)
        if discount_percent >= 50:
            base_rate = 95
        elif discount_percent >= 40:
            base_rate = 88
        elif discount_percent >= 30:
            base_rate = 78
        elif discount_percent >= 20:
            base_rate = 65
        elif discount_percent >= 10:
            base_rate = 50
        else:
            base_rate = 35
        
        # Adjust by days to expire (urgency increases willingness to buy cheap)
        if days_to_expire <= 1:
            base_rate = min(98, base_rate + 5)
        elif days_to_expire <= 3:
            base_rate = min(95, base_rate + 3)
        
        # Adjust by category (perishables sell faster at discount)
        perishable_categories = [ProductCategory.DAIRY, ProductCategory.MEAT, 
                                  ProductCategory.SEAFOOD, ProductCategory.BAKERY]
        if category in perishable_categories:
            base_rate = min(98, base_rate + 5)
        
        # Adjust by demand
        if demand_index is not None:
            if demand_index > 1.2:
                base_rate = min(98, base_rate + 8)
            elif demand_index < 0.8:
                base_rate = max(20, base_rate - 10)
        
        return round(base_rate, 0)

    def _estimate_time_to_sell(
        self,
        days_to_expire: int,
        expected_sell_rate: float,
        inventory_quantity: Optional[int],
    ) -> str:
        """
        Estimate time to sell based on various factors.
        
        Returns Vietnamese string description.
        """
        # Base estimate on sell rate and days to expire
        if expected_sell_rate >= 90:
            if days_to_expire <= 1:
                return "Trong ngày"
            elif days_to_expire <= 3:
                return "1-2 ngày"
            else:
                return "2-3 ngày"
        elif expected_sell_rate >= 75:
            if days_to_expire <= 2:
                return "1-2 ngày"
            elif days_to_expire <= 5:
                return "2-4 ngày"
            else:
                return "3-5 ngày"
        elif expected_sell_rate >= 50:
            if days_to_expire <= 3:
                return "2-3 ngày"
            else:
                return "4-7 ngày"
        else:
            return "Trên 7 ngày hoặc khó bán"

    def _calculate_competitiveness(
        self,
        suggested_price: float,
        base_price: float,
        competitor_price: Optional[float],
        discount_percent: float,
    ) -> float:
        """
        Calculate market competitiveness score (0-1).
        
        Higher score means more competitive pricing.
        """
        # Base competitiveness from discount
        if discount_percent >= 40:
            base_comp = 0.9
        elif discount_percent >= 30:
            base_comp = 0.75
        elif discount_percent >= 20:
            base_comp = 0.6
        elif discount_percent >= 10:
            base_comp = 0.45
        else:
            base_comp = 0.3
        
        # Adjust based on competitor price if available
        if competitor_price is not None:
            if suggested_price < competitor_price * 0.95:
                base_comp = min(1.0, base_comp + 0.15)
            elif suggested_price > competitor_price * 1.05:
                base_comp = max(0.1, base_comp - 0.2)
        
        return round(base_comp, 2)

    def _generate_pricing_reasons(
        self,
        request: PricingRequest,
        discount_percent: float,
        competitor_price: Optional[float],
        suggested_price: float,
        market_comparison: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate Vietnamese reasons explaining the pricing decision.
        """
        reasons = []
        
        # Reason 1: Days to expire
        if request.days_to_expire <= 1:
            reasons.append(f"Sản phẩm sẽ hết hạn trong vòng 24 giờ")
        elif request.days_to_expire <= 3:
            reasons.append(f"Sản phẩm chỉ còn {request.days_to_expire} ngày là hết hạn")
        elif request.days_to_expire <= 7:
            reasons.append(f"Sản phẩm còn {request.days_to_expire} ngày đến hạn sử dụng")
        else:
            reasons.append(f"Sản phẩm còn {request.days_to_expire} ngày hạn sử dụng")
        
        # Reason 2: Market price comparison (QUAN TRỌNG)
        if market_comparison and market_comparison.get("has_market_data"):
            price_vs_market = market_comparison.get("price_vs_market", 0)
            min_market = market_comparison.get("min_market_price", 0)
            
            if price_vs_market > 0:
                reasons.append(
                    f"Giá thấp hơn {price_vs_market:.0f}% so với giá thị trường "
                    f"({min_market:,.0f}đ)"
                )
            elif price_vs_market < 0:
                reasons.append(
                    f"Giá cao hơn thị trường {abs(price_vs_market):.0f}% nhưng phù hợp chất lượng"
                )
            else:
                reasons.append("Giá cạnh tranh với thị trường")
        
        # Reason 3: Discount level
        if discount_percent >= 40:
            reasons.append(f"Giảm giá {discount_percent:.0f}% để đảm bảo bán hết trước khi hết hạn")
        elif discount_percent >= 20:
            reasons.append(f"Áp dụng mức giảm {discount_percent:.0f}% phù hợp với thời hạn còn lại")
        else:
            reasons.append(f"Mức giảm {discount_percent:.0f}% vẫn đảm bảo biên lợi nhuận")
        
        # Reason 4: Category-specific
        category_reasons = {
            ProductCategory.DAIRY: "Sản phẩm sữa cần ưu tiên tiêu thụ nhanh",
            ProductCategory.MEAT: "Thịt tươi đòi hỏi vòng quay nhanh để đảm bảo chất lượng",
            ProductCategory.SEAFOOD: "Hải sản cần được tiêu thụ sớm để giữ độ tươi ngon",
            ProductCategory.BAKERY: "Bánh mì/bánh ngọt có hạn sử dụng ngắn",
            ProductCategory.PRODUCE: "Rau củ quả cần ưu tiên bán trước",
        }
        if request.product_type in category_reasons:
            reasons.append(category_reasons[request.product_type])
        
        # Reason 4: Competitor price
        if competitor_price is not None:
            diff_percent = ((competitor_price - suggested_price) / competitor_price) * 100
            if diff_percent > 5:
                reasons.append(f"Giá thấp hơn {diff_percent:.0f}% so với đối thủ cạnh tranh")
            elif diff_percent < -5:
                reasons.append(f"Giá cao hơn đối thủ {abs(diff_percent):.0f}% nhưng vẫn hợp lý")
        
        # Reason 5: Inventory
        if request.inventory_quantity is not None:
            if request.inventory_quantity > 100:
                reasons.append(f"Số lượng tồn kho cao ({request.inventory_quantity} sản phẩm) cần đẩy nhanh")
            elif request.inventory_quantity > 50:
                reasons.append(f"Tồn kho {request.inventory_quantity} sản phẩm - mức trung bình")
        
        # Reason 6: Demand
        if request.demand_index is not None:
            if request.demand_index > 1.2:
                reasons.append("Nhu cầu thị trường cao - có thể giữ giá tốt hơn")
            elif request.demand_index < 0.8:
                reasons.append("Nhu cầu thấp - cần giảm sâu hơn để kích cầu")
        
        # Reason 7: Strategy
        strategy_reasons = {
            PricingStrategy.AGGRESSIVE: "Chiến lược bán nhanh - ưu tiên doanh số",
            PricingStrategy.CONSERVATIVE: "Chiến lược bảo toàn biên lợi nhuận",
        }
        if request.strategy in strategy_reasons:
            reasons.append(strategy_reasons[request.strategy])
        
        return reasons[:6]  # Limit to 6 reasons

    def _get_recommended_action(
        self,
        days_to_expire: int,
        category: ProductCategory,
    ) -> str:
        """Generate recommended action based on product state."""
        urgency = self._get_urgency_level(days_to_expire)

        if urgency == "critical":
            return "Immediate clearance sale required. Consider bundling with other items."
        if urgency == "high":
            return "Prioritize for promotion. Feature in 'Last Chance' section."
        if urgency == "medium":
            return "Apply standard near-expiry discount. Monitor daily."
        return "Standard pricing acceptable. Review in 3 days."

    def _calculate_confidence(
        self,
        request: PricingRequest,
        decay: float,
    ) -> float:
        """Calculate confidence score for the pricing suggestion."""
        base_confidence = self.default_confidence

        # Increase confidence with more data
        if request.demand_index is not None:
            base_confidence += 0.05
        if request.competitor_price is not None:
            base_confidence += 0.05
        if request.inventory_quantity is not None:
            base_confidence += 0.03

        # Decrease confidence for edge cases
        if request.days_to_expire <= 1:
            base_confidence -= 0.1
        if decay <= self.min_decay + 0.05:
            base_confidence -= 0.05

        return max(0.0, min(1.0, base_confidence))

    def suggest_price(self, request: PricingRequest) -> PricingResponse:
        """
        Calculate suggested price for a near-expiry product.
        
        Args:
            request: Pricing request with product details
            
        Returns:
            Pricing response with suggested price and rationale
        """
        import time

        start_time = time.perf_counter()

        try:
            # Step 1: Calculate base decay
            base_decay = self._calculate_base_decay(request.days_to_expire)

            # Step 2: Apply category multiplier
            category_decay = self._apply_category_multiplier(
                base_decay,
                request.product_type,
            )

            # Step 3: Apply demand adjustment
            demand_decay, demand_adjustment = self._apply_demand_adjustment(
                category_decay,
                request.demand_index,
            )

            # Step 4: Apply strategy adjustment
            strategy_adjustment = self.STRATEGY_ADJUSTMENTS.get(
                request.strategy,
                0.0,
            )
            final_decay = demand_decay - strategy_adjustment
            final_decay = max(self.min_decay, min(self.max_decay, final_decay))

            # Step 5: Calculate base suggested price
            suggested = request.base_price * final_decay

            # Step 6: Apply competitor adjustment
            suggested, competitor_adjustment = self._apply_competitor_adjustment(
                suggested,
                request.base_price,
                request.competitor_price,
            )

            # Calculate discount percentage
            discount_percent = (1 - (suggested / request.base_price)) * 100

            # Calculate price range (±10% of suggested)
            min_price = suggested * 0.9
            max_price = suggested * 1.1

            # Ensure min_price doesn't go below absolute minimum
            absolute_min = request.base_price * self.min_decay
            min_price = max(min_price, absolute_min)

            # Build breakdown
            breakdown = PriceBreakdown(
                base_price=request.base_price,
                decay_factor=final_decay,
                demand_adjustment=demand_adjustment,
                strategy_adjustment=strategy_adjustment,
                competitor_adjustment=competitor_adjustment,
                final_discount_percent=discount_percent,
            )

            # Build rationale
            rationale: Dict[str, Any] = {
                "days_to_expire": request.days_to_expire,
                "category": request.product_type.value,
                "strategy": request.strategy.value,
                "base_decay_factor": round(base_decay, 3),
                "final_decay_factor": round(final_decay, 3),
            }

            if request.region:
                rationale["region"] = request.region
            if request.brand:
                rationale["brand"] = request.brand
            if request.demand_index is not None:
                rationale["demand_index"] = request.demand_index

            # Calculate confidence
            confidence = self._calculate_confidence(request, final_decay)
            
            # Step 7: Apply market price benchmark (NEW)
            market_comparison = None
            if request.min_market_price:
                suggested, market_comparison = self._apply_market_price_benchmark(
                    suggested,
                    request.base_price,
                    request.min_market_price,
                    request.avg_market_price,
                    request.days_to_expire,
                )
                # Recalculate discount after market adjustment
                discount_percent = (1 - (suggested / request.base_price)) * 100
                
                # Increase confidence if market data available
                confidence = min(1.0, confidence + 0.1)
            
            # Recalculate price range after market adjustment
            min_price = suggested * 0.9
            max_price = suggested * 1.1
            min_price = max(min_price, absolute_min)
            
            # Calculate new insights
            expected_sell_rate = self._calculate_expected_sell_rate(
                discount_percent,
                request.days_to_expire,
                request.product_type,
                request.demand_index,
            )
            
            estimated_time = self._estimate_time_to_sell(
                request.days_to_expire,
                expected_sell_rate,
                request.inventory_quantity,
            )
            
            # Competitiveness now considers market price
            competitiveness = self._calculate_competitiveness(
                suggested,
                request.base_price,
                request.min_market_price or request.competitor_price,
                discount_percent,
            )
            
            reasons = self._generate_pricing_reasons(
                request,
                discount_percent,
                request.competitor_price,
                suggested,
                market_comparison,
            )
            
            # Build market price info for response
            market_price_info = None
            if market_comparison:
                market_price_info = {
                    "min_market_price": request.min_market_price,
                    "avg_market_price": request.avg_market_price,
                    "source": request.market_price_source,
                    "price_vs_market_percent": market_comparison.get("price_vs_market"),
                    "adjustment_applied": market_comparison.get("adjustment_reason"),
                }

            calculation_time = (time.perf_counter() - start_time) * 1000

            return PricingResponse(
                suggested_price=suggested,
                discount_percent=round(discount_percent, 1),
                confidence=round(confidence, 3),
                min_suggested_price=min_price,
                max_suggested_price=max_price,
                expected_sell_rate=expected_sell_rate,
                estimated_time_to_sell=estimated_time,
                competitiveness=competitiveness,
                reasons=reasons,
                market_price_info=market_price_info,
                rationale=rationale,
                breakdown=breakdown,
                urgency_level=self._get_urgency_level(request.days_to_expire),
                recommended_action=self._get_recommended_action(
                    request.days_to_expire,
                    request.product_type,
                ),
                calculation_time_ms=round(calculation_time, 2),
                model_version="1.2.0",
            )

        except Exception as e:
            logger.error(f"Pricing calculation failed: {e}")
            raise PricingCalculationError(
                f"Failed to calculate price: {e}",
                details={"request": request.model_dump()},
            ) from e


# Singleton instance
pricing_service = PricingService()


def suggest_price(payload: PricingRequest) -> PricingResponse:
    """Suggest price for a near-expiry product (backward compatible function)."""
    return pricing_service.suggest_price(payload)
