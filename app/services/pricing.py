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

    def _get_urgency_level(self, days_to_expire: int) -> str:
        """Determine urgency level based on days to expiry."""
        if days_to_expire <= 1:
            return "critical"
        if days_to_expire <= 3:
            return "high"
        if days_to_expire <= 7:
            return "medium"
        return "low"

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

            calculation_time = (time.perf_counter() - start_time) * 1000

            return PricingResponse(
                suggested_price=suggested,
                discount_percent=round(discount_percent, 1),
                confidence=round(confidence, 3),
                min_suggested_price=min_price,
                max_suggested_price=max_price,
                rationale=rationale,
                breakdown=breakdown,
                urgency_level=self._get_urgency_level(request.days_to_expire),
                recommended_action=self._get_recommended_action(
                    request.days_to_expire,
                    request.product_type,
                ),
                calculation_time_ms=round(calculation_time, 2),
                model_version="1.0.0",
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
