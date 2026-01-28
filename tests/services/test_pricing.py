from datetime import date

import pytest

from app.models.pricing import PricingRequest, ProductCategory, PricingStrategy
from app.services.pricing import PricingService


@pytest.fixture
def pricing_service() -> PricingService:
    return PricingService()


class TestPricingService:
    """Test cases for pricing service."""

    def test_suggest_price_basic(self, pricing_service: PricingService) -> None:
        """Test basic price suggestion."""
        request = PricingRequest(
            product_type=ProductCategory.DAIRY,
            days_to_expire=3,
            base_price=50000,
        )

        response = pricing_service.suggest_price(request)

        assert response.suggested_price > 0
        assert response.suggested_price < request.base_price
        assert response.discount_percent > 0
        assert response.confidence > 0
        assert response.urgency_level == "high"

    def test_suggest_price_expired(self, pricing_service: PricingService) -> None:
        """Test price for already expired product."""
        request = PricingRequest(
            product_type=ProductCategory.MEAT,
            days_to_expire=0,
            base_price=100000,
        )

        response = pricing_service.suggest_price(request)

        assert response.urgency_level == "critical"
        assert response.discount_percent >= 50

    def test_suggest_price_long_expiry(self, pricing_service: PricingService) -> None:
        """Test price for product with long expiry."""
        request = PricingRequest(
            product_type=ProductCategory.BEVERAGE,
            days_to_expire=30,
            base_price=20000,
        )

        response = pricing_service.suggest_price(request)

        assert response.urgency_level == "low"
        assert response.discount_percent < 30

    def test_suggest_price_with_strategy(self, pricing_service: PricingService) -> None:
        """Test price suggestion with different strategies."""
        base_request = PricingRequest(
            product_type=ProductCategory.BEVERAGE,
            days_to_expire=7,
            base_price=30000,
        )

        # Test aggressive strategy
        aggressive_request = PricingRequest(
            product_type=ProductCategory.BEVERAGE,
            days_to_expire=7,
            base_price=30000,
            strategy=PricingStrategy.AGGRESSIVE,
        )
        aggressive = pricing_service.suggest_price(aggressive_request)

        # Test conservative strategy
        conservative_request = PricingRequest(
            product_type=ProductCategory.BEVERAGE,
            days_to_expire=7,
            base_price=30000,
            strategy=PricingStrategy.CONSERVATIVE,
        )
        conservative = pricing_service.suggest_price(conservative_request)

        assert aggressive.suggested_price < conservative.suggested_price

    def test_suggest_price_with_demand(self, pricing_service: PricingService) -> None:
        """Test price adjustment based on demand."""
        # High demand
        high_demand_request = PricingRequest(
            product_type=ProductCategory.SNACK,
            days_to_expire=5,
            base_price=20000,
            demand_index=1.5,
        )
        high_demand = pricing_service.suggest_price(high_demand_request)

        # Low demand
        low_demand_request = PricingRequest(
            product_type=ProductCategory.SNACK,
            days_to_expire=5,
            base_price=20000,
            demand_index=0.5,
        )
        low_demand = pricing_service.suggest_price(low_demand_request)

        assert high_demand.suggested_price > low_demand.suggested_price

    def test_suggest_price_category_impact(self, pricing_service: PricingService) -> None:
        """Test that different categories have different decay rates."""
        # Meat (perishable) should have higher discount
        meat_request = PricingRequest(
            product_type=ProductCategory.MEAT,
            days_to_expire=5,
            base_price=100000,
        )
        meat_response = pricing_service.suggest_price(meat_request)

        # Condiment (shelf-stable) should have lower discount
        condiment_request = PricingRequest(
            product_type=ProductCategory.CONDIMENT,
            days_to_expire=5,
            base_price=100000,
        )
        condiment_response = pricing_service.suggest_price(condiment_request)

        assert meat_response.discount_percent > condiment_response.discount_percent

    def test_suggest_price_breakdown(self, pricing_service: PricingService) -> None:
        """Test that breakdown is included in response."""
        request = PricingRequest(
            product_type=ProductCategory.DAIRY,
            days_to_expire=3,
            base_price=50000,
        )

        response = pricing_service.suggest_price(request)

        assert response.breakdown is not None
        assert response.breakdown.base_price == request.base_price
        assert 0 < response.breakdown.decay_factor < 1
        assert response.breakdown.final_discount_percent > 0

    def test_suggest_price_rationale(self, pricing_service: PricingService) -> None:
        """Test that rationale contains expected fields."""
        request = PricingRequest(
            product_type=ProductCategory.BAKERY,
            days_to_expire=2,
            base_price=25000,
            region="Hanoi",
            brand="ABC Bakery",
        )

        response = pricing_service.suggest_price(request)

        assert "days_to_expire" in response.rationale
        assert "category" in response.rationale
        assert "strategy" in response.rationale
        assert response.rationale["region"] == "Hanoi"
        assert response.rationale["brand"] == "ABC Bakery"
