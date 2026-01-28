from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProductCategory(str, Enum):
    """Product category for pricing."""

    DAIRY = "dairy"
    MEAT = "meat"
    SEAFOOD = "seafood"
    BAKERY = "bakery"
    PRODUCE = "produce"
    FROZEN = "frozen"
    BEVERAGE = "beverage"
    SNACK = "snack"
    CONDIMENT = "condiment"
    OTHER = "other"


class DemandLevel(str, Enum):
    """Demand level indicator."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PricingStrategy(str, Enum):
    """Pricing strategy type."""

    AGGRESSIVE = "aggressive"      # Maximum discount for quick sale
    BALANCED = "balanced"          # Balance between margin and turnover
    CONSERVATIVE = "conservative"  # Maintain reasonable margin


class PricingRequest(BaseModel):
    """Pricing suggestion request schema."""

    model_config = ConfigDict(extra="forbid")

    # Required fields
    product_type: ProductCategory = Field(
        description="Product category",
    )
    days_to_expire: int = Field(
        ge=0,
        le=365,
        description="Days until expiration",
    )
    base_price: float = Field(
        gt=0,
        le=1000000000,
        description="Original price in VND",
    )

    # Optional fields for better prediction
    expiry_date: Optional[date] = Field(
        default=None,
        description="Actual expiry date",
    )
    region: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Store region/location",
    )
    brand: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Product brand",
    )
    demand_index: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Demand index (1.0 = normal)",
    )
    competitor_price: Optional[float] = Field(
        default=None,
        gt=0,
        description="Competitor price if known",
    )
    inventory_quantity: Optional[int] = Field(
        default=None,
        ge=0,
        description="Current inventory quantity",
    )
    strategy: PricingStrategy = Field(
        default=PricingStrategy.BALANCED,
        description="Pricing strategy to apply",
    )

    @model_validator(mode="after")
    def validate_dates(self) -> "PricingRequest":
        if self.expiry_date and self.days_to_expire:
            expected_days = (self.expiry_date - date.today()).days
            if abs(expected_days - self.days_to_expire) > 1:
                # Auto-correct days_to_expire based on expiry_date
                object.__setattr__(self, 'days_to_expire', max(0, expected_days))
        return self


class PriceBreakdown(BaseModel):
    """Detailed price calculation breakdown."""

    base_price: float
    decay_factor: float
    demand_adjustment: float
    strategy_adjustment: float
    competitor_adjustment: float
    final_discount_percent: float


class PricingResponse(BaseModel):
    """Pricing suggestion response schema."""

    # Core output
    suggested_price: float = Field(
        description="Suggested selling price in VND",
    )
    discount_percent: float = Field(
        ge=0,
        le=100,
        description="Discount percentage from base price",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score of suggestion",
    )

    # Price range
    min_suggested_price: float = Field(
        description="Minimum recommended price",
    )
    max_suggested_price: float = Field(
        description="Maximum recommended price",
    )

    # Explanation
    rationale: Dict[str, Any] = Field(
        description="Explanation of pricing factors",
    )
    breakdown: Optional[PriceBreakdown] = Field(
        default=None,
        description="Detailed calculation breakdown",
    )

    # Recommendations
    urgency_level: str = Field(
        description="Urgency level (low/medium/high/critical)",
    )
    recommended_action: str = Field(
        description="Recommended action for the product",
    )

    # Metadata
    calculation_time_ms: Optional[float] = None
    model_version: Optional[str] = None

    @field_validator("suggested_price", "min_suggested_price", "max_suggested_price", mode="before")
    @classmethod
    def round_price(cls, v: float) -> float:
        # Round to nearest 100 VND
        return round(v / 100) * 100
