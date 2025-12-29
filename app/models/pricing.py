from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class PricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_type: str
    days_to_expire: int = Field(ge=0)
    base_price: float = Field(gt=0)
    region: Optional[str] = None
    brand: Optional[str] = None
    demand_index: Optional[float] = Field(default=None, ge=0.0)


class PricingResponse(BaseModel):
    suggested_price: float
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: Dict[str, Any]
