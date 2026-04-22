from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, Field


class PricingRequest(BaseModel):
    sku_id: str = Field(..., examples=["SKU-1001"])
    category: str
    brand: str
    customer_segment: str
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    is_festival: int = Field(..., ge=0, le=1)
    inventory_level: int = Field(..., ge=0)
    inventory_days_cover: float = Field(..., ge=0)
    competitor_price: float = Field(..., gt=0)
    click_through_rate: float = Field(..., ge=0)
    conversion_rate: float = Field(..., ge=0)
    units_sold_last_5m: int = Field(..., ge=0)
    units_sold_last_1h: int = Field(..., ge=0)
    base_cost: float = Field(..., gt=0)
    current_price: float = Field(..., gt=0)


class PricingResponse(BaseModel):
    sku_id: str
    recommended_price: float
    ml_price: float
    blended_price: float
    inventory_adjustment: float
    demand_adjustment: float
    flash_sale_multiplier: float
    confidence: float
    detected_flash_sale: bool
    reason: str
    generated_at: datetime


class KagglePricingRequest(BaseModel):
    product_id: str
    product_category_name: str
    qty: int = Field(..., ge=0)
    freight_price: float = Field(..., ge=0)
    product_name_lenght: int = Field(..., ge=0)
    product_description_lenght: int = Field(..., ge=0)
    product_photos_qty: int = Field(..., ge=0)
    product_weight_g: int = Field(..., ge=0)
    product_score: float = Field(..., ge=0)
    customers: int = Field(..., ge=0)
    weekday: int = Field(..., ge=0)
    weekend: int = Field(..., ge=0, le=1)
    holiday: int = Field(..., ge=0, le=1)
    volume: float = Field(..., ge=0)
    comp_1: float = Field(..., ge=0)
    ps1: float = Field(..., ge=0)
    fp1: float = Field(..., ge=0)
    comp_2: float = Field(..., ge=0)
    ps2: float = Field(..., ge=0)
    fp2: float = Field(..., ge=0)
    comp_3: float = Field(..., ge=0)
    ps3: float = Field(..., ge=0)
    fp3: float = Field(..., ge=0)
    lag_price: float = Field(..., ge=0)
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=2000)
    current_price: float = Field(..., gt=0)


class KagglePricingResponse(BaseModel):
    product_id: str
    product_category_name: str
    recommended_price: float
    current_price: float
    gap_to_current_price: float
    competitor_anchor_price: Optional[float] = None
    confidence: float
    reason: str
    generated_at: datetime


class OrderEvent(BaseModel):
    sku_id: str
    quantity: int = Field(..., ge=1)
    event_time: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MonitoringSummary(BaseModel):
    tracked_skus: int
    recent_order_events: int
    flash_sale_skus: list[str]
    average_recommended_price: Optional[float] = None
    last_price_update: Optional[datetime] = None
