from __future__ import annotations

import csv
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from app.cache import RedisCache
from app.config import Settings
from app.feature_engineering import KAGGLE_RETAIL_CATEGORICAL_FEATURES, KAGGLE_RETAIL_NUMERIC_FEATURES
from app.feature_engineering import add_derived_features
from app.modeling import load_model_bundle
from app.schemas import (
    KagglePricingRequest,
    KagglePricingResponse,
    OrderEvent,
    PricingRequest,
    PricingResponse,
)


@dataclass
class PricingRecommendation:
    response: PricingResponse
    ml_price: float
    blended_price: float


@dataclass
class KagglePricingRecommendation:
    response: KagglePricingResponse
    ml_price: float


class FlashSaleTracker:
    def __init__(self, threshold: int, lookback_minutes: int):
        self.threshold = threshold
        self.lookback = timedelta(minutes=lookback_minutes)
        self.events: dict[str, deque[datetime]] = defaultdict(deque)

    def register(self, event: OrderEvent) -> bool:
        queue = self.events[event.sku_id]
        queue.append(event.event_time)
        self._trim(event.sku_id, event.event_time)
        return len(queue) >= self.threshold

    def is_flash_sale(self, sku_id: str, now: datetime | None = None) -> bool:
        reference_time = now or datetime.now(UTC)
        self._trim(sku_id, reference_time)
        return len(self.events[sku_id]) >= self.threshold

    def recent_event_count(self) -> int:
        now = datetime.now(UTC)
        total = 0
        for sku_id in list(self.events.keys()):
            self._trim(sku_id, now)
            total += len(self.events[sku_id])
        return total

    def flash_sale_skus(self) -> list[str]:
        now = datetime.now(UTC)
        active = []
        for sku_id in list(self.events.keys()):
            self._trim(sku_id, now)
            if len(self.events[sku_id]) >= self.threshold:
                active.append(sku_id)
        return active

    def _trim(self, sku_id: str, reference_time: datetime) -> None:
        queue = self.events[sku_id]
        while queue and reference_time - queue[0] > self.lookback:
            queue.popleft()


class CompetitorPriceClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def get_price(self, sku_id: str, fallback_price: float) -> float:
        if not self.settings.competitor_api_url:
            return fallback_price
        try:
            response = requests.get(
                self.settings.competitor_api_url,
                params={"sku_id": sku_id},
                timeout=0.7,
            )
            response.raise_for_status()
            payload = response.json()
            return float(payload.get("competitor_price", fallback_price))
        except Exception:
            return fallback_price


class PricingEngine:
    def __init__(self, settings: Settings):
        if not settings.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {settings.model_path}. Train the model first."
            )
        self.settings = settings
        self.bundle = load_model_bundle(settings.model_path)
        self.dataset_profile = str(self.bundle.get("dataset_profile", "synthetic"))
        self.pipeline = self.bundle["pipeline"]
        self.flash_sale_tracker = FlashSaleTracker(
            threshold=settings.flash_sale_order_threshold,
            lookback_minutes=settings.flash_sale_lookback_minutes,
        )
        self.competitor_client = CompetitorPriceClient(settings)
        self.cache = RedisCache(settings.redis_url)

    def recommend_price(self, request: PricingRequest) -> PricingRecommendation:
        if self.dataset_profile != "synthetic":
            raise ValueError(
                "The loaded model is not compatible with the synthetic pricing request schema."
            )
        frame = pd.DataFrame([request.model_dump()])
        live_competitor_price = self.competitor_client.get_price(
            request.sku_id, request.competitor_price
        )
        frame["competitor_price"] = live_competitor_price
        enriched = add_derived_features(frame)
        ml_price = float(self.pipeline.predict(enriched)[0])

        blended_price = (
            ml_price * self.settings.model_weight
            + live_competitor_price * self.settings.competitor_weight
        )

        inventory_adjustment = self._inventory_adjustment(
            inventory_level=request.inventory_level,
            inventory_days_cover=request.inventory_days_cover,
        )
        demand_adjustment = self._demand_adjustment(
            units_sold_last_5m=request.units_sold_last_5m,
            units_sold_last_1h=request.units_sold_last_1h,
            conversion_rate=request.conversion_rate,
            is_festival=request.is_festival,
        )
        detected_flash_sale = self.flash_sale_tracker.is_flash_sale(request.sku_id)
        flash_sale_multiplier = 1.12 if detected_flash_sale else 1.0

        candidate_price = blended_price * inventory_adjustment * demand_adjustment
        candidate_price *= flash_sale_multiplier
        guardrailed_price = self._apply_guardrails(
            candidate_price=candidate_price,
            base_cost=request.base_cost,
            current_price=request.current_price,
        )
        confidence = self._confidence_score(request, live_competitor_price)
        reason = self._build_reason(
            inventory_adjustment=inventory_adjustment,
            demand_adjustment=demand_adjustment,
            detected_flash_sale=detected_flash_sale,
        )

        response = PricingResponse(
            sku_id=request.sku_id,
            recommended_price=round(guardrailed_price, 2),
            ml_price=round(ml_price, 2),
            blended_price=round(blended_price, 2),
            inventory_adjustment=round(inventory_adjustment, 3),
            demand_adjustment=round(demand_adjustment, 3),
            flash_sale_multiplier=round(flash_sale_multiplier, 3),
            confidence=round(confidence, 3),
            detected_flash_sale=detected_flash_sale,
            reason=reason,
            generated_at=datetime.now(UTC),
        )
        self._append_price_history(response, request.base_cost)
        return PricingRecommendation(
            response=response,
            ml_price=ml_price,
            blended_price=blended_price,
        )

    def recommend_kaggle_price(
        self, request: KagglePricingRequest
    ) -> KagglePricingRecommendation:
        if self.dataset_profile != "kaggle_retail":
            raise ValueError(
                "The loaded model is not compatible with the Kaggle retail pricing request schema."
            )

        payload = request.model_dump()
        current_price = payload.pop("current_price")
        feature_frame = pd.DataFrame([payload])[
            KAGGLE_RETAIL_NUMERIC_FEATURES + KAGGLE_RETAIL_CATEGORICAL_FEATURES
        ]
        ml_price = float(self.pipeline.predict(feature_frame)[0])

        competitor_values = [
            payload["comp_1"] or 0.0,
            payload["comp_2"] or 0.0,
            payload["comp_3"] or 0.0,
        ]
        non_zero_competitors = [value for value in competitor_values if value > 0]
        competitor_anchor = (
            sum(non_zero_competitors) / len(non_zero_competitors)
            if non_zero_competitors
            else None
        )
        confidence = self._kaggle_confidence_score(request, competitor_anchor)
        reason = self._build_kaggle_reason(current_price, ml_price, competitor_anchor)

        response = KagglePricingResponse(
            product_id=request.product_id,
            product_category_name=request.product_category_name,
            recommended_price=round(ml_price, 2),
            current_price=round(current_price, 2),
            gap_to_current_price=round(ml_price - current_price, 2),
            competitor_anchor_price=(
                round(competitor_anchor, 2) if competitor_anchor is not None else None
            ),
            confidence=round(confidence, 3),
            reason=reason,
            generated_at=datetime.now(UTC),
        )
        return KagglePricingRecommendation(response=response, ml_price=ml_price)

    def register_order_event(self, event: OrderEvent) -> bool:
        if event.event_time.tzinfo is None:
            event = OrderEvent(
                sku_id=event.sku_id,
                quantity=event.quantity,
                event_time=event.event_time.replace(tzinfo=UTC),
            )
        return self.flash_sale_tracker.register(event)

    def get_cached_recommendation(self, sku_id: str) -> dict[str, object] | None:
        return self.cache.get_json(f"price:{sku_id}")

    def _inventory_adjustment(self, inventory_level: int, inventory_days_cover: float) -> float:
        if inventory_level <= 15 or inventory_days_cover < 3:
            return 1.10
        if inventory_level >= 120 or inventory_days_cover > 21:
            return 0.93
        return 1.0

    def _demand_adjustment(
        self,
        units_sold_last_5m: int,
        units_sold_last_1h: int,
        conversion_rate: float,
        is_festival: int,
    ) -> float:
        rapid_demand = units_sold_last_5m >= 8 or units_sold_last_1h >= 40
        strong_conversion = conversion_rate >= 0.045
        if rapid_demand and strong_conversion:
            return 1.08 + (0.03 if is_festival else 0.0)
        if units_sold_last_1h <= 8 and conversion_rate < 0.02:
            return 0.94
        return 1.0

    def _apply_guardrails(self, candidate_price: float, base_cost: float, current_price: float) -> float:
        min_price = base_cost * (1.0 + self.settings.min_margin)
        max_price = current_price * self.settings.max_price_multiplier
        return max(min(candidate_price, max_price), min_price)

    def _confidence_score(self, request: PricingRequest, competitor_price: float) -> float:
        signal_score = min(request.conversion_rate * 12 + request.click_through_rate * 4, 0.45)
        inventory_score = 0.25 if request.inventory_level > 10 else 0.12
        competitor_score = 0.20 if competitor_price > 0 else 0.08
        recency_score = 0.10 if request.units_sold_last_1h > 0 else 0.04
        return min(signal_score + inventory_score + competitor_score + recency_score, 0.98)

    def _build_reason(
        self,
        inventory_adjustment: float,
        demand_adjustment: float,
        detected_flash_sale: bool,
    ) -> str:
        reasons = ["ML baseline with competitor blending"]
        if inventory_adjustment > 1.0:
            reasons.append("low inventory pressure")
        elif inventory_adjustment < 1.0:
            reasons.append("overstock discount")
        if demand_adjustment > 1.0:
            reasons.append("strong short-term demand")
        elif demand_adjustment < 1.0:
            reasons.append("soft demand correction")
        if detected_flash_sale:
            reasons.append("flash sale multiplier active")
        return ", ".join(reasons)

    def _kaggle_confidence_score(
        self,
        request: KagglePricingRequest,
        competitor_anchor: float | None,
    ) -> float:
        demand_score = min((request.qty / 40) + (request.customers / 80), 0.40)
        rating_score = min(request.product_score / 10, 0.20)
        competitor_score = 0.20 if competitor_anchor is not None else 0.08
        recency_score = 0.10 if request.lag_price > 0 else 0.04
        seasonal_score = 0.10 if request.holiday or request.weekend else 0.05
        return min(
            demand_score + rating_score + competitor_score + recency_score + seasonal_score,
            0.97,
        )

    def _build_kaggle_reason(
        self,
        current_price: float,
        ml_price: float,
        competitor_anchor: float | None,
    ) -> str:
        reasons = ["Kaggle retail model baseline"]
        if competitor_anchor is not None:
            if ml_price > competitor_anchor:
                reasons.append("positioned above competitor average")
            elif ml_price < competitor_anchor:
                reasons.append("positioned below competitor average")
            else:
                reasons.append("aligned with competitor average")
        if ml_price > current_price:
            reasons.append("upside versus current price")
        elif ml_price < current_price:
            reasons.append("defensive move versus current price")
        else:
            reasons.append("flat versus current price")
        return ", ".join(reasons)

    def _append_price_history(self, response: PricingResponse, base_cost: float) -> None:
        path = self.settings.price_history_path
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not file_exists:
                writer.writerow(
                    [
                        "generated_at",
                        "sku_id",
                        "recommended_price",
                        "ml_price",
                        "blended_price",
                        "confidence",
                        "detected_flash_sale",
                        "base_cost",
                    ]
                )
            writer.writerow(
                [
                    response.generated_at.isoformat(),
                    response.sku_id,
                    response.recommended_price,
                    response.ml_price,
                    response.blended_price,
                    response.confidence,
                    int(response.detected_flash_sale),
                    round(base_cost, 2),
                ]
            )
        self.cache.set_json(f"price:{response.sku_id}", response.model_dump(mode="json"))
