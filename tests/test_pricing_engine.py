from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings
from app.pricing_engine import PricingEngine
from app.schemas import KagglePricingRequest, OrderEvent, PricingRequest


class _StaticSyntheticPipeline:
    def predict(self, _frame):
        return [200.0]


class _StaticKagglePipeline:
    def predict(self, _frame):
        return [212.5]


def _build_settings(tmp_path: Path) -> Settings:
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    raw_data_path = tmp_path / "raw.csv"
    price_history_path = tmp_path / "history.csv"
    model_path.write_text("stub", encoding="utf-8")
    metrics_path.write_text("{}", encoding="utf-8")
    raw_data_path.write_text("", encoding="utf-8")
    return Settings(
        model_path=model_path,
        metrics_path=metrics_path,
        raw_data_path=raw_data_path,
        price_history_path=price_history_path,
        redis_url="",
    )


def test_synthetic_recommendation_applies_guardrails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "app.pricing_engine.load_model_bundle",
        lambda _path: {"dataset_profile": "synthetic", "pipeline": _StaticSyntheticPipeline()},
    )
    engine = PricingEngine(_build_settings(tmp_path))
    engine.competitor_client.get_price = lambda _sku_id, fallback_price: fallback_price

    result = engine.recommend_price(
        PricingRequest(
            sku_id="SKU-9",
            category="electronics",
            brand="brand_a",
            customer_segment="premium",
            hour_of_day=20,
            day_of_week=5,
            is_weekend=1,
            is_festival=1,
            inventory_level=10,
            inventory_days_cover=2.0,
            competitor_price=195.0,
            click_through_rate=0.08,
            conversion_rate=0.05,
            units_sold_last_5m=10,
            units_sold_last_1h=60,
            base_cost=100.0,
            current_price=130.0,
        )
    )

    assert result.response.recommended_price == 175.5
    assert result.response.flash_sale_multiplier == 1.0
    assert result.response.inventory_adjustment == 1.1
    assert result.response.demand_adjustment == 1.11


def test_flash_sale_detection_activates_multiplier(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "app.pricing_engine.load_model_bundle",
        lambda _path: {"dataset_profile": "synthetic", "pipeline": _StaticSyntheticPipeline()},
    )
    engine = PricingEngine(_build_settings(tmp_path))
    engine.competitor_client.get_price = lambda _sku_id, fallback_price: fallback_price

    now = datetime.now(UTC)
    for _ in range(engine.settings.flash_sale_order_threshold):
        engine.register_order_event(OrderEvent(sku_id="SKU-1", quantity=1, event_time=now))

    result = engine.recommend_price(
        PricingRequest(
            sku_id="SKU-1",
            category="electronics",
            brand="brand_a",
            customer_segment="premium",
            hour_of_day=20,
            day_of_week=5,
            is_weekend=1,
            is_festival=0,
            inventory_level=50,
            inventory_days_cover=8.0,
            competitor_price=200.0,
            click_through_rate=0.06,
            conversion_rate=0.05,
            units_sold_last_5m=8,
            units_sold_last_1h=50,
            base_cost=100.0,
            current_price=150.0,
        )
    )

    assert result.response.detected_flash_sale is True
    assert result.response.flash_sale_multiplier == 1.12


def test_kaggle_recommendation_uses_current_and_competitor_context(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "app.pricing_engine.load_model_bundle",
        lambda _path: {"dataset_profile": "kaggle_retail", "pipeline": _StaticKagglePipeline()},
    )
    engine = PricingEngine(_build_settings(tmp_path))

    result = engine.recommend_kaggle_price(
        KagglePricingRequest(
            product_id="P-55",
            product_category_name="home",
            qty=12,
            freight_price=4.5,
            product_name_lenght=24,
            product_description_lenght=90,
            product_photos_qty=3,
            product_weight_g=700,
            product_score=4.4,
            customers=8,
            weekday=2,
            weekend=0,
            holiday=0,
            volume=6480,
            comp_1=208.0,
            ps1=4.0,
            fp1=5.0,
            comp_2=210.0,
            ps2=4.1,
            fp2=5.0,
            comp_3=206.0,
            ps3=4.2,
            fp3=6.0,
            lag_price=198.0,
            month=4,
            year=2026,
            current_price=199.0,
        )
    )

    assert result.response.recommended_price == 212.5
    assert result.response.gap_to_current_price == 13.5
    assert result.response.competitor_anchor_price == 208.0
    assert "current price" in result.response.reason
