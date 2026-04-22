from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from app import api
from app.schemas import KagglePricingResponse, MonitoringSummary, PricingResponse


@dataclass
class _DummySettings:
    model_path: Path
    metrics_path: Path
    price_history_path: Path


class _DummyTracker:
    def __init__(self) -> None:
        self.events = {"SKU-1": []}

    def recent_event_count(self) -> int:
        return 3

    def flash_sale_skus(self) -> list[str]:
        return ["SKU-1"]


class _SyntheticEngine:
    dataset_profile = "synthetic"

    def __init__(self) -> None:
        self.flash_sale_tracker = _DummyTracker()

    def recommend_price(self, _request):
        response = PricingResponse(
            sku_id="SKU-1",
            recommended_price=125.0,
            ml_price=120.0,
            blended_price=123.0,
            inventory_adjustment=1.0,
            demand_adjustment=1.0,
            flash_sale_multiplier=1.0,
            confidence=0.81,
            detected_flash_sale=False,
            reason="test synthetic response",
            generated_at=datetime.now(UTC),
        )
        return type("SyntheticResult", (), {"response": response})()

    def register_order_event(self, _event) -> bool:
        return False


class _KaggleEngine:
    dataset_profile = "kaggle_retail"

    def __init__(self) -> None:
        self.flash_sale_tracker = _DummyTracker()

    def recommend_price(self, _request):
        raise ValueError("The loaded model is not compatible with the synthetic pricing request schema.")

    def recommend_kaggle_price(self, _request):
        response = KagglePricingResponse(
            product_id="P-1",
            product_category_name="electronics",
            recommended_price=199.0,
            current_price=189.0,
            gap_to_current_price=10.0,
            competitor_anchor_price=193.0,
            confidence=0.76,
            reason="test kaggle response",
            generated_at=datetime.now(UTC),
        )
        return type("KaggleResult", (), {"response": response})()

    def register_order_event(self, _event) -> bool:
        return False


def _build_client(monkeypatch, tmp_path: Path, engine_instance):
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    history_path = tmp_path / "history.csv"
    model_path.write_text("stub", encoding="utf-8")
    metrics_path.write_text('{"ok": true}', encoding="utf-8")
    history_path.write_text(
        "generated_at,recommended_price\n2026-04-22T10:00:00+00:00,150.0\n",
        encoding="utf-8",
    )

    settings = _DummySettings(
        model_path=model_path,
        metrics_path=metrics_path,
        price_history_path=history_path,
    )

    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "PricingEngine", lambda _settings: engine_instance)
    return TestClient(api.app)


def test_health_reports_active_profile(monkeypatch, tmp_path: Path) -> None:
    with _build_client(monkeypatch, tmp_path, _SyntheticEngine()) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["model_loaded"] is True
        assert payload["dataset_profile"] == "synthetic"
        assert payload["supported_endpoints"]["kaggle_retail"] == "/price/recommend/kaggle"


def test_synthetic_recommendation_endpoint(monkeypatch, tmp_path: Path) -> None:
    with _build_client(monkeypatch, tmp_path, _SyntheticEngine()) as client:
        response = client.post(
            "/price/recommend",
            json={
                "sku_id": "SKU-1",
                "category": "electronics",
                "brand": "brand_a",
                "customer_segment": "premium",
                "hour_of_day": 12,
                "day_of_week": 2,
                "is_weekend": 0,
                "is_festival": 0,
                "inventory_level": 30,
                "inventory_days_cover": 10,
                "competitor_price": 100,
                "click_through_rate": 0.05,
                "conversion_rate": 0.03,
                "units_sold_last_5m": 4,
                "units_sold_last_1h": 18,
                "base_cost": 70,
                "current_price": 115,
            },
        )
        assert response.status_code == 200
        assert response.json()["recommended_price"] == 125.0


def test_kaggle_recommendation_endpoint(monkeypatch, tmp_path: Path) -> None:
    with _build_client(monkeypatch, tmp_path, _KaggleEngine()) as client:
        response = client.post(
            "/price/recommend/kaggle",
            json={
                "product_id": "P-1",
                "product_category_name": "electronics",
                "qty": 10,
                "freight_price": 5,
                "product_name_lenght": 20,
                "product_description_lenght": 80,
                "product_photos_qty": 2,
                "product_weight_g": 800,
                "product_score": 4.2,
                "customers": 7,
                "weekday": 3,
                "weekend": 0,
                "holiday": 0,
                "volume": 3200,
                "comp_1": 195,
                "ps1": 4.0,
                "fp1": 5,
                "comp_2": 193,
                "ps2": 4.1,
                "fp2": 4,
                "comp_3": 191,
                "ps3": 4.3,
                "fp3": 6,
                "lag_price": 188,
                "month": 4,
                "year": 2026,
                "current_price": 189,
            },
        )
        assert response.status_code == 200
        assert response.json()["gap_to_current_price"] == 10.0


def test_profile_mismatch_returns_conflict(monkeypatch, tmp_path: Path) -> None:
    with _build_client(monkeypatch, tmp_path, _KaggleEngine()) as client:
        response = client.post(
            "/price/recommend",
            json={
                "sku_id": "SKU-1",
                "category": "electronics",
                "brand": "brand_a",
                "customer_segment": "premium",
                "hour_of_day": 12,
                "day_of_week": 2,
                "is_weekend": 0,
                "is_festival": 0,
                "inventory_level": 30,
                "inventory_days_cover": 10,
                "competitor_price": 100,
                "click_through_rate": 0.05,
                "conversion_rate": 0.03,
                "units_sold_last_5m": 4,
                "units_sold_last_1h": 18,
                "base_cost": 70,
                "current_price": 115,
            },
        )
        assert response.status_code == 409


def test_monitoring_summary_uses_history_file(monkeypatch, tmp_path: Path) -> None:
    with _build_client(monkeypatch, tmp_path, _SyntheticEngine()) as client:
        response = client.get("/monitoring/summary")
        assert response.status_code == 200
        payload = MonitoringSummary.model_validate(response.json())
        assert payload.average_recommended_price == 150.0
        assert payload.flash_sale_skus == ["SKU-1"]
