from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.pricing_engine import PricingEngine
from app.schemas import (
    KagglePricingRequest,
    KagglePricingResponse,
    MonitoringSummary,
    OrderEvent,
    PricingRequest,
    PricingResponse,
)


engine: PricingEngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    settings = get_settings()
    if settings.model_path.exists():
        engine = PricingEngine(settings)
    else:
        engine = None
    yield


app = FastAPI(title="Dynamic Pricing Engine", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "model_loaded": engine is not None,
        "dataset_profile": getattr(engine, "dataset_profile", None),
        "supported_endpoints": {
            "synthetic": "/price/recommend",
            "kaggle_retail": "/price/recommend/kaggle",
        },
        "model_path": str(settings.model_path),
        "metrics_path": str(settings.metrics_path),
    }


@app.post("/price/recommend", response_model=PricingResponse)
def recommend_price(request: PricingRequest) -> PricingResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")
    try:
        recommendation = engine.recommend_price(request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return recommendation.response


@app.post("/price/recommend/kaggle", response_model=KagglePricingResponse)
def recommend_kaggle_price(request: KagglePricingRequest) -> KagglePricingResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")
    try:
        recommendation = engine.recommend_kaggle_price(request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return recommendation.response


@app.post("/events/order")
def register_order(event: OrderEvent) -> dict[str, object]:
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")
    is_flash_sale = engine.register_order_event(event)
    return {
        "sku_id": event.sku_id,
        "flash_sale_active": is_flash_sale,
        "registered_at": datetime.now(UTC).isoformat(),
    }


@app.get("/monitoring/summary", response_model=MonitoringSummary)
def monitoring_summary() -> MonitoringSummary:
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")
    settings = get_settings()
    average_recommended_price = None
    last_price_update = None

    if settings.price_history_path.exists():
        history = pd.read_csv(settings.price_history_path)
        if not history.empty:
            average_recommended_price = float(history["recommended_price"].mean())
            last_price_update = pd.to_datetime(history["generated_at"].iloc[-1]).to_pydatetime()

    return MonitoringSummary(
        tracked_skus=len(engine.flash_sale_tracker.events),
        recent_order_events=engine.flash_sale_tracker.recent_event_count(),
        flash_sale_skus=engine.flash_sale_tracker.flash_sale_skus(),
        average_recommended_price=average_recommended_price,
        last_price_update=last_price_update,
    )


@app.get("/metrics")
def metrics() -> dict[str, object]:
    settings = get_settings()
    if not settings.metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found.")
    return json.loads(settings.metrics_path.read_text(encoding="utf-8"))
