from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "dynamic-pricing-engine"
    model_path: Path = BASE_DIR / "models" / "best_pricing_model.joblib"
    metrics_path: Path = BASE_DIR / "models" / "training_metrics.json"
    raw_data_path: Path = BASE_DIR / "data" / "raw" / "pricing_events.csv"
    price_history_path: Path = BASE_DIR / "data" / "processed" / "price_history.csv"
    competitor_api_url: str = ""
    competitor_weight: float = 0.30
    model_weight: float = 0.70
    min_margin: float = 0.08
    max_price_multiplier: float = 1.35
    redis_url: str = ""
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_orders: str = "pricing.orders"
    kafka_topic_clicks: str = "pricing.clicks"
    flash_sale_order_threshold: int = 20
    flash_sale_lookback_minutes: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    settings.price_history_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
