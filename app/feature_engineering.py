from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_festival",
    "inventory_level",
    "inventory_days_cover",
    "competitor_price",
    "click_through_rate",
    "conversion_rate",
    "units_sold_last_5m",
    "units_sold_last_1h",
    "base_cost",
    "current_price",
    "demand_index",
    "inventory_pressure",
    "competitor_gap",
]

CATEGORICAL_FEATURES = ["category", "brand", "customer_segment"]
TARGET_COLUMN = "optimal_price"

KAGGLE_RETAIL_NUMERIC_FEATURES = [
    "qty",
    "freight_price",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_score",
    "customers",
    "weekday",
    "weekend",
    "holiday",
    "volume",
    "comp_1",
    "ps1",
    "fp1",
    "comp_2",
    "ps2",
    "fp2",
    "comp_3",
    "ps3",
    "fp3",
    "lag_price",
    "month",
    "year",
]
KAGGLE_RETAIL_CATEGORICAL_FEATURES = ["product_id", "product_category_name"]
KAGGLE_RETAIL_TARGET_COLUMN = "unit_price"


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["demand_index"] = (
        enriched["units_sold_last_1h"] * 0.55
        + enriched["units_sold_last_5m"] * 0.35
        + enriched["conversion_rate"] * 100 * 0.10
    )
    enriched["inventory_pressure"] = np.where(
        enriched["inventory_level"] <= 20,
        1.25,
        np.where(enriched["inventory_level"] <= 60, 1.05, 0.92),
    )
    enriched["competitor_gap"] = (
        enriched["current_price"] - enriched["competitor_price"]
    ) / enriched["competitor_price"].clip(lower=1.0)
    return enriched


def load_training_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return add_derived_features(frame)


def load_kaggle_retail_training_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    if "month_year" in frame.columns:
        parsed_month_year = pd.to_datetime(frame["month_year"], dayfirst=True, errors="coerce")
        if "month" not in frame.columns:
            frame["month"] = parsed_month_year.dt.month
        if "year" not in frame.columns:
            frame["year"] = parsed_month_year.dt.year

    numeric_defaults = [
        "comp_1",
        "ps1",
        "fp1",
        "comp_2",
        "ps2",
        "fp2",
        "comp_3",
        "ps3",
        "fp3",
        "lag_price",
    ]
    for column in numeric_defaults:
        if column not in frame.columns:
            frame[column] = np.nan

    if "volume" not in frame.columns:
        frame["volume"] = (
            frame.get("product_name_lenght", 0).fillna(0)
            * frame.get("product_description_lenght", 0).fillna(0)
            * frame.get("product_photos_qty", 0).fillna(0).clip(lower=1)
        )

    if "weekday" not in frame.columns:
        frame["weekday"] = 0
    if "weekend" not in frame.columns:
        frame["weekend"] = 0
    if "holiday" not in frame.columns:
        frame["holiday"] = 0

    ensure_columns(
        frame,
        KAGGLE_RETAIL_NUMERIC_FEATURES
        + KAGGLE_RETAIL_CATEGORICAL_FEATURES
        + [KAGGLE_RETAIL_TARGET_COLUMN],
    )
    return frame


def split_xy(
    frame: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    selected_numeric = numeric_features or NUMERIC_FEATURES
    selected_categorical = categorical_features or CATEGORICAL_FEATURES
    features = frame[selected_numeric + selected_categorical].copy()
    target = frame[target_column].copy()
    return features, target


def ensure_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
