from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings


def generate_dataset(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = np.array(["electronics", "fashion", "home", "beauty", "grocery"])
    brands = np.array(["brand_a", "brand_b", "brand_c", "brand_d"])
    segments = np.array(["budget", "standard", "premium", "loyal"])

    category = rng.choice(categories, size=rows, p=[0.24, 0.22, 0.18, 0.16, 0.20])
    brand = rng.choice(brands, size=rows)
    customer_segment = rng.choice(segments, size=rows, p=[0.25, 0.40, 0.20, 0.15])
    hour_of_day = rng.integers(0, 24, size=rows)
    day_of_week = rng.integers(0, 7, size=rows)
    is_weekend = (day_of_week >= 5).astype(int)
    is_festival = rng.binomial(1, 0.12, size=rows)
    inventory_level = rng.integers(5, 180, size=rows)
    inventory_days_cover = np.round(rng.uniform(1.0, 28.0, size=rows), 2)
    base_cost = np.round(rng.uniform(150.0, 2200.0, size=rows), 2)

    category_factor = {
        "electronics": 1.55,
        "fashion": 1.35,
        "home": 1.25,
        "beauty": 1.48,
        "grocery": 1.18,
    }
    segment_factor = {
        "budget": 0.95,
        "standard": 1.0,
        "premium": 1.16,
        "loyal": 1.08,
    }

    base_markup = np.array([category_factor[item] for item in category]) * np.array(
        [segment_factor[item] for item in customer_segment]
    )
    competitor_price = np.round(base_cost * base_markup * rng.uniform(0.92, 1.08, size=rows), 2)
    current_price = np.round(competitor_price * rng.uniform(0.96, 1.08, size=rows), 2)
    click_through_rate = np.round(rng.uniform(0.01, 0.15, size=rows), 4)
    conversion_rate = np.round(rng.uniform(0.008, 0.08, size=rows), 4)
    units_sold_last_5m = rng.poisson(4 + is_festival * 2 + is_weekend, size=rows)
    units_sold_last_1h = rng.poisson(18 + is_festival * 10 + is_weekend * 4, size=rows)

    demand_multiplier = (
        1
        + is_festival * 0.10
        + is_weekend * 0.04
        + (hour_of_day >= 18).astype(int) * 0.05
        + (conversion_rate * 2.5)
        + (units_sold_last_1h / 300)
    )
    inventory_multiplier = np.where(
        inventory_level < 20,
        1.11,
        np.where(inventory_level > 120, 0.93, 1.0),
    )
    optimal_price = (
        base_cost
        * base_markup
        * demand_multiplier
        * inventory_multiplier
        * rng.uniform(0.97, 1.03, size=rows)
    )
    optimal_price = np.minimum(optimal_price, current_price * 1.30)
    optimal_price = np.maximum(optimal_price, base_cost * 1.08)

    frame = pd.DataFrame(
        {
            "sku_id": [f"SKU-{1000 + idx}" for idx in range(rows)],
            "category": category,
            "brand": brand,
            "customer_segment": customer_segment,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_festival": is_festival,
            "inventory_level": inventory_level,
            "inventory_days_cover": inventory_days_cover,
            "competitor_price": competitor_price,
            "click_through_rate": click_through_rate,
            "conversion_rate": conversion_rate,
            "units_sold_last_5m": units_sold_last_5m,
            "units_sold_last_1h": units_sold_last_1h,
            "base_cost": base_cost,
            "current_price": current_price,
            "optimal_price": np.round(optimal_price, 2),
        }
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic pricing events data.")
    parser.add_argument("--rows", type=int, default=25000, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    settings = get_settings()
    frame = generate_dataset(rows=args.rows, seed=args.seed)
    settings.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(settings.raw_data_path, index=False)
    print(f"Saved {len(frame)} rows to {settings.raw_data_path}")


if __name__ == "__main__":
    main()
