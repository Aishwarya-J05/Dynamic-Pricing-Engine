from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.modeling import train_best_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pricing model on synthetic or Kaggle data.")
    parser.add_argument(
        "--profile",
        default="synthetic",
        choices=["synthetic", "kaggle_retail"],
        help="Dataset profile to train on.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional override for the CSV file path.",
    )
    args = parser.parse_args()

    settings = get_settings()
    data_path = Path(args.data_path) if args.data_path else settings.raw_data_path
    result = train_best_model(
        data_path=data_path,
        artifact_path=settings.model_path,
        metrics_path=settings.metrics_path,
        dataset_profile=args.profile,
    )
    print(
        f"Profile: {result.dataset_profile} | Best model: {result.model_name} | "
        f"MAE={result.mae:.2f} | RMSE={result.rmse:.2f} | R2={result.r2:.3f}"
    )


if __name__ == "__main__":
    main()
