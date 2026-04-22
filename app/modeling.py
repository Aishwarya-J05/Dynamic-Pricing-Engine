from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from app.feature_engineering import (
    CATEGORICAL_FEATURES,
    KAGGLE_RETAIL_CATEGORICAL_FEATURES,
    KAGGLE_RETAIL_NUMERIC_FEATURES,
    KAGGLE_RETAIL_TARGET_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_kaggle_retail_training_data,
    load_training_data,
    split_xy,
)


@dataclass
class TrainingResult:
    model_name: str
    mae: float
    rmse: float
    r2: float
    artifact_path: Path
    dataset_profile: str


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_models() -> dict[str, object]:
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=16,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=350,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        ),
    }


def get_dataset_profile(dataset_profile: str) -> dict[str, object]:
    profiles = {
        "synthetic": {
            "loader": load_training_data,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target_column": TARGET_COLUMN,
        },
        "kaggle_retail": {
            "loader": load_kaggle_retail_training_data,
            "numeric_features": KAGGLE_RETAIL_NUMERIC_FEATURES,
            "categorical_features": KAGGLE_RETAIL_CATEGORICAL_FEATURES,
            "target_column": KAGGLE_RETAIL_TARGET_COLUMN,
        },
    }
    if dataset_profile not in profiles:
        raise ValueError(
            f"Unsupported dataset profile '{dataset_profile}'. "
            f"Expected one of: {', '.join(profiles)}"
        )
    return profiles[dataset_profile]


def train_best_model(
    data_path: Path,
    artifact_path: Path,
    metrics_path: Path,
    dataset_profile: str = "synthetic",
) -> TrainingResult:
    profile = get_dataset_profile(dataset_profile)
    frame = profile["loader"](data_path)
    numeric_features = profile["numeric_features"]
    categorical_features = profile["categorical_features"]
    target_column = profile["target_column"]
    x, y = split_xy(
        frame,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_column=target_column,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    candidates = build_models()
    best_result: TrainingResult | None = None
    serialized_bundle = None
    metrics_summary: dict[str, dict[str, float]] = {}

    for model_name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions))
        metrics_summary[model_name] = {"mae": mae, "rmse": rmse, "r2": r2}

        if best_result is None or mae < best_result.mae:
            best_result = TrainingResult(
                model_name=model_name,
                mae=mae,
                rmse=rmse,
                r2=r2,
                artifact_path=artifact_path,
                dataset_profile=dataset_profile,
            )
            serialized_bundle = {
                "pipeline": pipeline,
                "model_name": model_name,
                "dataset_profile": dataset_profile,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "target_column": target_column,
                "features": numeric_features + categorical_features,
            }

    if best_result is None or serialized_bundle is None:
        raise RuntimeError("No model was trained.")

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(serialized_bundle, artifact_path)

    metrics_payload = {
        "dataset_profile": dataset_profile,
        "data_path": str(data_path),
        "best_model": best_result.model_name,
        "best_model_metrics": {
            "mae": best_result.mae,
            "rmse": best_result.rmse,
            "r2": best_result.r2,
        },
        "all_models": metrics_summary,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return best_result


def load_model_bundle(artifact_path: Path) -> dict[str, object]:
    return joblib.load(artifact_path)
