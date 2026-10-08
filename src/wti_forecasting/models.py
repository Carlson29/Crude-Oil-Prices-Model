"""Deterministic baselines and compact benchmark model definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def baseline_predictions(
    train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Build baselines without reading test targets."""
    test_rows = len(test)
    return {
        "Zero": np.zeros(test_rows, dtype=float),
        "Training mean": np.full(test_rows, train["Target"].mean(), dtype=float),
        "Persistence": test["WTI"].to_numpy(dtype=float),
    }


def benchmark_models(random_state: int = 42, include_xgboost: bool = False) -> dict[str, object]:
    """Return the fast, deterministic subset of models used for reproducible checks."""
    models: dict[str, object] = {
        "Linear Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=102,
            max_depth=7,
            min_samples_split=7,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if include_xgboost:
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise RuntimeError(
                "XGBoost was requested but is not installed; install .[xgboost]"
            ) from error
        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
        )
    return models
