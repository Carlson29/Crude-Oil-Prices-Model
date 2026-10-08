"""Data contracts for the processed weekly WTI modelling table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"Date", "WTI", "Target", "Direction"}
NON_FEATURE_COLUMNS = {"Date", "Target", "Direction", "WTI_roll4"}


def load_model_data(path: str | Path) -> pd.DataFrame:
    """Load, order and validate the committed model-ready dataset."""
    frame = pd.read_csv(Path(path), parse_dates=["Date"])
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"model data is missing required columns: {sorted(missing)}")

    frame = frame.sort_values("Date").reset_index(drop=True)
    if frame["Date"].duplicated().any():
        raise ValueError("model data contains duplicate dates")

    numeric_columns = [column for column in frame.columns if column != "Date"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    missing_values = frame[numeric_columns].isna().sum()
    invalid = missing_values[missing_values > 0]
    if not invalid.empty:
        raise ValueError(f"model data contains missing or non-numeric values: {invalid.to_dict()}")
    if len(frame) < 10:
        raise ValueError("model data must contain at least 10 observations")
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return dissertation-aligned predictors, excluding metadata and ``WTI_roll4``.

    The final dissertation modelling setup removed ``WTI_roll4`` because of its exact relationship
    with contemporaneous WTI returns and their early lags. Keeping the exclusion here prevents the
    compact companion benchmark from silently using a feature rejected by the main study.
    """
    return [column for column in frame.columns if column not in NON_FEATURE_COLUMNS]


def temporal_train_test_split(
    frame: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an ordered time series without shuffling future observations into training."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    ordered = frame.sort_values("Date").reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_fraction))
    if split_index < 2 or len(ordered) - split_index < 1:
        raise ValueError("not enough observations for the requested temporal split")
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def validate_target_alignment(frame: pd.DataFrame, tolerance: float = 1e-12) -> bool:
    """Confirm each target equals the next observed weekly WTI return where observable."""
    if len(frame) < 2:
        return False
    current_targets = frame["Target"].to_numpy(dtype=float)[:-1]
    next_returns = frame["WTI"].to_numpy(dtype=float)[1:]
    return bool(np.allclose(current_targets, next_returns, atol=tolerance, rtol=0.0))
