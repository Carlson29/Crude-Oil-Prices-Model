from __future__ import annotations

import pandas as pd
import pytest

from wti_forecasting.data import (
    feature_columns,
    load_model_data,
    temporal_train_test_split,
    validate_target_alignment,
)


def sample_frame(rows: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-05", periods=rows, freq="W-FRI")
    wti = [float(index) / 100 for index in range(rows)]
    return pd.DataFrame(
        {
            "Date": dates,
            "WTI": wti,
            "WTI_lag1": [0.0, *wti[:-1]],
            "WTI_roll4": [0.0, *wti[:-1]],
            "Target": [*wti[1:], 0.25],
            "Direction": [int(value > 0) for value in [*wti[1:], 0.25]],
        }
    )


def test_load_model_data_sorts_dates_and_keeps_numeric_features(tmp_path):
    path = tmp_path / "model.csv"
    sample_frame().sort_values("Date", ascending=False).to_csv(path, index=False)

    loaded = load_model_data(path)

    assert loaded["Date"].is_monotonic_increasing
    assert loaded["Date"].is_unique
    assert feature_columns(loaded) == ["WTI", "WTI_lag1"]


def test_load_model_data_rejects_duplicate_dates(tmp_path):
    frame = sample_frame()
    frame.loc[1, "Date"] = frame.loc[0, "Date"]
    path = tmp_path / "duplicates.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        load_model_data(path)


def test_target_alignment_matches_next_observed_wti_return():
    assert validate_target_alignment(sample_frame())

    misaligned = sample_frame()
    misaligned.loc[2, "Target"] = 99.0
    assert not validate_target_alignment(misaligned)


def test_temporal_split_uses_the_latest_rows_only_for_test():
    frame = sample_frame()

    train, test = temporal_train_test_split(frame, test_fraction=0.2)

    assert len(train) == 8
    assert len(test) == 2
    assert train["Date"].max() < test["Date"].min()
