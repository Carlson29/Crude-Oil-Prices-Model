from __future__ import annotations

import json

import numpy as np
import pandas as pd

from wti_forecasting.benchmark import run_benchmark


def test_benchmark_writes_recruiter_facing_artifacts(tmp_path):
    rows = 60
    rng = np.random.default_rng(7)
    wti = rng.normal(0, 0.04, rows)
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-03", periods=rows, freq="W-FRI"),
            "WTI": wti,
            "Macro": rng.normal(0, 1, rows),
            "WTI_lag1": np.r_[0.0, wti[:-1]],
            "Target": np.r_[wti[1:], 0.01],
            "Direction": (np.r_[wti[1:], 0.01] > 0).astype(int),
        }
    )
    input_path = tmp_path / "model.csv"
    output_dir = tmp_path / "results"
    frame.to_csv(input_path, index=False)

    metrics = run_benchmark(input_path, output_dir, include_xgboost=False)

    assert {"Zero", "Training mean", "Persistence", "Linear Regression", "Random Forest"} <= set(
        metrics["model"]
    )
    assert (output_dir / "benchmark_metrics.csv").exists()
    assert (output_dir / "benchmark_metrics.json").exists()
    assert (output_dir / "benchmark_predictions.csv").exists()
    assert (output_dir / "model_comparison.png").exists()
    payload = json.loads((output_dir / "benchmark_metrics.json").read_text(encoding="utf-8"))
    assert payload["split"]["strategy"] == "chronological_80_20"
    assert payload["target_alignment_verified"] is True
