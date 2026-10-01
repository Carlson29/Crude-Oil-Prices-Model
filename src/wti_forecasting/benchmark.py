"""Run a reproducible, leakage-aware companion benchmark on the committed dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from .data import (
    feature_columns,
    load_model_data,
    temporal_train_test_split,
    validate_target_alignment,
)
from .evaluation import diebold_mariano, regression_metrics
from .models import baseline_predictions, benchmark_models


def _json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _write_comparison_chart(metrics: pd.DataFrame, destination: Path) -> None:
    ordered = metrics.sort_values("rmse", ascending=True)
    best_model = ordered.iloc[0]["model"]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    colours = ["#E8B04A" if name == best_model else "#2D6A73" for name in ordered["model"]]
    bars = axis.barh(ordered["model"], ordered["rmse"], color=colours)
    axis.set_title("One-week-ahead WTI return benchmark")
    axis.set_xlabel("RMSE (lower is better)")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, ordered["rmse"], strict=True):
        axis.text(value, bar.get_y() + bar.get_height() / 2, f"  {value:.4f}", va="center")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_benchmark(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    include_xgboost: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train a compact model suite and write auditable comparison artefacts."""
    frame = load_model_data(input_path)
    aligned = validate_target_alignment(frame)
    if not aligned:
        raise ValueError("Target does not align with the next observed weekly WTI return")

    train, test = temporal_train_test_split(frame, test_fraction=0.2)
    predictors = feature_columns(frame)
    x_train = train[predictors]
    y_train = train["Target"].to_numpy(dtype=float)
    x_test = test[predictors]
    y_test = test["Target"].to_numpy(dtype=float)

    predictions = baseline_predictions(train, test)
    for name, model in benchmark_models(random_state, include_xgboost).items():
        model.fit(x_train, y_train)
        predictions[name] = np.asarray(model.predict(x_test), dtype=float)

    persistence = predictions["Persistence"]
    rows: list[dict[str, float | str | None]] = []
    for name, predicted in predictions.items():
        metrics = regression_metrics(y_test, predicted)
        comparison = diebold_mariano(y_test, predicted, persistence)
        rows.append(
            {
                "model": name,
                **metrics,
                "dm_vs_persistence_statistic": _json_number(comparison["statistic"]),
                "dm_vs_persistence_p_value": comparison["p_value"],
            }
        )
    metrics_frame = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(destination / "benchmark_metrics.csv", index=False)

    prediction_frame = pd.DataFrame(
        {
            "Date": test["Date"].dt.strftime("%Y-%m-%d"),
            "Actual": y_test,
            **predictions,
        }
    )
    prediction_frame.to_csv(destination / "benchmark_predictions.csv", index=False)

    payload = {
        "dataset": {
            "observations": len(frame),
            "features": len(predictors),
            "start_date": frame["Date"].min().date().isoformat(),
            "end_date": frame["Date"].max().date().isoformat(),
        },
        "split": {
            "strategy": "chronological_80_20",
            "training_observations": len(train),
            "test_observations": len(test),
            "training_end": train["Date"].max().date().isoformat(),
            "test_start": test["Date"].min().date().isoformat(),
        },
        "target_alignment_verified": aligned,
        "metrics": metrics_frame.where(pd.notnull(metrics_frame), None).to_dict(orient="records"),
    }
    (destination / "benchmark_metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_comparison_chart(metrics_frame, destination / "model_comparison.png")
    return metrics_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/model_data/model_data.csv"),
        help="Path to the committed model-ready CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Directory for benchmark metrics, predictions and chart",
    )
    parser.add_argument("--include-xgboost", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run_benchmark(args.input, args.output, include_xgboost=args.include_xgboost)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
