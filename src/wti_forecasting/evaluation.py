"""Evaluation functions shared by the benchmark runner and dashboard."""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate the headline regression and directional metrics used by the study."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted arrays must have the same shape")
    return {
        "rmse": float(sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "directional_accuracy": float(np.mean((actual > 0) == (predicted > 0))),
    }


def diebold_mariano(
    y_true: np.ndarray,
    candidate_prediction: np.ndarray,
    benchmark_prediction: np.ndarray,
) -> dict[str, float]:
    """Compare one-step squared forecast losses using a two-sided normal approximation.

    A positive mean-loss improvement and statistic favour the candidate forecast.
    """
    actual = np.asarray(y_true, dtype=float)
    candidate = np.asarray(candidate_prediction, dtype=float)
    benchmark = np.asarray(benchmark_prediction, dtype=float)
    if not (actual.shape == candidate.shape == benchmark.shape):
        raise ValueError("all forecast arrays must have the same shape")
    if actual.size < 3:
        raise ValueError("Diebold-Mariano comparison requires at least three observations")

    loss_improvement = np.square(actual - benchmark) - np.square(actual - candidate)
    mean_improvement = float(np.mean(loss_improvement))
    variance = float(np.var(loss_improvement, ddof=1))
    if variance == 0:
        statistic = float("inf") if mean_improvement > 0 else 0.0
    else:
        statistic = mean_improvement / sqrt(variance / actual.size)
    p_value = 0.0 if np.isinf(statistic) else erfc(abs(statistic) / sqrt(2.0))
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean_loss_improvement": mean_improvement,
    }
