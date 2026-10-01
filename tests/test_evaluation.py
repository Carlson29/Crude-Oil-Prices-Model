from __future__ import annotations

import numpy as np

from wti_forecasting.evaluation import diebold_mariano, regression_metrics


def test_regression_metrics_include_directional_accuracy():
    metrics = regression_metrics(
        np.array([0.2, -0.1, 0.3]),
        np.array([0.2, -0.1, 0.3]),
    )

    assert metrics == {
        "rmse": 0.0,
        "mae": 0.0,
        "r2": 1.0,
        "directional_accuracy": 1.0,
    }


def test_diebold_mariano_reports_positive_improvement_for_better_candidate():
    actual = np.array([0.2, -0.1, 0.3, -0.2, 0.15, -0.05])
    candidate = actual.copy()
    benchmark = np.zeros_like(actual)

    result = diebold_mariano(actual, candidate, benchmark)

    assert result["mean_loss_improvement"] > 0
    assert result["statistic"] > 0
    assert 0 <= result["p_value"] <= 1
