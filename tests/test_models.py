from __future__ import annotations

import numpy as np
import pandas as pd

from wti_forecasting.models import baseline_predictions


def test_baselines_use_training_history_and_current_wti_only():
    train = pd.DataFrame({"Target": [0.1, 0.2, -0.1], "WTI": [0.0, 0.1, 0.2]})
    test = pd.DataFrame({"Target": [0.9, 0.8], "WTI": [-0.4, 0.5]})

    predictions = baseline_predictions(train, test)

    np.testing.assert_allclose(predictions["Zero"], [0.0, 0.0])
    np.testing.assert_allclose(predictions["Training mean"], [train["Target"].mean()] * 2)
    np.testing.assert_allclose(predictions["Persistence"], test["WTI"])
