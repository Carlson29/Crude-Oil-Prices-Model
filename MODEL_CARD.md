# Model card: WTI weekly-return forecasting

## Intended use

This research project evaluates whether macro-financial variables improve one-week-ahead forecasts
of West Texas Intermediate crude-oil returns. It is an educational forecasting study and portfolio
artefact. It is **not** a trading system or financial advice.

## Data

- **Period:** 12 March 2010 to 27 December 2024
- **Frequency:** weekly model-ready observations
- **Observations:** 773
- **Predictors:** 19
- **Target:** next observed weekly WTI return
- **Sources:** FRED and Yahoo Finance, represented by the committed processed dataset

Predictors cover WTI lags and rolling behaviour, the S&P 500, US dollar, gold, Brent, volatility
indices, the Treasury yield spread and industrial production. The repository also preserves the raw
daily dataset and the complete research notebook.

## Evaluation contract

The reproducible companion benchmark uses an 80/20 chronological split:

- Training: 618 observations, ending 7 January 2022
- Test: 155 observations, beginning 14 January 2022
- No random shuffling
- Target alignment checked against the next observed WTI return
- Scaling and fitting use training data only
- Zero, training-mean and persistence baselines are reported alongside fitted models
- Diebold–Mariano comparisons use one-step squared forecast loss against persistence

The original notebook contains the broader Random Forest, XGBoost, Prophet and sequence-LSTM
experiments, supplementary regime and classification work, SHAP analysis and diagnostics. The
companion benchmark intentionally stays compact enough for CI and reviewer reproduction.

## Reproducible companion results

| Model | RMSE | MAE | R² | Directional accuracy |
| --- | ---: | ---: | ---: | ---: |
| Zero | **0.0522** | 0.0395 | -0.000 | 47.1% |
| Training mean | 0.0522 | **0.0395** | -0.000 | **52.9%** |
| Random Forest | 0.0537 | 0.0410 | -0.059 | 41.9% |
| Linear Regression | 0.0574 | 0.0438 | -0.211 | 51.6% |
| Persistence | 0.0801 | 0.0615 | -1.359 | 47.1% |

The simple zero-return baseline has the lowest test RMSE. The fitted models do not beat that strong
baseline on this holdout. This is a meaningful negative result consistent with the difficulty of
forecasting short-horizon financial returns. The benchmark prevents model complexity from being
mistaken for genuine predictive improvement.

## Limitations

- Results depend on the selected period, variables, transformations and evaluation horizon.
- Macroeconomic series can be revised after publication; the original dataset is not a full vintage
  database.
- A single chronological holdout does not describe every future market regime.
- Transaction costs, tradability and portfolio construction are outside the project scope.
- The companion benchmark does not rerun the notebook's computationally heavier Prophet and LSTM
  experiments.
- Statistical association and forecast performance do not establish causality.

## Reproducibility and security

The committed processed CSV lets reviewers reproduce the companion benchmark without network calls
or credentials. Refreshing FRED data requires a user-provided `FRED_API_KEY` environment variable;
secrets must not be committed. Tests enforce chronological splitting, target alignment, deterministic
baselines and the absence of a hard-coded FRED key in the notebook.
