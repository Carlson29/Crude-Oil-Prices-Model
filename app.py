"""Interactive evidence dashboard for the WTI forecasting study."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from wti_forecasting.data import load_model_data  # noqa: E402

st.set_page_config(
    page_title="WTI Forecast Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(145deg, #06171d 0%, #0a222b 55%, #07191f 100%); }
      .block-container { max-width: 1220px; padding-top: 2.3rem; padding-bottom: 4rem; }
      .hero {
        padding: 2.4rem 2.5rem;
        border: 1px solid rgba(232, 176, 74, .34);
        border-radius: 24px;
        background: radial-gradient(circle at 90% 10%, rgba(232,176,74,.16), transparent 36%),
                    linear-gradient(135deg, rgba(16,55,65,.95), rgba(6,26,33,.96));
        box-shadow: 0 22px 70px rgba(0,0,0,.28);
        margin-bottom: 1.6rem;
      }
      .eyebrow { color: #e8b04a; font-size: .78rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
      .hero h1 { color: #f6fbfa; font-size: clamp(2.2rem, 5vw, 4.6rem); line-height: .98; margin: .7rem 0 1rem; letter-spacing: -.045em; }
      .hero p { color: #b8cbca; max-width: 760px; font-size: 1.05rem; line-height: 1.7; margin: 0; }
      div[data-testid="stMetric"] {
        background: rgba(13, 44, 53, .86); border: 1px solid rgba(154,190,187,.18);
        border-radius: 16px; padding: 1rem 1.15rem; min-height: 118px;
      }
      div[data-testid="stMetricValue"] { color: #f5c86b; }
      div[data-testid="stMetricLabel"] { color: #b8cbca; }
      .finding { border-left: 4px solid #e8b04a; padding: 1rem 1.2rem; background: rgba(232,176,74,.08); border-radius: 0 14px 14px 0; }
      .finding strong { color: #f5c86b; }
      [data-testid="stSidebar"] { background: #07191f; border-right: 1px solid rgba(154,190,187,.14); }
      h2, h3 { letter-spacing: -.02em; }
      @media (max-width: 720px) { .hero { padding: 1.6rem; border-radius: 18px; } .block-container { padding-top: 1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_evidence() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    data = load_model_data(ROOT / "data" / "model_data" / "model_data.csv")
    metrics = pd.read_csv(ROOT / "results" / "benchmark_metrics.csv")
    predictions = pd.read_csv(ROOT / "results" / "benchmark_predictions.csv", parse_dates=["Date"])
    summary = json.loads((ROOT / "results" / "benchmark_metrics.json").read_text(encoding="utf-8"))
    return data, metrics, predictions, summary


data, metrics, predictions, summary = load_evidence()
best = metrics.sort_values("rmse").iloc[0]

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Applied machine learning · Time-series evaluation</div>
      <h1>WTI Forecast Lab</h1>
      <p>Can macro-financial signals improve one-week-ahead forecasts of West Texas Intermediate
      returns? Explore the held-out benchmark, compare models with honest baselines, and inspect the
      evidence behind the conclusion.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(4)
metric_columns[0].metric("Weekly observations", f"{summary['dataset']['observations']:,}")
metric_columns[1].metric("Predictors", summary["dataset"]["features"])
metric_columns[2].metric("Held-out weeks", summary["split"]["test_observations"])
metric_columns[3].metric(f"Best test RMSE · {best['model']}", f"{best['rmse']:.4f}")

with st.sidebar:
    st.markdown("## Explore the evidence")
    st.caption("All headline numbers come from the committed chronological holdout.")
    metric_label = st.selectbox(
        "Comparison metric",
        ["RMSE", "MAE", "Directional accuracy", "R²"],
    )
    available_models = metrics["model"].tolist()
    selected_models = st.multiselect(
        "Models",
        available_models,
        default=available_models,
    )
    trace_model = st.selectbox(
        "Forecast trace",
        [column for column in predictions.columns if column not in {"Date", "Actual"}],
        index=0,
    )
    st.divider()
    st.markdown("**Evaluation contract**")
    st.caption("80/20 chronological split · no shuffling · target-alignment test · DM comparison")

benchmark_tab, trace_tab, design_tab, study_tab = st.tabs(
    ["Benchmark", "Forecast trace", "Research design", "Original study"]
)

metric_map = {
    "RMSE": ("rmse", False),
    "MAE": ("mae", False),
    "Directional accuracy": ("directional_accuracy", True),
    "R²": ("r2", False),
}

with benchmark_tab:
    st.markdown("## Held-out model comparison")
    column, as_percentage = metric_map[metric_label]
    comparison = metrics[metrics["model"].isin(selected_models)][["model", column]].copy()
    if as_percentage:
        comparison[column] *= 100
    if comparison.empty:
        st.info("Select at least one model in the sidebar to display the comparison chart.")
    else:
        st.bar_chart(comparison.set_index("model"), color="#e8b04a", horizontal=True)
    st.dataframe(
        metrics.style.format(
            {
                "rmse": "{:.4f}",
                "mae": "{:.4f}",
                "r2": "{:.3f}",
                "directional_accuracy": "{:.1%}",
                "dm_vs_persistence_statistic": "{:.3f}",
                "dm_vs_persistence_p_value": "{:.2e}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        """
        <div class="finding"><strong>Honest result:</strong> the zero-return baseline has the lowest
        RMSE on the 2022–2024 holdout. The nonlinear models do not beat that difficult baseline in
        this compact companion benchmark. That negative result is useful evidence—not a failed demo.</div>
        """,
        unsafe_allow_html=True,
    )

with trace_tab:
    st.markdown(f"## Actual returns vs {trace_model}")
    trace = predictions.set_index("Date")[["Actual", trace_model]].rename(
        columns={trace_model: "Forecast"}
    )
    st.line_chart(trace, color=["#dcebea", "#e8b04a"])
    st.caption(
        f"Test period: {summary['split']['test_start']} to {summary['dataset']['end_date']}. "
        "Returns are weekly decimal changes, not prices."
    )

with design_tab:
    st.markdown("## Designed to resist optimistic time-series results")
    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown(
            """
            - Observations remain in chronological order; the test period is never shuffled.
            - The committed target contract verifies that each target matches the next weekly WTI return.
            - Mean and fitted-model parameters are learned from training data only.
            - Baselines are reported beside ML models, not hidden.
            - Diebold–Mariano statistics compare squared forecast loss against persistence.
            - The original notebook retains Prophet, XGBoost, LSTM, SHAP and supplementary experiments.
            """
        )
        st.markdown("### Model-ready data")
        st.dataframe(data.tail(12), hide_index=True, width="stretch")
    with right:
        st.image(ROOT / "documentation" / "flowchart.png", caption="Original research workflow")
        st.metric("Training period ends", summary["split"]["training_end"])
        st.metric("Target alignment", "Verified")

with study_tab:
    st.markdown("## Saved evidence from the original 169-cell study")
    st.caption(
        "The companion benchmark does not replace the original experiments; it adds a compact, "
        "testable path for reviewers."
    )
    images = [
        ("Main comparison", "Main_results.png"),
        ("Model roles", "Model_roles.png"),
        ("Classification", "classification_results.png"),
        ("Regime analysis", "regime_based_results.png"),
    ]
    for row_start in range(0, len(images), 2):
        columns = st.columns(2)
        for column_widget, (title, filename) in zip(columns, images[row_start : row_start + 2]):
            with column_widget:
                st.markdown(f"### {title}")
                st.image(ROOT / "outputs" / filename, width="stretch")

st.divider()
st.caption(
    "Research prototype · Not financial advice · Forecast performance may change across market regimes."
)
