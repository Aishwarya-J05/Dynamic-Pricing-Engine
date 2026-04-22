from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.feature_engineering import load_kaggle_retail_training_data
from app.modeling import load_model_bundle


st.set_page_config(
    page_title="Retail Pricing Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        /* ── Base ─────────────────────────────────────────── */
        .stApp { background: var(--background-color); }

        /* ── Metric card ──────────────────────────────────── */
        .mcard {
            background: var(--secondary-background-color);
            border: 1px solid rgba(148,163,184,0.15);
            border-radius: 12px;
            padding: 14px 16px;
        }
        .mcard-label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: var(--text-color);
            opacity: 0.5;
            margin-bottom: 4px;
        }
        .mcard-value {
            font-size: 22px;
            font-weight: 700;
            color: var(--text-color);
            line-height: 1.2;
        }
        .mcard-sub {
            font-size: 12px;
            color: var(--text-color);
            opacity: 0.45;
            margin-top: 3px;
        }

        /* ── Signal banner ────────────────────────────────── */
        .signal-banner {
            border-radius: 10px;
            padding: 12px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0;
        }
        .signal-banner.green  { background: rgba( 16,185,129,0.10); border: 1px solid rgba( 16,185,129,0.25); }
        .signal-banner.orange { background: rgba(234, 88, 12,0.10); border: 1px solid rgba(234, 88, 12,0.25); }
        .signal-banner.gray   { background: rgba(100,116,139,0.10); border: 1px solid rgba(100,116,139,0.22); }

        .signal-title { font-size: 14px; font-weight: 700; }
        .signal-banner.green  .signal-title  { color: #10b981; }
        .signal-banner.orange .signal-title  { color: #ea580c; }
        .signal-banner.gray   .signal-title  { color: #64748b; }

        .signal-text { font-size: 13px; color: var(--text-color); opacity: 0.7; margin-top: 2px; }

        .signal-badge {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.04em;
            padding: 4px 10px;
            border-radius: 999px;
            white-space: nowrap;
        }
        .signal-banner.green  .signal-badge { background: rgba(16,185,129,0.18); color:#10b981; }
        .signal-banner.orange .signal-badge { background: rgba(234,88,12,0.18);  color:#ea580c; }
        .signal-banner.gray   .signal-badge { background: rgba(100,116,139,0.18);color:#64748b; }

        /* ── Section divider label ────────────────────────── */
        .section-sep {
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-color);
            opacity: 0.35;
            margin: 20px 0 6px;
        }

        /* ── KV table in right panel ──────────────────────── */
        .kv-table { width: 100%; border-collapse: collapse; }
        .kv-table td {
            font-size: 13px;
            padding: 7px 0;
            border-bottom: 1px solid rgba(148,163,184,0.12);
            vertical-align: middle;
        }
        .kv-table tr:last-child td { border-bottom: none; }
        .kv-table .kv-key   { color: var(--text-color); opacity: 0.55; width: 55%; }
        .kv-table .kv-val   { color: var(--text-color); font-weight: 600; text-align: right; }

        /* ── Page header ──────────────────────────────────── */
        .page-header {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .page-title {
            font-size: 20px;
            font-weight: 700;
            color: var(--text-color);
        }
        .page-context {
            font-size: 12px;
            color: var(--text-color);
            opacity: 0.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def candidate_kaggle_paths() -> list[Path]:
    return [
        ROOT_DIR / "data" / "raw" / "kaggle" / "retail_price.csv",
        ROOT_DIR / "data" / "raw" / "retail_price.csv",
    ]


@st.cache_resource
def load_bundle() -> dict[str, object] | None:
    model_path = ROOT_DIR / "models" / "best_pricing_model.joblib"
    if not model_path.exists():
        return None
    return load_model_bundle(model_path)


@st.cache_data
def load_metrics() -> dict[str, object]:
    metrics_path = ROOT_DIR / "models" / "training_metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@st.cache_data
def load_kaggle_data() -> tuple[pd.DataFrame, Path | None]:
    for path in candidate_kaggle_paths():
        if path.exists():
            return load_kaggle_retail_training_data(path), path
    return pd.DataFrame(), None


def fmt_inr(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"₹ {value:,.2f}"


def safe_float(value: object, fallback: float = 0.0) -> float:
    return fallback if pd.isna(value) else float(value)


def safe_int(value: object, fallback: int = 0) -> int:
    return fallback if pd.isna(value) else int(value)


def price_signal(predicted: float, actual: float) -> tuple[str, str, str]:
    """Returns (title, text, color_class) where color_class is green/orange/gray."""
    if actual == 0:
        return "Hold Price", "Insufficient data to compute signal.", "gray"
    delta_pct = ((predicted - actual) / actual) * 100
    if delta_pct >= 5:
        return (
            "Increase Opportunity",
            f"Model suggests a {delta_pct:.1f}% upward repricing opportunity.",
            "green",
        )
    if delta_pct <= -5:
        return (
            "Defensive Discount",
            f"Model suggests a {abs(delta_pct):.1f}% lower price to stay competitive.",
            "orange",
        )
    return "Hold Price", "Current price is already close to the model recommendation.", "gray"


def render_metric(label: str, value: str, sub: str) -> None:
    st.markdown(
        f'<div class="mcard">'
        f'  <div class="mcard-label">{label}</div>'
        f'  <div class="mcard-value">{value}</div>'
        f'  <div class="mcard-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_signal(title: str, text: str, color: str) -> None:
    st.markdown(
        f'<div class="signal-banner {color}">'
        f'  <div>'
        f'    <div class="signal-title">{title}</div>'
        f'    <div class="signal-text">{text}</div>'
        f'  </div>'
        f'  <span class="signal-badge">{title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-sep">{text}</div>', unsafe_allow_html=True)


# ── Plotly theme helpers ───────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#94a3b8",
    margin=dict(l=8, r=8, t=36, b=8),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=False, linecolor="rgba(148,163,184,0.15)"),
    yaxis=dict(gridcolor="rgba(148,163,184,0.10)", linecolor="rgba(148,163,184,0.15)"),
)


# ── Bootstrap ─────────────────────────────────────────────────────────────────

inject_styles()
bundle  = load_bundle()
metrics = load_metrics()
data, kaggle_path = load_kaggle_data()

# ── Guards ─────────────────────────────────────────────────────────────────────

if bundle is None:
    st.error("No trained model artifact found. Train the Kaggle model first.")
    st.code(
        "python scripts/train_model.py --profile kaggle_retail "
        "--data-path data/raw/kaggle/retail_price.csv"
    )
    st.stop()

if bundle.get("dataset_profile") != "kaggle_retail":
    st.warning(
        "The loaded model is not the Kaggle retail profile. "
        "Retrain with `--profile kaggle_retail` to use this dashboard."
    )
    st.code(
        "python scripts/train_model.py --profile kaggle_retail "
        "--data-path data/raw/kaggle/retail_price.csv"
    )
    st.stop()

if data.empty or kaggle_path is None:
    st.error(
        "Kaggle retail dataset not found. "
        "Place `retail_price.csv` in `data/raw/kaggle/` or `data/raw/`."
    )
    st.stop()

pipeline            = bundle["pipeline"]
numeric_features    = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]
target_column       = bundle["target_column"]


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Scenario Builder")

    category_options  = sorted(data["product_category_name"].dropna().unique())
    selected_category = st.selectbox("Category", category_options)

    category_frame  = data[data["product_category_name"] == selected_category].copy()
    product_options = sorted(category_frame["product_id"].dropna().unique())
    selected_product = st.selectbox("Product", product_options)

    product_frame = category_frame[category_frame["product_id"] == selected_product].copy()
    if "month_year" in product_frame.columns:
        product_frame["_label"] = product_frame["month_year"].astype(str)
        month_options       = product_frame["_label"].tolist()
        selected_month_label = st.selectbox("Period", month_options, index=len(month_options) - 1)
        selected_row = product_frame[product_frame["_label"] == selected_month_label].iloc[-1].copy()
    else:
        selected_month_label = ""
        selected_row = product_frame.iloc[-1].copy()

    st.divider()
    st.caption("COMMERCIAL")
    qty = st.slider(
        "Units sold", 0, int(max(data["qty"].max(), 20)), safe_int(selected_row["qty"], 1)
    )
    customers = st.slider(
        "Customers", 0, int(max(data["customers"].max(), 20)), safe_int(selected_row["customers"], 1)
    )
    freight_price = st.number_input(
        "Freight price", min_value=0.0, value=safe_float(selected_row["freight_price"]), step=0.5
    )
    lag_price = st.number_input(
        "Previous price", min_value=0.0, value=safe_float(selected_row["lag_price"]), step=0.5
    )
    product_score = st.slider(
        "Product rating", 0.0, 5.0, safe_float(selected_row["product_score"]), 0.1
    )

    st.divider()
    st.caption("COMPETITIVE")
    comp_1 = st.number_input("Competitor 1", min_value=0.0, value=safe_float(selected_row["comp_1"]), step=0.5)
    comp_2 = st.number_input("Competitor 2", min_value=0.0, value=safe_float(selected_row["comp_2"]), step=0.5)
    comp_3 = st.number_input("Competitor 3", min_value=0.0, value=safe_float(selected_row["comp_3"]), step=0.5)
    ps1 = st.slider("Competitor 1 rating", 0.0, 5.0, safe_float(selected_row["ps1"]), 0.1)
    ps2 = st.slider("Competitor 2 rating", 0.0, 5.0, safe_float(selected_row["ps2"]), 0.1)
    ps3 = st.slider("Competitor 3 rating", 0.0, 5.0, safe_float(selected_row["ps3"]), 0.1)

    st.divider()
    st.caption("PRODUCT")
    product_name_lenght = st.slider(
        "Name length", 0, int(max(data["product_name_lenght"].max(), 50)),
        safe_int(selected_row["product_name_lenght"])
    )
    product_description_lenght = st.slider(
        "Description length", 0, int(max(data["product_description_lenght"].max(), 200)),
        safe_int(selected_row["product_description_lenght"])
    )
    product_photos_qty = st.slider(
        "Photos", 0, int(max(data["product_photos_qty"].max(), 10)),
        safe_int(selected_row["product_photos_qty"])
    )
    product_weight_g = st.slider(
        "Weight (g)", 0, int(max(data["product_weight_g"].max(), 5000)),
        safe_int(selected_row["product_weight_g"])
    )


# ── Compute scenario ───────────────────────────────────────────────────────────

scenario = selected_row.copy()
scenario.update({
    "qty": qty, "customers": customers, "freight_price": freight_price,
    "lag_price": lag_price, "product_score": product_score,
    "comp_1": comp_1, "comp_2": comp_2, "comp_3": comp_3,
    "ps1": ps1, "ps2": ps2, "ps3": ps3,
    "product_name_lenght": product_name_lenght,
    "product_description_lenght": product_description_lenght,
    "product_photos_qty": product_photos_qty,
    "product_weight_g": product_weight_g,
})

feature_frame   = pd.DataFrame([scenario])[numeric_features + categorical_features]
predicted_price = float(pipeline.predict(feature_frame)[0])
actual_price    = float(selected_row[target_column])
price_gap       = predicted_price - actual_price

comp_series          = pd.Series([comp_1, comp_2, comp_3]).replace(0, pd.NA).dropna()
avg_competitor_price = float(comp_series.mean()) if not comp_series.empty else float("nan")

sig_title, sig_text, sig_color = price_signal(predicted_price, actual_price)


# ── Page header ────────────────────────────────────────────────────────────────

context_parts = [c for c in [selected_category, selected_product, selected_month_label] if c]
st.markdown(
    f'<div class="page-header">'
    f'  <span class="page-title">Retail Pricing Studio</span>'
    f'  <span class="page-context">{" · ".join(context_parts)}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

render_signal(sig_title, sig_text, sig_color)
st.write("")  # spacing


# ── KPI row ────────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric("Recommendation", fmt_inr(predicted_price), "Predicted unit price")
with c2:
    render_metric("Current price", fmt_inr(actual_price), "Reference record")
with c3:
    delta_sign = "+" if price_gap >= 0 else ""
    render_metric("Delta", f"{delta_sign}{fmt_inr(price_gap)}", "Predicted vs current")
with c4:
    render_metric("Avg competitor", fmt_inr(avg_competitor_price), "Across 3 references")

st.write("")


# ── Main content ───────────────────────────────────────────────────────────────

left_col, right_col = st.columns([1.5, 1], gap="medium")

# ── Left: charts ───────────────────────────────────────────────────────────────

with left_col:
    # Price history
    if "month_year" in product_frame.columns:
        hist = product_frame[["month_year", target_column]].copy()
        hist["month_year"] = pd.to_datetime(hist["month_year"], dayfirst=True, errors="coerce")
        hist = hist.sort_values("month_year")

        if not hist.empty:
            scenario_date = hist["month_year"].max() + pd.offsets.MonthBegin(1)
            combined = pd.concat(
                [
                    hist.rename(columns={target_column: "unit_price"}).assign(series="Historical"),
                    pd.DataFrame({
                        "month_year": [scenario_date],
                        "unit_price": [predicted_price],
                        "series": ["Scenario"],
                    }),
                ],
                ignore_index=True,
            )
            fig_line = px.line(
                combined, x="month_year", y="unit_price", color="series", markers=True,
                color_discrete_map={"Historical": "#0ea5e9", "Scenario": "#f97316"},
                title="Price history",
            )
            fig_line.update_layout(height=260, **CHART_LAYOUT)
            fig_line.update_traces(line_width=2)
            st.plotly_chart(fig_line, use_container_width=True)

    # Competitive comparison
    comp_df = pd.DataFrame({
        "label": ["Current", "Predicted", "Comp 1", "Comp 2", "Comp 3"],
        "price": [actual_price, predicted_price, comp_1, comp_2, comp_3],
        "color": ["#0ea5e9", "#f97316", "#94a3b8", "#94a3b8", "#94a3b8"],
    })
    fig_bar = go.Figure(
        go.Bar(
            x=comp_df["label"],
            y=comp_df["price"],
            marker_color=comp_df["color"],
            text=[fmt_inr(v) for v in comp_df["price"]],
            textposition="outside",
            textfont_size=11,
        )
    )
    fig_bar.update_layout(height=260, title="Market comparison", **CHART_LAYOUT)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Category distribution
    cat_data = data[data["product_category_name"] == selected_category]
    fig_hist = px.histogram(
        cat_data, x=target_column, nbins=24,
        title=f"{selected_category.replace('_', ' ').title()} — price distribution",
        color_discrete_sequence=["#0ea5e9"],
    )
    fig_hist.add_vline(
        x=predicted_price, line_dash="dash", line_color="#f97316",
        annotation_text="Predicted", annotation_position="top right",
        annotation_font_size=11,
    )
    fig_hist.update_layout(height=240, **CHART_LAYOUT)
    st.plotly_chart(fig_hist, use_container_width=True)


# ── Right: panels ──────────────────────────────────────────────────────────────

with right_col:

    # Scenario summary
    section_label("Scenario")
    summary_rows = [
        ("Units sold",      qty),
        ("Customers",       customers),
        ("Freight price",   fmt_inr(freight_price)),
        ("Previous price",  fmt_inr(lag_price)),
        ("Product rating",  f"{product_score:.1f}"),
    ]
    rows_html = "".join(
        f'<tr><td class="kv-key">{k}</td><td class="kv-val">{v}</td></tr>'
        for k, v in summary_rows
    )
    st.markdown(
        f'<table class="kv-table">{rows_html}</table>',
        unsafe_allow_html=True,
    )

    # Competitive pressure
    section_label("Competitive pressure")
    comp_rows = [
        ("Competitor 1", fmt_inr(comp_1), f"{ps1:.1f} ★"),
        ("Competitor 2", fmt_inr(comp_2), f"{ps2:.1f} ★"),
        ("Competitor 3", fmt_inr(comp_3), f"{ps3:.1f} ★"),
    ]
    comp_html = "".join(
        f'<tr>'
        f'  <td class="kv-key">{name}</td>'
        f'  <td class="kv-val">{price}</td>'
        f'  <td class="kv-val" style="opacity:0.5;font-weight:400">{rating}</td>'
        f'</tr>'
        for name, price, rating in comp_rows
    )
    st.markdown(f'<table class="kv-table">{comp_html}</table>', unsafe_allow_html=True)

    if not pd.isna(avg_competitor_price):
        pressure = "below" if predicted_price < avg_competitor_price else "above"
        gap_abs  = abs(predicted_price - avg_competitor_price)
        st.caption(
            f"Predicted price is {fmt_inr(gap_abs)} {pressure} the average competitor price."
        )

    # Repricing opportunities
    section_label("Top repricing opportunities")
    sample_n     = min(len(data), 250)
    sample_df    = data.sample(sample_n, random_state=42).copy()
    sample_feats = sample_df[numeric_features + categorical_features]
    sample_df["predicted_unit_price"] = pipeline.predict(sample_feats)
    sample_df["opportunity_gap"]      = sample_df["predicted_unit_price"] - sample_df[target_column]

    top_ops = (
        sample_df.sort_values("opportunity_gap", ascending=False)
        .head(8)[["product_id", "product_category_name", target_column, "predicted_unit_price", "opportunity_gap"]]
        .rename(columns={
            "product_id":             "Product",
            "product_category_name":  "Category",
            target_column:            "Current",
            "predicted_unit_price":   "Predicted",
            "opportunity_gap":        "Upside",
        })
    )
    top_ops["Current"]   = top_ops["Current"].map(fmt_inr)
    top_ops["Predicted"] = top_ops["Predicted"].map(fmt_inr)
    top_ops["Upside"]    = top_ops["Upside"].map(lambda x: f"+{fmt_inr(x)}" if x >= 0 else fmt_inr(x))

    st.dataframe(top_ops, use_container_width=True, hide_index=True, height=260)