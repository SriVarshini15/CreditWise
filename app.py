from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RAW_DATA_PATH, REPORTS_DIR, ROOT_DIR

st.set_page_config(
    page_title="CreditWise | Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#11243A"
BLUE = "#3867F5"
TEAL = "#0D9B83"
AMBER = "#D98917"
RED = "#D75151"
MUTED = "#738198"

st.markdown(
    """
    <style>
    :root {
        --cw-ink: #14253c;
        --cw-muted: #738198;
        --cw-blue: #3867f5;
        --cw-line: #e7ecf3;
        --cw-surface: #ffffff;
    }
    html, body, [class*="css"] { font-family: 'Avenir Next', Avenir, sans-serif; }
    .stApp { background: #f5f7fb; color: var(--cw-ink); }
    [data-testid="stHeader"] { background: rgba(245,247,251,.88); }
    [data-testid="stSidebar"] {
        background: #101f33;
        border-right: 1px solid rgba(255,255,255,.06);
    }
    [data-testid="stSidebar"] * { color: #eaf0fa; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: .55rem .65rem;
        border-radius: 10px;
        margin: .1rem 0;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-size: .91rem;
        font-weight: 500;
    }
    .block-container { max-width: 1480px; padding-top: 2.1rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--cw-ink); font-family: 'Avenir Next', Avenir, sans-serif; letter-spacing: -.035em; }
    h1 { font-size: 2.2rem !important; font-weight: 750 !important; }
    h2 { font-weight: 750 !important; }
    [data-testid="stMetric"] {
        background: white; border: 1px solid var(--cw-line); border-radius: 15px;
        padding: 17px 20px; box-shadow: 0 4px 14px rgba(20,37,60,.035);
    }
    [data-testid="stMetricLabel"] p { color: var(--cw-muted); font-size: .8rem; font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--cw-ink); font-family: 'Avenir Next', Avenir, sans-serif; font-size: 1.65rem; }
    [data-testid="stMetricDelta"] { font-size: .75rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: white; border: 1px solid var(--cw-line); border-radius: 16px;
        box-shadow: 0 4px 14px rgba(20,37,60,.035);
    }
    div[data-testid="stDataFrame"] { border: 1px solid var(--cw-line); border-radius: 12px; overflow: hidden; }
    .cw-brand { font: 800 1.25rem 'Avenir Next', Avenir, sans-serif; color: #fff; letter-spacing: -.04em; }
    .cw-brand span { color: #82a1ff; }
    .cw-sidebar-copy { color: #a9b6c8 !important; font-size: .78rem; line-height: 1.5; }
    .cw-eyebrow { color: var(--cw-blue); text-transform: uppercase; letter-spacing: .13em; font-size: .68rem; font-weight: 700; margin-bottom: .35rem; }
    .cw-title { color: var(--cw-ink); font: 800 2rem 'Avenir Next', Avenir, sans-serif; letter-spacing: -.05em; line-height: 1.15; margin: 0; }
    .cw-subtitle { color: var(--cw-muted); font-size: .96rem; margin-top: .45rem; margin-bottom: 1.35rem; }
    .cw-section { color: var(--cw-ink); font: 750 1.15rem 'Avenir Next', Avenir, sans-serif; letter-spacing: -.03em; margin: .4rem 0 .2rem; }
    .cw-card { background: white; border: 1px solid var(--cw-line); border-radius: 16px; padding: 18px 20px; box-shadow: 0 4px 14px rgba(20,37,60,.035); height: 100%; }
    .cw-card-label { color: var(--cw-muted); font-size: .76rem; font-weight: 600; }
    .cw-card-value { color: var(--cw-ink); font: 800 1.55rem 'Manrope', sans-serif; margin: .25rem 0; }
    .cw-card-note { color: var(--cw-muted); font-size: .75rem; }
    .cw-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 7px; }
    .cw-callout { padding: 13px 16px; border-radius: 12px; background: #edf3ff; border: 1px solid #dbe6ff; color: #31518e; font-size: .86rem; line-height: 1.55; }
    .cw-warning { padding: 13px 16px; border-radius: 12px; background: #fff7e9; border: 1px solid #f4e3c2; color: #77531d; font-size: .85rem; line-height: 1.5; }
    .cw-reason { padding: 13px 15px; background: #f8faff; border: 1px solid #e4ebfb; border-radius: 11px; margin: .5rem 0; color: #293c55; font-size: .88rem; line-height: 1.55; }
    .cw-footer { color: #8793a4; font-size: .72rem; border-top: 1px solid var(--cw-line); padding-top: 14px; margin-top: 2rem; }
    .stButton button, .stDownloadButton button { border-radius: 10px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_csv(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path)


def render_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="cw-eyebrow">{eyebrow}</div>'
        f'<div class="cw-title">{title}</div>'
        f'<div class="cw-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def render_section(title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="cw-section">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def render_account_explanation(customer: pd.Series) -> None:
    risk = float(customer["predicted_default_probability"])
    rank = int(customer["risk_rank"])
    risk_band = str(customer["risk_band"])
    st.markdown(
        f'<div class="cw-callout"><b>Why this person is flagged:</b> The model estimates a '
        f'{percent(risk)} chance of default next month ({risk_band} risk band). This account is '
        f'ranked #{rank} in the TOP-500 review queue because the queue prioritizes accounts with '
        'the highest model-estimated risk.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    render_section(
        "Signals behind the risk estimate",
        "These account-specific model comparisons show which values raised the estimate. "
        "They are not proof that a customer will default or that a factor caused the risk.",
    )
    reason_columns = ("primary_risk_reason", "secondary_risk_reason", "tertiary_risk_reason")
    found_reason = False
    for column in reason_columns:
        reason = customer.get(column, "")
        if pd.notna(reason) and str(reason).strip():
            found_reason = True
            st.markdown(f'<div class="cw-reason">{reason}</div>', unsafe_allow_html=True)
    if not found_reason:
        st.info("No single feature crossed the explanation threshold; the score reflects combined model signals.")


def plot_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=22, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=NAVY, size=12),
        hoverlabel=dict(bgcolor="white", font_color=NAVY),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e7ecf3", zeroline=False)
    fig.update_yaxes(gridcolor="#edf0f5", zeroline=False)
    return fig


def percent(value: float) -> str:
    return f"{value:.1%}"


summary_path = REPORTS_DIR / "training_summary.json"
queue_path = ROOT_DIR / "outputs" / "top_500_review_queue.csv"
if not summary_path.exists():
    st.warning("Training artifacts are missing. Run `PYTHONPATH=. python3 src/train.py` from the project root.")
    st.stop()

summary = load_summary(str(summary_path))
selected_model = summary["selected_model"]
results = summary["results"]
selected_metrics = results[selected_model]
queue = (
    load_csv(str(queue_path), queue_path.stat().st_mtime)
    if queue_path.exists()
    else pd.DataFrame()
)

with st.sidebar:
    st.markdown('<div class="cw-brand">◈ Credit<span>Wise</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="cw-sidebar-copy">Credit risk intelligence<br/>Local demonstration workspace</p>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "WORKSPACE",
        [
            "Executive overview",
            "Risk review queue",
            "Action center",
            "Customer profile",
            "Model & governance",
        ],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown('<p class="cw-sidebar-copy">ACTIVE MODEL</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:#fff;font-weight:700">{selected_model.replace("_", " ").title()}</div>'
        f'<p class="cw-sidebar-copy">Validation ROC-AUC {selected_metrics["val"]["roc_auc"]:.3f}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="cw-sidebar-copy">Decision support only. Human review is required; model scores are not confirmed outcomes.</p>',
        unsafe_allow_html=True,
    )

if page == "Executive overview":
    render_header(
        "Portfolio snapshot",
        "Risk intelligence overview",
        "A capacity-aware view of predicted default risk and held-out model performance.",
    )
    if queue.empty:
        st.warning("The review queue has not been generated. Run `PYTHONPATH=. python3 src/score_queue.py`.")
        st.stop()

    test = selected_metrics["test"]
    capacity = 500
    avg_queue_risk = float(queue["predicted_default_probability"].mean())
    queue_capacity = f"{len(queue):,} / {capacity:,}"
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Queue size", f"{len(queue):,}", help="Historical offline demonstration queue.")
    kpi_cols[1].metric("Mean queue risk", percent(avg_queue_risk))
    kpi_cols[2].metric("Precision@500 · test", percent(test["precision_at_500"]))
    kpi_cols[3].metric("Test ROC-AUC", f"{test['roc_auc']:.3f}")

    st.markdown(
        '<div class="cw-callout">Offline demonstration using the historical UCI dataset—not a live production scoring run. '
        'Queue membership is based on predicted risk rank. Use the reasons as model signals for review, not as causal explanations or decisions.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    left, right = st.columns([1.05, 1])
    with left:
        render_section("Queue risk profile", f"Predicted risk distribution across {queue_capacity} available review slots.")
        order = ["Critical", "High", "Moderate", "Low"]
        band_counts = queue["risk_band"].value_counts().reindex(order, fill_value=0).rename_axis("Risk band").reset_index(name="Accounts")
        fig = px.bar(
            band_counts,
            x="Risk band",
            y="Accounts",
            color="Risk band",
            color_discrete_map={"Critical": RED, "High": AMBER, "Moderate": BLUE, "Low": TEAL},
            category_orders={"Risk band": order},
        )
        fig.update_layout(showlegend=False)
        fig.update_traces(marker_line_width=0, hovertemplate="%{x}<br>%{y} accounts<extra></extra>")
        st.plotly_chart(plot_layout(fig, 315), use_container_width=True, config={"displayModeBar": False})
    with right:
        render_section("Risk score spread", "Each point represents a ranked account; higher scores indicate higher estimated risk.")
        score_frame = queue[["risk_rank", "predicted_default_probability", "risk_band"]].copy()
        fig = px.scatter(
            score_frame,
            x="risk_rank",
            y="predicted_default_probability",
            color="risk_band",
            color_discrete_map={"Critical": RED, "High": AMBER, "Moderate": BLUE, "Low": TEAL},
            category_orders={"risk_band": order},
            labels={"risk_rank": "Queue rank", "predicted_default_probability": "Predicted probability", "risk_band": "Risk band"},
            hover_data={"risk_rank": True, "predicted_default_probability": ":.1%", "risk_band": True},
        )
        fig.update_traces(marker=dict(size=7, opacity=.7))
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(plot_layout(fig, 315), use_container_width=True, config={"displayModeBar": False})

    bottom_left, bottom_right = st.columns([1, 1])
    with bottom_left:
        render_section("Top review accounts", "Highest-ranked accounts in the current offline queue.")
        preview = queue.head(5)[["risk_rank", "customer_id", "predicted_default_probability", "risk_band"]].copy()
        preview["predicted_default_probability"] = preview["predicted_default_probability"].map(percent)
        st.dataframe(preview, hide_index=True, use_container_width=True)
    with bottom_right:
        render_section("Model snapshot", "Selected using validation ROC-AUC; test results are held-out reporting only.")
        model_rows = []
        for name, record in results.items():
            model_rows.append({
                "Model": name.replace("_", " ").title(),
                "Val ROC-AUC": record["val"]["roc_auc"],
                "Test ROC-AUC": record["test"]["roc_auc"],
                "Test Recall": record["test"]["recall"],
            })
        st.dataframe(
            pd.DataFrame(model_rows).style.format({
                "Val ROC-AUC": "{:.3f}",
                "Test ROC-AUC": "{:.3f}",
                "Test Recall": "{:.1%}",
            }),
            hide_index=True,
            use_container_width=True,
        )

elif page == "Risk review queue":
    render_header(
        "Review operations",
        "TOP-500 risk review queue",
        "Search, filter, and export the highest-ranked accounts for human review.",
    )
    if queue.empty:
        st.warning("No review queue found. Generate one with `PYTHONPATH=. python3 src/score_queue.py`.")
        st.stop()
    total = len(queue)
    avg_risk = queue["predicted_default_probability"].mean()
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Queue size", f"{total:,}", f"Capacity: 500")
    kpi_cols[1].metric("Highest risk", percent(queue["predicted_default_probability"].max()))
    kpi_cols[2].metric("Median risk", percent(queue["predicted_default_probability"].median()))
    st.markdown(
        '<div class="cw-warning"><b>Why accounts are flagged:</b> the queue ranks accounts by their predicted probability of '
        'default and includes up to 500 highest-ranked accounts; queue placement is relative to this review capacity, not a '
        'confirmed outcome. For each person, the reason text highlights account-specific inputs that raise the model score '
        'compared with a training-set reference. These are model signals, not causes or proof of future default. '
        'This is an offline historical demonstration; account actions require human review.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    f1, f2, f3 = st.columns([1.3, 1, 1])
    query = f1.text_input("Find customer ID", placeholder="e.g. 10871", label_visibility="visible")
    available_bands = ["All bands"] + [band for band in ["Critical", "High", "Moderate", "Low"] if band in set(queue["risk_band"])]
    band = f2.selectbox("Risk band", available_bands)
    status_values = ["All statuses"] + sorted(queue["review_status"].dropna().astype(str).unique().tolist())
    status = f3.selectbox("Review status", status_values)

    filtered = queue.copy()
    if query.strip():
        filtered = filtered[filtered["customer_id"].astype(str).str.contains(query.strip(), case=False, na=False)]
    if band != "All bands":
        filtered = filtered[filtered["risk_band"] == band]
    if status != "All statuses":
        filtered = filtered[filtered["review_status"].astype(str) == status]
    st.caption(f"Showing {len(filtered):,} of {total:,} ranked accounts · sorted by model score, highest first.")

    display_columns = [
        column for column in [
            "risk_rank",
            "customer_id",
            "predicted_default_probability",
            "risk_band",
            "primary_risk_reason",
            "secondary_risk_reason",
            "tertiary_risk_reason",
            "review_status",
        ] if column in filtered.columns
    ]
    display_queue = filtered[display_columns].copy()
    display_queue["predicted_default_probability"] *= 100
    st.dataframe(
        display_queue,
        hide_index=True,
        use_container_width=True,
        height=520,
        column_config={
            "risk_rank": st.column_config.NumberColumn("Rank", format="#%d"),
            "customer_id": st.column_config.TextColumn("Customer ID"),
            "predicted_default_probability": st.column_config.ProgressColumn(
                "Predicted risk", min_value=0, max_value=100, format="%.1f%%"
            ),
            "risk_band": st.column_config.TextColumn("Risk band"),
            "primary_risk_reason": st.column_config.TextColumn("Why flagged"),
            "secondary_risk_reason": st.column_config.TextColumn("Additional signal"),
            "tertiary_risk_reason": st.column_config.TextColumn("Additional signal"),
        },
    )
    dl1, dl2 = st.columns([1, 5])
    dl1.download_button(
        "Download filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="creditwise_filtered_queue.csv",
        mime="text/csv",
        use_container_width=True,
    )
    dl2.download_button(
        "Download full TOP-500",
        queue.to_csv(index=False).encode("utf-8"),
        file_name="creditwise_top_500_review_queue.csv",
        mime="text/csv",
    )

elif page == "Action center":
    render_header(
        "Priority accounts",
        "TOP-20 action center",
        "A focused worklist for the accounts with the highest estimated risk.",
    )
    if queue.empty:
        st.warning("No risk queue is available yet.")
        st.stop()
    top20 = queue.head(20).copy()
    st.markdown(
        '<div class="cw-warning">Suggestions are provisional and require a reviewer. A score alone does not establish hardship '
        'or justify reducing a credit limit. The final decision belongs to an authorized human reviewer.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    action_counts = top20.get("recommended_action", pd.Series(dtype=str)).fillna("Unassigned").value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Priority accounts", f"{len(top20):,}")
    c2.metric("High / critical", int(top20["risk_band"].isin(["High", "Critical"]).sum()))
    c3.metric("Action tiers assigned", int((top20.get("recommended_action", "Unassigned") != "Unassigned").sum()) if isinstance(top20.get("recommended_action"), pd.Series) else 0)
    if not action_counts.empty and set(action_counts.index.astype(str)) == {"Unassigned"}:
        st.info("No action tier is assigned in this generated queue yet. Review the account-specific signals below; do not interpret an unassigned tier as a recommendation.")

    selected_id = st.selectbox("Open a priority account", top20["customer_id"].tolist(), format_func=lambda value: f"#{int(top20.loc[top20['customer_id'] == value, 'risk_rank'].iloc[0])} · Customer {value}")
    selected = top20[top20["customer_id"] == selected_id].iloc[0]
    detail_left, detail_right = st.columns([1, 1.25])
    with detail_left:
        st.markdown(
            f'<div class="cw-card"><div class="cw-card-label">CUSTOMER · RANK #{int(selected["risk_rank"]):,}</div>'
            f'<div class="cw-card-value">{selected_id}</div>'
            f'<div class="cw-card-note">Predicted default probability: <b>{percent(float(selected["predicted_default_probability"]))}</b></div>'
            f'<div class="cw-card-note">Risk band: <b>{selected["risk_band"]}</b></div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        render_account_explanation(selected)
    with detail_right:
        render_section("Review status", "Local demonstration state; it is not saved to an enterprise case system.")
        status_key = f"review_status_{selected_id}"
        status_options = ["Pending", "In Review", "Reviewed", "Escalated"]
        current_status = st.session_state.get(status_key, str(selected.get("review_status", "Pending")))
        if current_status not in status_options:
            current_status = "Pending"
        st.session_state[status_key] = st.selectbox(
            "Set local status",
            status_options,
            index=status_options.index(current_status),
            key=f"status_select_{selected_id}",
        )
        suggested_action = str(selected.get("recommended_action", "Unassigned"))
        st.markdown(
            f'<div class="cw-card"><div class="cw-card-label">CURRENT QUEUE ACTION FIELD</div>'
            f'<div class="cw-card-value" style="font-size:1.1rem">{suggested_action}</div>'
            '<div class="cw-card-note">No automated adverse action is taken. Verify circumstances and policy before any decision.</div></div>',
            unsafe_allow_html=True,
        )
    render_section("All top-20 accounts")
    action_columns = [
        column for column in [
            "risk_rank", "customer_id", "predicted_default_probability",
            "risk_band", "recommended_action", "review_status",
        ] if column in top20.columns
    ]
    st.dataframe(top20[action_columns], hide_index=True, use_container_width=True)

elif page == "Customer profile":
    render_header(
        "Account deep dive",
        "Customer risk profile",
        "Explore one ranked account’s score, model signals, and historical financial behavior.",
    )
    if queue.empty:
        st.warning("No risk queue is available yet.")
        st.stop()
    if not RAW_DATA_PATH.exists():
        st.error(f"Raw customer data file not found: {RAW_DATA_PATH}")
        st.stop()
    raw = load_csv(str(RAW_DATA_PATH), RAW_DATA_PATH.stat().st_mtime)
    id_column = summary["id_column"]
    customer_id = st.selectbox(
        "Select an account from the TOP-500 queue",
        queue["customer_id"].tolist(),
        format_func=lambda value: f"Customer {value}",
    )
    customer = queue[queue["customer_id"] == customer_id].iloc[0]
    customer_data = raw[raw[id_column] == customer_id]
    profile_left, profile_right = st.columns([1, 1.7])
    with profile_left:
        st.markdown(
            f'<div class="cw-card"><div class="cw-card-label">PREDICTED DEFAULT RISK</div>'
            f'<div class="cw-card-value">{percent(float(customer["predicted_default_probability"]))}</div>'
            f'<div class="cw-card-note">Queue rank <b>#{int(customer["risk_rank"])}</b> · {customer["risk_band"]} band</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        render_account_explanation(customer)
    with profile_right:
        render_section("Historical bill and payment amounts", "Dataset periods shown from oldest available (6) to latest available (1). Amounts are in the dataset's original currency units.")
        if customer_data.empty:
            st.warning("Customer details were not found in the local raw data.")
        else:
            row = customer_data.iloc[0]
            periods = [6, 5, 4, 3, 2, 1]
            trend = pd.DataFrame({
                "Dataset period": [f"Period {period}" for period in periods],
                "Bill amount": [row.get(f"BILL_AMT{period}", 0) for period in periods],
                "Payment amount": [row.get(f"PAY_AMT{period}", 0) for period in periods],
            })
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend["Dataset period"], y=trend["Bill amount"], mode="lines+markers", name="Bill amount", line=dict(color=BLUE, width=3)))
            fig.add_trace(go.Scatter(x=trend["Dataset period"], y=trend["Payment amount"], mode="lines+markers", name="Payment amount", line=dict(color=TEAL, width=3)))
            fig.update_layout(legend=dict(orientation="h", y=1.13, x=0), yaxis_title="Amount")
            st.plotly_chart(plot_layout(fig, 330), use_container_width=True, config={"displayModeBar": False})

    if not customer_data.empty:
        row = customer_data.iloc[0]
        render_section("Repayment status history", "Original status codes are shown; positive codes indicate payment delay months.")
        pay_features = ["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0"]
        available_pay = [feature for feature in pay_features if feature in row.index]
        pay_frame = pd.DataFrame({
            "Dataset period": [feature.replace("PAY_", "Period ") for feature in available_pay],
            "Status code": [int(row[feature]) for feature in available_pay],
            "Feature": available_pay,
        })
        fig = px.bar(
            pay_frame,
            x="Dataset period",
            y="Status code",
            color="Status code",
            color_continuous_scale=["#6fb9a7", "#dce6f8", "#e3a34a", "#d75151"],
            hover_data=["Feature"],
        )
        fig.add_hline(y=0, line_color="#9aa5b5", line_width=1)
        fig.update_layout(coloraxis_showscale=False, yaxis_title="Status code")
        st.plotly_chart(plot_layout(fig, 260), use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            '<div class="cw-callout">Demographic attributes are not shown as model reasons. Local score comparisons indicate model association only—not causality, intent, or certain future default.</div>',
            unsafe_allow_html=True,
        )

elif page == "Model & governance":
    render_header(
        "Model governance",
        "Performance & responsible use",
        "Compare validation and held-out test metrics without treating accuracy as the only success measure.",
    )
    selected_metrics = results[selected_model]
    test = selected_metrics["test"]
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Selected model", selected_model.replace("_", " ").title())
    kpi_cols[1].metric("Test ROC-AUC", f"{test['roc_auc']:.3f}")
    kpi_cols[2].metric("Test PR-AUC", f"{test['average_precision']:.3f}")
    kpi_cols[3].metric("Precision@500", percent(test["precision_at_500"]))

    render_section("Model comparison", "Models are selected using validation ROC-AUC; test metrics are reported on the held-out test set.")
    comparison_rows = []
    for name, record in results.items():
        comparison_rows.append({
            "Model": name.replace("_", " ").title(),
            "Validation ROC-AUC": record["val"]["roc_auc"],
            "Test ROC-AUC": record["test"]["roc_auc"],
            "Test PR-AUC": record["test"]["average_precision"],
            "Test precision": record["test"]["precision"],
            "Test recall": record["test"]["recall"],
            "Precision@500": record["test"]["precision_at_500"],
            "Recall@500": record["test"]["recall_at_500"],
        })
    compare = pd.DataFrame(comparison_rows)
    left, right = st.columns([1.3, 1])
    with left:
        metric_choice = st.selectbox("Compare models by", ["ROC-AUC", "PR-AUC", "Recall", "Precision@500", "Recall@500"])
        metric_lookup = {
            "ROC-AUC": ("Test ROC-AUC", ".3f"),
            "PR-AUC": ("Test PR-AUC", ".3f"),
            "Recall": ("Test recall", ".1%"),
            "Precision@500": ("Precision@500", ".1%"),
            "Recall@500": ("Recall@500", ".1%"),
        }
        field, number_format = metric_lookup[metric_choice]
        fig = px.bar(
            compare,
            x="Model",
            y=field,
            color="Model",
            color_discrete_sequence=[BLUE, "#8ea7fa"],
            text=compare[field].map(lambda value: format(value, number_format)),
        )
        fig.update_layout(showlegend=False, yaxis_title=metric_choice)
        st.plotly_chart(plot_layout(fig, 320), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown("#### Confusion matrix · held-out test set")
        matrix = test["confusion_matrix"]
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Predicted: repay", "Predicted: default"],
                y=["Actual: repay", "Actual: default"],
                text=matrix,
                texttemplate="%{text}",
                colorscale=[[0, "#f3f6fd"], [1, BLUE]],
                showscale=False,
                hovertemplate="%{y}<br>%{x}<br>%{z} accounts<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(plot_layout(fig, 320), use_container_width=True, config={"displayModeBar": False})
    st.dataframe(
        compare.style.format({
            "Validation ROC-AUC": "{:.3f}",
            "Test ROC-AUC": "{:.3f}",
            "Test PR-AUC": "{:.3f}",
            "Test precision": "{:.1%}",
            "Test recall": "{:.1%}",
            "Precision@500": "{:.1%}",
            "Recall@500": "{:.1%}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Feature and data governance"):
        st.markdown(f"- **Primary model features ({len(summary['feature_columns'])}):** {', '.join(summary['feature_columns'])}")
        st.markdown("- **Excluded from primary model:** SEX, AGE, EDUCATION, and MARRIAGE.")
        st.markdown("- **Target:** default payment next month. The target is not included in the feature matrix.")
        st.markdown("- **Queue:** historical offline demonstration; not prospective production performance.")
        st.markdown("- **Fairness:** demographic exclusion does not guarantee fairness; financial behavior can proxy for protected characteristics. Group error-rate audits are still required before operational use.")
        st.markdown("- **Human review:** predictions are prioritization signals. They must not automatically trigger credit-limit reductions, adverse action, or hardship assumptions.")
        st.markdown("- **Limitations:** historical data, model calibration and drift require further evaluation; production deployment needs policy, privacy, and regulatory review.")
    st.caption(f"Dataset: UCI Default of Credit Card Clients · Selected model: {selected_model} · Run: {selected_metrics.get('run_id', 'not recorded')}")

st.markdown(
    '<div class="cw-footer">CreditWise · Local decision-support demonstration · Predictions are estimates, not confirmed outcomes.</div>',
    unsafe_allow_html=True,
)
