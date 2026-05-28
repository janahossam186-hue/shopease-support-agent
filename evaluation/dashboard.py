"""
Streamlit Evaluation Dashboard
Run with:  streamlit run evaluation/dashboard.py

Displays:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  🛍️  ShopEase Customer Support — Evaluation Dashboard               │
  ├──────────────┬───────────────┬───────────────┬────────────────────────┤
  │ Total        │ Resolution    │ Avg Latency   │ Policy Compliance      │
  │ Interactions │ Rate          │               │ Rate                   │
  ├──────────────┴───────────────┴───────────────┴────────────────────────┤
  │  Intent Distribution (pie) │ Agent Distribution (pie)                 │
  ├─────────────────────────────────────────────────────────────────────── │
  │  Latency over time (line) │ Retrieval Quality (bar)                   │
  ├─────────────────────────────────────────────────────────────────────── │
  │  Resolution Status (bar)   │ Guardrail Events (bar)                   │
  ├─────────────────────────────────────────────────────────────────────── │
  │  Raw Interactions Table                                                │
  └──────────────────────────────────────────────────────────────────────┘
"""

import sys
from pathlib import Path

# Allow importing project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from evaluation.metrics import get_metrics_df, compute_summary

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopEase Support Dashboard",
    page_icon="🛍️",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Filters")
    hours = st.slider("Time window (hours)", min_value=1, max_value=168, value=24, step=1)
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    if st.button("🔄 Refresh Now"):
        st.rerun()

if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────
st.title("🛍️ ShopEase Customer Support — Evaluation Dashboard")

df = get_metrics_df(hours=hours)
kpis = compute_summary(df)

# ── KPI cards ─────────────────────────────────────────────────────────────────
if df.empty:
    st.warning(
        "📭 No interaction data yet. Run `python main.py` to start the agent "
        "and generate some conversations."
    )
    st.stop()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("💬 Total Interactions", kpis.get("total_interactions", 0))
with col2:
    rate = kpis.get("resolution_rate", 0)
    st.metric("✅ Resolution Rate", f"{rate:.1%}", delta=f"{rate - 0.75:.1%} vs target")
with col3:
    lat = kpis.get("avg_latency_ms", 0)
    st.metric("⚡ Avg Latency", f"{lat:.0f} ms")
with col4:
    compliance = kpis.get("policy_compliance_rate", 1)
    st.metric("📋 Policy Compliance", f"{compliance:.1%}")
with col5:
    retrieval = kpis.get("avg_retrieval_score", 0)
    st.metric("🔍 Avg Retrieval Score", f"{retrieval:.3f}")

st.divider()

# ── Row 1: Distribution charts ────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    intent_counts = df["intent"].value_counts().reset_index()
    intent_counts.columns = ["Intent", "Count"]
    fig = px.pie(
        intent_counts, names="Intent", values="Count",
        title="🎯 Intent Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    agent_counts = df["agent_used"].value_counts().reset_index()
    agent_counts.columns = ["Agent", "Count"]
    fig = px.bar(
        agent_counts, x="Agent", y="Count",
        title="🤖 Agent Utilisation",
        color="Agent",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Row 2: Latency + Resolution ───────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    df_sorted = df.sort_values("timestamp")
    fig = px.line(
        df_sorted,
        x="timestamp",
        y="latency_ms",
        title="⚡ Response Latency Over Time",
        labels={"latency_ms": "Latency (ms)", "timestamp": "Time"},
        markers=True,
    )
    fig.add_hline(
        y=kpis.get("p90_latency_ms", 0),
        line_dash="dash",
        line_color="red",
        annotation_text=f"P90 = {kpis.get('p90_latency_ms', 0):.0f}ms",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    res_counts = df["resolution_status"].value_counts().reset_index()
    res_counts.columns = ["Status", "Count"]
    color_map = {
        "resolved": "#2ecc71",
        "escalated": "#f39c12",
        "blocked": "#e74c3c",
        "pending": "#95a5a6",
    }
    fig = px.bar(
        res_counts, x="Status", y="Count",
        title="📊 Resolution Status Distribution",
        color="Status",
        color_discrete_map=color_map,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Row 3: Retrieval Quality + Guardrail Events ───────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    fig = px.histogram(
        df,
        x="avg_retrieval_score",
        nbins=20,
        title="🔍 Retrieval Score Distribution",
        labels={"avg_retrieval_score": "Avg Reranker Score"},
        color_discrete_sequence=["#3498db"],
    )
    fig.add_vline(
        x=df["avg_retrieval_score"].mean(),
        line_dash="dash",
        annotation_text=f"mean={df['avg_retrieval_score'].mean():.3f}",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    guardrail_blocked = int((~df["guardrail_passed"]).sum())
    guardrail_passed = int(df["guardrail_passed"].sum())
    toxic_flagged = int((df["toxicity_score"] >= 0.6).sum())

    guard_df = pd.DataFrame({
        "Event": ["Passed", "Input Blocked", "Toxicity Flagged"],
        "Count": [guardrail_passed, guardrail_blocked, toxic_flagged],
    })
    fig = px.bar(
        guard_df, x="Event", y="Count",
        title="🛡️ Guardrail Events",
        color="Event",
        color_discrete_sequence=["#2ecc71", "#e74c3c", "#f39c12"],
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Latency percentile summary ────────────────────────────────────────────────
st.subheader("⚡ Latency Percentiles")
lat_cols = st.columns(3)
with lat_cols[0]:
    st.metric("P50 (Median)", f"{kpis.get('p50_latency_ms', 0):.0f} ms")
with lat_cols[1]:
    st.metric("P90", f"{kpis.get('p90_latency_ms', 0):.0f} ms")
with lat_cols[2]:
    st.metric("Max", f"{df['latency_ms'].max():.0f} ms")

st.divider()

# ── Raw interaction table ─────────────────────────────────────────────────────
st.subheader("📋 Recent Interactions")
display_cols = [
    "timestamp", "customer_id", "intent", "agent_used",
    "resolution_status", "latency_ms", "guardrail_passed",
    "retrieved_doc_count", "avg_retrieval_score", "policy_compliant",
]
available = [c for c in display_cols if c in df.columns]
st.dataframe(
    df[available].head(50),
    use_container_width=True,
    hide_index=True,
)

# ── Export ─────────────────────────────────────────────────────────────────────
st.download_button(
    label="📥 Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="shopease_eval_metrics.csv",
    mime="text/csv",
)
