"""
Streamlit Evaluation Dashboard
Run with:  streamlit run evaluation/dashboard.py

Tab 1 — Live Metrics: latency, resolution rate, guardrail events, retrieval scores.
Tab 2 — RAG Evaluation: context precision & recall, naive vs hybrid comparison.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from evaluation.metrics import get_metrics_df, compute_summary, get_llm_scores_df


# ── RAG Evaluation tab ────────────────────────────────────────────────────────

def _render_rag_tab():
    """Render the RAG Evaluation tab: IR metrics, RAGAS scores, and key findings."""
    import json
    from pathlib import Path

    st.header("🔍 RAG Pipeline Evaluation")
    st.caption(
        "Comparison of retrieval pipelines across "
        "P@k, R@k, MRR, Latency, and RAGAS metrics."
    )

    # ── Section 1: IR Metrics ─────────────────────────────────────────────────
    st.subheader("📊 Information Retrieval Metrics (top_k=5)")

    ir_data = {
        "Retriever":        ["Naive", "BM25+RRF", "Step-Back", "Hybrid", "Agentic"],
        "P@1":              [0.750, 0.783, 0.700, 0.783, 0.883],
        "P@3":              [0.656, 0.711, 0.667, 0.744, 0.856],
        "P@5":              [0.647, 0.683, 0.617, 0.713, 0.837],
        "R@1":              [0.558, 0.564, 0.472, 0.581, 0.656],
        "R@3":              [0.808, 0.819, 0.697, 0.819, 0.769],
        "R@5":              [0.864, 0.869, 0.778, 0.878, 0.803],
        "MRR":              [0.824, 0.851, 0.777, 0.850, 0.904],
        "P50 Latency (ms)": [13,    15,    2515,  345,   1065],
    }
    ir_df = pd.DataFrame(ir_data)
    st.dataframe(ir_df.set_index("Retriever"), use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.bar(
            ir_df, x="Retriever", y=["P@1", "P@3", "P@5"], barmode="group",
            title="Precision@k by Retriever",
            labels={"value": "Precision", "variable": "Metric"},
            color_discrete_sequence=["#3498db", "#2ecc71", "#e74c3c"],
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = px.bar(
            ir_df, x="Retriever", y=["R@1", "R@3", "R@5"], barmode="group",
            title="Recall@k by Retriever",
            labels={"value": "Recall", "variable": "Metric"},
            color_discrete_sequence=["#9b59b6", "#f39c12", "#1abc9c"],
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.bar(
            ir_df, x="Retriever", y="MRR", title="Mean Reciprocal Rank (MRR)",
            color="Retriever", color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = px.bar(
            ir_df, x="Retriever", y="P50 Latency (ms)", title="P50 Latency (ms)",
            color="Retriever", color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Section 2: RAGAS Scores ───────────────────────────────────────────────
    st.subheader("🧪 RAGAS Evaluation Scores")

    ragas_results = {}
    for name in ["naive", "hybrid", "agentic"]:
        path = Path(f"evaluation/ragas_results_{name}.json")
        if path.exists():
            ragas_results[name.capitalize()] = json.loads(path.read_text())["scores"]

    if not ragas_results:
        st.info(
            "No RAGAS results yet. "
            "Run `.venv/Scripts/python.exe evaluation/ragas_eval.py` to generate them."
        )
    else:
        RAGAS_METRICS = ["Context Precision", "Context Recall", "Faithfulness", "Answer Relevancy"]

        ragas_rows = [
            {
                "Retriever":         retriever,
                "Context Precision": scores.get("context_precision", 0),
                "Context Recall":    scores.get("context_recall", 0),
                "Faithfulness":      scores.get("faithfulness", 0),
                "Answer Relevancy":  scores.get("answer_relevancy", 0),
                "Average":           sum(scores.values()) / len(scores),
            }
            for retriever, scores in ragas_results.items()
        ]
        ragas_df = pd.DataFrame(ragas_rows)

        best = ragas_df.loc[ragas_df["Average"].idxmax()]
        st.caption(f"Best overall: **{best['Retriever']}** (avg: {best['Average']:.3f})")

        kpi_cols = st.columns(4)
        icons = ["🎯", "📚", "✅", "💬"]
        for col, metric, icon in zip(kpi_cols, RAGAS_METRICS, icons):
            with col:
                best_val = ragas_df[metric].max()
                best_ret = ragas_df.loc[ragas_df[metric].idxmax(), "Retriever"]
                st.metric(f"{icon} {metric}", f"{best_val:.3f}", help=f"Best: {best_ret}")

        st.dataframe(
            ragas_df.set_index("Retriever").style.format("{:.3f}"),
            use_container_width=True,
        )

        fig = px.bar(
            ragas_df.melt(id_vars="Retriever", value_vars=RAGAS_METRICS,
                          var_name="Metric", value_name="Score"),
            x="Metric", y="Score", color="Retriever", barmode="group",
            title="RAGAS Scores by Retriever",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Section 3: Key Findings ───────────────────────────────────────────────
    st.subheader("💡 Key Findings")
    st.markdown("""
| Finding | Detail |
|---------|--------|
| **Best Precision** | Agentic (P@1=0.883) — query rewriting makes every doc count |
| **Best Recall** | Hybrid (R@5=0.878) — widest coverage |
| **Best MRR** | Agentic (0.904) — relevant docs appear highest |
| **Fastest** | Naive / BM25+RRF (~13–15 ms P50) |
| **Production** | Hybrid (345 ms) for speed; Agentic (1065 ms) for hard queries |
| **Step-back** | Not adopted — worse than naive on most queries |
""")


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

st.title("🛍️ ShopEase Customer Support — Evaluation Dashboard")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Live Metrics", "🔍 RAG Evaluation"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Metrics
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    df = get_metrics_df(hours=hours)
    kpis = compute_summary(df)

    if df.empty:
        st.warning(
            "📭 No interaction data yet. Run `python main.py` to start the agent "
            "and generate some conversations."
        )
    else:
        # ── KPI cards ─────────────────────────────────────────────────────────
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

        # ── Row 1: Distribution charts ────────────────────────────────────────
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

        # ── Row 2: Latency + Resolution ───────────────────────────────────────
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

        # ── Row 3: Retrieval Quality + Guardrail Events ───────────────────────
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

        # ── Latency percentiles ───────────────────────────────────────────────
        st.subheader("⚡ Latency Percentiles")
        lat_cols = st.columns(3)
        with lat_cols[0]:
            st.metric("P50 (Median)", f"{kpis.get('p50_latency_ms', 0):.0f} ms")
        with lat_cols[1]:
            st.metric("P90", f"{kpis.get('p90_latency_ms', 0):.0f} ms")
        with lat_cols[2]:
            st.metric("Max", f"{df['latency_ms'].max():.0f} ms")

        st.divider()

        # ── Raw interaction table ─────────────────────────────────────────────
        st.subheader("📋 Recent Interactions")
        display_cols = [
            "timestamp", "customer_id", "intent", "agent_used",
            "resolution_status", "latency_ms", "guardrail_passed",
            "retrieved_doc_count", "avg_retrieval_score", "policy_compliant",
        ]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available].head(50), use_container_width=True, hide_index=True)

        st.download_button(
            label="📥 Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="shopease_eval_metrics.csv",
            mime="text/csv",
        )

        st.divider()

        # ── LLM Judge Scores ──────────────────────────────────────────────────
        st.subheader("🧑‍⚖️ LLM Judge Scores")
        st.caption(
            "Per-turn RAG quality scores produced by the judge node: "
            "faithfulness, answer relevancy, and context precision (each 0–1)."
        )

        scores_df = get_llm_scores_df(hours=hours)

        if scores_df.empty:
            st.info("No judge scores yet — run the agent to generate conversations.")
        else:
            avg_faith = scores_df["faithfulness"].mean()
            avg_relev = scores_df["answer_relevancy"].mean()
            avg_prec  = scores_df["context_precision"].mean()
            avg_all   = (avg_faith + avg_relev + avg_prec) / 3

            jcol1, jcol2, jcol3, jcol4 = st.columns(4)
            with jcol1:
                st.metric("✅ Faithfulness", f"{avg_faith:.3f}",
                          help="Response grounded in retrieved docs (1 = no hallucination)")
            with jcol2:
                st.metric("💬 Answer Relevancy", f"{avg_relev:.3f}",
                          help="Response directly answers the question (1 = perfect)")
            with jcol3:
                st.metric("🎯 Context Precision", f"{avg_prec:.3f}",
                          help="Retrieved docs are relevant to the question (1 = all relevant)")
            with jcol4:
                st.metric("⭐ Overall Average", f"{avg_all:.3f}")

            # Time-series of all three scores
            scores_sorted = scores_df.sort_values("timestamp")
            fig = px.line(
                scores_sorted,
                x="timestamp",
                y=["faithfulness", "answer_relevancy", "context_precision"],
                title="🧑‍⚖️ LLM Judge Scores Over Time",
                labels={"value": "Score (0–1)", "timestamp": "Time", "variable": "Metric"},
                markers=True,
                color_discrete_map={
                    "faithfulness":      "#2ecc71",
                    "answer_relevancy":  "#3498db",
                    "context_precision": "#9b59b6",
                },
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            # Score distribution
            col_l, col_r = st.columns(2)
            with col_l:
                fig = px.histogram(
                    scores_df.melt(value_vars=["faithfulness", "answer_relevancy", "context_precision"],
                                   var_name="Metric", value_name="Score"),
                    x="Score", color="Metric", nbins=20, barmode="overlay",
                    title="Score Distribution",
                    opacity=0.7,
                    color_discrete_map={
                        "faithfulness":      "#2ecc71",
                        "answer_relevancy":  "#3498db",
                        "context_precision": "#9b59b6",
                    },
                )
                fig.update_layout(xaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            with col_r:
                # Recent Q&A with scores
                display_score_cols = [
                    "timestamp", "faithfulness", "answer_relevancy",
                    "context_precision", "question", "response",
                ]
                available_score = [c for c in display_score_cols if c in scores_df.columns]
                st.dataframe(
                    scores_df[available_score].head(20),
                    use_container_width=True,
                    hide_index=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RAG Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    _render_rag_tab()
