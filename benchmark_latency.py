"""
benchmark_latency.py — ShopEase Agent Latency Benchmark

Runs 20 queries across all 4 agent types, measures per-agent wall-clock
latency, reports mean / P50 / P90 / P95, and appends results to
data/benchmark_results.json for future comparison.

Usage:
    .\.venv\Scripts\python.exe benchmark_latency.py --label "current_baseline"
    .\.venv\Scripts\python.exe benchmark_latency.py --label "after_optimisation"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── Force UTF-8 on Windows ────────────────────────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()
logging.basicConfig(level=logging.ERROR)  # silence agent logs during benchmark

RESULTS_PATH = Path("data/benchmark_results.json")

# ── Benchmark query set (5 per target agent = 20 total) ──────────────────────
# Format: (customer_id, target_agent, query)
# target_agent is the expected route — actual route recorded from result.
BENCHMARK_QUERIES: list[tuple[str, str, str]] = [
    # General / Layla — product info, store info, recommendations
    ("CUST-001", "general", "What skincare products do you recommend for oily skin?"),
    ("CUST-001", "general", "How do I use the EcoBrew Coffee Maker?"),
    ("CUST-001", "general", "What are your store locations in Cairo?"),
    ("CUST-002", "general", "Do you have any current promotions or discount codes?"),
    ("CUST-002", "general", "What are the trending products this week?"),
    # Order Lookup — specific order ID queries
    ("CUST-001", "order_lookup", "Where is my order ORD-10001?"),
    ("CUST-001", "order_lookup", "Has my order ORD-10002 shipped yet?"),
    ("CUST-001", "order_lookup", "What items are in my order ORD-10001?"),
    ("CUST-001", "order_lookup", "When was ORD-10002 delivered?"),
    ("CUST-001", "order_lookup", "Can you give me the tracking details for ORD-10001?"),
    # Policy & Returns — return eligibility, refunds, policy questions
    ("CUST-001", "policy_returns", "I want to return my FitTrack Smart Watch, I bought it 5 days ago"),
    ("CUST-002", "policy_returns", "Can I get a refund for a defective coffee maker?"),
    ("CUST-003", "policy_returns", "What is your return policy for electronics?"),
    ("CUST-001", "policy_returns", "I received the wrong item, how do I return it?"),
    ("CUST-002", "policy_returns", "How long do I have to return a product after delivery?"),
    # Escalation — manager / supervisor requests
    ("CUST-001", "escalation", "I want to speak to a manager immediately"),
    ("CUST-002", "escalation", "This is completely unacceptable, I need to escalate this now"),
    ("CUST-003", "escalation", "I want to file a formal complaint about my experience"),
    ("CUST-004", "escalation", "I have been waiting 2 weeks and nobody helped me, I need a supervisor"),
    ("CUST-005", "escalation", "I am extremely dissatisfied and demand to speak with a senior representative"),
]

# Warm-up queries — one per major path; results discarded
WARMUP_QUERIES: list[tuple[str, str]] = [
    ("CUST-001", "Hi, what products do you sell?"),
    ("CUST-001", "Where is my order ORD-10001?"),
    ("CUST-001", "What is your return policy?"),
]


# ── Statistics helpers ────────────────────────────────────────────────────────

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * pct / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 1)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean":  round(sum(values) / len(values), 1),
        "p50":   _percentile(values, 50),
        "p90":   _percentile(values, 90),
        "p95":   _percentile(values, 95),
        "min":   round(min(values), 1),
        "max":   round(max(values), 1),
    }


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap() -> None:
    from config.settings import settings
    settings.apply_langsmith_env()
    settings.ensure_dirs()

    from evaluation.metrics import init_db
    init_db()

    console.print("[dim]Compiling LangGraph workflow…[/dim]")
    from graph.workflow import get_graph
    get_graph()

    console.print("[dim]Warming up RAG retriever…[/dim]")
    try:
        from rag.retriever import get_retriever
        get_retriever()
    except Exception as e:
        console.print(f"[yellow]⚠ RAG warm-up skipped: {e}[/yellow]")

    console.print("[green]✓ Ready.[/green]\n")


# ── Single query runner ───────────────────────────────────────────────────────

def _run_query(customer_id: str, query: str) -> tuple[float, str]:
    """Invoke the graph for one query. Returns (latency_ms, agent_used)."""
    from graph.workflow import get_graph, make_initial_state
    from memory.short_term import get_session_config

    session_id = f"bench_{uuid.uuid4().hex[:8]}"
    state = make_initial_state(customer_id, session_id, query)
    config = get_session_config(session_id)

    t0 = time.perf_counter()
    result = get_graph().invoke(state, config=config)
    latency_ms = (time.perf_counter() - t0) * 1000

    agent_used = result.get("agent_used", "unknown")
    return round(latency_ms, 1), agent_used


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(label: str) -> None:
    bootstrap()

    # ── Warm-up ───────────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold]Warm-up[/bold] — {len(WARMUP_QUERIES)} queries (results discarded)",
        border_style="dim",
    ))
    for i, (cid, q) in enumerate(WARMUP_QUERIES, 1):
        console.print(f"  [dim]warm-up {i}/{len(WARMUP_QUERIES)}: {q[:60]}…[/dim]", end=" ")
        try:
            ms, agent = _run_query(cid, q)
            console.print(f"[dim]{ms:.0f}ms → {agent}[/dim]")
        except Exception as e:
            console.print(f"[red]error: {e}[/red]")

    console.print()

    # ── Benchmark ─────────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold]Benchmark[/bold] — {len(BENCHMARK_QUERIES)} queries · label: [cyan]{label}[/cyan]",
        border_style="cyan",
    ))

    records: list[dict] = []
    latencies_by_agent: dict[str, list[float]] = {}

    for i, (cid, target, query) in enumerate(BENCHMARK_QUERIES, 1):
        console.print(
            f"  [{i:02d}/{len(BENCHMARK_QUERIES)}] "
            f"[dim]{cid}[/dim] [yellow]{target:>14}[/yellow]  {query[:55]}…",
            end=" ",
        )
        try:
            ms, actual_agent = _run_query(cid, query)
            match = "✓" if actual_agent == target else f"→{actual_agent}"
            console.print(f"[green]{ms:7.0f}ms[/green]  [dim]{match}[/dim]")

            latencies_by_agent.setdefault(actual_agent, []).append(ms)
            records.append({
                "query_index": i,
                "customer_id": cid,
                "target_agent": target,
                "actual_agent": actual_agent,
                "query": query,
                "latency_ms": ms,
            })
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
            records.append({
                "query_index": i,
                "customer_id": cid,
                "target_agent": target,
                "actual_agent": "error",
                "query": query,
                "latency_ms": None,
                "error": str(e),
            })

    # ── Per-agent statistics table ─────────────────────────────────────────────
    console.print()
    tbl = Table(
        title=f"Latency Results — [cyan]{label}[/cyan]",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )
    tbl.add_column("Agent",  style="bold", min_width=18)
    tbl.add_column("Runs",   justify="center", min_width=5)
    tbl.add_column("Mean",   justify="right", min_width=8)
    tbl.add_column("P50",    justify="right", min_width=8)
    tbl.add_column("P90",    justify="right", min_width=8)
    tbl.add_column("P95",    justify="right", min_width=8)
    tbl.add_column("Min",    justify="right", min_width=8)
    tbl.add_column("Max",    justify="right", min_width=8)

    agent_stats: dict[str, dict] = {}
    agent_order = ["general", "order_lookup", "policy_returns", "escalation"]
    all_agents = agent_order + [a for a in latencies_by_agent if a not in agent_order]

    overall_latencies: list[float] = []

    for agent in all_agents:
        vals = latencies_by_agent.get(agent, [])
        if not vals:
            continue
        s = _stats(vals)
        agent_stats[agent] = s
        overall_latencies.extend(vals)
        tbl.add_row(
            agent,
            str(len(vals)),
            f"{s['mean']:.0f} ms",
            f"{s['p50']:.0f} ms",
            f"{s['p90']:.0f} ms",
            f"{s['p95']:.0f} ms",
            f"{s['min']:.0f} ms",
            f"{s['max']:.0f} ms",
        )

    # Overall row
    if overall_latencies:
        ov = _stats(overall_latencies)
        tbl.add_section()
        tbl.add_row(
            "[bold]OVERALL[/bold]",
            str(len(overall_latencies)),
            f"[bold]{ov['mean']:.0f} ms[/bold]",
            f"[bold]{ov['p50']:.0f} ms[/bold]",
            f"[bold]{ov['p90']:.0f} ms[/bold]",
            f"[bold]{ov['p95']:.0f} ms[/bold]",
            f"[bold]{ov['min']:.0f} ms[/bold]",
            f"[bold]{ov['max']:.0f} ms[/bold]",
        )

    console.print(tbl)

    # Routing accuracy
    routed_correctly = sum(1 for r in records if r.get("actual_agent") == r["target_agent"])
    total_ran = sum(1 for r in records if r.get("actual_agent") != "error")
    if total_ran:
        console.print(
            f"\nRouting accuracy: [cyan]{routed_correctly}/{total_ran}[/cyan] "
            f"({routed_correctly/total_ran*100:.0f}%)"
        )

    # ── Save results ──────────────────────────────────────────────────────────
    existing: list[dict] = []
    if RESULTS_PATH.exists():
        try:
            existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

    run_record = {
        "label":      label,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "query_count": len(records),
        "agent_stats": agent_stats,
        "overall":    _stats(overall_latencies),
        "queries":    records,
    }
    existing.append(run_record)
    RESULTS_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(f"\n[green]✓ Results saved to {RESULTS_PATH}[/green]")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShopEase agent latency benchmark")
    parser.add_argument("--label", default="run", help="Label for this benchmark run")
    args = parser.parse_args()
    run_benchmark(args.label)
