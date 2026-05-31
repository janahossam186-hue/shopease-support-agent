# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**No system `pip`** — always use the project venv on the E drive:

```powershell
# Run all tests (no API key needed)
.\.venv\Scripts\python.exe -m pytest tests/ -v

# Run a single test class or individual test
.\.venv\Scripts\python.exe -m pytest tests/test_agents.py::TestInputGuardrail -v
.\.venv\Scripts\python.exe -m pytest tests/test_agents.py::TestPolicyGuardrail::test_refund_exceeds_limit -v

# Index documents — run once, or after changing any file in data/
# Downloads ~250 MB of embedding models on first run
.\.venv\Scripts\python.exe scripts/index_documents.py

# Run the agent
.\.venv\Scripts\python.exe main.py --customer CUST-001     # interactive chat
.\.venv\Scripts\python.exe main.py --demo                  # scripted 6-query demo

# Launch the evaluation dashboard
.\.venv\Scripts\streamlit.exe run evaluation/dashboard.py

# Install packages (redirect TEMP to avoid filling the C drive)
$env:TEMP = "E:\tmp"; $env:TMP = "E:\tmp"
.\.venv\Scripts\python.exe -m pip install --cache-dir E:\pip_cache <package>
```

## Required environment variables

Copy `.env.example` to `.env` and fill in:
- `GROQ_API_KEY` — Groq API key (free at console.groq.com)
- `LANGCHAIN_API_KEY` + `LANGCHAIN_TRACING_V2=true` — LangSmith tracing (auto-enabled once set)

## Architecture

### State machine (`graph/workflow.py`)

The graph passes a single `CustomerSupportState` TypedDict through every node. Each node returns a **partial dict** — only the keys it modifies. The sole exception is `messages`, which uses LangGraph's `add_messages` reducer (append-only). Use `make_initial_state()` to construct the correct starting state for each turn.

```
input_guardrail → supervisor ──► order_lookup ─────────────────┐
                             ──► policy_returns ───────────────►│ (or ──► escalation)
                             ──► escalation ───────────────────►│
                                                                ▼
                                                      output_guardrail → finalize_metrics → END
```

The compiled graph is a **process-level singleton** (`get_graph()`). Session isolation uses LangGraph's thread scoping: every `graph.invoke()` call must pass `{"configurable": {"thread_id": session_id}}` — see `memory/short_term.get_session_config()`.

### Routing

- **`route_after_guardrail`** (`input_guardrail` → `END` or `supervisor`): if `guardrail_passed` is False, a rejection `AIMessage` is already in state and the graph terminates.
- **`route_after_supervisor`** reads `state["intent"]`; both `"general"` and `"unknown"` route to `escalation` (by design — offers human handoff).
- **`route_after_policy`** reads `state["requires_escalation"]`, which is set **inline inside `policy_returns_node`** by calling `policy_guardrail_check()` — the policy guardrail is not its own graph node.

### RAG (`rag/`)

`HybridRetriever` is instantiated and `build_bm25_index()` is called **per request** inside each agent node — the BM25 index is in-memory only. If the ChromaDB `all_docs` collection is empty (indexer not yet run), retrieval silently returns `[]` and agents respond with "no articles found."

Score fusion: 40 % BM25 + 60 % dense (tunable via `BM25_WEIGHT` / `DENSE_WEIGHT` in `.env`).

### Memory

| Layer | Backend | Scope |
|-------|---------|-------|
| Short-term | `SqliteSaver` at `data/memory.db`, falls back to `MemorySaver` | Per `thread_id` (session) |
| Long-term | ChromaDB collection `customer_memory` | Per `customer_id`, semantic search |

Long-term memory is **written only by `escalation_node`** (stores ticket summaries) and **recalled only by `supervisor_node`** (injected into `metadata["past_context"]`).

### Guardrails

| Guardrail | Where | Effect |
|-----------|-------|--------|
| Input (injection patterns) | `input_guardrail_node` — graph node | Routes to `END`; rejection message appended to state |
| Policy (refund/return limits) | Inline in `policy_returns_node` via `policy_guardrail_check()` | Sets `requires_escalation=True`; `route_after_policy` branches to `escalation` |
| Toxicity + PII | `output_guardrail_node` — graph node | Sanitises or replaces the last `AIMessage` in state |

### Evaluation metrics

`evaluation/metrics.py` holds a **module-level SQLite singleton** (`_conn`). When testing code that touches metrics, patch the `settings` instance (not the module path):
```python
from config.settings import settings as s
monkeypatch.setattr(s, "eval_db_path", str(tmp_path / "test.db"))
import evaluation.metrics as em; em._conn = None
```

### Key mock data

- **Orders**: `data/mock_orders.json` — IDs `ORD-10001`–`ORD-10008`, customers `CUST-001`–`CUST-005`
- **ChromaDB collections**: `product_catalog`, `faqs`, `policies`, `all_docs` (unified for hybrid search)

### Extension patterns

| Task | Where |
|------|-------|
| New agent node | Create `agents/my_agent.py`, add `builder.add_node()` + branch in `route_after_supervisor()` in `graph/workflow.py` |
| New knowledge source | Add file to `data/`, add loader in `rag/indexer.py`, call from `index_all()` |
| New policy rule | Add `check_*` method to `PolicyGuardrail` in `guardrails/policy_guardrail.py`, wire into `policy_guardrail_check()` |
