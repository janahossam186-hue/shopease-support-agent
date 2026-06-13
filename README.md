# 🛍️ ShopEase — E-Commerce Customer Support Agent

> **CSAI 422 Course Project** — Multi-agent AI system with RAG, memory, guardrails, and observability

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          LangGraph State Machine                         │
│                                                                          │
│  Input ──► [input_guardrail] ──► [supervisor]                            │
│                │ (blocked)             │                                  │
│                ▼                       ├──► [order_lookup]               │
│               END                      ├──► [policy_returns]             │
│                                        ├──► [escalation]                 │
│                                        └──► [general_agent]              │
│                                                    │                     │
│                                        ┌───────────┘                     │
│                                        │ (multi-intent: loop back)       │
│                                        ▼                                 │
│                                [response_combiner]  (if decomposed)      │
│                                        │                                 │
│                                [output_guardrail]                        │
│                                        │                                 │
│                                [finalize_metrics]                        │
│                                        │                                 │
│                                   [judge_node] ──► END                   │
└──────────────────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Implementation |
|---------|---------------|
| **Multi-agent architecture** | 5 specialist LangGraph nodes: Supervisor, Order Lookup, Policy & Returns, Escalation, General |
| **Multi-intent decomposition** | Supervisor detects multiple intents in one message and routes sequentially; `response_combiner` merges partial responses |
| **OTP identity verification** | Order lookup requires a 6-digit OTP sent to the customer's email before revealing order data |
| **Advanced RAG** | Hybrid BM25 + dense vector search + cross-encoder reranking; agentic retriever with step-back prompting |
| **Short-term memory** | LangGraph `SqliteSaver` — per-session conversation history |
| **Long-term memory** | ChromaDB semantic store — cross-session customer context recalled by supervisor |
| **Input guardrail** | Regex + LLM-based prompt injection and jailbreak detection |
| **Policy guardrail** | Refund limits ($500 cap), return window (30 days), non-returnable items |
| **Output guardrail** | Profanity detection, threat detection, PII redaction (email, credit card, SSN) |
| **LLM judge node** | Scores every response on faithfulness, answer relevancy, and context precision (0–1) |
| **Evaluation dashboard** | Streamlit dashboard with Plotly charts; P50/P90/P95/Max latency metrics |
| **Latency benchmarking** | `benchmark_latency.py` — 20-query benchmark with percentile breakdown per agent |
| **LangSmith tracing** | Automatic tracing of all agent runs |

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- A Groq API key (free at [console.groq.com](https://console.groq.com))
- A LangSmith API key (free at [smith.langchain.com](https://smith.langchain.com))

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and fill in:
#   GROQ_API_KEY=gsk_...
#   LANGCHAIN_API_KEY=lsv2_...
```

### 4. Index documents (run once)
```bash
python scripts/index_documents.py
```
This downloads the embedding model (~250 MB) and indexes all data sources into ChromaDB.

### 5. Run the agent
```bash
# Interactive chat
python main.py --customer CUST-001

# Scripted demo (shows all agent types)
python main.py --demo
```

### 6. View the dashboard
```bash
streamlit run evaluation/dashboard.py
```

### 7. Run evaluation scenarios
```bash
# Run first 10 scenarios (token-efficient)
python scripts/run_eval_conversations.py

# Run latency benchmark
python benchmark_latency.py --label my_run
```

### 8. Run tests (no API key needed)
```bash
pytest tests/ -v
```

---

## Project Structure

```
data-mining_project/
├── config/
│   └── settings.py               # Pydantic settings from .env
├── data/
│   ├── product_catalog.json      # 12 sample products
│   ├── faqs.json                 # 20 FAQs
│   ├── shipping_policy.md        # Shipping policy document
│   ├── returns_policy.md         # Returns & refunds policy
│   ├── mock_orders.json          # 8 sample orders
│   ├── customers.json            # Customer profiles
│   ├── chroma_db/                # ChromaDB persistent store
│   ├── memory.db                 # SQLite short-term memory
│   └── eval_metrics.db           # SQLite evaluation metrics
├── rag/
│   ├── embeddings.py             # SentenceTransformer bi-encoder + cross-encoder
│   ├── indexer.py                # ChromaDB document indexer
│   ├── retriever.py              # Hybrid BM25 + dense + reranking pipeline
│   ├── agentic_retriever.py      # LLM-driven decide/retrieve/grade loop
│   ├── stepback_retriever.py     # Step-back prompting for broader queries
│   ├── bm25_rrf_retriever.py     # BM25 + Reciprocal Rank Fusion
│   └── naive_retriever.py        # Simple dense-only retriever
├── agents/
│   ├── supervisor.py             # Orchestrator: intent classification + decomposition
│   ├── order_lookup.py           # Order status, tracking, OTP verification
│   ├── policy_returns.py         # Returns, refunds, policy enforcement
│   ├── escalation.py             # Escalations, ticket creation, long-term memory write
│   ├── general_agent.py          # General KB questions, troubleshooting, recommendations
│   └── judge_node.py             # LLM judge: faithfulness / relevancy / precision scores
├── memory/
│   ├── short_term.py             # LangGraph SqliteSaver checkpointer
│   └── long_term.py              # ChromaDB semantic memory
├── guardrails/
│   ├── input_guardrail.py        # Prompt injection + jailbreak detection
│   ├── policy_guardrail.py       # Business rule enforcement
│   └── toxicity_guardrail.py     # Profanity, threats, PII redaction
├── graph/
│   └── workflow.py               # LangGraph StateGraph + response_combiner
├── evaluation/
│   ├── metrics.py                # SQLite interaction logger
│   ├── dashboard.py              # Streamlit dashboard (Plotly charts)
│   └── ragas_eval.py             # RAGAS evaluation pipeline
├── tools/
│   ├── order_tools.py            # LangChain tools: order lookup, OTP
│   └── policy_tools.py           # LangChain tools: policy/FAQ search
├── scripts/
│   ├── index_documents.py        # One-time indexing script
│   ├── run_eval_conversations.py # Runs 30 scripted eval scenarios
│   ├── live_agent_test.py        # Manual live testing helper
│   └── test_otp_flow.py          # OTP flow smoke test
├── tests/
│   ├── test_agents.py            # General agent unit tests
│   ├── test_supervisor.py        # Supervisor: routing, decomposition, OTP (63 tests)
│   ├── test_order_lookup.py      # Order lookup + OTP flow tests
│   ├── test_policies.py          # Policy guardrail tests
│   ├── test_guardrail.py         # Input/output guardrail tests
│   ├── test_escalation.py        # Escalation agent tests
│   ├── test_memory.py            # Memory layer tests
│   ├── test_adversarial.py       # Adversarial / injection tests
│   └── Generalagent_test.py      # General agent extended tests
├── benchmark_latency.py          # P50/P90/P95/Max latency benchmarking
├── main.py                       # CLI entry point
├── requirements.txt
├── .env.example
└── CLAUDE.md                     # Developer context for Claude Code
```

---

## RAG Pipeline Details

```
Customer query
      │
      ├─── Step-back prompting                      → broader reformulated query
      │
      ├─── BM25 sparse search (rank-bm25)           → top-10 candidates
      │
      ├─── Dense vector search (ChromaDB)           → top-10 candidates
      │           (all-MiniLM-L6-v2 embeddings)
      │
      ├─── Score fusion                             → merged candidates
      │    (40% BM25 + 60% dense, normalised)
      │
      └─── Cross-encoder reranking                 → top-3 final docs
               (ms-marco-MiniLM-L-6-v2)
```

Knowledge sources:
- **Product catalog** (12 products with descriptions, prices, specs)
- **FAQs** (20 Q&A pairs across 7 categories)
- **Shipping policy** (chunked markdown — 500 word chunks)
- **Returns policy** (chunked markdown — 500 word chunks)

---

## Guardrails

| Guardrail | Type | Trigger | Action |
|-----------|------|---------|--------|
| Prompt injection | Input | Regex + LLM patterns | Block + safe rejection |
| Jailbreak | Input | Role-play / override attempts | Block + safe rejection |
| Refund limit | Policy | Amount > $500 | Escalate to supervisor |
| Return window | Policy | > 30 days since delivery | Escalate to manager |
| Non-returnable items | Policy | Digital / final-sale items | Deny with explanation |
| Toxicity | Output | Threats, profanity | Sanitise or replace |
| PII leakage | Output | SSN, CC number, email in output | Redact |

---

## Memory

### Short-term (per-session)
- Implemented via LangGraph's `SqliteSaver`
- Scoped by `thread_id = session_id`
- Full conversation state persisted between turns

### Long-term (cross-session)
- ChromaDB collection `customer_memory`
- Escalation events + summaries embedded and stored
- Recalled by semantic search at session start via supervisor

---

## Evaluation Metrics

| Metric | How Measured |
|--------|-------------|
| **Resolution Rate** | % of interactions with `resolution_status = resolved` |
| **Faithfulness** | LLM judge: are all claims grounded in retrieved docs? (0–1) |
| **Answer Relevancy** | LLM judge: does the response answer the question? (0–1) |
| **Context Precision** | LLM judge: are retrieved docs relevant to the query? (0–1) |
| **Retrieval Quality** | Average cross-encoder rerank score |
| **Policy Compliance** | % of policy checks passing without escalation |
| **Latency P50/P90/P95/Max** | Wall-clock time from input to final response (ms) |
| **Guardrail Block Rate** | % of inputs blocked by input guardrail |
| **Routing Accuracy** | % of queries routed to the correct specialist agent |

> Set `JUDGE_ENABLED=false` in `.env` to skip LLM judge calls and save ~60% of token usage per eval run.

---

## LangSmith Integration

All graph runs are traced automatically when `LANGCHAIN_TRACING_V2=true`.

View traces at: [smith.langchain.com](https://smith.langchain.com)

Each trace captures:
- The full node execution sequence
- Input/output at every node
- Token usage per LLM call
- Latency per step

---

## Sample Interactions

```
Customer: Where is my order ORD-10002?
Agent:    [sends OTP to customer email for identity verification]
          Please enter the 6-digit code we just sent to your email.

Customer: 482910
Agent:    Your order ORD-10002 (SoundWave Headphones + SpeedRun Gaming Mouse)
          shipped on May 19th via USPS (tracking: 940011189922...).
          Estimated delivery: May 23rd.

Customer: I want to return it AND check the return policy for electronics.
Agent:    [supervisor decomposes → order_lookup + policy_returns → combiner]
          Your order ORD-10002 is eligible for return within the 30-day window.
          For electronics, our standard policy applies: items must be unopened
          or defective. Here's how to start your return: ...

Customer: I bought this coffee maker 55 days ago and it stopped working.
          I want a full refund.
Agent:    I completely understand your frustration. Unfortunately, 55 days is
          outside our 30-day return window. I'm escalating this to a senior
          specialist (Ticket: TKT-X7R2M) who will contact you within 1 business day.
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent framework | LangGraph 0.2+ |
| LLM orchestration | LangChain 0.3+ |
| LLM provider | Groq (`llama-3.3-70b-versatile`) |
| Vector database | ChromaDB (persistent) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Reranking | sentence-transformers CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| Sparse search | rank-bm25 |
| Short-term memory | LangGraph SqliteSaver |
| Long-term memory | ChromaDB |
| Observability | LangSmith |
| Dashboard | Streamlit + Plotly |
| Evaluation store | SQLite |
| Configuration | pydantic-settings |
| CLI | rich |

---

*CSAI 422 — Data Mining & AI Systems | ShopEase Customer Support Agent*
