# 🛍️ ShopEase — E-Commerce Customer Support Agent

> **CSAI 422 Course Project** — Multi-agent AI system with RAG, memory, guardrails, and observability

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LangGraph State Machine                         │
│                                                                     │
│  Input  ──►  [input_guardrail]  ──►  [supervisor]                  │
│                   │ (blocked)              │                         │
│                   ▼                        ├──► [order_lookup]      │
│                  END                       ├──► [policy_returns]    │
│                                            └──► [escalation]        │
│                                                       │             │
│                                            [output_guardrail]       │
│                                                       │             │
│                                            [finalize_metrics] ──► END│
└─────────────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Implementation |
|---------|---------------|
| **Multi-agent architecture** | 4 LangGraph nodes: Supervisor, Order Lookup, Policy & Returns, Escalation |
| **Advanced RAG** | Hybrid BM25 + dense vector search + cross-encoder reranking |
| **Short-term memory** | LangGraph `SqliteSaver` — per-session conversation history |
| **Long-term memory** | ChromaDB semantic store — cross-session customer context |
| **Input guardrail** | Regex + pattern matching for prompt injection detection |
| **Policy guardrail** | Refund limits, return window, non-returnable items |
| **Toxicity guardrail** | Profanity detection, threat detection, PII redaction |
| **Evaluation dashboard** | Streamlit dashboard with Plotly charts |
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
This downloads the embedding model (~90 MB) and indexes all data sources into ChromaDB.

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

### 7. Run tests (no API key needed)
```bash
pytest tests/ -v
```

---

## Project Structure

```
E:\data-mining_project\
├── config/
│   └── settings.py          # Pydantic settings from .env
├── data/
│   ├── product_catalog.json # 12 sample products
│   ├── faqs.json            # 20 FAQs
│   ├── shipping_policy.md   # Shipping policy document
│   ├── returns_policy.md    # Returns & refunds policy
│   └── mock_orders.json     # 8 sample orders
├── rag/
│   ├── embeddings.py        # SentenceTransformer bi-encoder + cross-encoder
│   ├── indexer.py           # ChromaDB document indexer
│   └── retriever.py         # Hybrid BM25 + dense + reranking pipeline
├── agents/
│   ├── supervisor.py        # Node 4: Orchestrator / intent classifier
│   ├── order_lookup.py      # Node 1: Order status + tracking
│   ├── policy_returns.py    # Node 2: Returns, refunds, policy
│   └── escalation.py        # Node 3: Escalations, ticket creation
├── memory/
│   ├── short_term.py        # LangGraph SqliteSaver checkpointer
│   └── long_term.py         # ChromaDB semantic memory
├── guardrails/
│   ├── input_guardrail.py   # Prompt injection detection
│   ├── policy_guardrail.py  # Business rule enforcement
│   └── toxicity_guardrail.py# Profanity, threats, PII
├── graph/
│   └── workflow.py          # LangGraph StateGraph definition
├── evaluation/
│   ├── metrics.py           # SQLite interaction logger
│   └── dashboard.py         # Streamlit dashboard
├── tools/
│   ├── order_tools.py       # LangChain tools: order lookup
│   └── policy_tools.py      # LangChain tools: policy/FAQ search
├── scripts/
│   └── index_documents.py   # One-time indexing script
├── tests/
│   └── test_agents.py       # Unit tests (no API key needed)
├── main.py                  # CLI entry point
├── requirements.txt
├── .env.example
└── CLAUDE.md               # Developer context for Claude Code
```

---

## RAG Pipeline Details

```
Customer query
      │
      ├─── BM25 sparse search (rank-bm25)        → top-10 candidates
      │
      ├─── Dense vector search (ChromaDB)         → top-10 candidates
      │           (all-MiniLM-L6-v2 embeddings)
      │
      ├─── Score fusion                           → merged candidates
      │    (40% BM25 + 60% dense, normalised)
      │
      └─── Cross-encoder reranking               → top-3 final docs
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
| Prompt injection | Input | Regex + patterns | Block + safe rejection |
| Refund limit | Policy | Amount > $500 | Escalate to supervisor |
| Return window | Policy | > 30 days since delivery | Escalate to manager |
| Non-returnable items | Policy | Digital/final-sale items | Deny with explanation |
| Toxicity | Output | Threats, profanity | Sanitise or replace |
| PII leakage | Output | SSN, CC, email in output | Redact |

---

## Memory

### Short-term (per-session)
- Implemented via LangGraph's `SqliteSaver`
- Scoped by `thread_id = session_id`
- Full conversation state persisted between turns

### Long-term (cross-session)
- ChromaDB collection `customer_memory`
- Escalation events + summaries embedded and stored
- Recalled by semantic search at session start

---

## Evaluation Metrics

| Metric | How Measured |
|--------|-------------|
| **Resolution Rate** | % of interactions with `resolution_status = resolved` |
| **Retrieval Quality** | Average cross-encoder rerank score |
| **Policy Compliance** | % of policy checks passing without escalation |
| **Latency** | Wall-clock time from input to final response (ms) |
| **Guardrail Block Rate** | % of inputs blocked by input guardrail |
| **Toxicity Rate** | % of outputs with toxicity score ≥ threshold |

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
Agent: Your order ORD-10002 (SoundWave Headphones + SpeedRun Gaming Mouse)
       shipped on May 19th via USPS (tracking: 940011189922...).
       It's currently in transit with an estimated delivery of May 23rd.

Customer: I want to return my UltraTab Pro 11, it's defective
Agent: I'm sorry to hear that! Since your order ORD-10005 was delivered
       on May 6th (18 days ago), you're well within the 30-day return window.
       Here's how to start your return: ...

Customer: I bought this coffee maker 55 days ago and it stopped working.
          I want a full refund.
Agent: I completely understand your frustration with the EcoBrew Coffee Maker.
       Unfortunately, 55 days is outside our standard 30-day return window.
       I'm escalating this to a senior specialist (Ticket: TKT-X7R2M)
       who will contact you within 1 business day...
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent framework | LangGraph 0.2+ |
| LLM orchestration | LangChain 0.3+ |
| LLM provider | Groq (llama-3.3-70b-versatile) |
| Vector database | ChromaDB (persistent) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Reranking | sentence-transformers CrossEncoder (ms-marco-MiniLM-L-6-v2) |
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
#   s h o p e a s e - s u p p o r t - a g e n t  
 