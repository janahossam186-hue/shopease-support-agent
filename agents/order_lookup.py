"""
Node — Order Lookup Agent (Nora) — LLMCompiler DAG Architecture

Architecture:
    planner_node → scheduler_node → joiner_node
                        ↑                |
                        |           REPLAN (max 2)
                        └───────────────┘

The main LangGraph graph calls order_lookup_node, which runs the full
planner → scheduler → joiner subgraph internally and returns final state.

Identity flow (stored in state["metadata"]):
  - "identity_verified" : bool — set True once OTP is confirmed
  - "pending_otp"       : str  — 6-digit code sent in the current session
  - "compiler_state"    : dict — plan, completed_tasks, trace, replan_count
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Any, Dict, List, Optional, Set, TypedDict

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from config.settings import settings
from tools.order_tools import (
    get_order_tool,
    list_customer_orders_tool,
    cancel_order_tool,
    update_address_tool,
    update_quantity_tool,
    remove_item_tool,
    send_otp_email,
    _load_orders,
)

logger = logging.getLogger(__name__)


# ── State schema ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Typed state shared by planner, scheduler, and joiner nodes."""
    messages: Annotated[list, add_messages]
    customer_id: str
    session_id: str
    order_id: str | None
    agent_used: str
    resolution_status: str
    requires_escalation: bool
    retrieved_docs: list
    retrieval_scores: list
    metadata: dict


MAX_REPLANS = 2
COMPILE_REF_REGEX = r"\$(T[A-Za-z0-9_]*)"

# Keys written into tool results that must be merged back into metadata after each wave
_METADATA_SIDE_EFFECT_KEYS = {
    "pending_otp", "identity_verified",
    "return_context", "escalation_handoff", "detected_emotion",
}


# ── LLM Singleton ──────────────────────────────────────────────────────────────

_llm_instance = None


def _get_llm():
    """Return shared ChatGroq instance — created once, reused every call."""
    global _llm_instance
    if _llm_instance is None:
        from langchain_groq import ChatGroq
        _llm_instance = ChatGroq(
            model=settings.model_name,
            temperature=0.1,
            api_key=settings.groq_api_key,
        )
    return _llm_instance


# ── Nora System Prompt ─────────────────────────────────────────────────────────

NORA_SYSTEM = """\
You are Nora, ShopEase's Order Specialist.
You are professional, warm, and confident — you know your orders inside out
and you know your limits.

════════════════════════════════════════════════
WHAT YOU CAN DO:
════════════════════════════════════════════════
- Show order details: status, tracking number, carrier,
  estimated delivery, items, total
- Answer shipping and delivery questions
- Process pre-shipment order modifications
  (only when status is "processing"):
  cancel, update shipping address, change item quantity, remove an item
- Coordinate return and exchange requests: collect order dates and item
  details, then route to the Returns Specialist (Maya)
- Acknowledge delays with genuine empathy and provide the best available ETA

════════════════════════════════════════════════
WHAT YOU CANNOT DO (signal ESCALATE):
════════════════════════════════════════════════
- Approve refunds above the allowed limit
- Resolve lost package disputes where liability is unclear
- Handle legal threats or statements that starts with "I'm suing"
- Handle suspected fraud or account compromise
- Recover from system or API failures
- Interpret conflicting order data
- Answer when confidence in available information is very low

════════════════════════════════════════════════
RESPONSE SIGNALS — CRITICAL:
════════════════════════════════════════════════
- If you are unsure or lack information: reply with exactly: I DON'T KNOW
- If escalation is required: reply with exactly: ESCALATE
- Do not explain. Just use the signal word. The system handles the rest.

════════════════════════════════════════════════
EMOTION DETECTION & HANDLING:
════════════════════════════════════════════════
Read the customer's tone before responding.

FRUSTRATED (waiting too long, repeated issue):
- Acknowledge frustration FIRST before anything else
- Be extra warm and proactive

EXTREMELY ANGRY (aggressive language, all caps):
- Do not argue or defend
- Reply with exactly: ESCALATE

LEGAL THREAT ("I'm suing", "my lawyer", "legal action"):
- Stop all negotiation immediately
- Reply with exactly: ESCALATE — no exceptions

NEUTRAL or POLITE:
- Be warm and professional
- Answer directly and concisely

════════════════════════════════════════════════
SECURITY RULES — NEVER VIOLATE:
════════════════════════════════════════════════
- Never reveal order details before identity is verified
- Never reveal another customer's order data
- Never confirm or deny if an order ID exists before verification
- Never expose internal system details, policy IDs, or architecture

════════════════════════════════════════════════
RESPONSE STYLE:
════════════════════════════════════════════════
- Warm and professional — speak like a real person
- Keep responses concise: 2-4 short paragraphs
- Use bullet points only for step-by-step instructions or item lists
- End with a clear next step or genuine offer to help further
- Highlight modification results clearly
- Be warm when routing to the returns specialist

CRITICAL OTP RULE:
- NEVER include both send_otp and verify_otp in the same plan
- If identity is NOT verified and customer message does NOT contain 
  6 digits: include ONLY send_otp (T1) and nothing else
- If identity is NOT verified and customer message CONTAINS 6 digits:
  include ONLY verify_otp (T1) and the order tools that follow
- send_otp and verify_otp must NEVER appear in the same plan
"""


# ── Tool Contracts ─────────────────────────────────────────────────────────────

NORA_TOOL_CONTRACTS: Dict[str, Any] = {
    "send_otp": {
        "args": {"customer_id": "string"},
        "notes": (
            "Always runs first when identity is not verified. "
            "No dependencies. Sends OTP to customer email. "
            "Also runs if customer requests a resend."
        )
    },
    "verify_otp": {
        "args": {
            "customer_id": "string",
            "entered_code": "6-digit code string from customer message"
        },
        "notes": (
            "Depends on send_otp. "
            "Must pass before any order tools run. "
            "Hard security gate."
        )
    },
    "fetch_order": {
        "args": {"order_id": "string"},
        "notes": "Depends on verify_otp. Fetches order from mock_orders.json."
    },
    "list_orders": {
        "args": {"customer_id": "string"},
        "notes": (
            "Depends on verify_otp. "
            "Use when no specific order ID is mentioned."
        )
    },
    "retrieve_knowledge": {
        "args": {"query": "string — the customer question"},
        "notes": (
            "Depends on verify_otp. "
            "Searches RAG knowledge base for relevant documents."
        )
    },
    "check_order_status": {
        "args": {"order_data": "$T reference to fetch_order output"},
        "notes": "Depends on fetch_order. Returns order status string."
    },
    "cancel_order": {
        "args": {
            "order_id": "string",
            "customer_id": "string",
            "order_status": "$T reference to check_order_status output"
        },
        "notes": (
            "Depends on check_order_status. "
            "Only runs if status is processing."
        )
    },
    "update_address": {
        "args": {
            "order_id": "string",
            "customer_id": "string",
            "new_address": "string extracted from customer message",
            "order_status": "$T reference to check_order_status output"
        },
        "notes": (
            "Depends on check_order_status. "
            "Only runs if status is processing."
        )
    },
    "update_quantity": {
        "args": {
            "order_id": "string",
            "customer_id": "string",
            "product_id": "string",
            "new_qty": "integer",
            "order_status": "$T reference to check_order_status output"
        },
        "notes": (
            "Depends on check_order_status. "
            "Only runs if status is processing."
        )
    },
    "remove_item": {
        "args": {
            "order_id": "string",
            "customer_id": "string",
            "product_id": "string",
            "order_status": "$T reference to check_order_status output"
        },
        "notes": (
            "Depends on check_order_status. "
            "Only runs if status is processing."
        )
    },
    "collect_return_context": {
        "args": {
            "order_data": "$T reference to fetch_order output"
        },
        "notes": (
            "Depends on fetch_order. "
            "Collects purchase date, delivery date, items for Maya."
        )
    },
    "build_handoff": {
        "args": {
            "reason": "string explaining why escalation is needed",
            "order_data": "$T reference to fetch_order output or empty string"
        },
        "notes": "Only used when joiner decides ESCALATE."
    }
}

NORA_FALLBACK_PLAN: Dict[str, Any] = {
    "tasks": [
        {
            "id": "T1",
            "tool": "send_otp",
            "args": {"customer_id": "PLACEHOLDER_CUSTOMER_ID"},
            "deps": []
        },
        {
            "id": "T2",
            "tool": "verify_otp",
            "args": {
                "customer_id": "PLACEHOLDER_CUSTOMER_ID",
                "entered_code": "PLACEHOLDER_CODE"
            },
            "deps": ["T1"]
        },
        {
            "id": "T3",
            "tool": "fetch_order",
            "args": {"order_id": "PLACEHOLDER_ORDER_ID"},
            "deps": ["T2"]
        },
        {
            "id": "T4",
            "tool": "retrieve_knowledge",
            "args": {"query": "PLACEHOLDER_QUESTION"},
            "deps": ["T2"]
        },
        {
            "id": "T5",
            "tool": "check_order_status",
            "args": {"order_data": "$T3"},
            "deps": ["T3"]
        }
    ]
}


# ── DAG Utilities ──────────────────────────────────────────────────────────────

def find_compile_refs(obj: Any) -> Set[str]:
    """Find all $T references inside nested args."""
    refs: Set[str] = set()
    if isinstance(obj, str):
        refs.update(re.findall(COMPILE_REF_REGEX, obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            refs.update(find_compile_refs(value))
    elif isinstance(obj, list):
        for value in obj:
            refs.update(find_compile_refs(value))
    return refs


def resolve_compile_refs(obj: Any, results: Dict[str, Any]) -> Any:
    """Replace $T references with actual task results."""
    if isinstance(obj, str):
        # Exact match — return the result object directly, not serialised
        exact = re.fullmatch(COMPILE_REF_REGEX, obj.strip())
        if exact:
            return results[exact.group(1)]
        # Partial match — inline as JSON string
        def repl(match: re.Match) -> str:
            return json.dumps(results[match.group(1)], ensure_ascii=False)
        return re.sub(COMPILE_REF_REGEX, repl, obj)
    if isinstance(obj, dict):
        return {k: resolve_compile_refs(v, results) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_compile_refs(v, results) for v in obj]
    return obj


def normalize_nora_tasks(raw_plan: Any) -> List[Dict[str, Any]]:
    """Normalize raw LLM plan output into a clean list of task dicts."""
    if isinstance(raw_plan, list):
        tasks = raw_plan
    else:
        tasks = raw_plan.get("tasks", [])
    return [
        {
            "id": task.get("id", ""),
            "tool": task.get("tool", ""),
            "args": task.get("args", {}),
            "deps": task.get("deps", [])
        }
        for task in tasks
    ]


def _get_all_deps(
    task_id: str,
    id_to_task: Dict[str, Any],
    visited: Optional[Set[str]] = None,
) -> Set[str]:
    """Recursively get all transitive dependencies of a task."""
    if visited is None:
        visited = set()
    for dep in id_to_task.get(task_id, {}).get("deps", []):
        if dep not in visited:
            visited.add(dep)
            _get_all_deps(dep, id_to_task, visited)
    return visited


def _detect_cycles(tasks: List[Dict[str, Any]]) -> None:
    """Raise ValueError if the DAG contains a cycle (DFS)."""
    graph = {t["id"]: t.get("deps", []) for t in tasks}
    visiting: Set[str] = set()
    visited: Set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"Cycle detected at task {node}.")
        if node in visited:
            return
        visiting.add(node)
        for dep in graph[node]:
            visit(dep)
        visiting.remove(node)
        visited.add(node)

    for tid in graph:
        visit(tid)


def validate_nora_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate the Nora DAG is executable.
    Checks: valid IDs, valid tool names, $T refs in deps,
    all deps exist, no cycles, verify_otp transitive dependency on order tools.
    """
    tasks = normalize_nora_tasks(tasks)
    task_ids = [t["id"] for t in tasks]

    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate task IDs in DAG.")

    id_to_task = {t["id"]: t for t in tasks}
    order_tools = {
        "fetch_order", "list_orders", "retrieve_knowledge",
        "cancel_order", "update_address", "update_quantity",
        "remove_item", "collect_return_context", "build_handoff",
    }
    verify_task_ids = [t["id"] for t in tasks if t["tool"] == "verify_otp"]

    for task in tasks:
        task_id = task["id"]
        tool = task["tool"]
        deps = task.get("deps", [])
        args = task.get("args", {})

        if not re.fullmatch(r"T[A-Za-z0-9_]*", task_id):
            raise ValueError(f"Invalid task ID: {task_id}")
        if tool not in NORA_TOOL_CONTRACTS:
            raise ValueError(f"{task_id}: unknown tool '{tool}'")

        for dep in deps:
            if dep not in id_to_task:
                raise ValueError(f"{task_id}: dependency '{dep}' does not exist.")
            if dep == task_id:
                raise ValueError(f"{task_id}: cannot depend on itself.")

        for ref in find_compile_refs(args):
            if ref not in id_to_task:
                raise ValueError(f"{task_id}: reference '${ref}' does not exist.")
            if ref not in deps:
                raise ValueError(
                    f"{task_id}: reference '${ref}' used in args but missing from deps."
                )

        if tool in order_tools and verify_task_ids:
            all_deps = _get_all_deps(task_id, id_to_task)
            if not any(v in all_deps for v in verify_task_ids):
                raise ValueError(
                    f"{task_id}: order tool '{tool}' must depend on "
                    f"verify_otp (directly or transitively)."
                )

    _detect_cycles(tasks)
    return tasks


# ── Tool Executor ──────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Any:
    """Strip markdown fences and parse JSON from LLM output."""
    clean = re.sub(r"```json|```", "", text).strip()
    return json.loads(clean)


def _get_order_status(args: Dict[str, Any]) -> str:
    """Extract order status string from the resolved order_status arg."""
    data = args.get("order_status", {})
    return data.get("order_status", "") if isinstance(data, dict) else str(data)


def _execute_tool(
    tool_name: str,
    args: Dict[str, Any],
    state: Dict[str, Any],
) -> Any:
    """
    Execute a single Nora tool by name with resolved args.
    Returns the result dict or raises on unrecoverable failure.
    All $T references must be resolved before calling this function.
    """
    customer_id = state.get("customer_id", "unknown")
    metadata = state.get("metadata", {})

    if tool_name == "send_otp":
        otp, _masked = send_otp_email(customer_id)
        logger.info("OTP dispatched for customer %s", customer_id)
        return {
            "status": "otp_sent",
            "message": "OTP sent to registered email",
            "pending_otp": otp,
            "identity_verified": False,
        }

    elif tool_name == "verify_otp":
        entered = str(args.get("entered_code", "")).strip()
        # Extract 6 consecutive digits — customer may include extra text
        otp_match = re.search(r"\d{6}", entered)
        entered_digits = otp_match.group() if otp_match else entered
        pending = str(metadata.get("pending_otp", ""))

        if metadata.get("identity_verified"):
            return {"status": "already_verified"}

        if pending and entered_digits == pending:
            logger.info("Identity verified for customer %s", customer_id)
            return {
                "status": "verified",
                "identity_verified": True,
                "pending_otp": None,  # invalidate after use
            }
        else:
            logger.warning("OTP mismatch for customer %s", customer_id)
            return {"status": "failed", "message": "Code does not match"}

    elif tool_name == "fetch_order":
        order_id = args.get("order_id", "")
        result = get_order_tool.invoke({"order_id": order_id})
        try:
            raw = _load_orders()
            order_dict = raw.get(order_id.strip().upper(), {})
            if not order_dict:  # Order doesn't exist at all
                return {"status": "not_found", "message": f"Order '{order_id}' not found"}
            if order_dict.get("customer_id") != customer_id:
                # FRAUD SIGNAL — customer trying to access another customer's order
                logger.warning(
                    "FRAUD SIGNAL: customer %s attempted to access order %s "
                    "belonging to customer %s — escalating immediately",
                    customer_id, order_id, order_dict.get("customer_id")
                )
                return {
                    "status": "fraud_detected",
                    "message": "Order belongs to a different account"
                }
            return {"status": "ok", "data": result, "raw": order_dict}
        except Exception as e:
            logger.warning("Could not verify order ownership: %s", e)
            return {"status": "ok", "data": result, "raw": {}}

    elif tool_name == "list_orders":
        result = list_customer_orders_tool.invoke({"customer_id": customer_id})
        return {"status": "ok", "data": result}

    elif tool_name == "retrieve_knowledge":
        try:
            from rag.retriever import get_retriever
            retriever = get_retriever()
            query = args.get("query", "")
            docs = retriever.retrieve(query=query, top_k_final=settings.top_k_rerank)
            kb_context = retriever.format_for_prompt(docs) if docs else ""
            scores = [d.rerank_score for d in docs] if docs else []
            return {"status": "ok", "context": kb_context, "scores": scores}
        except Exception as e:
            logger.warning("RAG retrieval failed: %s", e)
            return {"status": "ok", "context": "", "scores": []}

    elif tool_name == "check_order_status":
        order_data = args.get("order_data", {})
        if isinstance(order_data, dict):
            raw = order_data.get("raw", {})
            status = raw.get("status", "unknown")
        else:
            status = "unknown"
        return {"status": "ok", "order_status": status}

    elif tool_name == "cancel_order":
        status = _get_order_status(args)
        if status != "processing":
            return {"status": "denied", "message": f"Cannot cancel — order is '{status}'"}
        return {"status": "ok", "result": cancel_order_tool.invoke(
            {"order_id": args.get("order_id", ""), "customer_id": customer_id}
        )}

    elif tool_name == "update_address":
        status = _get_order_status(args)
        if status != "processing":
            return {"status": "denied", "message": f"Cannot update address — order is '{status}'"}
        return {"status": "ok", "result": update_address_tool.invoke({
            "order_id": args.get("order_id", ""),
            "customer_id": customer_id,
            "new_address": args.get("new_address", ""),
        })}

    elif tool_name == "update_quantity":
        status = _get_order_status(args)
        if status != "processing":
            return {"status": "denied", "message": f"Cannot update quantity — order is '{status}'"}
        return {"status": "ok", "result": update_quantity_tool.invoke({
            "order_id": args.get("order_id", ""),
            "customer_id": customer_id,
            "product_id": args.get("product_id", ""),
            "new_qty": int(args.get("new_qty", 1)),
        })}

    elif tool_name == "remove_item":
        status = _get_order_status(args)
        if status != "processing":
            return {"status": "denied", "message": f"Cannot remove item — order is '{status}'"}
        return {"status": "ok", "result": remove_item_tool.invoke({
            "order_id": args.get("order_id", ""),
            "customer_id": customer_id,
            "product_id": args.get("product_id", ""),
        })}

    elif tool_name == "collect_return_context":
        order_data = args.get("order_data", {})
        if isinstance(order_data, dict):
            raw = order_data.get("raw", {})
            return_context = {
                "order_id": raw.get("order_id", ""),
                "purchase_date": raw.get("created_at", ""),
                "delivered_at": raw.get("delivered_at", ""),
                "items": raw.get("items", []),
                "customer_id": customer_id,
            }
        else:
            return_context = {}
        return {"status": "ok", "return_context": return_context}

    elif tool_name == "build_handoff":
        reason = args.get("reason", "Escalation required")
        order_data = args.get("order_data", {})
        raw = order_data.get("raw", {}) if isinstance(order_data, dict) else {}
        handoff = {
            "issue_summary": reason,
            "actions_attempted": list(
                metadata.get("compiler_state", {})
                .get("completed_tasks", {}).keys()
            ),
            "customer_sentiment": metadata.get("detected_emotion", "neutral"),
            "relevant_order_info": raw,
            "escalation_reason": reason,
            "confidence_score": 0.3,
            "agent": "nora_order_lookup",
        }
        return {"status": "ok", "handoff": handoff, "escalation_handoff": handoff}

    else:
        raise ValueError(f"Unknown tool: {tool_name}")


# ── Planner Node ───────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """
    LLMCompiler planner — creates a DAG of tasks for Nora to execute.

    Reads:  messages, customer_id, order_id, metadata
    Writes: metadata["compiler_state"]["plan"]

    Uses NORA_TOOL_CONTRACTS to constrain the LLM plan.
    Falls back to NORA_FALLBACK_PLAN if validation fails.
    Skips OTP tasks if identity already verified this session.
    """
    messages = state.get("messages", [])
    customer_id = state.get("customer_id", "unknown")
    order_id = state.get("order_id")
    metadata = dict(state.get("metadata", {}))

    last_human = next(
        (m.content for m in reversed(messages) if getattr(m, "type", "") == "human"),
        "",
    )

    identity_verified = bool(metadata.get("identity_verified", False))
    replan_count = metadata.get("compiler_state", {}).get("replan_count", 0)
    previous_results = metadata.get("compiler_state", {}).get("completed_tasks", {})

    contracts_text = json.dumps(NORA_TOOL_CONTRACTS, indent=2, ensure_ascii=False)

    # ── OTP verification fast path ────────────────────────────────────────────
    # When an OTP is already pending and the customer's message contains a
    # 6-digit code, skip the LLM entirely and build the plan deterministically.
    # This prevents the LLM from generating a new send_otp task, which would
    # overwrite the pending code and loop the customer back to "enter your code".
    pending_otp = metadata.get("pending_otp")
    logger.info(
        "Nora planner debug: pending_otp=%s identity_verified=%s last_human=%s",
        metadata.get("pending_otp"),
        metadata.get("identity_verified"),
        last_human[:20],
    )
    if pending_otp and not identity_verified and re.search(r"\d{6}", last_human):
        pending_intent = metadata.get("pending_intent", {})
        original_order_id = pending_intent.get("order_id") or order_id
        original_question = pending_intent.get("question") or last_human

        fast_tasks: List[Dict[str, Any]] = [
            {
                "id": "T1",
                "tool": "verify_otp",
                "args": {"customer_id": customer_id, "entered_code": last_human},
                "deps": [],
            }
        ]
        if original_order_id:
            fast_tasks += [
                {
                    "id": "T2",
                    "tool": "fetch_order",
                    "args": {"order_id": original_order_id},
                    "deps": ["T1"],
                },
                {
                    "id": "T3",
                    "tool": "retrieve_knowledge",
                    "args": {"query": original_question},
                    "deps": ["T1"],
                },
                {
                    "id": "T4",
                    "tool": "check_order_status",
                    "args": {"order_data": "$T2"},
                    "deps": ["T2"],
                },
            ]
        else:
            fast_tasks.append(
                {
                    "id": "T2",
                    "tool": "list_orders",
                    "args": {"customer_id": customer_id},
                    "deps": ["T1"],
                }
            )

        try:
            fast_tasks = validate_nora_tasks(fast_tasks)
            logger.info(
                "Nora planner: OTP fast-path — %d tasks for customer %s",
                len(fast_tasks), customer_id,
            )
            compiler_state = metadata.get("compiler_state", {})
            compiler_state["plan"] = fast_tasks
            compiler_state["replan_count"] = replan_count
            compiler_state["completed_tasks"] = {}   # reset — never inherit from previous turn
            compiler_state["trace"] = []
            metadata["compiler_state"] = compiler_state
            return {"metadata": metadata, "agent_used": "order_lookup"}
        except Exception as e:
            logger.warning(
                "Nora planner: OTP fast-path failed (%s) — falling through to LLM", e
            )
            # Fall through to the LLM planner below

    # Build context notes to guide the planner
    context_notes: List[str] = []
    if identity_verified:
        context_notes.append(
            "Identity is already verified this session. "
            "Do NOT include send_otp or verify_otp in the plan."
        )
    if order_id:
        context_notes.append(f"Customer mentioned order ID: {order_id}")
    if previous_results:
        context_notes.append(
            f"Previous task results available: {list(previous_results.keys())}"
        )
    if replan_count > 0:
        context_notes.append(
            f"This is replan #{replan_count}. Previous plan had issues. Adjust accordingly."
        )
    context_str = "\n".join(context_notes) if context_notes else "No prior context."

    prompt = f"""You are an LLMCompiler planner for Nora, ShopEase's Order Specialist.

Return ONLY valid JSON. No markdown. No explanation.

Output schema:
{{
  "tasks": [
    {{
      "id": "T1",
      "tool": "tool_name",
      "args": {{}},
      "deps": []
    }}
  ]
}}

Tool contracts:
{contracts_text}

Critical rules:
- Use only tool names from the contracts above
- Task IDs must be T1, T2, T3 etc.
- Every $T reference in args must appear in deps
- The graph must be acyclic
- All order tools must depend on verify_otp (directly or transitively)
  UNLESS identity is already verified (check context notes)
- Only include tools needed to answer this specific question
- If customer says "resend", "new code", "send again", "didn't receive code":
  include send_otp as T1 with no dependencies
- Do not include unnecessary tools

Context:
{context_str}

Customer ID: {customer_id}
Order ID: {order_id or "not mentioned"}
Customer message: {last_human}

Create the minimal DAG needed to answer this message."""

    llm = _get_llm()

    try:
        response = llm.invoke(prompt).content
        raw_plan = _extract_json(response)
        tasks = normalize_nora_tasks(raw_plan)
        tasks = validate_nora_tasks(tasks)
        logger.info(
            "Nora planner: %d tasks planned for customer %s", len(tasks), customer_id
        )
    except Exception as e:
        logger.warning("Nora planner failed validation, using fallback: %s", e)
        # Fill placeholders in the fallback plan
        fallback = json.loads(
            json.dumps(NORA_FALLBACK_PLAN)
            .replace("PLACEHOLDER_CUSTOMER_ID", customer_id)
            .replace("PLACEHOLDER_CODE", last_human[:10])
            .replace("PLACEHOLDER_ORDER_ID", order_id or "UNKNOWN")
            .replace("PLACEHOLDER_QUESTION", last_human[:100])
        )
        # If identity already verified remove send_otp and verify_otp from fallback
        if identity_verified:
            fallback["tasks"] = [
                t for t in fallback["tasks"]
                if t["tool"] not in ("send_otp", "verify_otp")
            ]
            # Remove deps on T1/T2 since they're gone
            for t in fallback["tasks"]:
                t["deps"] = [d for d in t["deps"] if d not in ("T1", "T2")]
        tasks = validate_nora_tasks(fallback["tasks"])

    compiler_state = metadata.get("compiler_state", {})
    compiler_state["plan"] = tasks
    compiler_state["replan_count"] = replan_count
    compiler_state["completed_tasks"] = {}   # always reset — never carry over from a previous turn
    compiler_state["trace"] = []
    metadata["compiler_state"] = compiler_state

    return {
        "metadata": metadata,
        "agent_used": "order_lookup",
    }


# ── Scheduler Node ─────────────────────────────────────────────────────────────

def scheduler_node(state: AgentState) -> dict:
    """
    LLMCompiler scheduler — executes the DAG produced by planner_node.

    Reads:  metadata["compiler_state"]["plan"]
    Writes: metadata["compiler_state"]["completed_tasks"]
            metadata["compiler_state"]["trace"]

    Runs independent tasks in parallel via ThreadPoolExecutor.
    Each wave contains all tasks whose deps are already complete.
    Resolves $T references before each task executes.
    """
    metadata = dict(state.get("metadata", {}))
    compiler_state = dict(metadata.get("compiler_state", {}))
    tasks = compiler_state.get("plan", [])
    results: Dict[str, Any] = dict(compiler_state.get("completed_tasks", {}))
    trace: List[Dict[str, Any]] = list(compiler_state.get("trace", []))

    # Tasks not yet executed
    remaining = {t["id"]: t for t in tasks if t["id"] not in results}
    wave = 0

    while remaining:
        wave += 1

        # Tasks whose dependencies are all satisfied
        ready = [
            t for t in remaining.values()
            if set(t.get("deps", [])).issubset(results.keys())
        ]

        if not ready:
            logger.error(
                "Nora scheduler: DAG deadlock. Remaining: %s", list(remaining.keys())
            )
            break

        def run_task(task: Dict[str, Any]) -> tuple:
            """Execute one task in the thread pool and return result tuple."""
            t_start = time.perf_counter()
            try:
                resolved_args = resolve_compile_refs(task.get("args", {}), results)
                result = _execute_tool(task["tool"], resolved_args, state)
                elapsed = time.perf_counter() - t_start
                logger.info(
                    "Nora scheduler wave=%d task=%s tool=%s %.2fs",
                    wave, task["id"], task["tool"], elapsed,
                )
                return task["id"], result, elapsed, None
            except Exception as e:
                elapsed = time.perf_counter() - t_start
                logger.error(
                    "Nora scheduler task=%s tool=%s FAILED: %s",
                    task["id"], task["tool"], e,
                )
                return task["id"], {"status": "error", "error": str(e)}, elapsed, str(e)

        # Run the current wave in parallel (cap at 4 workers)
        with ThreadPoolExecutor(max_workers=min(len(ready), 4)) as pool:
            futures = [pool.submit(run_task, t) for t in ready]
            for future in as_completed(futures):
                task_id, result, elapsed, error = future.result()
                results[task_id] = result
                trace.append({
                    "wave": wave,
                    "task_id": task_id,
                    "tool": remaining[task_id]["tool"],
                    "deps": remaining[task_id].get("deps", []),
                    "seconds": round(elapsed, 3),
                    "error": error,
                })
                del remaining[task_id]

        # Merge metadata side-effects from this wave into metadata (main thread, no race).
        for task_result in results.values():
            if not isinstance(task_result, dict):
                continue
            for key in _METADATA_SIDE_EFFECT_KEYS:
                if key in task_result:
                    metadata[key] = task_result[key]

    compiler_state["completed_tasks"] = results
    compiler_state["trace"] = trace
    metadata["compiler_state"] = compiler_state

    logger.info(
        "Nora scheduler complete: %d tasks in %d waves", len(results), wave
    )

    return {"metadata": metadata}


# ── Joiner Node ────────────────────────────────────────────────────────────────

def joiner_node(state: AgentState) -> dict:
    """
    LLMCompiler joiner — reviews all task results and decides the next action.

    Decisions:
      ANSWER   → generate final response to customer
      CLARIFY  → ask customer for more information, pause plan
      ESCALATE → build handoff, route to supervisor
      REPLAN   → something failed, create new plan (max MAX_REPLANS)

    Reads:  metadata["compiler_state"]["completed_tasks"]
    Writes: messages (appends AIMessage), resolution_status,
            requires_escalation, metadata
    """
    messages = state.get("messages", [])
    customer_id = state.get("customer_id", "unknown")
    session_id = state.get("session_id", "unknown")
    metadata = dict(state.get("metadata", {}))
    compiler_state = dict(metadata.get("compiler_state", {}))
    results = compiler_state.get("completed_tasks", {})
    replan_count = compiler_state.get("replan_count", 0)

    last_human = next(
        (m.content for m in reversed(messages) if getattr(m, "type", "") == "human"),
        "",
    )

    # If OTP was just verified this turn, use the original question
    # instead of the OTP code as the customer message for the joiner
    pending_intent = metadata.get("pending_intent", {})
    original_question = pending_intent.get("question", "")

    results_text = json.dumps(results, indent=2, ensure_ascii=False)

    # ── Fraud check — runs before everything else ─────────────────────────────
    # If fetch_order returned fraud_detected it means the customer tried
    # to access an order belonging to a different customer.
    # Stop immediately — never reveal the order exists, never say "fraud".
    # Escalate silently with a neutral "not found" message to the customer.
    fraud_detected = any(
        isinstance(v, dict) and v.get("status") == "fraud_detected"
        for v in results.values()
    )

    if fraud_detected:
        logger.warning(
            "Nora joiner: fraud signal detected for customer %s — escalating immediately",
            customer_id,
        )
        handoff = {
            "issue_summary": (
                "Possible fraud — customer attempted to access another customer's order"
            ),
            "actions_attempted": list(results.keys()),
            "customer_sentiment": "suspicious",
            "relevant_order_info": {},
            "escalation_reason": "ownership_violation",
            "confidence_score": 0.95,
            "agent": "nora_order_lookup",
        }
        metadata["escalation_handoff"] = handoff

        try:
            from memory.long_term import LongTermMemory
            LongTermMemory().save_interaction(
                customer_id=customer_id,
                session_id=session_id,
                summary=(
                    f"FRAUD SIGNAL: customer {customer_id} attempted "
                    f"to access another customer's order. "
                    f"Message: {last_human[:120]}"
                ),
                metadata={"intent": "fraud", "agent": "nora_order_lookup"},
            )
        except Exception as e:
            logger.warning("Could not save fraud signal to memory: %s", e)

        # Neutral "not found" reply — gives nothing away about the real owner
        return {
            "messages": [AIMessage(content=(
                "I'm sorry, I wasn't able to locate that order in your account. "
                "Please double-check the order ID. If you believe this is an error, "
                "contact us at support@shopease.com or call 19123."
            ))],
            "agent_used": "order_lookup",
            "resolution_status": "escalated",
            "requires_escalation": True,
            "retrieved_docs": [],
            "retrieval_scores": [],
            "metadata": metadata,
        }

    # ── Check for OTP status ───────────────────────────────────────────────────
    verify_result = next(
        (v for k, v in results.items()
         if isinstance(v, dict) and v.get("status") in
         ("verified", "already_verified", "failed", "otp_sent")),
        None,
    )

    # Use original question if OTP was just verified this turn
    just_verified = verify_result and verify_result.get("status") == "verified"
    display_question = (
        original_question
        if just_verified and original_question
        else last_human
    )

    # OTP just sent — save the original intent so the next turn can resume it,
    # then ask the customer to enter the code.
    if verify_result and verify_result.get("status") == "otp_sent":
        metadata["identity_verified"] = False
        metadata["pending_intent"] = {
            "order_id": state.get("order_id"),
            "question": last_human,
        }
        return {
            "messages": [AIMessage(content=(
                "For your security, I've sent a 6-digit verification code "
                "to your registered email address. "
                "Please enter it here to continue."
            ))],
            "agent_used": "order_lookup",
            "resolution_status": "pending_verification",
            "requires_escalation": False,
            "metadata": metadata,
            "retrieved_docs": [],
            "retrieval_scores": [],
        }

    # Wrong OTP — ask to try again
    if verify_result and verify_result.get("status") == "failed":
        return {
            "messages": [AIMessage(content=(
                "That code doesn't match what I have on file. "
                "Please double-check and try again, or let me know "
                "if you'd like me to resend a new code."
            ))],
            "agent_used": "order_lookup",
            "resolution_status": "pending_verification",
            "requires_escalation": False,
            "metadata": metadata,
            "retrieved_docs": [],
            "retrieval_scores": [],
        }

    # If customer's last message was just an OTP code (6 digits)
    # and identity was just verified — skip LLM joiner decision,
    # go straight to ANSWER using list_orders result
    just_otp = bool(re.fullmatch(r"\d{6}", last_human.strip())) or \
               bool(re.search(r"\d{6}", last_human.strip()) and len(last_human.strip()) <= 10)
    just_verified = verify_result and verify_result.get("status") == "verified"
    llm = _get_llm()

    if just_otp and just_verified:
        action = "ANSWER"
        decision = {"action": "ANSWER", "reason": "OTP verified — show order list"}
        logger.info("Nora joiner: skipping LLM — OTP just verified, going straight to ANSWER")

    else:
        # ── Ask LLM joiner to decide action ───────────────────────────────────
        joiner_prompt = f"""You are the joiner for Nora, ShopEase's Order Specialist.

You have just executed a plan. Review all task results and decide the next action.

CUSTOMER MESSAGE:
{display_question}

TASK RESULTS:
{results_text}

Decide one of these actions:
- ANSWER: You have enough information to respond to the customer fully.
- CLARIFY: You need more information from the customer to proceed.
  Specify exactly what question to ask.
- ESCALATE: The situation is outside Nora's authority.
  Specify the exact reason.
- REPLAN: A critical task failed and the plan needs to be adjusted.
  Only use if replan_count < {MAX_REPLANS}.
  Current replan count: {replan_count}

Respond in this exact JSON format with no extra text:
{{
  "action": "ANSWER or CLARIFY or ESCALATE or REPLAN",
  "reason": "brief explanation",
  "clarify_question": "question to ask customer if action is CLARIFY",
  "escalate_reason": "reason for escalation if action is ESCALATE",
  "replan_notes": "what to change in the new plan if action is REPLAN"
}}"""

        try:
            joiner_result = llm.invoke(joiner_prompt).content.strip()
            joiner_result = re.sub(r"```json|```", "", joiner_result).strip()
            decision = json.loads(joiner_result)
            action = decision.get("action", "ANSWER").upper()
            logger.info(
                "Nora joiner decision: %s reason: %s",
                action, decision.get("reason", ""),
            )
        except Exception as e:
            logger.warning("Nora joiner parsing failed, defaulting to ANSWER: %s", e)
            action = "ANSWER"
            decision = {"action": "ANSWER", "reason": "joiner parse failed"}

    # ── CLARIFY ───────────────────────────────────────────────────────────────
    if action == "CLARIFY":
        clarify_q = decision.get("clarify_question", "Could you please provide more details?")
        compiler_state["pending_clarification"] = clarify_q
        compiler_state["awaiting_clarification"] = True
        metadata["compiler_state"] = compiler_state
        return {
            "messages": [AIMessage(content=clarify_q)],
            "agent_used": "order_lookup",
            "resolution_status": "pending_clarification",
            "requires_escalation": False,
            "metadata": metadata,
            "retrieved_docs": [],
            "retrieval_scores": [],
        }

    # ── ESCALATE ──────────────────────────────────────────────────────────────
    if action == "ESCALATE":
        escalate_reason = decision.get("escalate_reason", "Situation outside Nora's authority")
        handoff = {
            "issue_summary": last_human[:200],
            "actions_attempted": list(results.keys()),
            "customer_sentiment": metadata.get("detected_emotion", "neutral"),
            "relevant_order_info": next(
                (v.get("raw", {}) for v in results.values()
                 if isinstance(v, dict) and "raw" in v),
                {},
            ),
            "escalation_reason": escalate_reason,
            "confidence_score": 0.3,
            "agent": "nora_order_lookup",
        }
        metadata["escalation_handoff"] = handoff

        try:
            from memory.long_term import LongTermMemory
            LongTermMemory().save_interaction(
                customer_id=customer_id,
                session_id=session_id,
                summary=(
                    f"Escalated from Nora: {escalate_reason}. "
                    f"Preview: {last_human[:120]}"
                ),
                metadata={"intent": "escalation", "agent": "nora_order_lookup"},
            )
        except Exception as e:
            logger.warning("Could not save escalation to memory: %s", e)

        return {
            "messages": [AIMessage(content=(
                "I want to make sure you get the best possible help. "
                "I'm connecting you with a senior specialist who can "
                "resolve this for you right away."
            ))],
            "agent_used": "order_lookup",
            "resolution_status": "escalated",
            "requires_escalation": True,
            "metadata": metadata,
            "retrieved_docs": [],
            "retrieval_scores": [],
        }

    # ── REPLAN ────────────────────────────────────────────────────────────────
    if action == "REPLAN" and replan_count < MAX_REPLANS:
        compiler_state["replan_count"] = replan_count + 1
        compiler_state["replan_notes"] = decision.get("replan_notes", "")
        metadata["compiler_state"] = compiler_state
        logger.info(
            "Nora joiner: REPLAN #%d — %s",
            replan_count + 1, decision.get("replan_notes", ""),
        )
        # Routing back to planner is handled by _route_after_joiner
        return {
            "agent_used": "order_lookup",
            "resolution_status": "replanning",
            "requires_escalation": False,
            "metadata": metadata,
            "retrieved_docs": [],
            "retrieval_scores": [],
        }

    # ── ANSWER ────────────────────────────────────────────────────────────────
    # Collect knowledge base context if retrieve_knowledge ran
    kb_result = next(
        (v for v in results.values() if isinstance(v, dict) and "context" in v),
        {},
    )
    kb_context = kb_result.get("context", "No relevant articles found.")
    retrieval_scores = kb_result.get("scores", [])

    # Check for return context stored by collect_return_context.
    # When present, re-route via supervisor → policy_returns (Maya handles the formal return).
    return_context = metadata.get("return_context")
    resolution_status = "needs_rerouting" if return_context else "resolved"

    # Build conversation history for the solver (all turns except the last)
    history_lines: List[str] = []
    for m in messages[:-1]:
        role = getattr(m, "type", "")
        if role == "human":
            history_lines.append(f"Customer: {m.content}")
        elif role == "ai":
            history_lines.append(f"Nora: {m.content}")
    history_text = "\n".join(history_lines) or "No prior conversation."

    solver_prompt = ChatPromptTemplate.from_messages([
        ("system", NORA_SYSTEM),
        ("human",
         "TASK RESULTS SUMMARY:\n{results}\n\n"
         "KNOWLEDGE BASE:\n{kb_context}\n\n"
         "PAST CONTEXT:\n{past_context}\n\n"
         "CONVERSATION HISTORY:\n{history}\n\n"
         "Customer's Question: {question}"),
    ])

    # Call llm.invoke() directly (not through a chain) so mock LLMs work in tests
    # and so we avoid LangChain's internal _generate() machinery which requires
    # a fully implemented BaseChatModel rather than a simple Runnable.
    try:
        formatted = solver_prompt.format_messages(
            results=results_text,
            kb_context=kb_context,
            past_context=metadata.get("past_context", "No prior interactions."),
            history=history_text,
            question=display_question,
        )
        llm_out = llm.invoke(formatted)
        response = llm_out.content if hasattr(llm_out, "content") else str(llm_out)
    except Exception as e:
        logger.error("Nora joiner solver failed: %s", e)
        response = (
            "I'm having trouble retrieving your details right now. "
            "Please try again or contact us at support@shopease.com or call 19123."
        )

    logger.info(
        "Nora joiner ANSWER | customer=%s status=%s", customer_id, resolution_status
    )

    return {
        "messages": [AIMessage(content=response)],
        "agent_used": "order_lookup",
        "resolution_status": resolution_status,
        "requires_escalation": False,
        "retrieved_docs": [{"context": kb_context}],
        "retrieval_scores": retrieval_scores,
        "metadata": metadata,
    }


# ── Subgraph + Public Entry Point ──────────────────────────────────────────────

def _route_after_joiner(state: AgentState) -> str:
    """Route joiner output: back to planner on REPLAN, otherwise END."""
    resolution = state.get("resolution_status", "resolved")
    metadata = state.get("metadata", {})
    replan_count = metadata.get("compiler_state", {}).get("replan_count", 0)

    if resolution == "replanning" and replan_count <= MAX_REPLANS:
        return "planner"
    return END


def build_order_lookup_subgraph():
    """
    Build the LLMCompiler subgraph for Nora.

    Nodes: planner → scheduler → joiner
    Joiner can route back to planner for replanning (max MAX_REPLANS times).
    """
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("scheduler", scheduler_node)
    builder.add_node("joiner", joiner_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "scheduler")
    builder.add_edge("scheduler", "joiner")
    builder.add_conditional_edges(
        "joiner",
        _route_after_joiner,
        {"planner": "planner", END: END},
    )

    return builder.compile()


# Singleton subgraph — compiled once per process
_subgraph = None


def _get_subgraph():
    """Return the singleton compiled subgraph, building it on first call."""
    global _subgraph
    if _subgraph is None:
        _subgraph = build_order_lookup_subgraph()
    return _subgraph


def order_lookup_node(state: AgentState) -> dict:
    """
    Public entry point called by the main LangGraph graph.
    Runs planner → scheduler → joiner in a loop, replanning up to MAX_REPLANS times.

    Uses a manual loop rather than subgraph.invoke() to avoid double-accumulation
    of messages: subgraph.invoke() with AgentState would apply the add_messages
    reducer inside the subgraph AND again in the main graph, duplicating every
    AIMessage. The manual loop returns only joiner_out["messages"] (the new
    AIMessage alone), which the main graph's add_messages reducer appends once.

    Signature is unchanged from the previous sequential agent.
    """
    current = dict(state)   # local copy — nodes mutate metadata in-place via state ref
    joiner_out: dict = {}

    for _ in range(MAX_REPLANS + 2):
        current.update(planner_node(current))
        current.update(scheduler_node(current))
        joiner_out = joiner_node(current)
        current.update(joiner_out)
        if current.get("resolution_status") != "replanning":
            break
    resolution = current.get("resolution_status", "resolved")
    if resolution == "resolved":
        try:
            from memory.long_term import LongTermMemory
            last_ai = next(
                (m.content for m in reversed(joiner_out.get("messages", []))
                 if hasattr(m, "type") and m.type == "ai"),
                "",
            )
            LongTermMemory().save_interaction(
                customer_id=current.get("customer_id", "unknown"),
                session_id=current.get("session_id", "unknown"),
                summary=last_ai[:200] if last_ai else "Order lookup resolved.",
                metadata={"intent": "order_lookup", "agent": "nora_order_lookup"},
            )
        except Exception as e:
            logger.warning("Could not save order lookup to long-term memory: %s", e)

    return {
        "messages": joiner_out.get("messages", []),
        "agent_used": current.get("agent_used", "order_lookup"),
        "resolution_status": current.get("resolution_status", "resolved"),
        "requires_escalation": current.get("requires_escalation", False),
        "retrieved_docs": current.get("retrieved_docs", []),
        "retrieval_scores": current.get("retrieval_scores", []),
        "metadata": current.get("metadata", {}),
    }
