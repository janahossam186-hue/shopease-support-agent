"""
ShopEase Customer Support — Gradio Chat Interface

Launch with:
    .\.venv\Scripts\python.exe app.py
    # or
    .\.venv\Scripts\python.exe -m gradio app.py
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from pathlib import Path

# ── Force UTF-8 on Windows ────────────────────────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler("./data/logs/agent.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
for mod in ["agents", "guardrails", "rag", "memory", "graph", "evaluation"]:
    logging.getLogger(mod).setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# ── Bootstrap ──────────────────────────────────────────────────────────────────

def bootstrap():
    """Initialise settings, ensure dirs, prime the graph."""
    from config.settings import settings
    settings.apply_langsmith_env()
    settings.ensure_dirs()

    from evaluation.metrics import init_db
    init_db()

    from graph.workflow import get_graph
    get_graph()
    print("✓ ShopEase agent ready.")

bootstrap()

import gradio as gr
from graph.workflow import get_graph, make_initial_state
from memory.short_term import get_session_config

# ── Agent label mapping ────────────────────────────────────────────────────────
AGENT_LABELS = {
    "order_lookup":   "📦 Order Agent",
    "policy_returns": "📋 Policy & Returns Agent",
    "escalation":     "🎫 Escalation Specialist",
    "general":        "💬 Layla (General Assistant)",
    "none":           "🤖 ShopEase Agent",
    "unknown":        "🤖 ShopEase Agent",
}

# ── Store info for sidebar ─────────────────────────────────────────────────────
SIDEBAR_HTML = """
<div style="font-family: Arial, sans-serif; padding: 12px;">

  <div style="text-align:center; margin-bottom: 18px;">
    <div style="font-size:2.2em; font-weight:800; color:#1a56db; letter-spacing:-1px;">
      🛍️ ShopEase
    </div>
    <div style="color:#6b7280; font-size:0.85em; margin-top:2px;">
      Your Smart Shopping Destination — Egypt
    </div>
  </div>

  <hr style="border-color:#e5e7eb; margin:12px 0;" />

  <div style="margin-bottom:14px;">
    <div style="font-weight:700; color:#374151; margin-bottom:6px;">📞 Contact Us</div>
    <div style="font-size:0.88em; color:#4b5563; line-height:1.7;">
      ☎️ Hotline: <strong>19123</strong><br/>
      💬 WhatsApp: +20 100 123 4567<br/>
      📧 support@shopease.eg<br/>
      🕐 Sat–Thu: 9AM–10PM | Fri: 12PM–8PM
    </div>
  </div>

  <hr style="border-color:#e5e7eb; margin:12px 0;" />

  <div style="margin-bottom:14px;">
    <div style="font-weight:700; color:#374151; margin-bottom:6px;">🏪 Store Locations</div>
    <div style="font-size:0.85em; color:#4b5563; line-height:1.8;">
      📍 Cairo Festival City Mall<br/>
      📍 Maadi — Degla Square<br/>
      📍 Heliopolis — City Stars<br/>
      📍 Giza — Dandy Mall<br/>
      📍 Alexandria — San Stefano
    </div>
  </div>

  <hr style="border-color:#e5e7eb; margin:12px 0;" />

  <div style="margin-bottom:14px;">
    <div style="font-weight:700; color:#374151; margin-bottom:6px;">🎉 Current Promotions</div>
    <div style="font-size:0.85em; color:#4b5563; line-height:1.8;">
      ☀️ <strong>Summer Glow Sale</strong> — 20% off SPF & skincare<br/>
      &nbsp;&nbsp;&nbsp;<code style="background:#f3f4f6;padding:1px 4px;border-radius:3px;">SUMMER20</code><br/>
      🎓 <strong>Student Deal</strong> — 15% off tech<br/>
      &nbsp;&nbsp;&nbsp;<code style="background:#f3f4f6;padding:1px 4px;border-radius:3px;">STUDENT15</code><br/>
      🎁 <strong>New Customer</strong> — 10% off + free shipping<br/>
      &nbsp;&nbsp;&nbsp;<code style="background:#f3f4f6;padding:1px 4px;border-radius:3px;">WELCOME10</code>
    </div>
  </div>

  <hr style="border-color:#e5e7eb; margin:12px 0;" />

  <div style="margin-bottom:14px;">
    <div style="font-weight:700; color:#374151; margin-bottom:6px;">🔥 Trending Now</div>
    <div style="font-size:0.85em; color:#4b5563; line-height:1.8;">
      1. HydraGlow Vitamin C Serum<br/>
      2. FitTrack Smart Watch<br/>
      3. VelvetMatte Foundation<br/>
      4. EcoBrew Coffee Maker<br/>
      5. SoundWave Headphones
    </div>
  </div>

  <hr style="border-color:#e5e7eb; margin:12px 0;" />

  <div style="font-size:0.78em; color:#9ca3af; text-align:center; line-height:1.5;">
    🚀 Same-day delivery in Cairo & Giza<br/>
    (order before 2 PM)<br/>
    Free shipping on orders over EGP 500
  </div>

</div>
"""

# ── Core chat function ─────────────────────────────────────────────────────────

def chat(
    user_message: str,
    history: list[dict],
    customer_id: str,
    session_id: str,
) -> tuple[list[dict], str, str]:
    """
    Process one chat turn.

    Returns:
        updated_history  — the full chat history (gradio messages format)
        agent_label      — which agent responded
        session_id       — (possibly initialised on first turn)
    """
    if not user_message.strip():
        return history, "—", session_id

    # Initialise session ID on first message
    if not session_id:
        session_id = f"session_{customer_id.strip() or 'guest'}_{uuid.uuid4().hex[:8]}"

    cid = customer_id.strip() or "CUST-GUEST"
    graph = get_graph()
    config = get_session_config(session_id)

    state = make_initial_state(
        customer_id=cid,
        session_id=session_id,
        user_message=user_message,
    )

    try:
        result = graph.invoke(state, config=config)
    except Exception as e:
        logger.error("Graph invocation error: %s", e, exc_info=True)
        ai_msg = "I'm sorry, I ran into a small technical issue. Please try again or call us on 19123! 😊"
        agent_label = AGENT_LABELS["none"]
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ai_msg},
        ]
        return history, agent_label, session_id

    # Extract AI reply
    ai_reply = next(
        (m.content for m in reversed(result.get("messages", []))
         if hasattr(m, "type") and m.type == "ai"),
        "I'm here to help! Could you please rephrase your question? 😊",
    )

    agent_used = result.get("agent_used", "none")
    intent = result.get("intent", "?")
    resolution = result.get("resolution_status", "?")
    latency_ms = result.get("latency_ms", 0.0)

    agent_label = AGENT_LABELS.get(agent_used, f"🤖 {agent_used}")

    # Build the agent info footer
    ticket_id = result.get("escalation_ticket_id")
    footer_parts = [f"Intent: `{intent}`", f"Status: `{resolution}`"]
    if ticket_id:
        footer_parts.append(f"Ticket: `{ticket_id}`")
    if not result.get("guardrail_passed", True):
        footer_parts.append("⚠️ Guardrail triggered")

    footer = "  \n*" + " · ".join(footer_parts) + f" · {latency_ms:.0f}ms*"

    # Append to history
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ai_reply + footer},
    ]

    return history, agent_label, session_id


def reset_session(customer_id: str) -> tuple[list, str, str, str]:
    """Clear chat history and start a fresh session."""
    new_session_id = f"session_{customer_id.strip() or 'guest'}_{uuid.uuid4().hex[:8]}"
    return [], "—", new_session_id, ""


# ── Build Gradio UI ────────────────────────────────────────────────────────────

_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
)

_APP_CSS = """
    #shopease-header {
        background: linear-gradient(135deg, #1a56db 0%, #3b82f6 100%);
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 4px;
        color: white;
    }
    #agent-badge {
        font-size: 0.9em;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        display: inline-block;
    }
    .message-wrap { max-height: 520px; overflow-y: auto; }
    footer { display: none !important; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="ShopEase Customer Support",
    ) as demo:

        # ── State ─────────────────────────────────────────────────────────────
        session_state = gr.State(value="")

        # ── Header ────────────────────────────────────────────────────────────
        with gr.Row(elem_id="shopease-header"):
            gr.HTML("""
                <div style="display:flex; align-items:center; gap:16px; color:white;">
                    <div style="font-size:2.2em;">🛍️</div>
                    <div>
                        <div style="font-size:1.4em; font-weight:800; margin:0;">ShopEase Egypt</div>
                        <div style="font-size:0.85em; opacity:0.85;">
                            AI-Powered Customer Support · Available 24/7
                        </div>
                    </div>
                </div>
            """)

        # ── Main layout ───────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left sidebar
            with gr.Column(scale=1, min_width=230):
                gr.HTML(SIDEBAR_HTML)

            # Chat area
            with gr.Column(scale=3):

                # Customer ID row
                with gr.Row():
                    customer_id_input = gr.Textbox(
                        value="CUST-001",
                        label="Customer ID",
                        placeholder="e.g. CUST-001",
                        scale=2,
                        max_lines=1,
                        info="Change to switch customer context",
                    )
                    agent_badge = gr.Textbox(
                        value="—",
                        label="Last Responded By",
                        scale=2,
                        interactive=False,
                        elem_id="agent-badge",
                    )
                    reset_btn = gr.Button(
                        "🔄 New Session",
                        scale=1,
                        variant="secondary",
                        size="sm",
                    )

                # Chat window
                chatbot = gr.Chatbot(
                    label="Chat with ShopEase Support",
                    height=480,
                    avatar_images=(
                        None,  # user avatar (None = default)
                        "https://em-content.zobj.net/source/twitter/348/shopping-bags_1f6cd-fe0f.png",
                    ),
                    placeholder=(
                        "<div style='text-align:center; color:#9ca3af; padding:40px 20px;'>"
                        "<div style='font-size:2em; margin-bottom:8px;'>👋</div>"
                        "<div style='font-size:1.1em; font-weight:600;'>Hi! I'm Layla from ShopEase.</div>"
                        "<div style='margin-top:8px;'>How can I help you today?</div>"
                        "<div style='margin-top:16px; font-size:0.85em;'>"
                        "Ask me about orders, product usage, beauty tips, store locations, or anything else!</div>"
                        "</div>"
                    ),
                )

                # Message input
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message here… (press Enter or click Send)",
                        label="Your Message",
                        scale=5,
                        max_lines=3,
                        lines=1,
                        show_label=False,
                        autofocus=True,
                    )
                    send_btn = gr.Button(
                        "Send →",
                        scale=1,
                        variant="primary",
                        size="lg",
                    )

                # Example messages
                gr.Examples(
                    examples=[
                        ["Hi! I'm looking for a good skincare routine for oily skin 😊"],
                        ["Where is my order ORD-10002?"],
                        ["How do I use the EcoBrew Coffee Maker?"],
                        ["What are your store locations in Cairo?"],
                        ["I want to return my FitTrack Smart Watch"],
                        ["What are the trending products this week?"],
                        ["My CookMaster Instant Pot shows a 'Burn' warning — help!"],
                        ["Do you have any current promotions or promo codes?"],
                    ],
                    inputs=msg_input,
                    label="💡 Quick Questions",
                )

        # ── Event wiring ──────────────────────────────────────────────────────

        def submit_message(user_msg, history, cid, sid):
            if not user_msg.strip():
                return history, "—", sid, ""
            new_history, agent_lbl, new_sid = chat(user_msg, history, cid, sid)
            return new_history, agent_lbl, new_sid, ""   # clear input box

        # Send on button click
        send_btn.click(
            fn=submit_message,
            inputs=[msg_input, chatbot, customer_id_input, session_state],
            outputs=[chatbot, agent_badge, session_state, msg_input],
        )

        # Send on Enter key
        msg_input.submit(
            fn=submit_message,
            inputs=[msg_input, chatbot, customer_id_input, session_state],
            outputs=[chatbot, agent_badge, session_state, msg_input],
        )

        # Reset session
        reset_btn.click(
            fn=reset_session,
            inputs=[customer_id_input],
            outputs=[chatbot, agent_badge, session_state, msg_input],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=_APP_THEME,
        css=_APP_CSS,
    )
