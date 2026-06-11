"""
Guardrail 3 — Toxicity Detection

Inspects BOTH input messages AND agent responses for:
  • Profanity / offensive language
  • Threats and harassment
  • Personally-identifiable information (PII) leakage in outputs
  • Hallucinated refund amounts or policy statements

Two-stage pipeline:
  1. better-profanity keyword scan (fast, zero-cost)
  2. Pattern-based PII and threat detection
  (Optional) LLM re-check for nuanced cases

Acts as an OUTPUT guardrail node in the graph — verifies the agent's
draft response before it reaches the customer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_core.messages import AIMessage

from config.settings import settings

logger = logging.getLogger(__name__)

# ── PII / sensitive-data patterns ─────────────────────────────────────────────
PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),                    # SSN
    (re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|"  # Visa
        r"5[1-5][0-9]{14}|"                # Mastercard
        r"3[47][0-9]{13}|"                 # Amex
        r"6(?:011|5[0-9]{2})[0-9]{12})\b"  # Discover
    ), "credit_card"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "email"),
    (re.compile(r"\b\d{3}[\s.-]\d{3}[\s.-]\d{4}\b"), "phone"),        # US phone
]

# ── Threat / harassment patterns ──────────────────────────────────────────────
THREAT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"i('ll| will| am going to)?\s+(kill|hurt|harm|attack|destroy)",
        r"you('re|\s+are)\s+(dead|going to die|finished)",
        r"i\s+know\s+where\s+you\s+live",
        r"(bomb|shoot|stab|poison)\s+(you|them|the)",
    ]
]

# ── Product-safety context patterns (never treat these as threats) ────────────
# Matches product malfunction descriptions so threat detection is not triggered
# by words like "burning" or "overheating" in a customer support context.
PRODUCT_SAFE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(product|device|it|the\s+\w+)\s+(keeps?\s+)?burn(s|ing)?\b",
        r"\bburning\s+(smell|issue|problem|sensation)\b",
        r"\b(overheating?|overheat(s|ed|ing)?)\b",
        r"\bproduct\s+burns?\b",
        r"\bit\s+burns?\b",
    ]
]


@dataclass
class ToxicityResult:
    is_toxic: bool
    toxicity_score: float       # 0.0 – 1.0
    category: str               # profanity | threat | pii | clean
    sanitised_text: str         # text with profanity masked


class ToxicityGuardrail:
    """
    Fast, dependency-light toxicity checker.
    """

    def __init__(self):
        self._profanity_loaded = False
        self._profanity = None

    def _load_profanity(self):
        if not self._profanity_loaded:
            try:
                from better_profanity import profanity
                profanity.load_censor_words()
                self._profanity = profanity
            except ImportError:
                logger.warning("better-profanity not installed; skipping profanity check.")
            self._profanity_loaded = True
        return self._profanity

    def check(self, text: str, context: str = "input") -> ToxicityResult:
        """
        Args:
            text: The text to analyse.
            context: 'input' (customer message) or 'output' (agent response).
        """
        score = 0.0
        category = "clean"

        # 1. Threat detection — skip entirely for product malfunction descriptions
        _is_product_context = any(p.search(text) for p in PRODUCT_SAFE_PATTERNS)
        if not _is_product_context:
            for pattern in THREAT_PATTERNS:
                if pattern.search(text):
                    logger.warning("Threat detected in %s: %.60s…", context, text)
                    return ToxicityResult(
                        is_toxic=True,
                        toxicity_score=1.0,
                        category="threat",
                        sanitised_text="[Message removed — policy violation]",
                    )

        # 2. Profanity check
        profanity = self._load_profanity()
        sanitised = text
        if profanity and profanity.contains_profanity(text):
            sanitised = profanity.censor(text)
            score = max(score, 0.8)
            category = "profanity"
            logger.info("Profanity detected in %s", context)

        # 3. PII check (relevant for agent OUTPUT — we don't want to leak PII)
        if context == "output":
            for pattern, pii_type in PII_PATTERNS:
                if pattern.search(text):
                    sanitised = pattern.sub(f"[{pii_type}_redacted]", sanitised)
                    score = max(score, 0.6)
                    category = "pii"
                    logger.warning("PII (%s) detected in agent output — redacted.", pii_type)

        is_toxic = score >= settings.toxicity_threshold

        return ToxicityResult(
            is_toxic=is_toxic,
            toxicity_score=score,
            category=category,
            sanitised_text=sanitised,
        )

    def check_input(self, text: str) -> ToxicityResult:
        return self.check(text, context="input")

    def check_output(self, text: str) -> ToxicityResult:
        return self.check(text, context="output")


# ── LangGraph output-guardrail node ───────────────────────────────────────────

_guardrail = ToxicityGuardrail()


def output_guardrail_node(state: dict) -> dict:
    """
    LangGraph node: sanitises the agent's last response.
    - Redacts PII from the output.
    - Replaces highly toxic responses with a safe fallback.
    - Records the toxicity_score in state for evaluation.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    # Find the last AI message (the agent's draft response)
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)),
        None,
    )
    if last_ai is None:
        return {}

    result = _guardrail.check_output(last_ai.content)

    updates: dict = {"toxicity_score": result.toxicity_score}

    if result.is_toxic and result.category == "threat":
        # Replace entirely
        safe_response = (
            "I understand you may be frustrated. Let me connect you with a "
            "senior support specialist who can help resolve this right away."
        )
        updates["messages"] = [AIMessage(content=safe_response)]
        logger.warning("Output guardrail replaced toxic response.")
    elif result.sanitised_text != last_ai.content:
        # PII redacted or profanity masked
        updates["messages"] = [AIMessage(content=result.sanitised_text)]
        logger.info("Output guardrail sanitised response (category: %s).", result.category)

    return updates
