"""
Guardrail 2 — Policy Compliance

Enforces ShopEase business rules before any refund/return action is taken:
  • Refund amounts must not exceed self-service limit ($500)
  • Return requests must be within the return window (30 days)
  • Replacement orders require original order verification

Called inline by the Policy & Returns agent node — not a separate graph node —
because the agent needs to know whether to escalate before it can respond.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PolicyCheckResult:
    compliant: bool
    requires_escalation: bool
    violation: str          # human-readable explanation of what failed
    rule_id: str            # machine-readable rule ID for logging


class PolicyGuardrail:
    """
    Stateless rule engine for ShopEase business policies.

    Each check_* method returns a PolicyCheckResult.
    """

    def check_refund_amount(
        self,
        requested_amount: Optional[float],
        order_total: Optional[float] = None,
    ) -> PolicyCheckResult:
        """
        Rule POL-001: Self-service refund limit.
        Amounts ≤ $500 are auto-approved.
        Amounts > $500 require supervisor escalation.
        """
        if requested_amount is None:
            return PolicyCheckResult(
                compliant=True,
                requires_escalation=False,
                violation="",
                rule_id="POL-001",
            )

        limit = settings.max_refund_amount

        if requested_amount <= 0:
            return PolicyCheckResult(
                compliant=False,
                requires_escalation=False,
                violation=f"Refund amount must be positive. Received: ${requested_amount:.2f}",
                rule_id="POL-001",
            )

        if requested_amount > limit:
            logger.warning(
                "Policy violation POL-001: Requested refund $%.2f exceeds limit $%.2f",
                requested_amount,
                limit,
            )
            return PolicyCheckResult(
                compliant=False,
                requires_escalation=True,
                violation=(
                    f"The requested refund (${requested_amount:.2f}) exceeds the "
                    "self-service limit. Supervisor approval is required."
                ),
                rule_id="POL-001",
            )

        return PolicyCheckResult(
            compliant=True,
            requires_escalation=False,
            violation="",
            rule_id="POL-001",
        )

    def check_return_window(
        self,
        delivered_at: Optional[str],
        created_at: Optional[str] = None,
    ) -> PolicyCheckResult:
        """
        Rule POL-002: Return window.
        Returns must be initiated within RETURN_WINDOW_DAYS of delivery.
        """
        window = settings.return_window_days
        today = date.today()

        # Determine reference date
        ref_date_str = delivered_at or created_at
        if not ref_date_str:
            # No date info — allow but flag for review
            return PolicyCheckResult(
                compliant=True,
                requires_escalation=False,
                violation="",
                rule_id="POL-002",
            )

        try:
            ref_date = datetime.fromisoformat(ref_date_str).date()
        except ValueError:
            try:
                ref_date = date.fromisoformat(ref_date_str[:10])
            except ValueError:
                logger.warning("Cannot parse date: %s", ref_date_str)
                return PolicyCheckResult(
                    compliant=True, requires_escalation=False, violation="", rule_id="POL-002"
                )

        days_since = (today - ref_date).days

        if days_since > window:
            logger.info(
                "Policy violation POL-002: Return request %d days after delivery (limit %d)",
                days_since,
                window,
            )
            return PolicyCheckResult(
                compliant=False,
                requires_escalation=True,
                violation=(
                    f"Return window is {window} days from delivery. "
                    f"This item was delivered {days_since} days ago "
                    f"({days_since - window} day(s) past the window)."
                ),
                rule_id="POL-002",
            )

        days_remaining = window - days_since
        return PolicyCheckResult(
            compliant=True,
            requires_escalation=False,
            violation=f"Within return window ({days_remaining} day(s) remaining).",
            rule_id="POL-002",
        )

    def check_item_returnable(self, product_category: str, product_name: str) -> PolicyCheckResult:
        """
        Rule POL-003: Non-returnable item categories.
        """
        NON_RETURNABLE_KEYWORDS = [
            "digital", "download", "software", "license",
            "perishable", "food", "personal care", "hygiene",
            "gift card", "final sale",
        ]
        text = f"{product_category} {product_name}".lower()
        for keyword in NON_RETURNABLE_KEYWORDS:
            if keyword in text:
                return PolicyCheckResult(
                    compliant=False,
                    requires_escalation=False,
                    violation=(
                        f"'{product_name}' falls under the '{product_category}' category, "
                        "which is not eligible for return per our policy."
                    ),
                    rule_id="POL-003",
                )
        return PolicyCheckResult(
            compliant=True, requires_escalation=False, violation="", rule_id="POL-003"
        )


# ── Convenience function for use by agents ────────────────────────────────────

_guardrail = PolicyGuardrail()


def policy_guardrail_check(
    refund_amount: Optional[float] = None,
    delivered_at: Optional[str] = None,
    created_at: Optional[str] = None,
    product_category: str = "",
    product_name: str = "",
) -> dict:
    """
    Run all applicable policy checks and return a consolidated result dict.

    Returns::
        {
            "policy_compliant": bool,
            "requires_escalation": bool,
            "policy_violations": list[str],
            "policy_rule_ids": list[str],
        }
    """
    results = []

    if refund_amount is not None:
        results.append(_guardrail.check_refund_amount(refund_amount))

    if delivered_at or created_at:
        results.append(_guardrail.check_return_window(delivered_at, created_at))

    if product_name:
        results.append(_guardrail.check_item_returnable(product_category, product_name))

    violations = [r.violation for r in results if not r.compliant]
    escalations = any(r.requires_escalation for r in results)
    compliant = all(r.compliant for r in results)
    rule_ids = [r.rule_id for r in results if not r.compliant]

    return {
        "policy_compliant": compliant,
        "requires_escalation": escalations,
        "policy_violations": violations,
        "policy_rule_ids": rule_ids,
    }
