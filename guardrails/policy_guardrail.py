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


_MANAGER_REVIEW_THRESHOLD = 1_000.0

_TIER_RANK: dict[str, int] = {"": 0, "supervisor": 1, "manager": 2}


def _parse_date(s: str) -> Optional[date]:
    """Parse ISO date/datetime string → date. Returns None on failure."""
    try:
        return datetime.fromisoformat(s).date()
    except (ValueError, TypeError):
        return None


@dataclass
class PolicyCheckResult:
    compliant: bool
    requires_escalation: bool
    violation: str          # human-readable explanation of what failed
    rule_id: str            # machine-readable rule ID for logging
    escalation_tier: str = ""   # "" | "supervisor" | "manager"
    restocking_fee_pct: int = 0  # 15 for late returns approved by exception


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
        Rule POL-001: Two-tier refund escalation.
        ≤ $500            → auto-approved (no escalation)
        $500.01 – $1,000  → supervisor approval (1 business day)
        > $1,000          → manager review (2–3 business days)
        """
        if requested_amount is None:
            return PolicyCheckResult(
                compliant=True,
                requires_escalation=False,
                violation="",
                rule_id="POL-001",
                escalation_tier="",
            )

        auto_limit = settings.max_refund_amount  # 500.0

        if requested_amount <= 0:
            return PolicyCheckResult(
                compliant=False,
                requires_escalation=False,
                violation=f"Refund amount must be positive. Received: ${requested_amount:.2f}",
                rule_id="POL-001",
                escalation_tier="",
            )

        if requested_amount <= auto_limit:
            return PolicyCheckResult(
                compliant=True,
                requires_escalation=False,
                violation="",
                rule_id="POL-001",
                escalation_tier="",
            )

        if requested_amount <= _MANAGER_REVIEW_THRESHOLD:
            logger.warning(
                "Policy violation POL-001 (supervisor tier): Requested refund $%.2f",
                requested_amount,
            )
            return PolicyCheckResult(
                compliant=False,
                requires_escalation=True,
                violation=(
                    f"The requested refund (${requested_amount:.2f}) requires "
                    "supervisor approval (resolved within 1 business day)."
                ),
                rule_id="POL-001",
                escalation_tier="supervisor",
            )

        logger.warning(
            "Policy violation POL-001 (manager tier): Requested refund $%.2f",
            requested_amount,
        )
        return PolicyCheckResult(
            compliant=False,
            requires_escalation=True,
            violation=(
                f"The requested refund (${requested_amount:.2f}) requires "
                "manager review (resolved within 2–3 business days)."
            ),
            rule_id="POL-001",
            escalation_tier="manager",
        )

    def check_return_window(
        self,
        delivered_at: Optional[str],
        created_at: Optional[str] = None,
    ) -> PolicyCheckResult:
        """
        Rule POL-002: Return window.
        Standard: returns within RETURN_WINDOW_DAYS of delivery.
        Holiday exception: items purchased Nov 1 – Dec 31 may be returned until Jan 31 of the following year.
        """
        window = settings.return_window_days
        today = date.today()

        ref_date_str = delivered_at or created_at
        if not ref_date_str:
            return PolicyCheckResult(
                compliant=True,
                requires_escalation=False,
                violation="",
                rule_id="POL-002",
            )

        def _parse(s: str) -> Optional[date]:
            try:
                return datetime.fromisoformat(s).date()
            except ValueError:
                try:
                    return date.fromisoformat(s[:10])
                except ValueError:
                    return None

        ref_date = _parse(ref_date_str)
        if ref_date is None:
            logger.warning("Cannot parse date: %s", ref_date_str)
            return PolicyCheckResult(
                compliant=True, requires_escalation=False, violation="", rule_id="POL-002"
            )

        # Holiday window: Nov/Dec purchases → deadline Jan 31 of the following year.
        # Use created_at as the purchase date; fall back to ref_date if not provided.
        order_date = _parse(created_at) if created_at else ref_date
        if order_date and order_date.month in (11, 12):
            # Avoid calling the (potentially mocked) date() constructor directly.
            holiday_deadline = order_date.replace(year=order_date.year + 1, month=1, day=31)
            if today <= holiday_deadline:
                days_remaining = (holiday_deadline - today).days
                return PolicyCheckResult(
                    compliant=True,
                    requires_escalation=False,
                    violation=f"Within holiday return window (deadline {holiday_deadline}, {days_remaining} day(s) remaining).",
                    rule_id="POL-002",
                )
            logger.info("Policy violation POL-002: Holiday return window expired on %s", holiday_deadline)
            return PolicyCheckResult(
                compliant=False,
                requires_escalation=True,
                violation=f"Holiday return window expired on {holiday_deadline}.",
                rule_id="POL-002",
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
                restocking_fee_pct=15,
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


_DAMAGE_REPORT_WINDOW_HOURS = 48
_DROPOFF_WINDOW_DAYS = 7


def policy_guardrail_check(
    refund_amount: Optional[float] = None,
    delivered_at: Optional[str] = None,
    created_at: Optional[str] = None,
    product_category: str = "",
    product_name: str = "",
    damage_reported_at: Optional[str] = None,
    return_initiated_at: Optional[str] = None,
) -> dict:
    """
    Run all applicable policy checks and return a consolidated result dict.
    """
    results = []

    if refund_amount is not None:
        results.append(_guardrail.check_refund_amount(refund_amount))

    # Undelivered order: block return if no delivery date but order date is known.
    # Note: check_return_window() itself still falls back to created_at when called
    # directly — this stricter check only applies via this convenience function.
    if delivered_at is None and created_at is not None:
        results.append(PolicyCheckResult(
            compliant=False,
            requires_escalation=False,
            violation="Returns can only be initiated after the order has been delivered.",
            rule_id="POL-002",
        ))
    elif delivered_at or created_at:
        results.append(_guardrail.check_return_window(delivered_at, created_at))

    if product_name:
        results.append(_guardrail.check_item_returnable(product_category, product_name))

    violations = [r.violation for r in results if not r.compliant]
    escalations = any(r.requires_escalation for r in results)
    compliant = all(r.compliant for r in results)
    rule_ids = [r.rule_id for r in results if not r.compliant]
    tier = max(
        (r.escalation_tier for r in results),
        key=lambda t: _TIER_RANK.get(t, 0),
        default="",
    )

    # Late damage report: defective item must be reported within 48 hours of delivery.
    late_damage_report = False
    if damage_reported_at and delivered_at:
        delivery = _parse_date(delivered_at)
        reported = _parse_date(damage_reported_at)
        if delivery and reported:
            hours_gap = (reported - delivery).total_seconds() / 3600
            if hours_gap > _DAMAGE_REPORT_WINDOW_HOURS:
                late_damage_report = True
                logger.info(
                    "Policy note: damage reported %.1f hours after delivery (limit %d h)",
                    hours_gap, _DAMAGE_REPORT_WINDOW_HOURS,
                )

    # Drop-off window: customer has 7 days after initiating return to drop off package.
    dropoff_window_expired = False
    if return_initiated_at:
        initiated = _parse_date(return_initiated_at)
        if initiated:
            days_since_initiation = (date.today() - initiated).days
            if days_since_initiation > _DROPOFF_WINDOW_DAYS:
                dropoff_window_expired = True
                logger.info(
                    "Policy note: drop-off window expired (%d days since return initiated)",
                    days_since_initiation,
                )

    return {
        "policy_compliant": compliant,
        "requires_escalation": escalations,
        "policy_violations": violations,
        "policy_rule_ids": rule_ids,
        "escalation_tier": tier,
        "late_damage_report": late_damage_report,
        "dropoff_window_expired": dropoff_window_expired,
    }
