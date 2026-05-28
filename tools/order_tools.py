"""
LangChain tools for order lookup and modification, backed by the mock order
database.  In production these would call a real OMS/ERP API.

Read tools  (get_order_tool, list_customer_orders_tool) — unchanged from v1.
Write tools (cancel_order_tool, update_address_tool, update_quantity_tool,
             remove_item_tool) — new; all verify ownership + processing status.
Plain helpers (send_otp_email, get_customer_info) — not LangChain tools.
"""

from __future__ import annotations

import json
import logging
import random
import smtplib
import string
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Module-level singletons ───────────────────────────────────────────────────

_orders: dict | None = None
_customers: dict | None = None

_ORDERS_PATH = Path("./data/mock_orders.json")
_CUSTOMERS_PATH = Path("./data/customers.json")

# Read from settings so the credentials come from .env
_GMAIL_ADDRESS = settings.gmail_address
_GMAIL_PASSWORD = settings.gmail_app_password


# ── Private loaders ───────────────────────────────────────────────────────────

def _load_orders() -> dict:
    """Return the in-memory order dict, loading from disk on first call."""
    global _orders
    if _orders is None:
        _orders = json.loads(_ORDERS_PATH.read_text(encoding="utf-8"))
    return _orders


def _save_orders(orders: dict) -> None:
    """Write the orders dict back to disk and reset the cache so the next
    read reflects the change."""
    global _orders
    _ORDERS_PATH.write_text(
        json.dumps(orders, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _orders = None  # force reload on next call


def _load_customers() -> dict:
    """Return the in-memory customer dict, loading from disk on first call."""
    global _customers
    if _customers is None:
        _customers = json.loads(_CUSTOMERS_PATH.read_text(encoding="utf-8"))
    return _customers


# ── Shared ownership check ────────────────────────────────────────────────────

def _verify_ownership(order: dict, customer_id: str, order_id: str) -> Optional[str]:
    """Return an error string if the order doesn't belong to customer_id,
    or None if ownership is confirmed."""
    if order["customer_id"] != customer_id:
        logger.warning(
            "Ownership check failed: customer %s tried to access order %s "
            "owned by %s",
            customer_id, order_id, order["customer_id"],
        )
        return "You are not authorised to access this order."
    return None


def _verify_processing(order: dict, order_id: str) -> Optional[str]:
    """Return an error string if the order is not in 'processing' status,
    or None if the modification is allowed."""
    status = order["status"]
    if status != "processing":
        return (
            f"Order {order_id} cannot be modified — its current status is "
            f"'{status.replace('_', ' ').title()}'. "
            "Only orders that are still processing can be changed."
        )
    return None


# ── Formatting helper (shared with read tools) ────────────────────────────────

def _format_order(order: dict) -> str:
    """Convert an order dict to a human-readable string."""
    lines = [
        f"Order ID: {order['order_id']}",
        f"Status: {order['status'].replace('_', ' ').title()}",
        f"Created: {order['created_at']}",
    ]

    items_str = ", ".join(
        f"{item['name']} x{item['qty']} (${item['price']:.2f})"
        for item in order["items"]
    )
    lines.append(f"Items: {items_str}")
    lines.append(f"Total: ${order['total']:.2f}")
    lines.append(f"Payment: {order['payment_method']}")
    lines.append(f"Ship to: {order['shipping_address']}")

    if order.get("tracking_number"):
        lines.append(f"Tracking: {order['tracking_number']} via {order['carrier']}")
    if order.get("shipped_at"):
        lines.append(f"Shipped: {order['shipped_at']}")
    if order.get("delivered_at"):
        lines.append(f"Delivered: {order['delivered_at']}")
    if order.get("estimated_delivery"):
        lines.append(f"Estimated delivery: {order['estimated_delivery']}")
    if order.get("notes"):
        lines.append(f"Notes: {order['notes']}")
    if order.get("return_initiated_at"):
        lines.append(f"Return initiated: {order['return_initiated_at']}")
        lines.append(f"Return reason: {order.get('return_reason', 'Not specified')}")

    return "\n".join(lines)


# ── Read tools (unchanged) ────────────────────────────────────────────────────

@tool
def get_order_tool(order_id: str) -> str:
    """
    Look up a specific order by its order ID (e.g., 'ORD-10001').
    Returns order status, items, tracking, and delivery information.
    """
    orders = _load_orders()
    order_id_upper = order_id.strip().upper()

    order = orders.get(order_id_upper)
    if order is None:
        # Try partial match for common variations
        for key in orders:
            if key.endswith(order_id_upper) or order_id_upper in key:
                order = orders[key]
                break

    if order is None:
        return (
            f"Order '{order_id}' not found. Please verify the order ID. "
            "Order IDs are in the format ORD-XXXXX (e.g., ORD-10001)."
        )

    return _format_order(order)


@tool
def list_customer_orders_tool(customer_id: str) -> str:
    """
    List all orders for a given customer ID. Returns a summary of each order.
    Use this when the customer asks about 'my orders' without specifying an order ID.
    """
    orders = _load_orders()
    customer_orders = [
        o for o in orders.values() if o["customer_id"] == customer_id
    ]

    if not customer_orders:
        return f"No orders found for customer ID '{customer_id}'."

    lines = [f"Found {len(customer_orders)} order(s) for customer {customer_id}:\n"]
    for order in sorted(customer_orders, key=lambda o: o["created_at"], reverse=True):
        status = order["status"].replace("_", " ").title()
        items_str = ", ".join(item["name"] for item in order["items"])
        lines.append(
            f"• {order['order_id']} — {status} — ${order['total']:.2f} — {items_str}"
        )

    return "\n".join(lines)


# ── Write tools ───────────────────────────────────────────────────────────────

@tool
def cancel_order_tool(order_id: str, customer_id: str) -> str:
    """
    Cancel an order on behalf of the customer.

    Only succeeds when:
    - The order exists and belongs to customer_id.
    - The order status is 'processing' (not yet shipped).

    Sets status to 'cancelled' and records a cancelled_at timestamp.
    Returns a confirmation message or a descriptive error.
    """
    orders = _load_orders()
    oid = order_id.strip().upper()
    order = orders.get(oid)

    if order is None:
        return f"Order '{order_id}' not found."

    if err := _verify_ownership(order, customer_id, oid):
        return err
    if err := _verify_processing(order, oid):
        return err

    orders[oid]["status"] = "cancelled"
    orders[oid]["cancelled_at"] = datetime.utcnow().strftime("%Y-%m-%d")
    orders[oid]["cancellation_reason"] = "Customer request"
    _save_orders(orders)

    logger.info("Order %s cancelled for customer %s", oid, customer_id)
    return (
        f"Order {oid} has been successfully cancelled. "
        "Any charges will be refunded within 3–5 business days."
    )


@tool
def update_address_tool(order_id: str, customer_id: str, new_address: str) -> str:
    """
    Update the shipping address on an order.

    Only succeeds when:
    - The order exists and belongs to customer_id.
    - The order status is 'processing' (not yet shipped).

    Returns a confirmation showing both old and new addresses, or an error.
    """
    orders = _load_orders()
    oid = order_id.strip().upper()
    order = orders.get(oid)

    if order is None:
        return f"Order '{order_id}' not found."

    if err := _verify_ownership(order, customer_id, oid):
        return err
    if err := _verify_processing(order, oid):
        return err

    old_address = orders[oid]["shipping_address"]
    orders[oid]["shipping_address"] = new_address.strip()
    _save_orders(orders)

    logger.info("Address updated for order %s (customer %s)", oid, customer_id)
    return (
        f"Shipping address updated successfully.\n"
        f"Previous: {old_address}\n"
        f"New: {new_address.strip()}"
    )


@tool
def update_quantity_tool(
    order_id: str, customer_id: str, product_id: str, new_qty: int
) -> str:
    """
    Update the quantity of a specific item in an order.

    Only succeeds when:
    - The order exists and belongs to customer_id.
    - The order status is 'processing'.
    - The product_id exists in the order.

    If new_qty is 0 the item is removed entirely.
    Recalculates subtotal and total after the change.
    Returns a confirmation or an error.
    """
    orders = _load_orders()
    oid = order_id.strip().upper()
    order = orders.get(oid)

    if order is None:
        return f"Order '{order_id}' not found."

    if err := _verify_ownership(order, customer_id, oid):
        return err
    if err := _verify_processing(order, oid):
        return err

    items = order["items"]
    item = next((i for i in items if i["product_id"] == product_id), None)
    if item is None:
        return f"Product '{product_id}' not found in order {oid}."

    if new_qty <= 0:
        # Delegate to remove logic
        return _remove_item_from_order(orders, oid, product_id)

    # Price in mock data is line total; derive unit price to recalculate.
    old_qty = item["qty"]
    unit_price = item["price"] / old_qty if old_qty else item["price"]
    item["qty"] = new_qty
    item["price"] = round(unit_price * new_qty, 2)

    subtotal = round(sum(i["price"] for i in items), 2)
    order["subtotal"] = subtotal
    order["total"] = round(subtotal + order.get("shipping_cost", 0.0), 2)
    _save_orders(orders)

    logger.info(
        "Quantity for %s in order %s changed %d→%d (customer %s)",
        product_id, oid, old_qty, new_qty, customer_id,
    )
    return (
        f"{item['name']} quantity updated from {old_qty} to {new_qty}.\n"
        f"New order total: ${order['total']:.2f}"
    )


@tool
def remove_item_tool(order_id: str, customer_id: str, product_id: str) -> str:
    """
    Remove a specific item from an order.

    Only succeeds when:
    - The order exists and belongs to customer_id.
    - The order status is 'processing'.
    - The product_id exists in the order.

    If the removed item is the last one, the entire order is cancelled.
    Recalculates subtotal and total.
    Returns a confirmation or an error.
    """
    orders = _load_orders()
    oid = order_id.strip().upper()
    order = orders.get(oid)

    if order is None:
        return f"Order '{order_id}' not found."

    if err := _verify_ownership(order, customer_id, oid):
        return err
    if err := _verify_processing(order, oid):
        return err

    return _remove_item_from_order(orders, oid, product_id)


def _remove_item_from_order(orders: dict, oid: str, product_id: str) -> str:
    """Core item-removal logic shared by remove_item_tool and update_quantity_tool.

    Operates on the already-loaded orders dict.  Saves back to disk and resets
    the singleton cache.
    """
    order = orders[oid]
    items = order["items"]
    item = next((i for i in items if i["product_id"] == product_id), None)

    if item is None:
        return f"Product '{product_id}' not found in order {oid}."

    items.remove(item)
    logger.info("Item %s removed from order %s", product_id, oid)

    if not items:
        # Last item gone — cancel the order automatically
        order["status"] = "cancelled"
        order["cancelled_at"] = datetime.utcnow().strftime("%Y-%m-%d")
        order["cancellation_reason"] = "All items removed by customer"
        _save_orders(orders)
        return (
            f"{item['name']} was the only item in order {oid}. "
            "The order has been cancelled automatically. "
            "Any charges will be refunded within 3–5 business days."
        )

    subtotal = round(sum(i["price"] for i in items), 2)
    order["subtotal"] = subtotal
    order["total"] = round(subtotal + order.get("shipping_cost", 0.0), 2)
    _save_orders(orders)

    return (
        f"{item['name']} has been removed from order {oid}.\n"
        f"Updated order total: ${order['total']:.2f}"
    )


# ── Email OTP sender ──────────────────────────────────────────────────────────

def send_otp_email(customer_id: str) -> tuple[str, str]:
    """
    Generate a 6-digit OTP and send it to the customer's registered email.

    Returns:
        (otp_code, masked_email) where masked_email is e.g. "a***@example.com".

    Raises:
        ValueError: customer_id not found in customers.json.
        RuntimeError: SMTP send failed.
    """
    customers = _load_customers()
    customer = customers.get(customer_id)
    if not customer:
        raise ValueError(f"Customer {customer_id} not found in customer database.")

    otp = str(random.randint(100000, 999999))
    email: str = customer["email"]
    name: str = customer["name"]

    # Build masked version for internal logging only — never shown to customer
    local, domain = email.split("@", 1)
    masked = local[0] + "***@" + domain

    body = (
        f"Hi {name},\n\n"
        f"Your ShopEase verification code is: {otp}\n\n"
        f"This code is valid for this session only.\n"
        f"If you did not request this, please ignore this email.\n\n"
        f"— ShopEase Support Team"
    )
    msg = MIMEText(body)
    msg["Subject"] = "ShopEase — Your Verification Code"
    msg["From"] = _GMAIL_ADDRESS
    msg["To"] = email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(_GMAIL_ADDRESS, _GMAIL_PASSWORD)
            server.send_message(msg)
    except Exception as exc:
        logger.error("Failed to send OTP email to customer %s: %s", customer_id, exc)
        raise RuntimeError(f"OTP email could not be delivered: {exc}") from exc

    logger.info("OTP sent to customer %s (%s)", customer_id, masked)
    return otp, masked


# ── Customer info helper ──────────────────────────────────────────────────────

def get_customer_info(customer_id: str) -> Optional[dict]:
    """Return the customer dict (name, email, phone) or None if not found."""
    return _load_customers().get(customer_id)
