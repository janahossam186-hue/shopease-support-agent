"""
scripts/generate_orders.py

Expands data/mock_orders.json from 8 to 200 orders.
  - ORD-10001–ORD-10008: preserved, addresses updated to Egyptian locations.
  - ORD-10009–ORD-10200: 192 freshly generated orders.

Run:
    .venv\\Scripts\\python.exe scripts/generate_orders.py
"""

import json
import random
from datetime import timedelta
from pathlib import Path

try:
    from faker import Faker
except ImportError:
    raise SystemExit(
        "Faker not installed.\n"
        "Run: .venv\\Scripts\\python.exe -m pip install faker"
    )

random.seed(42)
fake = Faker("en_US")
Faker.seed(42)

# ── Products (mirrored from data/product_catalog.json) ───────────────────────

PRODUCTS = [
    {"product_id": "PROD-001", "name": "ProBook Laptop 15",         "price": 899.99},
    {"product_id": "PROD-002", "name": "UltraTab Pro 11",           "price": 549.99},
    {"product_id": "PROD-003", "name": "SoundWave Headphones",      "price": 199.99},
    {"product_id": "PROD-004", "name": "EcoBrew Coffee Maker",      "price": 129.99},
    {"product_id": "PROD-005", "name": "FitTrack Smart Watch",      "price": 249.99},
    {"product_id": "PROD-006", "name": "ErgoDesk Chair",            "price": 399.99},
    {"product_id": "PROD-007", "name": "CleanAir Purifier",         "price": 279.99},
    {"product_id": "PROD-008", "name": "CookMaster Instant Pot",    "price": 89.99},
    {"product_id": "PROD-009", "name": "SpeedRun Gaming Mouse",     "price": 69.99},
    {"product_id": "PROD-010", "name": "SmartHome Hub",             "price": 149.99},
    {"product_id": "PROD-011", "name": "Canvas Pro Drawing Tablet", "price": 329.99},
    {"product_id": "PROD-012", "name": "ThermoGuard Water Bottle",  "price": 34.99},
]

# ── Egyptian addresses ────────────────────────────────────────────────────────

EG_ADDRESSES = [
    "15 Road 9, Maadi, Cairo",
    "47 El-Thawra Street, Heliopolis, Cairo",
    "23 Hassan Sabry Street, Zamalek, Cairo",
    "8 Street 90, New Cairo, Cairo",
    "32 Gamal Abd El-Nasser Street, 6th of October, Giza",
    "19 El-Nasr Street, Nasr City, Cairo",
    "55 El-Tahrir Square, Dokki, Giza",
    "12 El-Sudan Street, Mohandessin, Giza",
    "78 El-Haram Street, Giza",
    "3 El-Corniche, Maadi, Cairo",
    "101 Mostafa El-Nahas Street, Nasr City, Cairo",
    "67 El-Merghany Street, Heliopolis, Cairo",
    "29 El-Batal Ahmed Abdel Aziz Street, Mohandessin, Giza",
    "14 El-Obour Buildings, Nasr City, Cairo",
    "5 El-Sheikh Zayed, 6th of October, Giza",
    "88 Lebanon Street, Mohandessin, Giza",
    "22 Corniche El-Nil, Maadi, Cairo",
    "37 El-Geish Street, Heliopolis, Cairo",
    "9 Hassan Mohamed Street, Zamalek, Cairo",
    "63 South Academy, New Cairo, Cairo",
]

# Egyptian addresses assigned to the existing 8 orders
EXISTING_ADDR_MAP = {
    "ORD-10001": "15 Road 9, Maadi, Cairo",
    "ORD-10002": "15 Road 9, Maadi, Cairo",
    "ORD-10003": "47 El-Thawra Street, Heliopolis, Cairo",
    "ORD-10004": "23 Hassan Sabry Street, Zamalek, Cairo",
    "ORD-10005": "47 El-Thawra Street, Heliopolis, Cairo",
    "ORD-10006": "32 Gamal Abd El-Nasser Street, 6th of October, Giza",
    "ORD-10007": "15 Road 9, Maadi, Cairo",
    "ORD-10008": "8 Street 90, New Cairo, Cairo",
}

CUSTOMERS = [f"CUST-{i:03d}" for i in range(1, 11)]

CARRIERS = ["Aramex", "DHL Egypt", "FedEx Egypt", "Egypt Post", "Bosta", "Mylerz"]

# ── Status pool: exact counts summing to 192 ─────────────────────────────────

STATUS_POOL = (
    ["delivered"]        * 77 +
    ["in_transit"]       * 38 +
    ["processing"]       * 29 +
    ["delayed"]          * 19 +
    ["cancelled"]        * 15 +
    ["return_initiated"] * 14
)
assert len(STATUS_POOL) == 192, f"Expected 192, got {len(STATUS_POOL)}"
random.shuffle(STATUS_POOL)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(dt) -> str | None:
    return dt.strftime("%Y-%m-%d") if dt else None


def _tracking() -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(20))


def _payment() -> str:
    kind = random.choice(["visa", "mastercard", "meeza", "fawry", "cod", "vodafone", "instapay"])
    n = random.randint(1000, 9999)
    if kind == "visa":
        return f"Visa ending in {n}"
    if kind == "mastercard":
        return f"Mastercard ending in {n}"
    if kind == "meeza":
        return f"Meeza Card ending in {n}"
    if kind == "fawry":
        return "Fawry"
    if kind == "cod":
        return "Cash on Delivery"
    if kind == "vodafone":
        return "Vodafone Cash"
    return "InstaPay"


def _items() -> tuple[list[dict], float, float, float]:
    count    = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
    selected = random.sample(PRODUCTS, k=count)
    items    = []
    for p in selected:
        qty = random.choices([1, 2], weights=[85, 15])[0]
        items.append({
            "product_id": p["product_id"],
            "name":       p["name"],
            "qty":        qty,
            "price":      round(p["price"] * qty, 2),
        })
    subtotal      = round(sum(i["price"] for i in items), 2)
    shipping_cost = 0.00 if subtotal >= 50 else 12.99
    total         = round(subtotal + shipping_cost, 2)
    return items, subtotal, shipping_cost, total


# ── Order builder ─────────────────────────────────────────────────────────────

def build_order(order_num: int, status: str) -> dict:
    from datetime import date as date_type

    order_id = f"ORD-{order_num}"
    created  = fake.date_between(start_date=date_type(2024, 1, 1), end_date=date_type(2025, 6, 1))
    items, subtotal, shipping_cost, total = _items()

    base = {
        "order_id":         order_id,
        "customer_id":      random.choice(CUSTOMERS),
        "status":           status,
        "items":            items,
        "subtotal":         subtotal,
        "shipping_cost":    shipping_cost,
        "total":            total,
        "payment_method":   _payment(),
        "shipping_address": random.choice(EG_ADDRESSES),
        "created_at":       _fmt(created),
    }

    carrier = random.choice(CARRIERS)

    if status == "processing":
        base.update({
            "shipped_at":         None,
            "delivered_at":       None,
            "estimated_delivery": _fmt(created + timedelta(days=random.randint(3, 7))),
            "tracking_number":    None,
            "carrier":            carrier,
            "notes":              "Order is being prepared for shipment.",
        })

    elif status == "in_transit":
        shipped = created + timedelta(days=1)
        base.update({
            "shipped_at":         _fmt(shipped),
            "delivered_at":       None,
            "estimated_delivery": _fmt(shipped + timedelta(days=random.randint(2, 5))),
            "tracking_number":    _tracking(),
            "carrier":            carrier,
            "notes":              "Package is on its way.",
        })

    elif status == "delivered":
        shipped   = created + timedelta(days=1)
        delivered = shipped + timedelta(days=random.randint(3, 7))
        base.update({
            "shipped_at":     _fmt(shipped),
            "delivered_at":   _fmt(delivered),
            "tracking_number": _tracking(),
            "carrier":        carrier,
            "notes":          "",
        })

    elif status == "delayed":
        shipped  = created + timedelta(days=1)
        old_eta  = shipped + timedelta(days=random.randint(3, 5))
        new_eta  = old_eta + timedelta(days=random.randint(3, 7))
        base.update({
            "shipped_at":         _fmt(shipped),
            "delivered_at":       None,
            "estimated_delivery": _fmt(old_eta),
            "tracking_number":    _tracking(),
            "carrier":            carrier,
            "notes":              f"Package delayed in transit. New estimated delivery: {_fmt(new_eta)}.",
        })

    elif status == "cancelled":
        reason = random.choice([
            "Customer request",
            "Out of stock after order placement",
            "Payment verification failed",
            "Duplicate order",
            "Customer changed mind",
        ])
        base.update({
            "shipped_at":          None,
            "delivered_at":        None,
            "cancelled_at":        _fmt(created),
            "cancellation_reason": reason,
            "tracking_number":     None,
            "carrier":             None,
            "notes":               "Full refund issued.",
        })

    elif status == "return_initiated":
        shipped   = created + timedelta(days=1)
        delivered = shipped + timedelta(days=random.randint(3, 7))
        return_dt = delivered + timedelta(days=random.randint(1, 20))
        base.update({
            "shipped_at":          _fmt(shipped),
            "delivered_at":        _fmt(delivered),
            "return_initiated_at": _fmt(return_dt),
            "return_reason":       random.choice([
                "Item not as described",
                "Defective product",
                "Wrong item received",
                "Changed mind",
                "Better price found elsewhere",
            ]),
            "tracking_number":  _tracking(),
            "carrier":          carrier,
            "notes":            "Return label sent. Awaiting receipt at warehouse.",
        })

    return base


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    data_path = Path(__file__).parent.parent / "data" / "mock_orders.json"

    existing: dict = json.loads(data_path.read_text(encoding="utf-8"))

    # Update addresses on the original 8 orders
    for oid, addr in EXISTING_ADDR_MAP.items():
        if oid in existing:
            existing[oid]["shipping_address"] = addr

    # Append the 192 new orders
    all_orders = dict(existing)
    for i, status in enumerate(STATUS_POOL):
        order = build_order(10009 + i, status)
        all_orders[order["order_id"]] = order

    data_path.write_text(
        json.dumps(all_orders, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Written {len(all_orders)} orders to {data_path}")


if __name__ == "__main__":
    main()
