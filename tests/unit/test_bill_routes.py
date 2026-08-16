"""Integration tests for bill generation endpoint.

Tests the full POST /bills/{order_id} pipeline end-to-end through the HTTP
layer using the in-memory test database.

Architecture note: POST /orders/process is a *stateless* pipeline — it
returns a bill dict but does NOT persist orders to the database. To test
the /bills routes (which operate on persisted orders), we insert orders
directly via the test DB session.

BUG-01/08 regression: verifies that GenerateBillUseCase correctly:
  - builds a MatchResult from saved order items
  - passes it to BillGeneratorService
  - maps BillResult → BillModel using real field names
  - marks the order as 'billed'
  - prevents double-billing
"""

import pytest
from sqlalchemy.orm import Session

from tests.conftest import TestSessionLocal
from app.infrastructure.database.models.menu_item_model import MenuItemModel
from app.infrastructure.database.models.order_model import OrderItemModel, OrderModel

MENU_URL = "/api/v1/menu"
ORDERS_URL = "/api/v1/orders"
BILLS_URL = "/api/v1/bills"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def db_session(client) -> Session:
    """Yield a fresh session on the test DB (already set up by client fixture)."""
    db = TestSessionLocal()
    yield db
    db.close()


def _insert_order(db: Session, items: list[dict]) -> OrderModel:
    """Insert a pending order with the given items directly into the test DB.

    Each item dict: {"name": str, "menu_item_id": int, "unit_price": float, "quantity": int}
    """
    order = OrderModel(
        status="pending",
        raw_transcript="test",
        items=[
            OrderItemModel(
                menu_item_id=item["menu_item_id"],
                name=item["name"],
                quantity=item["quantity"],
                unit_price=item["unit_price"],
            )
            for item in items
        ],
    )
    db.add(order)
    db.commit()
    db.refresh(order)
    return order


def _insert_menu_item(db: Session, name: str, price: float) -> MenuItemModel:
    """Insert a menu item and return it."""
    m = MenuItemModel(name=name, price=price)
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


# ── List bills — empty ───────────────────────────────────────────────────────

def test_list_bills_empty(client):
    """GET /bills returns an empty list when no bills exist."""
    resp = client.get(BILLS_URL)
    assert resp.status_code == 200
    assert resp.json() == []


# ── Generate bill — happy path ───────────────────────────────────────────────

def test_generate_bill_creates_bill_model(client, db_session):
    """POST /bills/{order_id} must create and return a BillModel.

    BUG-01/08 regression: previously crashed because GenerateBillUseCase
    passed an Order entity (wrong type) to BillGeneratorService and then
    accessed non-existent fields on BillResult.
    """
    dosa = _insert_menu_item(db_session, "Dosa", 40.0)
    order = _insert_order(db_session, [
        {"name": "Dosa", "menu_item_id": dosa.id, "unit_price": 40.0, "quantity": 2},
    ])

    resp = client.post(f"{BILLS_URL}/{order.id}")
    assert resp.status_code == 201, resp.text

    bill = resp.json()
    assert bill["order_id"] == order.id
    assert "id" in bill
    assert "subtotal" in bill
    assert "tax_rate" in bill
    assert "tax_amount" in bill
    assert "total" in bill
    assert "generated_at" in bill


def test_generate_bill_subtotal_correct(client, db_session):
    """Subtotal = sum of (unit_price × quantity) across all items.
    2 Dosa (₹40) + 1 Tea (₹15) → subtotal = 95.0
    """
    dosa = _insert_menu_item(db_session, "Dosa", 40.0)
    tea  = _insert_menu_item(db_session, "Tea",  15.0)
    order = _insert_order(db_session, [
        {"name": "Dosa", "menu_item_id": dosa.id, "unit_price": 40.0, "quantity": 2},
        {"name": "Tea",  "menu_item_id": tea.id,  "unit_price": 15.0, "quantity": 1},
    ])

    bill = client.post(f"{BILLS_URL}/{order.id}").json()
    assert bill["subtotal"] == 95.0


def test_generate_bill_total_equals_subtotal_when_no_tax(client, db_session):
    """With default tax=0, total must equal subtotal."""
    coffee = _insert_menu_item(db_session, "Coffee", 20.0)
    order  = _insert_order(db_session, [
        {"name": "Coffee", "menu_item_id": coffee.id, "unit_price": 20.0, "quantity": 1},
    ])

    bill = client.post(f"{BILLS_URL}/{order.id}").json()
    assert bill["total"] == bill["subtotal"]
    assert bill["tax_amount"] == 0.0


def test_generate_bill_marks_order_as_billed(client, db_session):
    """After billing, the order status must change to 'billed'."""
    tea   = _insert_menu_item(db_session, "Tea", 15.0)
    order = _insert_order(db_session, [
        {"name": "Tea", "menu_item_id": tea.id, "unit_price": 15.0, "quantity": 1},
    ])

    client.post(f"{BILLS_URL}/{order.id}")

    order_resp = client.get(f"{ORDERS_URL}/{order.id}").json()
    assert order_resp["status"] == "billed"


def test_generate_bill_prevents_double_billing(client, db_session):
    """Billing an already-billed order must return HTTP 400.

    BUG-01/08 regression: previously crashed before reaching this guard.
    """
    dosa  = _insert_menu_item(db_session, "Dosa", 40.0)
    order = _insert_order(db_session, [
        {"name": "Dosa", "menu_item_id": dosa.id, "unit_price": 40.0, "quantity": 1},
    ])

    first = client.post(f"{BILLS_URL}/{order.id}")
    assert first.status_code == 201

    second = client.post(f"{BILLS_URL}/{order.id}")
    assert second.status_code == 400
    assert "already been billed" in second.json()["detail"]


def test_generate_bill_not_found(client):
    """Billing a non-existent order must return HTTP 404."""
    resp = client.post(f"{BILLS_URL}/99999")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ── Get bill by ID ──────────────────────────────────────────────────────────

def test_get_bill_by_id(client, db_session):
    """GET /bills/{bill_id} must return the bill that was created."""
    tea   = _insert_menu_item(db_session, "Tea", 15.0)
    order = _insert_order(db_session, [
        {"name": "Tea", "menu_item_id": tea.id, "unit_price": 15.0, "quantity": 1},
    ])

    created = client.post(f"{BILLS_URL}/{order.id}").json()
    bill_id = created["id"]

    resp = client.get(f"{BILLS_URL}/{bill_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == bill_id
    assert resp.json()["order_id"] == order.id


def test_get_bill_not_found(client):
    """GET /bills/{bill_id} for a non-existent id must return HTTP 404."""
    resp = client.get(f"{BILLS_URL}/99999")
    assert resp.status_code == 404


# ── List bills ──────────────────────────────────────────────────────────────

def test_list_bills_after_generation(client, db_session):
    """GET /bills must include all generated bills."""
    dosa = _insert_menu_item(db_session, "Dosa", 40.0)
    tea  = _insert_menu_item(db_session, "Tea",  15.0)

    order1 = _insert_order(db_session, [
        {"name": "Dosa", "menu_item_id": dosa.id, "unit_price": 40.0, "quantity": 1},
    ])
    order2 = _insert_order(db_session, [
        {"name": "Tea", "menu_item_id": tea.id, "unit_price": 15.0, "quantity": 2},
    ])

    client.post(f"{BILLS_URL}/{order1.id}")
    client.post(f"{BILLS_URL}/{order2.id}")

    bills = client.get(BILLS_URL).json()
    assert len(bills) == 2
    order_ids = {b["order_id"] for b in bills}
    assert order1.id in order_ids
    assert order2.id in order_ids
