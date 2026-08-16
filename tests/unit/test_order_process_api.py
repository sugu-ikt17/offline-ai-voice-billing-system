"""Unit tests for POST /api/v1/orders/process endpoint.

Verifies the full speech → parse → match → bill pipeline
via the HTTP layer using the test client's in-memory DB.
"""


def test_process_order_returns_bill_structure(client):
    """Endpoint must return a complete BillResult.to_dict() shaped response."""
    response = client.post(
        "/api/v1/orders/process",
        json={"speech": "2 dosa 1 tea"},
    )

    assert response.status_code == 200
    data = response.json()

    # Top-level keys
    assert "bill"            in data
    assert "warnings"        in data
    assert "unmatched_items" in data

    bill = data["bill"]
    for key in ("bill_number", "date_time", "items",
                "item_count", "total_quantity",
                "subtotal", "discount", "tax", "grand_total"):
        assert key in bill, f"Missing bill key: {key}"

    # Bill number format
    assert bill["bill_number"].startswith("BILL-")


def test_process_order_empty_speech_returns_zero_bill(client):
    """Empty speech string returns zeroed bill with no items."""
    response = client.post(
        "/api/v1/orders/process",
        json={"speech": ""},
    )

    assert response.status_code == 200
    data = response.json()

    bill = data["bill"]
    assert bill["items"]          == []
    assert bill["item_count"]     == 0
    assert bill["total_quantity"] == 0
    assert bill["grand_total"]    == 0.0
    assert data["warnings"]        == []
    assert data["unmatched_items"] == []
