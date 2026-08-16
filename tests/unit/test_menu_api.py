"""Menu CRUD API tests — covers happy paths and the documented business rules."""

MENU_URL = "/api/v1/menu"


def test_create_menu_item(client):
    response = client.post(MENU_URL, json={"name": "Masala Dosa", "price": 45.0})

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Masala Dosa"
    assert body["price"] == 45.0
    assert "id" in body
    assert "created_at" in body
    assert "updated_at" in body


def test_create_menu_item_rejects_empty_name(client):
    response = client.post(MENU_URL, json={"name": "", "price": 10.0})
    assert response.status_code == 422  # Pydantic schema validation (min_length=1)


def test_create_menu_item_rejects_non_positive_price(client):
    response = client.post(MENU_URL, json={"name": "Tea", "price": 0})
    assert response.status_code == 422  # Pydantic schema validation (gt=0)


def test_create_menu_item_rejects_duplicate_name(client):
    client.post(MENU_URL, json={"name": "Idli", "price": 20.0})
    response = client.post(MENU_URL, json={"name": "Idli", "price": 25.0})

    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_create_menu_item_rejects_duplicate_name_case_insensitive(client):
    client.post(MENU_URL, json={"name": "Idli", "price": 20.0})
    response = client.post(MENU_URL, json={"name": "idli", "price": 25.0})

    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_list_menu_items(client):
    client.post(MENU_URL, json={"name": "Idli", "price": 20.0})
    client.post(MENU_URL, json={"name": "Vada", "price": 15.0})

    response = client.get(MENU_URL)

    assert response.status_code == 200
    names = {item["name"] for item in response.json()}
    assert names == {"Idli", "Vada"}


def test_get_menu_item_by_id(client):
    created = client.post(MENU_URL, json={"name": "Filter Coffee", "price": 15.0}).json()

    response = client.get(f"{MENU_URL}/{created['id']}")

    assert response.status_code == 200
    assert response.json()["name"] == "Filter Coffee"


def test_get_menu_item_not_found(client):
    response = client.get(f"{MENU_URL}/9999")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_update_menu_item_price(client):
    created = client.post(MENU_URL, json={"name": "Uttapam", "price": 30.0}).json()

    response = client.put(f"{MENU_URL}/{created['id']}", json={"price": 35.0})

    assert response.status_code == 200
    body = response.json()
    assert body["price"] == 35.0
    assert body["name"] == "Uttapam"  # unchanged
    assert body["updated_at"] != created["updated_at"]


def test_update_menu_item_not_found(client):
    response = client.put(f"{MENU_URL}/9999", json={"price": 10.0})
    assert response.status_code == 404


def test_update_menu_item_rejects_duplicate_name(client):
    client.post(MENU_URL, json={"name": "Poori", "price": 25.0})
    second = client.post(MENU_URL, json={"name": "Chapati", "price": 20.0}).json()

    response = client.put(f"{MENU_URL}/{second['id']}", json={"name": "Poori"})

    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_update_menu_item_allows_renaming_to_same_name(client):
    """Renaming an item to its own current name (no real change) must not
    be treated as a false-positive duplicate."""
    created = client.post(MENU_URL, json={"name": "Sambar Vada", "price": 20.0}).json()

    response = client.put(f"{MENU_URL}/{created['id']}", json={"name": "Sambar Vada"})

    assert response.status_code == 200
    assert response.json()["name"] == "Sambar Vada"


def test_delete_menu_item(client):
    created = client.post(MENU_URL, json={"name": "Rava Kesari", "price": 25.0}).json()

    response = client.delete(f"{MENU_URL}/{created['id']}")
    assert response.status_code == 204

    follow_up = client.get(f"{MENU_URL}/{created['id']}")
    assert follow_up.status_code == 404


def test_delete_menu_item_not_found(client):
    response = client.delete(f"{MENU_URL}/9999")
    assert response.status_code == 404
