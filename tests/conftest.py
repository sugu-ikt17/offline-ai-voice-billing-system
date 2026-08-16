"""Shared pytest fixtures.

Uses an isolated in-memory SQLite database per test session, so the
test suite never touches the real database/voice_billing.db file and
tests don't leak state into each other via the filesystem.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.database.database import Base, get_db
from app.main import app
# Ensure all models are registered on Base.metadata before create_all().
from app.infrastructure.database.models import bill_model, menu_item_model, order_model  # noqa: F401

TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,  # keep one shared connection so :memory: data persists across requests
)
TestSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def _override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def client():
    """A TestClient wired to a fresh in-memory database for each test."""
    Base.metadata.create_all(bind=engine)

    app.dependency_overrides[get_db] = _override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
