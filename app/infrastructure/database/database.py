"""
Database configuration.

Sets up the SQLAlchemy engine, session factory, and declarative base
used across the application. No models, repositories, or business
logic belong in this file.
"""

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Database file location: <project_root>/database/voice_billing.db
BASE_DIR = Path(__file__).resolve().parents[3]
DB_DIR = BASE_DIR / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_DIR / 'voice_billing.db'}"

# check_same_thread=False is required for SQLite when the same connection
# may be accessed by different threads (as FastAPI does with its workers).
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session and ensures
    it is closed after the request completes.

    BUG-09 FIX: Return type was annotated as `Session` but the function
    uses `yield`, making it a Generator. Correct annotation added.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
