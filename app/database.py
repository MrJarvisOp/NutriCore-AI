"""
database.py
-----------
SQLAlchemy ORM models + async session factory.
Stores request history so you can audit / replay plans.
"""
from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import DATABASE_URL
from app.utils.logger import get_logger

log = get_logger(__name__)

# Convert sync SQLite URL → async if needed
_async_url = (
    DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")
    if DATABASE_URL.startswith("sqlite")
    else DATABASE_URL
)

engine        = create_async_engine(_async_url, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# ─── ORM models ──────────────────────────────────────────────────────────────

class HealthPlanRecord(Base):
    """Persists each generated health plan for auditing."""
    __tablename__ = "health_plans"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    created_at     = Column(DateTime, default=datetime.utcnow, nullable=False)

    # User inputs
    age            = Column(Integer,  nullable=False)
    weight_kg      = Column(Float,    nullable=False)
    height_cm      = Column(Float,    nullable=False)
    goal           = Column(String(32), nullable=False)
    gender         = Column(String(16), nullable=True)
    activity_level = Column(String(32), nullable=True)

    # Health metrics (JSON blob)
    health_metrics = Column(Text, nullable=True)

    # Full response (JSON blob)
    diet_plan      = Column(Text, nullable=True)
    exercise_plan  = Column(Text, nullable=True)
    warnings       = Column(Text, nullable=True)

    # Medical flags (JSON list)
    medical_flags  = Column(Text, nullable=True)

    # Validation
    validation_passed = Column(Integer, default=1)   # 1 = True, 0 = False
    retry_count       = Column(Integer, default=0)

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "created_at":      self.created_at.isoformat() if self.created_at else None,
            "age":             self.age,
            "weight_kg":       self.weight_kg,
            "height_cm":       self.height_cm,
            "goal":            self.goal,
            "health_metrics":  json.loads(self.health_metrics or "{}"),
            "medical_flags":   json.loads(self.medical_flags  or "[]"),
            "validation_passed": bool(self.validation_passed),
        }


# ─── DB helpers ──────────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create all tables (idempotent)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database tables initialised")


async def save_plan(record: HealthPlanRecord) -> HealthPlanRecord:
    async with AsyncSessionLocal() as session:
        session.add(record)
        await session.commit()
        await session.refresh(record)
        log.info(f"Saved health plan record id={record.id}")
        return record


async def get_session() -> AsyncSession:  # type: ignore[return]
    """FastAPI dependency - yield an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session
