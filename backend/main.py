"""ImmunoAgent FastAPI entry point (Phase 1).

Creates the ``app`` instance, mounts the gateway router, and runs start‑up
hooks that bootstrap Postgres extensions and create the schema (the four
core tables from PRD §4.1).

Typical usage:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio

from sqlalchemy import event

from backend.app.config import get_settings
from backend.database.connection import init_db_engine, create_extensions
from backend.database.models import Base

from .core.gateway import router as gateway_router

settings = get_settings()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
from fastapi import FastAPI

app = FastAPI(
    title="ImmunoAgent",
    version="0.1.0",
    description="Agent Gateway → Epistemic Provenance DAG → Sentinel quarantine",
)


# ---------------------------------------------------------------------------
# Startup: initialise DB engine + extensions + create tables
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    """One‑time initialisation when the worker process starts."""
    engine = init_db_engine(settings.database_url)

    # Bootstrap required Postgres extensions (vector, uuid-ossp) idempotently.
    # Extensions MUST exist before create_all() or the Vector(768) column DDL
    # fails — see connection.py.
    await create_extensions(engine)

    # Create all SQLAlchemy tables (idempotent — no data loss on restart).
    # Hackathon choice: create_all() is one idempotent call; Alembic would
    # add migration machinery we don't need for a 12‑day demo. In production
    # you would migrate via Alembic.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Store the engine on the app state so routers can access it if needed.
    # Single engine instance reused for the process lifetime — creating a new
    # engine per request would re-pool connections and wreck the fast path.
    app.state.db_engine = engine


# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------
app.include_router(gateway_router)

# ---------------------------------------------------------------------------
# Simple healthcheck
# ---------------------------------------------------------------------------
@app.get("/healthz", include_in_schema=False)
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}