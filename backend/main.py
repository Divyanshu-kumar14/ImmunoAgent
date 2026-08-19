"""ImmunoAgent FastAPI entry point (Phase 1–3).

Creates the ``app`` instance, mounts the gateway router, and runs start‑up
hooks that bootstrap Postgres extensions, create the schema (the four
core tables from PRD §4.1), initialise the shared Redis client, seed the
agent registry, and start the WebSocket telemetry consumer.
"""
from __future__ import annotations

import asyncio

import redis.asyncio as redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection

from backend.app.config import get_settings
from backend.database.connection import init_db_engine, create_extensions
from backend.database.models import AgentRegistry, Base

from .core.gateway import router as gateway_router
from .api.router import router as api_router
from .api.websocket import router as ws_router, start_telemetry_consumer

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

# Store the telemetry consumer task for shutdown
_telemetry_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# Startup: initialise DB engine + extensions + create tables + Redis + seed
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    """One‑time initialisation when the worker process starts."""
    global _telemetry_task

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

    # Seed the agent registry with demo agents (idempotent)
    await _seed_agent_registry(engine)

    # Store the engine on the app state so routers can access it if needed.
    # Single engine instance reused for the process lifetime — creating a new
    # engine per request would re-pool connections and wreck the fast path.
    app.state.db_engine = engine

    # Initialise shared Redis client (single pool, reused across requests)
    app.state.redis_client = redis.from_url(
        settings.redis_url,
        decode_responses=True,
        max_connections=20,
    )

    # Verify Redis connectivity
    await app.state.redis_client.ping()

    # Start WebSocket telemetry consumer (background task)
    _telemetry_task = await start_telemetry_consumer(app.state.redis_client)


# ---------------------------------------------------------------------------
# Shutdown: close Redis pool + cancel consumer
# ---------------------------------------------------------------------------
@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Clean up connections on shutdown."""
    global _telemetry_task
    if _telemetry_task:
        _telemetry_task.cancel()
        try:
            await _telemetry_task
        except asyncio.CancelledError:
            pass
    if hasattr(app.state, "redis_client"):
        await app.state.redis_client.close()
    if hasattr(app.state, "db_engine"):
        await app.state.db_engine.dispose()


# ---------------------------------------------------------------------------
# Seed agent registry (idempotent)
# ---------------------------------------------------------------------------
async def _seed_agent_registry(engine) -> None:
    """Insert the three demo agents if they don't exist."""
    agents = [
        {
            "agent_id": "ingest_agent",
            "name": "Document Ingest Agent",
            "department": "operations",
            "allowed_tools": ["read_file", "view_policy"],
            "max_action_tier": "INTERNAL_WRITE",
            "is_active": True,
        },
        {
            "agent_id": "payable_agent",
            "name": "Accounts Payable Agent",
            "department": "finance",
            "allowed_tools": [
                "read_file",
                "view_policy",
                "update_policy",
                "execute_wire_transfer",
                "fetch_vendor_invoice",
            ],
            "max_action_tier": "CRITICAL_EXEC",
            "is_active": True,
        },
        {
            "agent_id": "admin_agent",
            "name": "Admin Policy Agent",
            "department": "security",
            "allowed_tools": [
                "read_file",
                "view_policy",
                "update_policy",
                "execute_wire_transfer",
                "fetch_vendor_invoice",
            ],
            "max_action_tier": "CRITICAL_EXEC",
            "is_active": True,
        },
    ]

    async with engine.begin() as conn:
        for agent_data in agents:
            # Check if agent exists
            res = await conn.execute(
                select(AgentRegistry).where(AgentRegistry.agent_id == agent_data["agent_id"])
            )
            if res.scalar_one_or_none() is None:
                await conn.execute(
                    AgentRegistry.__table__.insert().values(**agent_data)
                )


# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------
app.include_router(gateway_router)
app.include_router(api_router)
app.include_router(ws_router)


# ---------------------------------------------------------------------------
# Simple healthcheck
# ---------------------------------------------------------------------------
@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict:
    return {"status": "ok"}