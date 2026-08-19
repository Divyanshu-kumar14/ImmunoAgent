"""pgvector Memory Bank with mandatory ``quarantine_status = 'ACTIVE'`` filter (Phase 2).

Contract (PRD §4.1):
  * Every semantic search must pre‑filter ``quarantine_status = 'ACTIVE'``.
  * EXCISED memories are never returned.
  * Excision is performed by updating ``quarantine_status`` to ``'EXCISED'``
    (not physical deletion) so the row stays auditable.

The module exposes three functions:
  * ``insert_memory(engine, memory)`` – insert a new memory row.
  * ``semantic_search(engine, query_embedding, agent_id, k=5)`` – retrieve the
    top‑k ACTIVE memories similar to the query vector.
  * ``excise_memories(engine, source_span_id)`` – mark all memories derived
    from a given span as EXCISED.
"""
from __future__ import annotations

from typing import Any, Dict, List
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncConnection

from backend.database.models import AgentMemoryBank


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

async def insert_memory(
    conn: AsyncConnection,
    *,
    agent_id: str,
    session_id: str,
    source_span_id: str,
    content_text: str,
    embedding: List[float],
    metadata: Dict[str, Any] | None = None,
) -> str:
    """Insert a new memory record and return the generated ``memory_id``.

    The ``quarantine_status`` defaults to ``'ACTIVE'`` per the contract.
    """
    mem_id = str(uuid4())
    stmt = (
        AgentMemoryBank.__table__.insert()
        .values(
            memory_id=mem_id,
            agent_id=agent_id,
            session_id=session_id,
            source_span_id=source_span_id,
            content_text=content_text,
            embedding=embedding,  # pgvector column expects a list/array
            quarantine_status="ACTIVE",
            metadata_=metadata or {},
            created_at=func.now(),
        )
        .returning(AgentMemoryBank.memory_id)
    )
    result = await conn.execute(stmt)
    return result.scalar() or mem_id


# ---------------------------------------------------------------------------
# Semantic search – ACTIVE‑only filter
# ---------------------------------------------------------------------------

async def semantic_search(
    conn: AsyncConnection,
    *,
    query_embedding: List[float],
    agent_id: str | None = None,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Return the top‑k *ACTIVE* memories similar to *query_embedding*.

    The SQL query enforces ``quarantine_status = 'ACTIVE'`` via a composite
    index (``idx_memory_agent_status``) so the HNSW vector search only runs
    on the small pre‑filtered subset.

    Returns a list of dicts with the memory fields (excluding the raw
    pgvector embedding bytes for readability; caller can access ``.embedding``
    if needed).
    """
    stmt = (
        select(AgentMemoryBank)
        .where(AgentMemoryBank.quarantine_status == "ACTIVE")
    )
    # Optional agent_id filter — pushed into SQL, NOT applied in Python.
    # Why (correctness + performance): the composite index
    # ``idx_memory_agent_status(agent_id, quarantine_status)`` turns this into
    # an O(log n) index seek, shrinking the set the HNSW vector search runs
    # over from the whole table to one agent's ACTIVE rows. Filtering in
    # Python would defeat the index AND the HNSW pruning.
    if agent_id is not None:
        stmt = stmt.where(AgentMemoryBank.agent_id == agent_id)

    stmt = stmt.order_by(
        # HNSW cosine distance; pgvector syntax.
        AgentMemoryBank.embedding.cosine_distance(query_embedding)
    ).limit(k)

    result = await conn.execute(stmt)
    memories = result.scalars().all()

    out: List[Dict[str, Any]] = []
    for m in memories:
        out.append(
            {
                "memory_id": str(m.memory_id),
                "agent_id": m.agent_id,
                "session_id": m.session_id,
                "source_span_id": m.source_span_id,
                "content_text": m.content_text,
                "quarantine_status": m.quarantine_status,
                "metadata_": m.metadata_,
                # Expose embedding as list for convenience; may be large.
                "embedding": list(m.embedding) if m.embedding else [],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Excision – mark memories EXCISED by source_span_id
# ---------------------------------------------------------------------------

async def excise_memories(
    conn: AsyncConnection,
    *,
    source_span_id: str,
) -> int:
    """Mark all memories derived from *source_span_id* as ``'EXCISED'``.

    Returns the number of rows updated.  Physical deletion is avoided so the
    audit trail remains; the ``quarantine_status`` change is what the
    ``semantic_search`` ``ACTIVE`` filter catches.
    """
    stmt = (
        AgentMemoryBank.__table__.update()
        .where(AgentMemoryBank.source_span_id == source_span_id)
        .values(quarantine_status="EXCISED")
    )
    result = await conn.execute(stmt)
    return result.rowcount or 0