"""Sentinel Quarantine Manager — revocation + excision + incident recording (Phase 3).

Orchestrates the full quarantine protocol:
  1. Revoke agent JWTs (Redis deny-list)
  2. Deactivate agent in registry
  3. Collect all descendant spans from patient zero
  4. Transactional memory excision (EXCISED)
  5. Write security_incidents row
  6. Broadcast quarantine event via WebSocket / Redis stream

Performance notes:
  * Descendant span collection uses shared adjacency cache (see causal_analyzer).
  * Memory excision batched into single UPDATE with IN clause (not N round-trips).
  * Provenance node updates batched similarly.
  * Memory restoration batched in rollback.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import redis.asyncio as redis
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from backend.core.auth import revoke_jwt
from backend.database.models import (
    AgentMemoryBank,
    AgentRegistry,
    ProvenanceNodes,
    SecurityIncidents,
)
from backend.memory.memory_bank import excise_memories
from backend.sentinel.causal_analyzer import _get_adjacency as _get_descendant_adjacency

# ---------------------------------------------------------------------------
# Redis stream key for telemetry broadcast
# ---------------------------------------------------------------------------
TELEMETRY_STREAM = "stream:telemetry_events"


# ---------------------------------------------------------------------------
# Helper: collect all descendant spans via forward BFS (Redis edges)
# ---------------------------------------------------------------------------

async def _collect_descendant_spans(
    redis_client: redis.Redis,
    trace_id: str,
    patient_zero_span_id: str,
) -> Set[str]:
    """Return the set of all span_ids reachable from patient_zero (inclusive).

    Uses the dag:edges:{trace_id} SET where members are "parent:child".
    Reuses the adjacency cache from causal_analyzer to avoid duplicate edge scans.
    """
    # Reuse cached adjacency if available (built by causal_analyzer during detection)
    children_of = _get_descendant_adjacency(redis_client, trace_id)
    if not children_of:
        # Build adjacency: parent -> list of children
        edge_key = f"dag:edges:{trace_id}"
        edge_pairs = await redis_client.smembers(edge_key)
        children_of: Dict[str, List[str]] = {}
        for pair in edge_pairs:
            parent, child = pair.split(":", 1)
            children_of.setdefault(parent, []).append(child)
        # Cache it for future calls (causal_analyzer's cache)
        import time
        from backend.sentinel.causal_analyzer import _TRACE_ADJACENCY_CACHE
        _TRACE_ADJACENCY_CACHE[trace_id] = (children_of, time.time())

    # Forward BFS using list as stack (DFS-like, but order doesn't matter for set)
    visited: Set[str] = set()
    stack: List[str] = [patient_zero_span_id]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        # Extend stack with children (DFS) — avoids deque overhead for small sets
        stack.extend(children_of.get(current, []))
    return visited


# ---------------------------------------------------------------------------
# Helper: broadcast to Redis stream (WebSocket consumers read from here)
# ---------------------------------------------------------------------------

async def _broadcast_event(
    redis_client: redis.Redis,
    event_type: str,
    payload: Dict[str, Any],
) -> None:
    """Append an event to the telemetry stream for WebSocket consumers."""
    event = {
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": json.dumps(payload),
    }
    await redis_client.xadd(TELEMETRY_STREAM, event)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def execute_quarantine(
    *,
    db_conn: AsyncConnection,
    redis_client: redis.Redis,
    trace_id: str,
    trigger_span_id: str,
    triggering_agent_id: str,
    patient_zero_span_id: str,
    anomalous_payload: Dict[str, Any],
    descendant_spans: Set[str],
) -> Dict[str, Any]:
    """Execute the full quarantine protocol transactionally.

    All DB mutations run in the caller's transaction (AsyncConnection).
    Redis operations (revocation, stream) are best-effort and run outside
    the DB transaction — if they fail, the incident is still recorded.

    Performance: Batches all multi-row updates into single statements.
    """
    excision_summary: Dict[str, Any] = {
        "patient_zero_span_id": patient_zero_span_id,
        "excised_memory_count": 0,
        "deactivated_agent": False,
        "revoked_jti_count": 0,
        "descendant_spans": list(descendant_spans),
    }

    # 1️⃣ Deactivate agent in registry — single UPDATE
    await db_conn.execute(
        update(AgentRegistry)
        .where(AgentRegistry.agent_id == triggering_agent_id)
        .values(is_active=False)
    )
    excision_summary["deactivated_agent"] = True

    # 2️⃣ Revoke JWTs — only the trigger's jti (deny-list has no agent_id index)
    # For complete revocation, a secondary index agent_id -> {jti} would be needed.
    # Agent deactivation (step 1) blocks future token issuance.
    trigger_jti = anomalous_payload.get("jti")
    if trigger_jti:
        await revoke_jwt(trigger_jti, redis_client=redis_client)
        excision_summary["revoked_jti_count"] = 1

    # 3️⃣ Excise memories for ALL descendant spans — BATCHED single UPDATE
    # Instead of N calls to excise_memories(), use one IN clause.
    # This reduces DB round-trips from O(N) to O(1).
    if descendant_spans:
        stmt = (
            update(AgentMemoryBank)
            .where(
                AgentMemoryBank.source_span_id.in_(list(descendant_spans)),
                AgentMemoryBank.quarantine_status == "ACTIVE",
            )
            .values(quarantine_status="EXCISED")
        )
        result = await db_conn.execute(stmt)
        total_excised = result.rowcount or 0
        excision_summary["excised_memory_count"] = total_excised

        # 4️⃣ Mark provenance nodes as EXCISED — BATCHED single UPDATE
        await db_conn.execute(
            update(ProvenanceNodes)
            .where(
                ProvenanceNodes.span_id.in_(list(descendant_spans)),
                ProvenanceNodes.taint_status != "EXCISED",  # idempotent
            )
            .values(taint_status="EXCISED", taint_score=1.0)
        )
    else:
        total_excised = 0

    # 5️⃣ Create security_incidents row
    incident_id = uuid.uuid4()
    await db_conn.execute(
        SecurityIncidents.__table__.insert().values(
            incident_id=incident_id,
            trace_id=trace_id,
            triggering_agent_id=triggering_agent_id,
            patient_zero_span_id=patient_zero_span_id,
            anomalous_action_payload=anomalous_payload,
            excision_summary=excision_summary,
            status="QUARANTINED",
            created_at=datetime.now(timezone.utc),
        )
    )

    # 6️⃣ Broadcast quarantine event (best-effort, outside DB transaction)
    try:
        await _broadcast_event(
            redis_client,
            "QUARANTINE_EXECUTED",
            {
                "incident_id": str(incident_id),
                "trace_id": trace_id,
                "agent_id": triggering_agent_id,
                "patient_zero_span_id": patient_zero_span_id,
                "excised_count": total_excised,
            },
        )
    except Exception:
        # Stream failure doesn't roll back DB — incident is recorded
        pass

    return {
        "incident_id": str(incident_id),
        "status": "QUARANTINED",
        "excision_summary": excision_summary,
    }


async def rollback_quarantine(
    *,
    db_conn: AsyncConnection,
    redis_client: redis.Redis,
    incident_id: str,
) -> Dict[str, Any]:
    """Rollback a quarantine: reactivate agent, restore memories to ACTIVE.

    This is the "clean recovery" path after manual review.

    Performance: Batches all restorations into single UPDATE statements.
    """
    # Fetch incident
    result = await db_conn.execute(
        select(SecurityIncidents).where(SecurityIncidents.incident_id == incident_id)
    )
    incident = result.scalar_one_or_none()
    if not incident:
        raise ValueError(f"Incident {incident_id} not found")

    if incident.status == "ROLLED_BACK":
        return {"status": "already_rolled_back", "incident_id": incident_id}

    descendant_spans = incident.excision_summary.get("descendant_spans", [])

    # 1️⃣ Reactivate agent — single UPDATE
    await db_conn.execute(
        update(AgentRegistry)
        .where(AgentRegistry.agent_id == incident.triggering_agent_id)
        .values(is_active=True)
    )

    # 2️⃣ Restore EXCISED memories back to ACTIVE — BATCHED single UPDATE
    restored = 0
    if descendant_spans:
        stmt = (
            update(AgentMemoryBank)
            .where(
                AgentMemoryBank.source_span_id.in_(descendant_spans),
                AgentMemoryBank.quarantine_status == "EXCISED",
            )
            .values(quarantine_status="ACTIVE")
        )
        res = await db_conn.execute(stmt)
        restored = res.rowcount or 0

        # 3️⃣ Restore provenance node taint status — BATCHED single UPDATE
        await db_conn.execute(
            update(ProvenanceNodes)
            .where(
                ProvenanceNodes.span_id.in_(descendant_spans),
                ProvenanceNodes.taint_status == "EXCISED",
            )
            .values(taint_status="CLEAN", taint_score=0.0)
        )

    # 4️⃣ Update incident status
    await db_conn.execute(
        update(SecurityIncidents)
        .where(SecurityIncidents.incident_id == incident_id)
        .values(status="ROLLED_BACK")
    )

    # 5️⃣ Broadcast rollback event
    try:
        await _broadcast_event(
            redis_client,
            "QUARANTINE_ROLLED_BACK",
            {
                "incident_id": incident_id,
                "trace_id": incident.trace_id,
                "agent_id": incident.triggering_agent_id,
                "restored_memory_count": restored,
            },
        )
    except Exception:
        pass

    return {
        "incident_id": incident_id,
        "status": "ROLLED_BACK",
        "restored_memory_count": restored,
    }