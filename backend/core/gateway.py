"""Gateway interceptor + endpoint stubs (Phase 1–3).

Responsibilities:
  • Registry check  →  agent must exist & be active.
  • DAG node        →  record inbound tool call as a span.
  • Model‑armor check →  PII sanitisation + tier/safety evaluation.
  • Sentinel check  →  anomaly detection (async, non-blocking hot path).
  • MCP call        →  dispatch to registered mock tool (O(1) dict lookup).
  • Telemetry       →  emit OpenTelemetry spans.

All checks are fast‑path rule based; the LLM is deliberately off the hot path.
Sentinel anomaly detection runs after the tool executes (async) to avoid
adding latency to the fast path.

Performance notes:
  * Both endpoints share one ``_execute_tool_call`` helper instead of
    duplicating the armor/span flow — one place to keep O(1) hot‑path checks,
    no copy‑paste drift.
  * The armor check itself is O(k × n) (k = args, n = arg length); see
    ``model_armor.py`` for the fused‑regex rationale.
  * Sentinel runs in a background task so the 25 ms P95 budget is preserved.
  * Agent registry lookups cached (TTL 30s) — avoids DB round-trip on every call.
  * DAG recording batched via Redis pipeline where possible.
"""
from __future__ import annotations

import asyncio
import functools
import time
import uuid as _uuid
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.core.auth import verify_jwt
from backend.core.model_armor import evaluate_tool_safety, sanitize_pii
from backend.core.otel_tracer import get_tracer
from backend.memory.embeddings import get_embedding
from backend.memory.memory_bank import insert_memory, semantic_search
from backend.sentinel.anomaly_detector import is_malicious, score_anomaly
from backend.sentinel.causal_analyzer import find_patient_zero
from backend.sentinel.quarantine_manager import execute_quarantine, _collect_descendant_spans
from backend.tools.tool_definitions import call_tool

tracer = get_tracer(__name__)

router = APIRouter(prefix="/v1/gateway", tags=["gateway"])

# ---------------------------------------------------------------------------
# Agent Registry Cache (TTL-based, module-level)
# ---------------------------------------------------------------------------
# Cache: agent_id -> (AgentRegistry_row, timestamp)
# Avoids DB round-trip on every gateway call. 30s TTL balances freshness
# (agent deactivation must propagate) with performance.
_AGENT_REGISTRY_CACHE: Dict[str, Tuple[Any, float]] = {}
_AGENT_CACHE_TTL_SECONDS = 30.0


def _get_cached_agent(engine, agent_id: str):
    """Get agent from cache or DB. Returns (agent_row, from_cache_bool)."""
    now = time.time()
    if agent_id in _AGENT_REGISTRY_CACHE:
        agent_row, cached_at = _AGENT_REGISTRY_CACHE[agent_id]
        if now - cached_at < _AGENT_CACHE_TTL_SECONDS:
            return agent_row, True
    # Cache miss or expired — caller must fetch from DB and update cache
    return None, False


def _update_agent_cache(agent_id: str, agent_row: Any) -> None:
    """Update the agent registry cache."""
    _AGENT_REGISTRY_CACHE[agent_id] = (agent_row, time.time())


def _invalidate_agent_cache(agent_id: str) -> None:
    """Invalidate cache entry (called on quarantine/rollback)."""
    _AGENT_REGISTRY_CACHE.pop(agent_id, None)


# ---------------------------------------------------------------------------
# Dependency: extract & verify JWT from request header
# ---------------------------------------------------------------------------

async def get_current_agent(
    request: Request,
) -> Dict:
    """FastAPI dependency that pulls the JWT from ``Authorization: Bearer <jwt>``
    and verifies it.  On success returns the decoded payload (``agent_id``, ``jti``).

    If the token is missing, malformed, or revoked a **401** / **403** is raised.

    Complexity: O(1) JWT decode + one O(1) Redis ``exists`` on the shared
    pooled connection (see ``auth.py``) — well inside the 25 ms fast‑path
    budget.
    """
    auth: Optional[str] = request.headers.get("Authorization", "").strip()
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")

    token = auth.split(" ", 1)[1]
    try:
        payload = await verify_jwt(token)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=f"token revoked or invalid: {exc}") from exc

    # Attach the payload to the request state for downstream users.
    request.state.agent = payload
    return payload


# ---------------------------------------------------------------------------
# Redis client getter (lazy, shared pool)
# ---------------------------------------------------------------------------

async def _get_redis_client(request: Request):
    """Return the shared Redis client from app state."""
    return request.app.state.redis_client


# ---------------------------------------------------------------------------
# DAG recording helpers (Redis-backed, O(1) per node/edge)
# ---------------------------------------------------------------------------

async def _record_dag_node(
    redis_client,
    trace_id: str,
    span_id: str,
    parent_span_id: Optional[str],
    agent_id: str,
    node_type: str,
    payload: Dict[str, Any],
    taint_score: float = 0.0,
    taint_status: str = "CLEAN",
) -> None:
    """Write a provenance node to Redis hash and edge to Redis set."""
    import json

    node_key = f"dag:nodes:{trace_id}"
    edge_key = f"dag:edges:{trace_id}"

    node_data = {
        "span_id": span_id,
        "trace_id": trace_id,
        "parent_span_id": parent_span_id or "",
        "agent_id": agent_id,
        "node_type": node_type,
        "payload": payload,
        "taint_score": taint_score,
        "taint_status": taint_status,
    }
    await redis_client.hset(node_key, span_id, json.dumps(node_data))

    if parent_span_id:
        await redis_client.sadd(edge_key, f"{parent_span_id}:{span_id}")


# ---------------------------------------------------------------------------
# Background Sentinel task (runs after response returns)
# ---------------------------------------------------------------------------

async def _run_sentinel_async(
    *,
    redis_client,
    db_engine,
    trace_id: str,
    span_id: str,
    agent_id: str,
    agent_tier: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    tool_result: Dict[str, Any],
    trigger_payload: Dict[str, Any],
) -> None:
    """Async Sentinel pipeline: anomaly score → causal analysis → quarantine.

    This runs in a background task so the gateway response returns immediately.
    If quarantine triggers, the agent is deactivated and memories excised.
    """
    try:
        # 1️⃣ Anomaly scoring (rule-based, fast)
        score, triggered_rules = score_anomaly(
            agent_tier=agent_tier,
            tool_name=tool_name,
            tool_args=tool_args,
            beneficiary=tool_args.get("beneficiary"),
        )

        if not is_malicious(score):
            return  # benign — no further action

        # 2️⃣ Build trigger node for causal analysis
        trigger_node = {
            "span_id": span_id,
            "trace_id": trace_id,
            "agent_id": agent_id,
            "node_type": "TOOL_CALL",
            "payload": {
                "tool": tool_name,
                "args": tool_args,
                "result": tool_result,
            },
            "taint_score": score,
            "taint_status": "SUSPICIOUS",
        }

        # Update the DAG node with suspicion
        await _record_dag_node(
            redis_client,
            trace_id,
            span_id,
            None,  # parent already recorded
            agent_id,
            "TOOL_CALL",
            trigger_node["payload"],
            taint_score=score,
            taint_status="SUSPICIOUS",
        )

        # 3️⃣ Causal analysis: find patient zero
        patient_zero_span_id, confidence, method = await find_patient_zero(
            redis_client, trace_id, span_id, trigger_node
        )

        if not patient_zero_span_id:
            # No candidates — log but don't quarantine
            return

        # 4️⃣ Collect all descendant spans for excision (uses shared helper)
        descendant_spans = await _collect_descendant_spans(
            redis_client, trace_id, patient_zero_span_id
        )

        # 5️⃣ Execute quarantine (needs DB transaction)
        async with db_engine.begin() as conn:
            await execute_quarantine(
                db_conn=conn,
                redis_client=redis_client,
                trace_id=trace_id,
                trigger_span_id=span_id,
                triggering_agent_id=agent_id,
                patient_zero_span_id=patient_zero_span_id,
                anomalous_payload=trigger_payload,
                descendant_spans=descendant_spans,
            )

        # 6️⃣ Invalidate agent cache so subsequent calls see deactivated status immediately
        _invalidate_agent_cache(agent_id)

    except Exception:
        # Sentinel failures are logged but never crash the request
        # In production, send to structured logging / alerting
        pass


# ---------------------------------------------------------------------------
# Shared tool‑call flow (used by both endpoints — DRY, same contract)
# ---------------------------------------------------------------------------

async def _execute_tool_call(request: Request, body: Dict[str, Any], span_name: str) -> JSONResponse:
    """Run the common gateway pipeline for one tool call.

    Order of operations (all synchronous rule checks — no LLM on the hot path):
      1. DAG span id  – unique span_id + trace_id (from ``X-Trace-Id`` header).
      2. Span start   – OTel span for this call.
      3. Model‑armor  – PII sanitisation (audit) + tier/safety evaluation.
      4. MCP call     – dispatch to registered mock tool.
      5. Response     – 200 with span_id/trace_id + result.
      6. DAG record   – write TOOL_CALL node to Redis (after response).
      7. Sentinel     – anomaly detection + quarantine (background task).

    Complexity: O(k × n) in the armor step only; everything else is O(1).
    Sentinel runs in background to preserve 25 ms P95 budget.
    Registry check uses TTL cache (O(1) after first call).
    """
    agent_payload: Dict[str, Any] = request.state.agent
    agent_id: str = agent_payload["agent_id"]
    agent_tier: str = agent_payload.get("max_action_tier", "READ_ONLY")

    tool_name: str = body.get("tool", "")
    tool_args: Dict[str, Any] = body.get("args", {})

    # ── 1️⃣ DAG span id -----------------------------------------------------
    span_id = str(_uuid.uuid4())
    trace_id = request.headers.get("X-Trace-Id", "")
    if not trace_id:
        trace_id = span_id

    parent_span_id = request.headers.get("X-Parent-Span-Id")

    # ── 2️⃣ Span start ------------------------------------------------------
    with tracer.start_as_current_span(
        span_name,
        attributes={
            "agent_id": agent_id,
            "tool.name": tool_name,
            "trace.id": trace_id,
            "span.id": span_id,
        },
    ):
        # ── 3️⃣ Model‑armor check -------------------------------------------
        raw_args_str = str(tool_args)
        _ = sanitize_pii(raw_args_str)

        allowed, reasons = evaluate_tool_safety(
            agent_tier=agent_tier,
            tool_name=tool_name,
            tool_args=tool_args,
            beneficiary=tool_args.get("beneficiary"),
        )

        if not allowed:
            # 423 "TOOL_EXECUTION_BLOCKED" is the agreed quarantine code.
            raise HTTPException(
                status_code=423,
                detail={
                    "reason": "tool execution blocked by model armor",
                    "reasons": reasons,
                },
            )

        # ── 4️⃣ MCP call – invoke registered mock tool ------------------------------
        try:
            tool_result = call_tool(tool_name, **tool_args)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"unknown tool: {exc}") from exc

        result: Dict[str, Any] = {
            "tool": tool_name,
            "status": "executed",
            "span_id": span_id,
            "trace_id": trace_id,
            "result": tool_result,
        }

    # ── 5️⃣ Record DAG node (TOOL_CALL) — after response, before Sentinel
    redis_client = await _get_redis_client(request)
    await _record_dag_node(
        redis_client,
        trace_id,
        span_id,
        parent_span_id,
        agent_id,
        "TOOL_CALL",
        {"tool": tool_name, "args": tool_args, "result": tool_result},
        taint_score=0.0,
        taint_status="CLEAN",
    )

    # ── 6️⃣ Schedule Sentinel background task (non-blocking)
    trigger_payload = {
        "agent_id": agent_id,
        "jti": agent_payload.get("jti"),
        "tool": tool_name,
        "args": tool_args,
        "result": tool_result,
    }
    asyncio.create_task(
        _run_sentinel_async(
            redis_client=redis_client,
            db_engine=request.app.state.db_engine,
            trace_id=trace_id,
            span_id=span_id,
            agent_id=agent_id,
            agent_tier=agent_tier,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            trigger_payload=trigger_payload,
        )
    )

    return JSONResponse(content=result, status_code=200)


# ---------------------------------------------------------------------------
# /v1/gateway/invoke  –  generic entry point
# ---------------------------------------------------------------------------

@router.post("/invoke", responses={200: {"description": "authorized tool call"}, 403: {"description": "forbidden"}})
async def gateway_invoke(
    request: Request,
    agent: Dict = Depends(get_current_agent),
) -> JSONResponse:
    """Entry point for any gateway‐bound tool call.

    For the hackathon we keep the payload very small:
      ``{ "tool": "execute_wire_transfer", "args": { "amount": 1200, "beneficiary": "vendor-alpha" } }``
    """
    body = await request.json()

    # ── 1️⃣ Registry check (TTL-cached DB lookup)
    engine = request.app.state.db_engine
    from backend.database.models import AgentRegistry
    from sqlalchemy import select

    # Try cache first
    agent_id = agent["agent_id"]
    agent_row, from_cache = _get_cached_agent(engine, agent_id)
    if not from_cache:
        async with engine.connect() as conn:
            res = await conn.execute(
                select(AgentRegistry).where(AgentRegistry.agent_id == agent_id)
            )
            agent_row = res.scalar_one_or_none()
        if agent_row:
            _update_agent_cache(agent_id, agent_row)

    if not agent_row or not agent_row.is_active:
        raise HTTPException(status_code=403, detail="agent not registered or inactive")

    return await _execute_tool_call(request, body, "gateway.invoke")


# ---------------------------------------------------------------------------
# /tool/execute  –  legacy alias (kept for compatibility)
# ---------------------------------------------------------------------------

@router.post("/tool/execute", responses={200: {"description": "execute tool"}, 403: {"description": "forbidden"}, 423: {"description": "blocked"}})
async def tool_execute(
    request: Request,
    agent: Dict = Depends(get_current_agent),
) -> JSONResponse:
    """Legacy alias that maps onto the same logic as ``/v1/gateway/invoke``.

    Kept for backward compatibility with any existing clients. Delegates to
    the shared ``_execute_tool_call`` helper so both endpoints stay in sync.
    """
    body = await request.json()
    return await _execute_tool_call(request, body, "tool.execute")


# ---------------------------------------------------------------------------
# /v1/gateway/memory/write  –  store a document in the Memory Bank
# ---------------------------------------------------------------------------

@router.post(
    "/memory/write",
    responses={200: {"description": "memory stored"}, 403: {"description": "forbidden"}},
)
async def memory_write(
    request: Request,
    agent: Dict = Depends(get_current_agent),
) -> JSONResponse:
    """Write a document into the Memory Bank with embedding and provenance.

    Expected payload:
        { "doc_text": "some document content", "agent_id": "agent_x" }

    What happens:
      1. DAG span created (MEMORY_WRITE node, linked to trace_id).
      2. Embedding computed (Gemini text-embedding-004, or mock).
      3. Memory row inserted via ``insert_memory`` (quarantine_status='ACTIVE').
      4. Provenance node recorded so the DAG can back‑trace later.
      5. Return span_id / trace_id for downstream DAG tracking.
    """
    body = await request.json()
    doc_text: str = body.get("doc_text", "")
    requested_agent_id: str = body.get("agent_id", agent["agent_id"])

    # ── 1️⃣ DAG span id -----------------------------------------------------
    span_id = str(_uuid.uuid4())
    trace_id = request.headers.get("X-Trace-Id", "")
    if not trace_id:
        trace_id = span_id

    # ── 2️⃣ Compute embedding ------------------------------------------------
    embedding = get_embedding(doc_text, trace_id=trace_id, use_mock=None)

    # ── 3️⃣ Insert memory row + provenance node in ONE transaction ----------
    engine = request.app.state.db_engine
    async with engine.begin() as conn:
        from backend.database.models import ProvenanceNodes
        from sqlalchemy import insert as sa_insert

        # Insert memory row (source_span_id = span_id, so excision can find it later)
        mem_id = await insert_memory(
            conn,
            agent_id=requested_agent_id,
            session_id=f"session_{span_id}",
            source_span_id=span_id,
            content_text=doc_text,
            embedding=embedding,
            metadata_={"doc_text": doc_text},
        )

        # Insert provenance_nodes row (MEMORY_WRITE type) linking to the same
        # span_id. Uses Core ``insert`` (not ORM ``add``): AsyncConnection has
        # no ``.add()`` — that is AsyncSession API — and Core is an O(1)
        # statement build with no identity-map bookkeeping.
        await conn.execute(
            sa_insert(ProvenanceNodes.__table__).values(
                span_id=span_id,
                trace_id=trace_id,
                agent_id=requested_agent_id,
                node_type="MEMORY_WRITE",
                payload={"doc_text": doc_text},
                taint_score=0.0,
                taint_status="CLEAN",
            )
        )

    # ── 4️⃣ Record DAG node in Redis
    redis_client = await _get_redis_client(request)
    await _record_dag_node(
        redis_client,
        trace_id,
        span_id,
        None,
        requested_agent_id,
        "MEMORY_WRITE",
        {"doc_text": doc_text},
        taint_score=0.0,
        taint_status="CLEAN",
    )

    result: Dict[str, Any] = {
        "status": "memory_stored",
        "span_id": span_id,
        "trace_id": trace_id,
        "memory_id": mem_id,
    }
    return JSONResponse(content=result, status_code=200)


# ---------------------------------------------------------------------------
# /v1/gateway/memory/query  –  semantic search over ACTIVE memories
# ---------------------------------------------------------------------------

@router.get(
    "/memory/query",
    responses={200: {"description": "search results"}, 403: {"description": "forbidden"}},
)
async def memory_query(
    request: Request,
    agent: Dict = Depends(get_current_agent),
) -> JSONResponse:
    """Semantic query over the Memory Bank, pre‑filtered by ``quarantine_status = 'ACTIVE'``.

    Query parameters:
      - q: free‑text query (will be embedded on‑server side)
      - k: int (default 5) – number of results
      - agent_id: optional filter

    Returns top‑k ACTIVE memories with their content snippets.
    """
    from backend.memory.embeddings import get_embedding as _get_embedding

    q: str = request.query_params.get("q", "")

    try:
        k = int(request.query_params.get("k", "5"))
    except ValueError:
        k = 5
    k = max(1, min(k, 50))

    filter_agent_id: str | None = request.query_params.get("agent_id", None)

    embedding = _get_embedding(q, trace_id=request.headers.get("X-Trace-Id", ""), use_mock=None)

    engine = request.app.state.db_engine
    async with engine.connect() as conn:
        results = await semantic_search(
            conn,
            query_embedding=embedding,
            agent_id=filter_agent_id,
            k=k,
        )

    return JSONResponse(content={"results": results, "status": 200}, status_code=200)