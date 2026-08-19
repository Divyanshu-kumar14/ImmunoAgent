"""Gateway interceptor + endpoint stubs (Phase 1).

Responsibilities:
  • Registry check  →  agent must exist & be active.
  • DAG node        →  record inbound tool call as a span.
  • Model‑armor check →  PII sanitisation + tier/safety evaluation.
  • MCP call        →  dispatch to registered mock tool (O(1) dict lookup).
  • Telemetry       →  emit OpenTelemetry spans.

All checks are fast‑path rule based; the LLM is deliberately off the hot path.

Performance notes:
  * Both endpoints share one ``_execute_tool_call`` helper instead of
    duplicating the armor/span flow — one place to keep O(1) hot‑path checks,
    no copy‑paste drift.
  * The armor check itself is O(k × n) (k = args, n = arg length); see
    ``model_armor.py`` for the fused‑regex rationale.
"""

from __future__ import annotations

import uuid as _uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.core.auth import verify_jwt
from backend.core.model_armor import evaluate_tool_safety, sanitize_pii
from backend.core.otel_tracer import get_tracer
from backend.memory.embeddings import get_embedding
from backend.memory.memory_bank import insert_memory, semantic_search
from backend.tools.tool_definitions import call_tool

tracer = get_tracer(__name__)

router = APIRouter(prefix="/v1/gateway", tags=["gateway"])


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
# Shared tool‑call flow (used by both endpoints — DRY, same contract)
# ---------------------------------------------------------------------------

async def _execute_tool_call(request: Request, body: Dict[str, Any], span_name: str) -> JSONResponse:
    """Run the common gateway pipeline for one tool call.

    Order of operations (all synchronous rule checks — no LLM on the hot path):
      1. DAG span id  – unique span_id + trace_id (from ``X-Trace-Id`` header).
      2. Span start   – OTel span for this call.
      3. Model‑armor  – PII sanitisation (audit) + tier/safety evaluation.
      4. Response     – 423 on blocked tool, else 200 with span_id/trace_id.

    Complexity: O(k × n) in the armor step only; everything else is O(1).
    """
    agent_payload: Dict[str, Any] = request.state.agent
    agent_id: str = agent_payload["agent_id"]

    tool_name: str = body.get("tool", "")
    tool_args: Dict[str, Any] = body.get("args", {})

    # ── 1️⃣ DAG span id -----------------------------------------------------
    # Unique span_id for this call; trace_id is pulled from the caller's
    # ``X-Trace-Id`` header when present, else we fall back to the span_id so
    # the DAG still has a stable trace to hang nodes on.
    span_id = str(_uuid.uuid4())
    trace_id = request.headers.get("X-Trace-Id", "")
    if not trace_id:
        trace_id = span_id

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
        # PII sanitisation is best‑effort audit (the sanitized copy would be
        # persisted/logged downstream); the Sentinel acts on the raw payload
        # in Phase 3. Kept in the hot path because it is one fused O(n) pass.
        raw_args_str = str(tool_args)
        _ = sanitize_pii(raw_args_str)

        # Safety evaluation using the rule‑based armor (O(k × n), no LLM).
        allowed, reasons = evaluate_tool_safety(
            agent_tier=agent_payload.get("max_action_tier", "READ_ONLY"),
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
        # In Phase 2 we dispatch to the registered mock implementations so
        # the end‑to‑end flow (tool → memory → agent response) works without
        # external services.  The result is enriched with span/trace IDs.
        # Dispatch is a single O(1) dict lookup inside call_tool — no scan.
        try:
            tool_result = call_tool(tool_name, **tool_args)
        except KeyError as exc:
            # Unknown tool = client error (400), NOT a 500: a predictable
            # registry miss on the hot path shouldn't pay for a traceback.
            raise HTTPException(status_code=400, detail=f"unknown tool: {exc}") from exc
        result: Dict[str, Any] = {
            "tool": tool_name,
            "status": "executed",
            "span_id": span_id,
            "trace_id": trace_id,
            "result": tool_result,
        }
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

    # ── 1️⃣ Registry check (simple in‑process stub; real DB lookup in Phase 2)
    # -----------------------------------------------------------------------
    # TODO: Replace with SQLAlchemy query against ``agent_registry``.
    # For now we allow any non‑empty agent_id; the real check will be in Phase 2.
    # -----------------------------------------------------------------------
    if not agent["agent_id"]:
        raise HTTPException(status_code=403, detail="agent not registered")

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
    # use_mock=None → the embedding module decides: real Gemini when
    # GEMINI_API_KEY is set, deterministic hash mock otherwise (Plan: mock
    # fallback — the demo never hard-depends on the LLM).
    embedding = get_embedding(doc_text, trace_id=trace_id, use_mock=None)

    # ── 3️⃣ Insert memory row + provenance node in ONE transaction ----------
    # engine.begin() (not acquire/connect): the memory row and its provenance
    # node must commit together or not at all, and begin() auto-commits on
    # successful exit — no manual commit bookkeeping on the hot path. The
    # pooled connection is reused across requests (no per-request handshake).
    engine = request.app.state.db_engine  # type: ignore[attr-defined]
    async with engine.begin() as conn:
        from backend.database.models import ProvenanceNodes  # noqa: F401 (local import keeps module load order safe)
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

    # k is clamped to [1, 50]: bounds the HNSW search radius and the JSON
    # payload size (perf: an unbounded k would stream the whole ACTIVE set).
    # Invalid input falls back to the default instead of raising a 500.
    try:
        k = int(request.query_params.get("k", "5"))
    except ValueError:
        k = 5
    k = max(1, min(k, 50))

    filter_agent_id: str | None = request.query_params.get("agent_id", None)

    # Query embedding uses the same env-driven mock/production decision as
    # memory_write, so write and query embeddings live in the same space.
    embedding = _get_embedding(q, trace_id=request.headers.get("X-Trace-Id", ""), use_mock=None)

    # Read-only path: engine.connect() (auto-closes, no transaction needed).
    engine = request.app.state.db_engine  # type: ignore[attr-defined]
    async with engine.connect() as conn:
        results = await semantic_search(
            conn,
            query_embedding=embedding,
            agent_id=filter_agent_id,
            k=k,
        )

    return JSONResponse(content={"results": results, "status": 200}, status_code=200)