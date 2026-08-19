"""Base agent class for all mock agents (Phase 2).

 Responsibilities:
   • Gateway HTTP client – POST to ``/v1/gateway/invoke`` or ``/tool/execute``.
   • Trace header propagation – inject ``X-Trace-Id`` and ``X-Parent-Span-Id``
     so the gateway can associate the call with the correct DAG span.
   • Shared :class:`Agent` base that concrete agents (ingest, payable, admin)
     subclass.

Performance notes:
   * The client uses a single ``httpx.AsyncClient`` instance per agent process,
     pooled across requests, to avoid TCP handshake overhead on every call.
   * Trace headers are auto‑generated from the current OpenTelemetry span if
     present; otherwise a fresh ``trace_id`` is created (``inject_traceparent``).
"""
from __future__ import annotations

import uuid as _uuid
from typing import Dict, Any, Optional

import httpx

from backend.core.otel_tracer import inject_traceparent


class BaseAgent:
    """Base class for all ImmunoAgent agents.

    Subclasses must define:
      * ``name`` – used as the ``agent_id`` when registering / issuing JWTs.
      * ``port`` – the HTTP port the agent listens on (for inbound calls).
      * ``endpoint`` – the gateway URL (typically ``http://host:<port>/v1/gateway/invoke``).
    """

    #: Override in subclasses
    name: str
    #: Override in subclasses
    port: int
    #: Override in subclasses – gateway base URL
    gateway_url: str = "http://localhost:8000"

    #: Claim embedded in the agent's JWT (see core/auth.issue_jwt). Kept on
    #: the class so the demo harness issues correctly-tiered tokens — the
    #: gateway reads this from the token (O(1), no per-request DB lookup).
    max_action_tier: str = "READ_ONLY"

    # Shared HTTP client across all requests in the process.
    _client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily create and return the pooled AsyncClient."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    # -----------------------------------------------------------------------
    # Trace header helpers
    # ---------------------------------------------------------------------------

    def _trace_headers(
        self,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Return ``X-Trace-Id`` and ``X-Parent-Span-Id`` for this call.

        - ``trace_id`` continues an existing DAG trace when supplied.
        - ``parent_span_id`` links this call to its parent span.
        - When neither is given, a fresh trace is generated.

        Both values are plain O(1) header constructions — no I/O, so this
        never adds latency to the agent → gateway fast path.
        """
        headers: Dict[str, str] = {}
        if trace_id:
            # Continue the caller's trace; the DAG stays coherent across hops.
            headers["X-Trace-Id"] = trace_id
        else:
            # Fresh trace – generate a traceparent and split it.
            tp = inject_traceparent()
            traceparent = tp["traceparent"]
            parts = traceparent.split("-")
            headers["X-Trace-Id"] = parts[0]
            # Fix: `_uuid.uuid4().hex` (NOT `str(...).hex` — str has no .hex
            # attribute, which would raise AttributeError on that branch).
            headers["X-Parent-Span-Id"] = (
                parts[1] if len(parts) > 1 else _uuid.uuid4().hex[:16]
            )
        if parent_span_id:
            # Explicit parent linkage (e.g. a memory read triggered by an
            # earlier ingest span) overrides the auto-generated span id.
            headers["X-Parent-Span-Id"] = parent_span_id
        return headers

    # -----------------------------------------------------------------------
    # Core request method
    # -----------------------------------------------------------------------

    async def invoke_tool(
        self,
        *,
        tool_name: str,
        tool_args: Dict[str, Any],
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a tool call through the gateway and return the result.

        The method:
          1. Builds the JSON payload ``{ "tool": <name>, "args": <args> }``.
          2. Adds trace headers for DAG lineage.
          3. Posts to ``/v1/gateway/invoke`` (or ``/tool/execute``).
          4. Returns the gateway's JSON response.

        Raises ``httpx.HTTPStatusError`` on non‑2xx responses (the caller
        can inspect the status code to decide next steps, e.g. quarantine).
        """
        client = self._get_client()
        payload = {"tool": tool_name, "args": tool_args}
        headers = self._trace_headers(parent_span_id, trace_id)

        resp = await client.post(
            f"{self.gateway_url}/v1/gateway/invoke",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()

    # -----------------------------------------------------------------------
    # Memory Bank client methods (Phase 2 flow: ingest → write, payable → query)
    # -----------------------------------------------------------------------

    async def write_memory(
        self,
        *,
        doc_text: str,
        agent_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a document into the Memory Bank via ``/v1/gateway/memory/write``.

        Separate from :meth:`invoke_tool` on purpose: memory ingestion is a
        write to the memory store, NOT a tool dispatch. Sending it through
        ``/v1/gateway/invoke`` would hit the O(1) tool-registry miss and be
        rejected, because ``memory.write`` is not a registered tool.
        """
        client = self._get_client()
        payload = {"doc_text": doc_text, "agent_id": agent_id or self.name}
        headers = self._trace_headers(parent_span_id, trace_id)

        resp = await client.post(
            f"{self.gateway_url}/v1/gateway/memory/write",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()

    async def query_memory(
        self,
        *,
        query: str,
        k: int = 5,
        agent_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Semantic-search the Memory Bank via ``/v1/gateway/memory/query``.

        Returns the gateway's ``{"results": [...]}`` payload. The ACTIVE-only
        filter is enforced server-side (memory_bank contract), so EXCISED
        memories never reach the agent. ``k`` is clamped server-side too,
        keeping the response payload bounded.
        """
        client = self._get_client()
        headers = self._trace_headers(parent_span_id, trace_id)
        params: Dict[str, Any] = {"q": query, "k": k}
        if agent_id:
            params["agent_id"] = agent_id

        resp = await client.get(
            f"{self.gateway_url}/v1/gateway/memory/query",
            params=params,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()