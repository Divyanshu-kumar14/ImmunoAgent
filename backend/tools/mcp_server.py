"""Minimal MCP (Model‑Control‑Plane) server for Phase 2.

Wraps the mock tool definitions from ``tool_definitions.py`` and provides
a tiny async dispatcher that the gateway can call instead of the hard‑canned
stub.  The design keeps the hot‑path O(1) lookup and delegates any heavy
I/O to background workers in later phases.

Typical usage from the gateway:
    from backend.tools.mcp_server import call_tool_via_mcp
    result = await call_tool_via_mcp("execute_wire_transfer", amount_cents=120000, beneficiary="vendor-alpha")
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.tools.tool_definitions import call_tool

router = APIRouter(prefix="/v1/tools", tags=["tools"])

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic MCP dispatcher
# ---------------------------------------------------------------------------

@router.post("/call", response_model=Dict[str, Any])
async def call_tool_endpoint(request: Request) -> JSONResponse:
    """Dispatch a tool call to the registered mock implementation.

    Expected payload:
        { "tool": "execute_wire_transfer", "args": { "amount_cents": 120000, "beneficiary": "vendor-alpha" } }
    """
    body: Dict[str, Any] = await request.json()
    tool_name: str = body.get("tool", "")
    args: Dict[str, Any] = body.get("args", {})

    try:
        # Single O(1) dict dispatch — the membership check + lookup happen
        # inside call_tool's one .get() probe, not as two separate lookups.
        result = call_tool(tool_name, **args)
        return JSONResponse(content={"result": result, "status": 200}, status_code=200)
    except KeyError as exc:
        # Unknown tool = 400 client error, not a 500: predictable miss on
        # the hot path should not pay for a traceback.
        return JSONResponse(
            content={"error": str(exc), "status": 400},
            status_code=400,
        )
    except Exception as exc:  # broad – keep the fast path from crashing
        log.exception("MCP tool call failed")
        return JSONResponse(
            content={"error": str(exc), "status": 500},
            status_code=500,
        )