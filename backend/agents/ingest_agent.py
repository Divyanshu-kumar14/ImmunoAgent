"""Ingest Agent (port 8001) — Phase 2.

Responsibility:
  • Accept a document (text) via a simple HTTP server.
  • POST to the gateway /memory/write endpoint to store it in the Memory Bank
    with embedding and provenance.
  • Propagate trace headers (X-Trace-Id, X-Parent-Span-Id) so the DAG remains
    coherent across agent boundaries.

Run this agent standalone:
    python -m agents.ingest_agent --port 8001
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.agents.base_agent import BaseAgent

# -----------------------------------------------------------------------
# Concrete agent subclass
# -----------------------------------------------------------------------

class IngestAgent(BaseAgent):
    name = "ingest_agent"
    port = 8001
    max_action_tier = "INTERNAL_WRITE"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.doc_buffer: str = ""
        self.app = FastAPI(title="IngestAgent")
        self._register_routes()

    # -------------------------------------------------------------------
    # Inbound doc receipt (simple POST)
    # -------------------------------------------------------------------

    def _register_routes(self) -> None:
        # Routes are closures over the instance (NOT bound methods decorated
        # in the class body): a class-body `@app.post` decorator would register
        # the method signature as-is, making FastAPI treat `self` as a query
        # parameter — a guaranteed validation failure on every request.
        agent = self

        @self.app.post("/doc")
        async def receive_doc(request: Request) -> JSONResponse:
            body = await request.json()
            agent.doc_buffer = body.get("text", "")
            return JSONResponse(
                content={"received": True, "doc_len": len(agent.doc_buffer)},
                status_code=200,
            )

    # -------------------------------------------------------------------
    # Trigger: write doc to memory via gateway
    # -------------------------------------------------------------------

    async def write_doc_to_memory(self, parent_span_id: Optional[str] = None) -> Dict[str, Any]:
        """Flush the buffered doc to the gateway /memory/write.

        Uses ``BaseAgent.write_memory`` (the memory route) rather than
        ``invoke_tool``: "memory.write" is not a registered tool, so sending
        it through /v1/gateway/invoke would hit the O(1) tool-registry miss
        and be rejected.
        """
        if not self.doc_buffer:
            raise ValueError("no document buffered")
        return await self.write_memory(
            doc_text=self.doc_buffer,
            agent_id=self.name,
            parent_span_id=parent_span_id,
        )


# -----------------------------------------------------------------------
# Server entry point
# -----------------------------------------------------------------------

def run_agent(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Start the ingest agent HTTP server."""
    agent = IngestAgent()
    agent.port = port
    agent.gateway_url = f"http://{host}:8000"
    uvicorn.run(agent.app, host=host, port=port, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Ingest Agent")
    parser.add_argument("--port", type=int, default=8001, help="HTTP port (default 8001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    args = parser.parse_args()
    run_agent(host=args.host, port=args.port)