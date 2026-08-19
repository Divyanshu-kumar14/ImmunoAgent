"""Payable Agent (port 8002) — Phase 2.

Responsibility:
  • Query the Memory Bank for ACTIVE memories (semantic search).
  • Based on the query result, invoke a tool via the gateway /tool/execute
    (or /v1/gateway/invoke) — e.g., fetch_vendor_invoice, execute_wire_transfer.
  • Propagate trace headers for DAG lineage.

Run this agent standalone:
    python -m agents.payable_agent --port 8002
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

class PayableAgent(BaseAgent):
    name = "payable_agent"
    port = 8002
    max_action_tier = "CRITICAL_EXEC"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.last_memories: list[dict] = []
        self.app = FastAPI(title="PayableAgent")
        self._register_routes()

    # -------------------------------------------------------------------
    # Inbound query receipt — runs a real semantic search via the gateway
    # -------------------------------------------------------------------

    def _register_routes(self) -> None:
        # Closure over the instance (see ingest_agent._register_routes for
        # why the class-body decorator pattern is avoided).
        agent = self

        @self.app.post("/query")
        async def receive_query(request: Request) -> JSONResponse:
            body = await request.json()
            query_text: str = body.get("query", "")
            # Phase 2 flow: payable reads memory via the gateway. The
            # ACTIVE-only filter is enforced server-side (memory_bank
            # contract), so EXCISED memories never reach the agent.
            search = await agent.query_memory(query=query_text, agent_id=agent.name)
            agent.last_memories = search.get("results", [])
            return JSONResponse(
                content={
                    "received": True,
                    "query_len": len(query_text),
                    "num_memories": len(agent.last_memories),
                },
                status_code=200,
            )

    # -------------------------------------------------------------------
    # Trigger: execute a tool based on a memory result
    # -------------------------------------------------------------------

    async def execute_tool_from_memory(
        self,
        tool_name: str,
        tool_args: dict,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> dict:
        """Call the gateway to execute a tool, passing memory-derived args.

        Delegates to ``BaseAgent.invoke_tool`` — the pooled HTTP client and
        trace-header logic live in exactly one place (DRY), so the hot path
        keeps a single O(1) header build + pooled connection reuse instead of
        duplicating the request plumbing here.
        """
        return await self.invoke_tool(
            tool_name=tool_name,
            tool_args=tool_args,
            parent_span_id=parent_span_id,
            trace_id=trace_id,
        )


# -----------------------------------------------------------------------
# Server entry point
# -----------------------------------------------------------------------

def run_agent(host: str = "0.0.0.0", port: int = 8002) -> None:
    """Start the payable agent HTTP server."""
    agent = PayableAgent()
    agent.port = port
    agent.gateway_url = f"http://{host}:8000"
    uvicorn.run(agent.app, host=host, port=port, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Payable Agent")
    parser.add_argument("--port", type=int, default=8002, help="HTTP port (default 8002)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    args = parser.parse_args()
    run_agent(host=args.host, port=args.port)