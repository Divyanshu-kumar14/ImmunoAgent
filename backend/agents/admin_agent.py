"""Admin Agent (port 8003) — Phase 2.

Responsibility:
  • High‑privilege agent that can touch IAM policies.
  • Exposes an endpoint to update policy documents via the gateway's
    ``update_policy`` mock tool.
  • All calls propagate trace headers for DAG consistency.

Run this agent standalone:
    python -m agents.admin_agent --port 8003
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

class AdminAgent(BaseAgent):
    name = "admin_agent"
    port = 8003
    max_action_tier = "CRITICAL_EXEC"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.policy_buffer: str = ""
        self.app = FastAPI(title="AdminAgent")
        self._register_routes()

    # -------------------------------------------------------------------
    # Inbound policy update receipt
    # -------------------------------------------------------------------

    def _register_routes(self) -> None:
        # Closure over the instance (see ingest_agent._register_routes for
        # why the class-body decorator pattern is avoided).
        agent = self

        @self.app.post("/policy")
        async def receive_policy(request: Request) -> JSONResponse:
            body = await request.json()
            agent.policy_buffer = body.get("text", "")
            return JSONResponse(
                content={"received": True, "policy_len": len(agent.policy_buffer)},
                status_code=200,
            )

    # -------------------------------------------------------------------
    # Trigger: update policy via gateway
    # -------------------------------------------------------------------

    async def update_policy_via_gateway(self, policy_id: str = "policy_default", parent_span_id: str | None = None) -> Dict[str, Any]:
        """Send the buffered policy text to the gateway for storage."""
        if not self.policy_buffer:
            raise ValueError("no policy buffered")
        payload = {"policy_id": policy_id, "new_text": self.policy_buffer}
        return await self.invoke_tool(
            tool_name="update_policy",
            tool_args={"policy_id": policy_id, "new_text": self.policy_buffer},
            parent_span_id=parent_span_id,
        )


# -----------------------------------------------------------------------
# Server entry point
# -----------------------------------------------------------------------

def run_agent(host: str = "0.0.0.0", port: int = 8003) -> None:
    """Start the admin agent HTTP server."""
    agent = AdminAgent()
    agent.port = port
    agent.gateway_url = f"http://{host}:8000"
    uvicorn.run(agent.app, host=host, port=port, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Admin Agent")
    parser.add_argument("--port", type=int, default=8003, help="HTTP port (default 8003)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    args = parser.parse_args()
    run_agent(host=args.host, port=args.port)