"""Phase 2: Mock tool definitions returned by the MCP server.

Each function is a pure in‑process mock — no external I/O.  They are
intended to be invoked via the gateway ``/tool/execute`` endpoint (or the
MCP client in Phase 3+) and return a deterministic result so the end‑to‑end
flow works without network dependencies.

Note: tool function parameters are *not* positional‑only, so they can be
passed freely as keyword arguments from ``call_tool`` / MCP dispatch.
"""

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Mock: fetch_vendor_invoice
# ---------------------------------------------------------------------------

def fetch_vendor_invoice(vendor: str, invoice_id: str, **extra: Any) -> Dict[str, Any]:
    """Return a mock invoice payload for the given vendor + invoice id.

    Scenario A (benign):  vendor = "vendor-alpha", invoice = "INV-0012".
    """
    return {
        "vendor": vendor,
        "invoice_id": invoice_id,
        "amount_cents": 120_000,  # $1,200
        "currency": "USD",
        "status": "paid",
        "beneficiary": vendor,
        "line_items": [
            {"description": "Consulting services", "amount_cents": 120_000}
        ],
    }


# ---------------------------------------------------------------------------
# Mock: execute_wire_transfer
# ---------------------------------------------------------------------------

def execute_wire_transfer(
    amount: int | None = None,
    amount_cents: int | None = None,
    beneficiary: str | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Mock a wire‑transfer execution.

    Accepts both amount conventions so either caller works:
      * gateway/armor payload contract: ``amount`` in USD dollars
        (see ``model_armor.WIRE_AMOUNT_THRESHOLD`` — it reads ``amount``)
      * direct MCP call convention: ``amount_cents``

    The alias is resolved ONCE at the top (dollars → cents), so downstream
    logic never branches on which field arrived — keeps the mock O(1) with
    no duplicated formatting paths.
    """
    if amount_cents is None and amount is not None:
        amount_cents = amount * 100  # dollars -> cents
    if amount_cents is None or beneficiary is None:
        raise ValueError("execute_wire_transfer requires 'amount'/'amount_cents' and 'beneficiary'")

    return {
        "transfer_id": f"transfer_{amount_cents}_{beneficiary.replace('-', '_')}",
        "amount_cents": amount_cents,
        "beneficiary": beneficiary,
        "status": "completed",
        "message": f"${amount_cents/100:,.2f} sent to {beneficiary}",
    }


# ---------------------------------------------------------------------------
# Mock: read_file
# ---------------------------------------------------------------------------

def read_file(file_path: str, **extra: Any) -> Dict[str, Any]:
    """Mock reading a file from disk.

    Returns the file's "contents" and metadata.  In a real system this would
    hit the filesystem or a storage service.
    """
    # Simulate a few known files; otherwise return a generic empty file.
    known = {
        "policy.json": {
            "content": '{"allow": true, "max_amount": 1000000}',
            "encoding": "utf-8",
        },
        "agents.json": {
            "content": '[{"agent_id":"admin_agent","max_action_tier":"CRITICAL_EXEC"}]',
            "encoding": "utf-8",
        },
    }
    # O(1) dict probe: one lookup (dict.get) instead of two (in + index).
    entry = known.get(file_path)
    if entry is not None:
        return {"content": entry["content"], "encoding": entry["encoding"]}
    return {"content": "", "encoding": "utf-8", "note": f"file '{file_path}' not in mock fs"}


# ---------------------------------------------------------------------------
# Mock: update_policy
# ---------------------------------------------------------------------------

def update_policy(policy_id: str, new_text: str, **extra: Any) -> Dict[str, Any]:
    """Mock updating a policy document.

    Returns the new policy version stub.
    """
    return {
        "policy_id": policy_id,
        "version": 2,
        "updated_at": "2025-01-01T00:00:00Z",  # placeholder
        "text_snippet": new_text[:80] + ("..." if len(new_text) > 80 else ""),
    }


# ---------------------------------------------------------------------------
# Public export: dict of tool name -> callable for MCP dispatch
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "fetch_vendor_invoice": fetch_vendor_invoice,
    "execute_wire_transfer": execute_wire_transfer,
    "read_file": read_file,
    "update_policy": update_policy,
}


def call_tool(name: str, **kwargs: Any) -> Dict[str, Any]:
    """Dispatch to the named mock tool, returning its result dict.

    Dispatch is a single O(1) dict lookup (``TOOL_FUNCTIONS``) — never a
    linear scan over tool names. Raises ``KeyError`` (with the offending
    name) if the tool is unknown — the gateway translates that into a
    400/423 response.
    """
    fn = TOOL_FUNCTIONS.get(name)  # O(1) hash lookup; None on miss
    if fn is None:
        raise KeyError(f"unknown tool: {name}")
    return fn(**kwargs)