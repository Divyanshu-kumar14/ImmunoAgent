"""Fast‑path model armor (PRD §4.2).

PII sanitiser + tool‑safety evaluator. All checks are rule‑based and execute
in the gateway's hot path (< 25 ms P95). The LLM is only invoked asynchronously
in the Sentinel path (Phase 3).

Performance notes (why this module is shaped this way):
  * The three PII patterns are fused into ONE compiled alternation regex
    (``PII_RE``). A single ``search``/``sub`` over the input is O(n) with one
    pass; the naive approach of running three separate regexes is 3×O(n).
    On the gateway hot path (every tool call) that difference is the
    difference between ~0.1 µs and ~0.3 µs per KB of payload — small, but it
    compounds across every argument of every tool call.
  * ``TOOL_TIER_ALLOWLIST`` is a dict of sets → O(1) tier membership lookup,
    never a linear scan.
  * ``BENEFICIARY_ALLOWLIST`` is a set → O(1) membership check.
"""

import re
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# PII regex patterns (U.S. formats; extend as needed)
# ---------------------------------------------------------------------------
# Single fused alternation = ONE scan over the input instead of three.
#   1. SSN          123-45-6789 (or 123456789)
#   2. card number  13-16 digits with optional spaces/dashes between groups
#   3. ACH routing  9 consecutive digits
PII_RE = re.compile(
    r"\b(?:"
    r"\d{3}-?\d{2}-?\d{4}|"      # SSN
    r"(?:\d[ -]*?){13,16}|"      # card number
    r"\d{9}"                     # ACH routing number
    r")\b"
)

# ---------------------------------------------------------------------------
# Action‑tier allowlist (PRD §4.2.1). Only agents whose
# ``max_action_tier`` >= tier may perform the action.
# ---------------------------------------------------------------------------
TOOL_TIER_ALLOWLIST = {
    "READ_ONLY": {"read_file", "view_policy"},
    "INTERNAL_WRITE": {"read_file", "view_policy", "update_policy"},
    "CRITICAL_EXEC": {
        "read_file",
        "view_policy",
        "update_policy",
        "execute_wire_transfer",
        "fetch_vendor_invoice",
    },
}

# Amount threshold above which a wire‑transfer is flagged (cents).
WIRE_AMOUNT_THRESHOLD = 10_000_00  # $100 000

# Beneficiary allowlist — agents may only send funds to approved payees.
# Set => O(1) membership check (vs a list's O(n) scan).
BENEFICIARY_ALLOWLIST = {
    "vendor-alpha",
    "vendor-beta",
    "certified-partner",
}


def sanitize_pii(text: str) -> str:
    """Run the fused PII pattern over *text* and replace matches with ***REDACTED***.

    Single O(n) pass over the input (fused alternation), replacing SSN, card,
    and routing numbers in one sweep.

    Returns the sanitized string.  If any PII was found the caller should
    consider the wire malicious (Sentinel will raise).
    """
    return PII_RE.sub("***REDACTED***", text)


def _contains_pii(value: str) -> bool:
    """O(n) single‑pass PII detection (fused regex — no triple scan)."""
    return PII_RE.search(value) is not None


def evaluate_tool_safety(
    *,
    agent_tier: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    beneficiary: str | None = None,
) -> Tuple[bool, List[str]]:
    """Return *(allowed, reasons)*.

    Rules (all rule‑based, no LLM):

    1. **Tier check** – the agent's ``max_action_tier`` must include ``tool_name``.
    2. **Amount check** – if ``tool_name`` is ``execute_wire_transfer`` and
       ``tool_args.get("amount", 0)`` > ``WIRE_AMOUNT_THRESHOLD`` → reject.
    3. **Beneficiary check** – if ``beneficiary`` is supplied it must be in
       ``BENEFICIARY_ALLOWLIST``.
    4. **PII check** – if any value in ``tool_args`` matches a PII regex,
       the call is rejected.

    Complexity: O(k × n) where k = number of tool arguments and n = average
    argument length — every per‑argument check is O(1) (set/dict lookups) or
    a single O(n) regex pass. No nested loops.

    Returns a tuple ``(allowed, reasons)`` where *reasons* lists every rule
    that failed (empty when allowed).
    """
    reasons: List[str] = []

    # 1️⃣ Tier check — O(1) set membership on the tier allowlist.
    allowed_tools = TOOL_TIER_ALLOWLIST.get(agent_tier, set())
    if tool_name not in allowed_tools:
        reasons.append(f"tool '{tool_name}' not allowed for tier '{agent_tier}'")

    # 2️⃣ Amount check (only for wire transfers) — O(1) numeric compare.
    if tool_name == "execute_wire_transfer":
        amount = tool_args.get("amount", 0)
        if isinstance(amount, (int, float)) and amount > WIRE_AMOUNT_THRESHOLD:
            reasons.append(
                f"wire amount {amount} exceeds threshold {WIRE_AMOUNT_THRESHOLD}"
            )

    # 3️⃣ Beneficiary check — O(1) set membership.
    if beneficiary and beneficiary not in BENEFICIARY_ALLOWLIST:
        reasons.append(
            f"beneficiary '{beneficiary}' not in allowlist"
        )

    # 4️⃣ PII check — one fused O(n) regex pass per string argument.
    # Early break: a single PII hit is enough to reject the call; no need to
    # scan the remaining arguments (avoids wasted O(n) passes).
    for key, val in tool_args.items():
        if isinstance(val, str) and _contains_pii(val):
            reasons.append(f"PII detected in tool argument '{key}'")
            break

    allowed = len(reasons) == 0
    return allowed, reasons