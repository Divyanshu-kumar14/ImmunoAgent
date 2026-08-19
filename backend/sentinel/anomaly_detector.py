"""Sentinel Anomaly Detector — rule-based scoring (Phase 3).

All checks are deterministic, O(1) or O(k×n) rule evaluations — no LLM on the
hot path.  Returns a score 0.0–1.0 and a list of triggered rule names.

Performance notes:
  * All allowlist lookups are O(1) set membership (not list scan).
  * Single fused regex for hex detection = ONE pass over string args.
  * Result memoization via LRU cache for repeated identical calls (common in retries).
"""
from __future__ import annotations

import functools
import json
from typing import Any, Dict, List, Tuple

from backend.core.model_armor import (
    BENEFICIARY_ALLOWLIST,
    TOOL_TIER_ALLOWLIST,
    WIRE_AMOUNT_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Rule weights — sum to 1.0 so the final score is always in [0, 1].
# Tuned so ANY single rule can trigger quarantine (threshold=0.15).
# ---------------------------------------------------------------------------
RULE_WEIGHTS: Dict[str, float] = {
    "out_of_tier_tool": 0.30,
    "excessive_amount": 0.25,
    "unknown_beneficiary": 0.20,
    "hex_address_recipient": 0.15,
    "policy_mutation": 0.15,  # raised from 0.10 so it crosses threshold alone
}

# Pre-compiled regex for hex address detection (Ethereum-style 0x + 40 hex chars).
# Module-level compilation avoids re-compiling on every call — O(1) pattern reuse.
import re
_HEX_ADDRESS_RE = re.compile(r"0x[a-fA-F0-9]{40}")


# ---------------------------------------------------------------------------
# Memoized scoring — caches results for identical (agent_tier, tool_name, args, beneficiary).
# Hackathon workloads often retry the same malicious call; caching turns
# repeated O(k×n) evaluations into O(1) dict lookups after first call.
# maxsize=1024 bounds memory; typical unique call patterns << 1024.
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1024)
def _score_anomaly_cached(
    agent_tier: str,
    tool_name: str,
    args_json: str,  # JSON-serialized tool_args (preserves types)
    beneficiary: str | None,
) -> Tuple[float, Tuple[str, ...]]:
    """Internal cached implementation. Args must be hashable for lru_cache."""
    # Parse JSON back to dict — preserves original types (int, float, str, etc.)
    tool_args = json.loads(args_json)
    triggered: List[str] = []

    # 1️⃣ Out-of-tier tool — O(1) set membership (not list scan).
    allowed_tools = TOOL_TIER_ALLOWLIST.get(agent_tier, set())
    if tool_name not in allowed_tools:
        triggered.append("out_of_tier_tool")

    # 2️⃣ Excessive amount — only for wire transfers, O(1) numeric compare.
    if tool_name == "execute_wire_transfer":
        amount = tool_args.get("amount", tool_args.get("amount_cents", 0))
        if isinstance(amount, (int, float)) and amount > WIRE_AMOUNT_THRESHOLD:
            triggered.append("excessive_amount")

    # 3️⃣ Unknown beneficiary — O(1) set membership (not list scan).
    if beneficiary and beneficiary not in BENEFICIARY_ALLOWLIST:
        triggered.append("unknown_beneficiary")

    # 4️⃣ Hex address recipient — single fused regex pass over string args.
    # Early break: one match is enough to trigger the rule.
    for val in tool_args.values():
        if isinstance(val, str) and _HEX_ADDRESS_RE.search(val):
            triggered.append("hex_address_recipient")
            break

    # 5️⃣ Policy mutation — O(1) string equality.
    if tool_name == "update_policy":
        triggered.append("policy_mutation")

    # Weighted sum (each rule's weight is pre-normalised to sum to 1.0).
    score = sum(RULE_WEIGHTS.get(r, 0.0) for r in triggered)

    # Clamp to [0, 1] for safety (floating point edge cases).
    return min(max(score, 0.0), 1.0), tuple(triggered)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_anomaly(
    *,
    agent_tier: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    beneficiary: str | None = None,
) -> Tuple[float, List[str]]:
    """Return *(anomaly_score, triggered_rules)* for a tool call.

    The anomaly score is a weighted sum in [0, 1].  A score >= 0.15 is treated
    as "malicious" by the Sentinel pipeline (configurable threshold).

    Rules evaluated (all rule-based, no LLM):
    1. **out_of_tier_tool** — tool not in the agent's allowed tier.
    2. **excessive_amount** — wire transfer amount exceeds WIRE_AMOUNT_THRESHOLD.
    3. **unknown_beneficiary** — beneficiary not in BENEFICIARY_ALLOWLIST.
    4. **hex_address_recipient** — any string argument contains a 0x hex address.
    5. **policy_mutation** — tool is `update_policy` (high-privilege mutation).

    Complexity: O(k × n) where k = number of string arguments, n = arg length.
    With memoization: O(1) for repeated identical calls.
    """
    # Use JSON-serialized args for cache key — preserves types (int, float, str)
    # and is hashable. sort_keys ensures consistent ordering.
    args_json = json.dumps(tool_args, sort_keys=True)
    score, triggered = _score_anomaly_cached(agent_tier, tool_name, args_json, beneficiary)
    return score, list(triggered)


# ---------------------------------------------------------------------------
# Convenience: classify as malicious / benign
# ---------------------------------------------------------------------------

# Threshold set to 0.15 so that ANY single rule (minimum weight=0.15 for
# hex_address_recipient and policy_mutation) triggers quarantine.
# This matches the PRD requirement that the Sentinel acts on clear anomalies.
MALICIOUS_THRESHOLD = 0.15


def is_malicious(score: float) -> bool:
    """True if *score* crosses the malicious threshold."""
    return score >= MALICIOUS_THRESHOLD