"""Sentinel Causal Analyzer — reverse BFS + Gemini analysis (Phase 3).

Given a trigger span_id, walks the DAG backwards to find candidate
patient-zero nodes (INPUT_INGEST / MEMORY_WRITE).  Uses Gemini for
semantic causal analysis when available; falls back to rule-based
heuristic (first INPUT_INGEST/MEMORY_WRITE encountered).

Returns the identified patient_zero_span_id and a confidence score.

Performance notes:
  * Redis pipeline for batched HGET of node payloads (reduces RTT from O(N) to O(1)).
  * Per-trace_id adjacency cache (TTL 60s) avoids rebuilding parents_of dict on repeated calls.
  * LRU cache on rule-based fallback for identical candidate sets.
"""
from __future__ import annotations

import functools
import json
import re
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import redis.asyncio as redis
from backend.app.config import get_settings

# ---------------------------------------------------------------------------
# Redis keys (must match provenance/dag_manager.py conventions)
# ---------------------------------------------------------------------------
# dag:edges:{trace_id} → Redis SET of "parent_span_id:child_span_id"
# dag:nodes:{trace_id} → Redis HASH span_id → JSON node payload


# ---------------------------------------------------------------------------
# Gemini prompt template
# ---------------------------------------------------------------------------
SENTINEL_CAUSAL_ANALYSIS_PROMPT = """
You are an expert security analyst for the ImmunoAgent system.
You are given a directed acyclic graph (DAG) of provenance nodes representing
the execution trace of a multi-agent system. An anomaly was detected at a
specific trigger span.

Your task: identify the **single most likely patient-zero span** — the
originating input or memory write that introduced the malicious payload
which ultimately caused the anomaly.

Input:
- `trigger_span`: the span where the anomaly was detected (tool call).
- `candidate_nodes`: list of upstream nodes of type INPUT_INGEST or
  MEMORY_WRITE, each with its span_id, node_type, and payload.

Output JSON (exactly this schema):
{
  "patient_zero_span_id": "<span_id>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<concise explanation>"
}

Rules:
- Only consider nodes of type INPUT_INGEST or MEMORY_WRITE as candidates.
- Confidence must be > 0.85 to be accepted; otherwise the rule-based
  fallback (first candidate in reverse topological order) will be used.
- If no candidates exist, return null for patient_zero_span_id and 0.0 confidence.
""".strip()


# ---------------------------------------------------------------------------
# Per-trace adjacency cache (module-level, TTL-based invalidation)
# ---------------------------------------------------------------------------
# Cache: trace_id -> (parents_of_dict, timestamp)
# Avoids rebuilding the child->parents adjacency map on repeated calls
# for the same trace (common during retries or multi-trigger traces).
_TRACE_ADJACENCY_CACHE: Dict[str, Tuple[Dict[str, List[str]], float]] = {}
_ADJACENCY_TTL_SECONDS = 60.0


def _get_adjacency(redis_client: redis.Redis, trace_id: str) -> Dict[str, List[str]]:
    """Get or build the parents_of adjacency map for a trace_id (with TTL cache)."""
    import time
    now = time.time()
    if trace_id in _TRACE_ADJACENCY_CACHE:
        parents_of, cached_at = _TRACE_ADJACENCY_CACHE[trace_id]
        if now - cached_at < _ADJACENCY_TTL_SECONDS:
            return parents_of
    # Cache miss or expired — will be populated by caller
    return {}


async def _build_adjacency_map(
    redis_client: redis.Redis,
    trace_id: str,
) -> Dict[str, List[str]]:
    """Build child->parents adjacency map from Redis edge SET (O(E) scan once)."""
    edge_key = f"dag:edges:{trace_id}"
    edge_pairs = await redis_client.smembers(edge_key)
    parents_of: Dict[str, List[str]] = {}
    for pair in edge_pairs:
        parent, child = pair.split(":", 1)
        parents_of.setdefault(child, []).append(parent)
    import time
    _TRACE_ADJACENCY_CACHE[trace_id] = (parents_of, time.time())
    return parents_of


# ---------------------------------------------------------------------------
# Reverse BFS up the DAG (Redis-backed with batched HGET)
# ---------------------------------------------------------------------------

async def _collect_upstream_nodes(
    redis_client: redis.Redis,
    trace_id: str,
    trigger_span_id: str,
) -> List[Dict[str, Any]]:
    """Walk the DAG backwards from trigger_span_id, returning candidate nodes.

    Candidates = nodes with node_type in {"INPUT_INGEST", "MEMORY_WRITE"}.

    Uses Redis SET for edges (O(1) per edge lookup) and Redis HASH for
    node payloads (O(1) per node fetch via pipelined HMGET).
    """
    node_key = f"dag:nodes:{trace_id}"

    # Get adjacency map (cached or fresh)
    parents_of = _get_adjacency(redis_client, trace_id)
    if not parents_of:
        parents_of = await _build_adjacency_map(redis_client, trace_id)

    # Reverse BFS from trigger_span_id — collect all visited span_ids first
    visited: Set[str] = set()
    queue: deque[str] = deque([trigger_span_id])
    all_spans: List[str] = []

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        all_spans.append(current)
        for parent in parents_of.get(current, []):
            if parent not in visited:
                queue.append(parent)

    # Batched fetch: single pipeline HMGET for all spans (O(1) RTT vs O(N))
    # This is the key optimization — avoids N round-trips to Redis.
    if not all_spans:
        return []

    pipe = redis_client.pipeline()
    for span_id in all_spans:
        pipe.hget(node_key, span_id)
    node_jsons = await pipe.execute()

    # Filter candidates from batched results
    candidates: List[Dict[str, Any]] = []
    for span_id, node_json in zip(all_spans, node_jsons):
        if not node_json:
            continue
        node = json.loads(node_json)
        node_type = node.get("node_type", "")
        if node_type in ("INPUT_INGEST", "MEMORY_WRITE"):
            candidates.append(
                {
                    "span_id": span_id,
                    "node_type": node_type,
                    "payload": node.get("payload", {}),
                }
            )

    return candidates


# ---------------------------------------------------------------------------
# Rule-based fallback (deterministic, no LLM) — memoized for identical inputs
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=256)
def _rule_based_fallback_cached(
    candidates_tuple: Tuple[Tuple[str, str, str], ...],
) -> Tuple[Optional[str], float]:
    """Cached rule-based fallback: first candidate in reverse BFS order.

    Args:
        candidates_tuple: hashable tuple of (span_id, node_type, payload_json_str)
    """
    if not candidates_tuple:
        return None, 0.0
    # candidates are already in reverse BFS order (closest first)
    first_span_id, _, _ = candidates_tuple[0]
    return first_span_id, 0.75  # moderate confidence for rule-based


def _rule_based_fallback(
    candidates: List[Dict[str, Any]],
) -> Tuple[Optional[str], float]:
    """Return the first candidate in reverse BFS order (closest to trigger)."""
    # Convert to hashable tuple for cache key
    candidates_tuple = tuple(
        (c["span_id"], c["node_type"], json.dumps(c["payload"], sort_keys=True))
        for c in candidates
    )
    return _rule_based_fallback_cached(candidates_tuple)


# ---------------------------------------------------------------------------
# Gemini-backed analysis (async, best-effort)
# ---------------------------------------------------------------------------

async def _gemini_analyze(
    trigger_span: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Tuple[Optional[str], float, str]:
    """Call Gemini to identify patient zero. Returns (span_id, confidence, reasoning).

    If Gemini is unavailable or confidence <= 0.85, returns (None, 0.0, "") so
    the caller can fall back.
    """
    settings = get_settings()
    if not settings.gemini_api_key:
        return None, 0.0, "Gemini API key not configured"

    try:
        # Lazy import — the SDK may not be installed
        import google.generativeai as genai
    except Exception:
        return None, 0.0, "Gemini SDK not available"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"{SENTINEL_CAUSAL_ANALYSIS_PROMPT}\n\n"
            f"Trigger span:\n{json.dumps(trigger_span, indent=2)}\n\n"
            f"Candidate nodes:\n{json.dumps(candidates, indent=2)}"
        )
        resp = await model.generate_content_async(prompt)
        text = resp.text or "{}"
        # Extract JSON from response (may be wrapped in markdown fences)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None, 0.0, "Failed to parse Gemini response"
        data = json.loads(m.group(0))
        span_id = data.get("patient_zero_span_id")
        confidence = float(data.get("confidence", 0.0))
        reasoning = data.get("reasoning", "")
        if confidence > 0.85 and span_id:
            return span_id, confidence, reasoning
        return None, 0.0, f"Gemini confidence {confidence:.2f} <= 0.85"
    except Exception as exc:  # broad — any failure falls back
        return None, 0.0, f"Gemini call failed: {exc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def find_patient_zero(
    redis_client: redis.Redis,
    trace_id: str,
    trigger_span_id: str,
    trigger_node: Dict[str, Any],
) -> Tuple[Optional[str], float, str]:
    """Main entry point: find patient zero for a trigger span.

    Returns (patient_zero_span_id, confidence, method) where method is one of:
    - "gemini" (confidence > 0.85)
    - "rule_based" (fallback)
    - "none" (no candidates found)
    """
    # 1. Collect upstream candidates via reverse BFS (with batched HGET)
    candidates = await _collect_upstream_nodes(redis_client, trace_id, trigger_span_id)

    if not candidates:
        return None, 0.0, "none"

    # 2. Try Gemini analysis (best-effort)
    span_id, confidence, reasoning = await _gemini_analyze(trigger_node, candidates)
    if confidence > 0.85:
        return span_id, confidence, "gemini"

    # 3. Rule-based fallback (cached)
    span_id, confidence = _rule_based_fallback(candidates)
    return span_id, confidence, "rule_based"