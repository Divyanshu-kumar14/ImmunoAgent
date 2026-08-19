"""Ephemeral JWT auth (HS256, 5‑min TTL, jti deny‑list in Redis for revocation).

Phase 1 core module — used by the gateway interceptor to attach a JWT to
outbound calls and to verify inbound tokens before registry/DAG/armor checks.

Performance notes (why this module is shaped this way):
  * A single module‑level Redis client is shared across all calls instead of
    opening a new connection per request. ``redis.from_url()`` allocates a
    whole connection pool (socket + handshake); doing that per call turns an
    O(1) lookup into O(connection setup) and dominates the fast‑path budget.
    With a shared pool every ``exists``/``setex`` is a reused connection, so
    auth adds only one O(1) Redis round‑trip to the hot path (< 25 ms P95).
  * No in‑memory memoization of verified JWTs is used on purpose: revocation
    must be effective *immediately* (Sentinel quarantines agents mid‑flight).
    A positive cache would delay revocation until TTL expiry — a security
    hole, not a performance win. Deny‑list lookups are already O(1).
"""
from __future__ import annotations

import time
import uuid as _uuid
from typing import Dict, Optional

import jwt  # PyJWT
import redis.asyncio as redis

from backend.app.config import get_settings

_SETTINGS = get_settings()

# --- Shared async Redis client (single connection pool, created lazily) ---
# Why module‑level: see module docstring. Created lazily (not at import time)
# so tests that inject their own client never touch it, and so we don't open
# a socket before the event loop is ready.
_redis: Optional[redis.Redis] = None


def _get_redis(redis_client: Optional[redis.Redis] = None) -> redis.Redis:
    """Return the shared Redis client, or the caller‑supplied one (tests)."""
    global _redis
    if redis_client is not None:
        return redis_client
    if _redis is None:
        # decode_responses=True: all keys we store are plain strings (jti).
        _redis = redis.from_url(_SETTINGS.redis_url, decode_responses=True)
    return _redis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jwt_payload(
    agent_id: str,
    jti: str,
    exp_offset: int = 300,
    max_action_tier: str = "READ_ONLY",
) -> Dict:
    """Build the JWT payload.

    * ``exp`` = now + 5 minutes (300 s) — matches the "ephemeral" requirement.
    * ``jti`` = unique identifier for revocation via Redis deny‑list.
    * ``max_action_tier`` = claims-based authz hint. Carrying the tier in the
      token (instead of a DB lookup per request) keeps the fast path O(1):
      the gateway reads the claim straight from the decoded payload with no
      extra round-trip (model_armor still enforces the tier allowlist).
    """
    now = int(time.time())
    return {
        "agent_id": agent_id,
        "jti": jti,
        "iat": now,
        "exp": now + exp_offset,
        "max_action_tier": max_action_tier,
    }


# ---------------------------------------------------------------------------
# Issue token
# ---------------------------------------------------------------------------

async def issue_jwt(
    agent_id: str,
    max_action_tier: str = "READ_ONLY",
    redis_client: Optional[redis.Redis] = None,
) -> str:
    """Create an ephemeral JWT for *agent_id*.

    The 5‑minute lifetime lives in the JWT ``exp`` claim — nothing is written
    to Redis here. The Redis deny‑list only ever contains *revoked* jtis
    (added by :func:`revoke_jwt`), so a freshly issued token passes
    :func:`verify_jwt` immediately. This is the deny‑list contract: absence
    in Redis == valid.

    ``max_action_tier`` is embedded as a claim so the gateway can authorize
    the tool call without a per-request DB lookup (O(1) fast path).

    Returns:
        Encoded HS256 JWT string.
    """
    jti = str(_uuid.uuid4())
    payload = _make_jwt_payload(agent_id, jti, max_action_tier=max_action_tier)

    token = jwt.encode(
        payload,
        _SETTINGS.jwt_secret,  # dedicated HS256 secret (>=32 bytes; see config.py)
        algorithm="HS256",
    )
    return token


# ---------------------------------------------------------------------------
# Verify token (fast‑path, no LLM)
# ---------------------------------------------------------------------------

async def verify_jwt(token: str, redis_client: Optional[redis.Redis] = None) -> Dict:
    """Verify an inbound JWT and check revocation.

    Returns the decoded payload if valid and not revoked.
    Raises ``ValueError`` if invalid, missing ``jti``, or revoked.

    Fast‑path logic (all O(1) / constant work):
      1. Decode HS256 with the app secret (constant time for fixed‑size token).
      2. Check ``jti`` against the Redis deny‑list — one O(1) ``exists``.
      3. Present in deny‑list → reject; absent → valid.
    """
    try:
        payload = jwt.decode(
            token,
            _SETTINGS.jwt_secret,
            algorithms=["HS256"],
        )
    except jwt.InvalidTokenError as exc:
        raise ValueError("invalid JWT") from exc

    jti = payload.get("jti")
    if not jti:
        raise ValueError("JWT missing jti")

    # Deny‑list check — one O(1) hash lookup on the shared pooled connection.
    is_revoked = await _get_redis(redis_client).exists(jti)
    if is_revoked:
        raise ValueError("revoked JWT")

    return payload


# ---------------------------------------------------------------------------
# Revocation (manual — Sentinel calls this on quarantine)
# ---------------------------------------------------------------------------

async def revoke_jwt(jti: str, redis_client: Optional[redis.Redis] = None) -> None:
    """Force‑revoke a JWT by its jti (e.g. on quarantine).

    Writes the jti to the deny‑list with a 5‑minute TTL matching the token
    lifetime — the entry becomes irrelevant the moment the token expires, so
    Redis never accumulates stale keys.
    """
    await _get_redis(redis_client).setex(jti, 300, "revoked")