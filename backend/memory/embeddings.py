"""Gemini embedding client with mock fallback (Phase 2).

Provides :func:`get_embedding` which returns a 768‑dim float vector.
In production the Gemini SDK is used; when the API key is missing or the
call fails a deterministic hash‑based mock is returned so the demo never
hard‑depends on an LLM quota.

The mock is seeded per‑trace_id so the same trace always gets the same vector
(useful for deterministic testing without hitting the quota limit).
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any, List

# The Gemini SDK may not be installed; guard the import gracefully.
try:  # pragma: no cover – run‑time guard
    import google.generativeai as genai  # type: ignore  # noqa: F401
except Exception:  # broad – missing module or import error
    genai = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Mock embedding – deterministic hash based on the trace_id so calls within
# the same trace are consistent but different across traces.
# ---------------------------------------------------------------------------

# Model handle is built lazily and reused (see _get_model).
_model_cache: dict[str, Any] = {}


def _get_model() -> Any:
    """Lazily build and cache the Gemini embedding model handle.

    Why (performance): constructing a model object per call re-parses the
    SDK config and re-allocates the handle — O(setup) work on every
    memory-write. A module-level cache turns the second call onward into an
    O(1) dict lookup.
    """
    if "_model" not in _model_cache and genai is not None:
        _model_cache["_model"] = genai.EmbeddingModel("text-embedding-004")
    return _model_cache.get("_model")


@lru_cache(maxsize=4096)
def _mock_embedding(trace_id: str, dim: int = 768) -> List[float]:
    """Deterministic 768‑dim vector derived from the trace_id hash.

    Memoized with ``lru_cache`` because the vector is a pure function of the
    trace_id: re-embedding the same trace (e.g. repeated queries in a test
    loop) becomes an O(1) hash lookup instead of re-running the 768-iteration
    build. ``maxsize=4096`` keeps memory bounded.

    We cycle through the 32‑byte SHA‑256 digest so that any length trace_id
    produces exactly ``dim`` floats in ``[-1, 1]``.
    """
    h = hashlib.sha256(trace_id.encode()).digest()
    vector: List[float] = []
    for i in range(dim):
        byte = h[i % 32]  # cycle through the 32‑byte block
        v = (byte / 255.0) * 2.0 - 1.0
        vector.append(round(v, 4))
    return vector


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_embedding(
    text: str,
    trace_id: str | None = None,
    *,
    use_mock: bool | None = None,
) -> List[float]:
    """Return a 768‑dim embedding for *text*.

    Parameters
    ----------
    text : str
        The input document / query.
    trace_id : str | None
        OpenTelemetry trace_id used to seed the mock vector.  If ``None`` a
        fresh random‑looking vector is returned (hash of empty string).
    use_mock : bool | None
        force mock mode (True) or production mode (False).  Defaults to ``True``
        when ``GEMINI_API_KEY`` is not set, otherwise ``False``.

    Returns
    -------
    List[float]
        Exactly 768 floating‑point values.
    """
    from backend.app.config import get_settings

    if use_mock is None:
        settings = get_settings()
        use_mock = not bool(settings.gemini_api_key)

    if not use_mock and genai is not None:
        # Production path – block until Gemini returns.
        try:
            # Reuse the cached model handle (O(1) after first call — _get_model).
            model = _get_model()
            resp = model.encode(text)
            # Normalise to 768‑dim list[float] if needed.
            vec = resp.get("embedding") or resp["values"]
            if isinstance(vec, list) and len(vec) == 768:
                return vec
            # If the SDK returns something else, fall back to mock.
            raise ValueError("unexpected embedding shape")
        except Exception as exc:  # broad – any failure falls back to mock
            print(f"[embedding] Gemini call failed ({exc}), falling back to mock")
    # Mock path (or fallback from production failure).
    return _mock_embedding(trace_id or hashlib.sha256(b"").hexdigest()[:32], dim=768)