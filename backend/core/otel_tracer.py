"""OpenTelemetry tracer + traceparent / X-Trace-Id extraction (Phase 1).

Provides a tracer instance that can be used to create spans.  Console exporter
wiring will be added when the OTel collector is available; for the hackathon we
only need the tracer object and the W3C traceparent helpers.

Typical usage:
    from backend.core.otel_tracer import get_tracer, extract_traceparent, inject_traceparent
    tracer = get_tracer("gateway")
    with tracer.start_as_current_span("my_operation", attributes={...}) as span:
        ...
"""

from __future__ import annotations

import re
import uuid as _uuid
from typing import Dict, Optional

from opentelemetry import trace as otrace
from opentelemetry.sdk.trace import TracerProvider  # type: ignore  # noqa: F401

# ---------------------------------------------------------------------------
# Minimal tracer provider – we expose the tracer without attaching a processor
# here; the app will add a console exporter later (or via Docker env).
# ---------------------------------------------------------------------------
_provider = TracerProvider()
otrace.set_tracer_provider(_provider)

_tracer = otrace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_tracer(name: str = "immunoagent") -> otrace.Tracer:
    """Return an OpenTelemetry Tracer instance.

    The returned tracer can be used with ``start_as_current_span`` or the
    context manager ``tracer.start_span``.  No exporter is configured in this
    minimal build; add a ``BatchSpanProcessor`` + ``OStreamExporter`` when
    the OTel collector is available.

    Complexity note: this returns the same module‑level ``_tracer`` every
    time (O(1)). We deliberately do NOT construct a new ``TracerProvider``
    per call — that re‑allocates the SDK pipeline (resource, span processors)
    on every request and would blow the fast‑path budget.
    """
    return _tracer


# ---------------------------------------------------------------------------
# W3C traceparent extraction (PRD §4.2 telemetry contract)
# ---------------------------------------------------------------------------

_TRAPARENT_RE = r"^traceparent:\s*([0-9a-f]{32})-([0-9a-f]{16})$"


def extract_traceparent(header_value: str) -> Dict[str, str]:
    """Parse a ``traceparent`` W3C header.

    Example value:
        "traceparent: 0af76594f38705555555555555555555-4bf92f35d69d11ee5555555555555555"

    Returns ``{"trace_id", "span_id"}`` hex strings, or empty strings if the
    header is missing/malformed.
    """
    m = re.match(_TRAPARENT_RE, header_value, flags=re.IGNORECASE)
    if m:
        return {"trace_id": m.group(1), "span_id": m.group(2)}
    return {"trace_id": "", "span_id": ""}


# ---------------------------------------------------------------------------
# Helper: inject traceparent into outbound requests (client side)
# ---------------------------------------------------------------------------

def inject_traceparent(span_context: Optional[object] = None) -> Dict[str, str]:
    """Generate a new traceparent header value for outbound calls.

    If ``span_context`` is provided (e.g. from an existing span), the new
    trace_id will be a derivative; otherwise a random 32‑hex‑char trace_id is
    generated.
    """
    if span_context and hasattr(span_context, "trace_id"):
        trace_id = str(span_context.trace_id).zfill(32)
        span_id = _uuid.uuid4().hex[:16].zfill(16)
    else:
        trace_id = _uuid.uuid4().hex[:32].lower()
        span_id = _uuid.uuid4().hex[:16].lower()
    return {"traceparent": f"{trace_id}-{span_id}"}