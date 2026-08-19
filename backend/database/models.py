"""SQLAlchemy models for the 4 core tables — schema mirrors PRD §4.1 exactly.

Architecture note: Redis is the fast-path DAG store (O(1) hash/set ops, PRD
§4.2); Postgres is the durable, queryable source of truth for audit trails,
memory-bank queries, and incident records. The indexes below mirror PRD §4.1
and are the ones the Sentinel hot paths actually hit — see per-index comments
for the query each one serves.
"""
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from .connection import Base


class AgentRegistry(Base):
    """1. Agent Registry — who may call what (PRD §4.1)."""

    __tablename__ = "agent_registry"

    agent_id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    department = Column(String(64), nullable=False)
    allowed_tools = Column(ARRAY(Text), nullable=False, server_default=text("'{}'"))
    # 'READ_ONLY', 'INTERNAL_WRITE', 'CRITICAL_EXEC' — gate for model_armor tiers.
    max_action_tier = Column(String(32), nullable=False, server_default="READ_ONLY")
    is_active = Column(Boolean, nullable=False, server_default=func.true())
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ProvenanceNodes(Base):
    """2. Provenance trace nodes — one row per span in the DAG (PRD §4.1)."""

    __tablename__ = "provenance_nodes"
    __table_args__ = (
        # Index on trace_id (PRD §4.1 idx_provenance_trace): DAG recovery reads
        # all nodes of one trace_id together (index range scan, not full scan).
        Index("idx_provenance_trace", "trace_id"),
        # Index on parent_span_id: the causal back-trace (reverse BFS from a
        # trigger span up the DAG) hot-paths this column. A dedicated index
        # turns each edge lookup into an O(log n) seek instead of scanning
        # the whole table (PRD §4.1 idx_provenance_parent).
        Index("idx_provenance_parent", "parent_span_id"),
    )

    span_id = Column(String(64), primary_key=True)
    trace_id = Column(String(64), nullable=False)
    parent_span_id = Column(
        String(64), ForeignKey("provenance_nodes.span_id"), nullable=True
    )
    agent_id = Column(String(64), ForeignKey("agent_registry.agent_id"), nullable=False)
    # 'INPUT_INGEST', 'THOUGHT', 'MEMORY_READ', 'MEMORY_WRITE', 'TOOL_CALL'
    node_type = Column(String(32), nullable=False)
    payload = Column(JSONB, nullable=False)
    taint_score = Column(Float, nullable=False, server_default=text("0.0"))
    # 'CLEAN', 'SUSPICIOUS', 'QUARANTINED', 'EXCISED'
    taint_status = Column(String(32), nullable=False, server_default="CLEAN")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AgentMemoryBank(Base):
    """3. Memory Bank — pgvector semantic store (PRD §4.1)."""

    __tablename__ = "agent_memory_bank"
    __table_args__ = (
        # Composite (agent_id, quarantine_status): every semantic search MUST
        # pre-filter quarantine_status='ACTIVE' (memory_bank contract). This
        # composite index serves that filter as an O(log n) index seek, so the
        # HNSW vector search only runs on the small pre-filtered set instead
        # of the whole table (PRD §4.1 idx_memory_agent_status).
        Index("idx_memory_agent_status", "agent_id", "quarantine_status"),
        # Memory excision walks memories by source_span_id (PRD §4.1
        # idx_memory_source_span) — used to find EXCISED rows during rollback.
        Index("idx_memory_source_span", "source_span_id"),
    )

    memory_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(String(64), ForeignKey("agent_registry.agent_id"), nullable=False)
    session_id = Column(String(64), nullable=False)
    source_span_id = Column(
        String(64), ForeignKey("provenance_nodes.span_id"), nullable=False
    )
    content_text = Column(Text, nullable=False)
    # 768-dim embedding from Gemini text-embedding-004 (Plan Phase 2).
    embedding = Column(Vector(768), nullable=False)
    # 'ACTIVE', 'QUARANTINED', 'EXCISED' — EXCISED rows are never returned.
    quarantine_status = Column(String(32), nullable=False, server_default="ACTIVE")
    # Python attr is `metadata_` because `metadata` is reserved by SQLAlchemy's
    # Declarative API; the underlying DB column stays named `metadata` (PRD).
    metadata_ = Column("metadata", JSONB, nullable=False, server_default=text("'{}'"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SecurityIncidents(Base):
    """4. Quarantine & security events (PRD §4.1)."""

    __tablename__ = "security_incidents"

    incident_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    # Indexed: incident lookup by trace_id is the demo/audit hot path.
    trace_id = Column(String(64), nullable=False, index=True)
    triggering_agent_id = Column(
        String(64), ForeignKey("agent_registry.agent_id"), nullable=False
    )
    patient_zero_span_id = Column(
        String(64), ForeignKey("provenance_nodes.span_id"), nullable=True
    )
    anomalous_action_payload = Column(JSONB, nullable=False)
    excision_summary = Column(JSONB, nullable=False)
    # Lifecycle: 'DETECTED' -> 'QUARANTINED' -> 'EXCISED' -> 'ROLLED_BACK'.
    # (PRD's DEFAULT 'RESOLVED' contradicts its own state list; a fresh
    # incident is by definition DETECTED, so we start there.)
    status = Column(String(32), nullable=False, server_default="DETECTED")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
