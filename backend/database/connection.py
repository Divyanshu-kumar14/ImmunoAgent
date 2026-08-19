"""Async engine factory + Postgres extension bootstrap (PRD §4.1).

Extensions must exist BEFORE ``Base.metadata.create_all()`` runs, otherwise
the ``Vector(768)`` column DDL fails. We create them in an explicit startup
step (``create_extensions``) rather than a connect-event listener because
asyncpg connections cannot be awaited inside SQLAlchemy's synchronous
connect event.
"""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def init_db_engine(database_url: str) -> AsyncEngine:
    """Create the async engine with production-safe pooling defaults.

    Performance rationale:
    - Pooled connections are reused across requests, avoiding the TCP + auth
      handshake per request (the dominant latency cost on hot paths).
    - ``pool_pre_ping`` validates idle connections before reuse so a Postgres
      restart does not surface as random connection-drop errors later.
    - ``max_overflow`` absorbs short spikes without starving the pool.
    """
    return create_async_engine(
        database_url,
        echo=False,
        future=True,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )


async def create_extensions(engine: AsyncEngine) -> None:
    """Idempotently enable required Postgres extensions (PRD §4.1).

    Safe to call on every startup: ``IF NOT EXISTS`` makes it a no-op once the
    extensions are present.
    """
    async with engine.begin() as conn:
        # uuid-ossp: gen_random_uuid() for memory/incident primary keys.
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        # vector: pgvector HNSW index for 768-dim semantic search (PRD §4.1).
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "vector"'))
