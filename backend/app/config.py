"""Central application configuration (pydantic-settings v2).

Performance note: constructing a `Settings` object re-reads and parses the
environment for every field, which is O(fields) work per instantiation. We
therefore expose a single cached singleton via `get_settings()` so hot paths
(every gateway request) read config with an O(1) dict lookup instead of
re-parsing env vars each time.
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration.

    Key names map 1:1 to ``.env.example`` (no prefix). A prefix such as
    ``IMMUNO_`` would silently break the documented ``.env`` contract, so we
    keep the bare names as the single source of truth.
    """

    app_name: str = "ImmunoAgent"
    debug: bool = False

    # Default is Postgres + pgvector (PRD §4.1). The Memory Bank's semantic
    # search runs on pgvector's HNSW index (approximate O(log n) search),
    # pre-filtered by quarantine_status — see models.py idx_memory_agent_status.
    database_url: str = Field(
        default="postgresql://immunoagent:secret@localhost:5432/immunoagent"
    )

    # Redis is the FAST-PATH store: the DAG (hash) + edges (set) + JWT jti
    # deny-list are all O(1) hash/set lookups, which is what keeps the
    # rule-based security fast path well under the 25 ms P95 budget.
    redis_url: str = Field(default="redis://localhost:6379/0")

    # Empty key => mock/rule-based fallback everywhere; the demo never
    # hard-depends on the LLM (Plan: Risks & Mitigations).
    gemini_api_key: str = Field(default="")

    # HMAC secret for ephemeral JWTs (core/auth.py). Kept separate from
    # app_name: PyJWT warns (and RFC 7518 §3.2 agrees) that keys shorter than
    # 32 bytes weaken HS256. The default is a long dev value; override in
    # production via .env — never commit the real one.
    jwt_secret: str = Field(
        default="dev-jwt-secret-change-me-in-prod-0123456789abcdef"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)  # parse env exactly once per process; cached for hot paths
def get_settings() -> Settings:
    return Settings()
