"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

APP_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = APP_DIR.parent


@dataclass(frozen=True)
class Settings:
    # "mock" serves the synthetic demo corpus; "production" serves data/processed.
    data_mode: str = os.getenv("DATA_MODE", "mock")
    # "auto" uses sentence-transformers when installed, else LSA. Or force "lsa" / "sentence-transformers".
    embedder: str = os.getenv("EMBEDDER", "auto")
    st_model: str = os.getenv("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # Answer generator: "extractive" (no LLM, default) or "anthropic" (needs requirements-llm.txt + credentials).
    llm_provider: str = os.getenv("LLM_PROVIDER", "extractive")
    llm_model: str = os.getenv("LLM_MODEL", "claude-opus-5-5")
    cors_origins: tuple[str, ...] = tuple(
        o.strip().rstrip("/")
        for o in os.getenv(
            "COSMOS_CORS_ORIGINS",
            os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:5174,http://localhost:4173"),
        ).split(",")
        if o.strip()
    )

    @property
    def data_dir(self) -> Path:
        if os.getenv("DATA_DIR"):  # explicit override, e.g. a staging copy of the corpus
            return Path(os.environ["DATA_DIR"])

        mode = "processed" if self.data_mode == "production" else "mock"
        candidates = (
            Path.cwd() / "data" / mode,
            APP_DIR / "data" / mode,
            PACKAGE_DIR / "data" / mode,
            ROOT / "data" / mode,
        )
        for candidate in candidates:
            if (candidate / "papers.json").is_file():
                return candidate
        return candidates[0]


settings = Settings()
