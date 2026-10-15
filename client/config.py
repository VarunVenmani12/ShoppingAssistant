"""
Configuration for the Shopping Assistant client.

Values are read (in priority order) from:
    1. real environment variables
    2. the ``.env`` file in the project root
    3. the defaults below
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def _load_env_file(path: Path) -> dict[str, str]:
    """Tiny .env parser (KEY=VALUE lines, # comments). No extra dependency needed."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    ollama_model: str
    ollama_host: str
    ollama_think: bool
    ollama_temperature: float
    ollama_timeout: float
    max_tool_rounds: int
    log_level: str
    server_log_level: str
    web_host: str
    web_port: int


def load_settings() -> Settings:
    file_values = _load_env_file(ENV_FILE)

    def get(key: str, default: str) -> str:
        return os.environ.get(key) or file_values.get(key) or default

    return Settings(
        ollama_model=get("OLLAMA_MODEL", "qwen3:8b"),
        ollama_host=get("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        ollama_think=_to_bool(get("OLLAMA_THINK", "false")),
        ollama_temperature=float(get("OLLAMA_TEMPERATURE", "0.2")),
        ollama_timeout=float(get("OLLAMA_TIMEOUT", "300")),
        max_tool_rounds=int(get("MAX_TOOL_ROUNDS", "5")),
        log_level=get("LOG_LEVEL", "INFO").upper(),
        server_log_level=get("SERVER_LOG_LEVEL", "INFO").upper(),
        web_host=get("WEB_HOST", "127.0.0.1"),
        web_port=int(get("WEB_PORT", "8000")),
    )
