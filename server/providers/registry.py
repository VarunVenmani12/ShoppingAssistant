"""
Provider selection from environment configuration.

``SHOPPING_PROVIDERS`` is a comma-separated list of provider names, e.g.
``local`` (the default) or ``local,amazon,flipkart``. The first provider is
the primary catalogue that answers tool calls; the others are queried too and
their status is reported next to the results. Each provider is enabled
independently; unknown names are ignored with a warning.

Values come from real environment variables first, then the project ``.env``
file (the same precedence as ``client/config.py``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from server.providers.base import ProductProvider
from server.providers.local_json import LocalJSONProvider
from server.providers.marketplace import AmazonProvider, FlipkartProvider

log = logging.getLogger("shopping.server")

ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"
DEFAULT_PROVIDERS = ("local",)
# Environment variables with these prefixes configure providers. The MCP client
# forwards them to the server subprocess.
PROVIDER_ENV_PREFIXES = ("SHOPPING_", "AMAZON_", "FLIPKART_")

PROVIDER_FACTORIES: dict[str, Callable[[Mapping[str, str]], ProductProvider]] = {
    "local": lambda env: LocalJSONProvider(),
    "amazon": AmazonProvider,
    "flipkart": FlipkartProvider,
}


@dataclass(frozen=True)
class ProviderConfig:
    enabled: tuple[str, ...]
    unknown: tuple[str, ...] = ()


def load_provider_env(
    environ: Mapping[str, str] | None = None, env_file: Path | None = ENV_FILE
) -> dict[str, str]:
    """Provider-related settings: environment variables override the .env file."""
    environ = os.environ if environ is None else environ
    merged: dict[str, str] = {}
    if env_file is not None and env_file.exists():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            merged[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in environ.items():
        if value:
            merged[key] = value
    return {k: v for k, v in merged.items() if k.startswith(PROVIDER_ENV_PREFIXES)}


def parse_provider_config(env: Mapping[str, str]) -> ProviderConfig:
    """Parse SHOPPING_PROVIDERS. Order is kept, duplicates and case are ignored."""
    enabled: list[str] = []
    unknown: list[str] = []
    for item in env.get("SHOPPING_PROVIDERS", "").split(","):
        name = item.strip().lower()
        if not name or name in enabled or name in unknown:
            continue
        (enabled if name in PROVIDER_FACTORIES else unknown).append(name)
    return ProviderConfig(tuple(enabled) or DEFAULT_PROVIDERS, tuple(unknown))


def build_providers(env: Mapping[str, str] | None = None) -> list[ProductProvider]:
    """Create the enabled providers, primary first."""
    env = load_provider_env() if env is None else env
    config = parse_provider_config(env)
    if config.unknown:
        log.warning(
            "[SERVER] Ignoring unknown provider(s) in SHOPPING_PROVIDERS: %s (known: %s)",
            ", ".join(config.unknown), ", ".join(PROVIDER_FACTORIES),
        )
    return [PROVIDER_FACTORIES[name](env) for name in config.enabled]
