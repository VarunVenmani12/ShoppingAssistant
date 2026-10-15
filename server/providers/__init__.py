"""Product data providers (local JSON today; marketplace placeholders for later phases)."""

from server.providers.base import (
    Offer,
    Product,
    ProductProvider,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderStatus,
    UnsupportedOperationError,
    normalize_product,
)
from server.providers.local_json import LocalJSONProvider
from server.providers.marketplace import AmazonProvider, FlipkartProvider
from server.providers.registry import build_providers, load_provider_env, parse_provider_config

__all__ = [
    "AmazonProvider", "FlipkartProvider", "LocalJSONProvider", "Offer", "Product",
    "ProductProvider", "ProviderError", "ProviderNotConfiguredError", "ProviderStatus",
    "UnsupportedOperationError", "build_providers", "load_provider_env",
    "normalize_product", "parse_provider_config",
]
