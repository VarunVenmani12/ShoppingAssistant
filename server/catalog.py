"""
Catalogue service behind the three MCP tools.

The primary provider (first in SHOPPING_PROVIDERS, ``local`` by default)
answers each call, and its products are formatted with the original
``product_store`` functions, so the tool payloads are unchanged.

When more than one provider is enabled, the others are asked too and a
``providers`` list is added to the payload with each provider's outcome.
Their results are not merged into the answer yet (that needs live providers
and comes in a later phase), so nothing is ever fabricated.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from server.product_store import (
    build_search_result,
    compare_prices,
    get_product_details,
    parse_search_filters,
    validate_product_id,
)
from server.providers.base import ProductProvider, ProviderError
from server.providers.registry import build_providers


class ProductCatalog:
    def __init__(self, providers: list[ProductProvider]) -> None:
        if not providers:
            raise ValueError("At least one provider is required.")
        self.providers = list(providers)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ProductCatalog":
        return cls(build_providers(env))

    @property
    def primary(self) -> ProductProvider:
        return self.providers[0]

    def statuses(self) -> list[dict[str, Any]]:
        return [p.status().to_dict() for p in self.providers]

    # ------------------------------------------------------------------ #
    # Tool operations (raise InvalidArgumentError for bad input)
    # ------------------------------------------------------------------ #
    def search(self, **kwargs: Any) -> dict[str, Any]:
        filters = parse_search_filters(**kwargs)
        return self._run(
            lambda p: p.search(filters),
            lambda found: build_search_result([x.to_dict() for x in found], filters),
        )

    def compare(self, product_id: Any) -> dict[str, Any]:
        pid = validate_product_id(product_id)
        return self._run(
            lambda p: p.get_product(pid),
            lambda product: compare_prices([product.to_dict()] if product else [], pid),
        )

    def details(self, product_id: Any) -> dict[str, Any]:
        pid = validate_product_id(product_id)
        return self._run(
            lambda p: p.get_product(pid),
            lambda product: get_product_details([product.to_dict()] if product else [], pid),
        )

    # ------------------------------------------------------------------ #
    def _run(self, operation: Callable[[ProductProvider], Any], render: Callable[[Any], dict]):
        try:
            result = render(operation(self.primary))
        except ProviderError as exc:
            result = {"error": str(exc)}
        if len(self.providers) > 1:
            result["providers"] = [self._outcome(self.primary, None)] + [
                self._outcome(p, operation) for p in self.providers[1:]
            ]
        return result

    @staticmethod
    def _outcome(provider: ProductProvider, operation: Callable | None) -> dict[str, Any]:
        outcome = provider.status().to_dict()
        if operation is None:
            outcome["role"] = "primary"
            return outcome
        outcome["role"] = "secondary"
        try:
            operation(provider)
        except ProviderError as exc:
            outcome["message"] = str(exc)
        else:  # only reachable once a real provider is implemented
            outcome["message"] = "Provider responded; merging its results is not enabled yet."
        return outcome
