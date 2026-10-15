"""Local JSON provider: the demo catalogue in ``products.json`` (the default source)."""

from __future__ import annotations

from pathlib import Path

from server.product_store import (
    DEFAULT_PRODUCTS_FILE,
    ProductDataError,
    SearchFilters,
    find_product,
    load_products,
    product_matches,
)
from server.providers.base import Product, ProductProvider, ProviderStatus, normalize_product


class LocalJSONProvider(ProductProvider):
    """Reads ``products.json`` on every call, exactly like the original server did."""

    name = "local"

    def __init__(self, path: Path | str = DEFAULT_PRODUCTS_FILE) -> None:
        self.path = Path(path)

    def status(self) -> ProviderStatus:
        try:
            count = len(load_products(self.path))
        except ProductDataError as exc:
            return ProviderStatus(self.name, available=False, live=False, message=str(exc))
        return ProviderStatus(
            self.name, available=True, live=False,
            message=f"Local demo catalogue ({self.path.name}, {count} products, sample prices).",
        )

    def search(self, filters: SearchFilters) -> list[Product]:
        return [
            normalize_product(p, self.name)
            for p in load_products(self.path)
            if product_matches(p, filters)
        ]

    def get_product(self, product_id: str) -> Product | None:
        raw = find_product(load_products(self.path), product_id)
        return normalize_product(raw, self.name) if raw else None
