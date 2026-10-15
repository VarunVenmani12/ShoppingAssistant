"""
Common product model and the provider interface.

Every data source (the local JSON catalogue today, marketplaces later)
implements ``ProductProvider`` and returns ``Product`` / ``Offer`` objects.
``normalize_product`` turns a raw record into a ``Product`` without inventing
data: optional fields that are missing stay empty, and records missing the
fields we cannot do without are rejected.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from server.product_store import (
    CATEGORY_ALIASES,
    InvalidArgumentError,
    ProductDataError,
    SearchFilters,
    parse_price,
)

# Keys normalize_product understands; anything else is kept in Product.extra.
_KNOWN_KEYS = {
    "id", "name", "category", "brand", "price", "currency", "rating",
    "description", "specifications", "stores", "offers",
}


class ProviderError(Exception):
    """Base class for provider failures. Messages are safe to show to users."""


class ProviderNotConfiguredError(ProviderError):
    """The provider is enabled but cannot serve data (no credentials / no live integration)."""


class UnsupportedOperationError(ProviderError):
    """The provider does not implement the requested operation."""


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Offer:
    """One store's price for a product."""

    store: str
    price: int | float
    in_stock: bool = True
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"store": self.store, "price": self.price, "in_stock": self.in_stock}
        if self.url:
            data["url"] = self.url
        return data


@dataclass(frozen=True)
class Product:
    """Provider-independent product record."""

    id: str
    name: str
    price: int | float
    category: str = ""
    brand: str = ""
    currency: str = "INR"
    rating: float | None = None
    description: str = ""
    specifications: dict[str, Any] = field(default_factory=dict)
    offers: tuple[Offer, ...] = ()
    source: str = "local"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Legacy product dict, the shape the existing tool functions expect."""
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "brand": self.brand,
            "price": self.price,
            "currency": self.currency,
        }
        if self.rating is not None:
            data["rating"] = self.rating
        if self.description:
            data["description"] = self.description
        data["specifications"] = dict(self.specifications)
        data["stores"] = [o.to_dict() for o in self.offers]
        data.update(self.extra)
        return data


@dataclass(frozen=True)
class ProviderStatus:
    """What a provider can do right now. ``live`` is only True for real marketplace data."""

    name: str
    available: bool
    live: bool
    message: str
    credentials_present: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "provider": self.name,
            "available": self.available,
            "live": self.live,
            "message": self.message,
        }
        if self.credentials_present is not None:
            data["credentials_present"] = self.credentials_present
        return data


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def _number(value: Any, field_name: str) -> int | float | None:
    """Parse a price, keeping whole numbers as int so JSON output is unchanged."""
    try:
        number = parse_price(value, field_name)
    except InvalidArgumentError as exc:
        raise ProductDataError(str(exc)) from exc
    if number is None:
        return None
    return int(number) if float(number).is_integer() else number


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalize_offer(raw: Any) -> Offer | None:
    """Return an Offer, or None when the entry has no usable store name or price."""
    if not isinstance(raw, dict):
        return None
    store = _text(raw.get("store"))
    try:
        price = _number(raw.get("price"), "store price")
    except ProductDataError:
        return None
    if not store or price is None:
        return None
    in_stock = raw.get("in_stock", True)
    return Offer(
        store=store,
        price=price,
        in_stock=in_stock if isinstance(in_stock, bool) else True,
        url=_text(raw.get("url")) or None,
    )


def normalize_product(raw: Any, source: str = "local") -> Product:
    """Convert one raw record into a Product. Raises ProductDataError if unusable.

    Required: ``id`` and ``name``, plus a ``price`` or at least one valid offer
    (the lowest offer price is then used). Everything else is optional.
    """
    if not isinstance(raw, dict):
        raise ProductDataError("Product record must be an object.")
    product_id, name = _text(raw.get("id")), _text(raw.get("name"))
    if not product_id or not name:
        raise ProductDataError("Product record is missing 'id' or 'name'.")

    raw_offers = raw.get("stores", raw.get("offers")) or []
    offers = tuple(
        o for o in (_normalize_offer(r) for r in (raw_offers if isinstance(raw_offers, list) else []))
        if o is not None
    )

    price = _number(raw.get("price"), f"price of {product_id}")
    if price is None:
        if not offers:
            raise ProductDataError(f"Product {product_id} has no price and no store offers.")
        price = min(o.price for o in offers)

    category = _text(raw.get("category")).lower()
    rating = raw.get("rating")
    specs = raw.get("specifications")

    return Product(
        id=product_id,
        name=name,
        price=price,
        category=CATEGORY_ALIASES.get(category, category),
        brand=_text(raw.get("brand")),
        currency=_text(raw.get("currency")) or "INR",
        rating=rating if isinstance(rating, (int, float)) and not isinstance(rating, bool) else None,
        description=_text(raw.get("description")),
        specifications=dict(specs) if isinstance(specs, dict) else {},
        offers=offers,
        source=source,
        extra={k: v for k, v in raw.items() if k not in _KNOWN_KEYS},
    )


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class ProductProvider(ABC):
    """A source of products. Override the operations the source supports."""

    name: str = "base"

    @abstractmethod
    def status(self) -> ProviderStatus:
        """Report whether this provider can serve data. Must not make network calls."""

    def search(self, filters: SearchFilters) -> list[Product]:
        raise UnsupportedOperationError(f"Provider '{self.name}' does not support search.")

    def get_product(self, product_id: str) -> Product | None:
        raise UnsupportedOperationError(f"Provider '{self.name}' does not support product details.")

    def get_offers(self, product_id: str) -> list[Offer]:
        """Offers for one product. Default: the offers attached to get_product()."""
        product = self.get_product(product_id)
        return list(product.offers) if product else []
