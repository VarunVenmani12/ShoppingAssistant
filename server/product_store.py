"""
Product data access layer used by the MCP server.

This module only reads ``products.json`` and filters its contents.
It contains NO LLM logic and performs NO network, shell or arbitrary
file-system access.

It holds the validation helpers and the query functions that work on plain
product dicts. The provider layer (``server/providers``) and the catalogue
service (``server/catalog.py``) build on these; the MCP server uses the
catalogue. The public functions below keep their original signatures so
existing callers and tests continue to work.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# The only data file the tools are allowed to read.
DEFAULT_PRODUCTS_FILE = Path(__file__).resolve().parent.parent / "products.json"

# Friendly aliases -> canonical category names used in products.json
CATEGORY_ALIASES = {
    "laptop": "laptop", "laptops": "laptop", "notebook": "laptop", "notebooks": "laptop",
    "smartphone": "smartphone", "smartphones": "smartphone", "phone": "smartphone",
    "phones": "smartphone", "mobile": "smartphone", "mobiles": "smartphone",
    "mobile phone": "smartphone", "mobile phones": "smartphone",
    "monitor": "monitor", "monitors": "monitor", "display": "monitor", "displays": "monitor",
    "keyboard": "keyboard", "keyboards": "keyboard",
    "mouse": "mouse", "mice": "mouse", "mouses": "mouse",
    "headphone": "headphones", "headphones": "headphones", "headset": "headphones",
    "headsets": "headphones", "earphones": "headphones",
}

PRODUCT_ID_PATTERN = re.compile(r"^[A-Z]{1,3}\d{3}$")
MAX_TEXT_LENGTH = 100
MAX_PRICE = 10_000_000  # sanity limit for price filters (₹1 crore)


class ProductDataError(Exception):
    """Raised when products.json is missing or malformed."""


class InvalidArgumentError(ValueError):
    """Raised when a tool argument fails validation."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_products(path: Path | str = DEFAULT_PRODUCTS_FILE) -> list[dict[str, Any]]:
    """Load and minimally validate the product catalogue."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        raise ProductDataError(f"Product file not found: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ProductDataError(
            f"Product file {path.name} contains invalid JSON (line {exc.lineno})."
        ) from exc

    products = data.get("products") if isinstance(data, dict) else data
    if not isinstance(products, list):
        raise ProductDataError("Product file must contain a list of products.")

    required = {"id", "name", "category", "brand", "price", "specifications", "stores"}
    for index, product in enumerate(products):
        if not isinstance(product, dict) or not required.issubset(product):
            raise ProductDataError(f"Product #{index + 1} is missing required fields.")
    return products


# --------------------------------------------------------------------------- #
# Argument parsing / validation helpers
# --------------------------------------------------------------------------- #
def _clean_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"any", "none", "null", "all"}:
        return None
    if len(text) > MAX_TEXT_LENGTH:
        raise InvalidArgumentError(f"'{field}' is too long (max {MAX_TEXT_LENGTH} characters).")
    return text


def parse_price(value: Any, field: str) -> float | None:
    """Accept 50000, "50000", "₹50,000", "50k", "Rs. 50000"."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().lower()
        if text in {"", "any", "none", "null"}:
            return None
        text = text.replace("₹", "").replace("rs.", "").replace("rs", "")
        text = text.replace("inr", "").replace(",", "").replace(" ", "")
        multiplier = 1
        if text.endswith("k"):
            multiplier, text = 1_000, text[:-1]
        elif text.endswith("lakh"):
            multiplier, text = 100_000, text[:-4]
        try:
            number = float(text) * multiplier
        except ValueError as exc:
            raise InvalidArgumentError(f"'{field}' must be a number, got {value!r}.") from exc
    if number < 0 or number > MAX_PRICE:
        raise InvalidArgumentError(f"'{field}' must be between 0 and {MAX_PRICE}.")
    return number


def parse_size_gb(value: Any, field: str) -> float | None:
    """Convert "16GB", "16", 16, "1TB SSD", "512 GB" into gigabytes."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number < 0:
            raise InvalidArgumentError(f"'{field}' must be positive.")
        return number
    text = str(value).strip().upper()
    if text in {"", "ANY", "NONE", "NULL"}:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(TB|GB)?", text)
    if not match:
        raise InvalidArgumentError(f"'{field}' must look like '16GB' or '1TB', got {value!r}.")
    number = float(match.group(1))
    if match.group(2) == "TB":
        number *= 1024
    return number


def normalize_category(value: Any) -> str | None:
    text = _clean_text(value, "category")
    if text is None:
        return None
    key = text.lower()
    if key not in CATEGORY_ALIASES:
        raise InvalidArgumentError(
            f"Unknown category {text!r}. Valid categories: "
            "laptop, smartphone, monitor, keyboard, mouse, headphones."
        )
    return CATEGORY_ALIASES[key]


def validate_product_id(product_id: Any) -> str:
    text = _clean_text(product_id, "product_id")
    if text is None:
        raise InvalidArgumentError("'product_id' is required (for example 'L001').")
    text = text.upper()
    if not PRODUCT_ID_PATTERN.match(text):
        raise InvalidArgumentError(
            f"'{product_id}' is not a valid product ID. IDs look like 'L001' or 'S002'."
        )
    return text


def format_inr(amount: float | int) -> str:
    """Format a number the Indian way: 1234567 -> '₹12,34,567'."""
    value = int(round(amount))
    digits = str(abs(value))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [tail])
    return f"{'-' if value < 0 else ''}₹{digits}"


def _with_display_prices(store: dict[str, Any]) -> dict[str, Any]:
    return {**store, "price_formatted": format_inr(store["price"])}


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def _summary(product: dict[str, Any]) -> dict[str, Any]:
    """Compact product view returned by search results."""
    specs = product.get("specifications", {})
    prices = [s["price"] for s in product.get("stores", []) if "price" in s]
    summary = {
        "id": product["id"],
        "name": product["name"],
        "category": product["category"],
        "brand": product["brand"],
        "price": product["price"],
        "price_formatted": format_inr(product["price"]),
        "lowest_store_price": min(prices) if prices else product["price"],
        "currency": product.get("currency", "INR"),
        "rating": product.get("rating"),
    }
    for key in ("processor", "ram", "storage", "display", "type"):
        if key in specs:
            summary[key] = specs[key]
    return summary


@dataclass(frozen=True)
class SearchFilters:
    """Validated search filters, shared by search_products() and the providers."""

    category: str | None = None
    max_price: float | None = None
    min_price: float | None = None
    brand: str | None = None
    min_ram_gb: float | None = None
    min_storage_gb: float | None = None
    keyword: str | None = None

    def applied(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def parse_search_filters(
    category: Any = None,
    max_price: Any = None,
    min_price: Any = None,
    brand: Any = None,
    ram: Any = None,
    storage: Any = None,
    keyword: Any = None,
) -> SearchFilters:
    """Validate raw tool arguments. Raises InvalidArgumentError."""
    filters = SearchFilters(
        category=normalize_category(category),
        max_price=parse_price(max_price, "max_price"),
        min_price=parse_price(min_price, "min_price"),
        brand=_clean_text(brand, "brand"),
        min_ram_gb=parse_size_gb(ram, "ram"),
        min_storage_gb=parse_size_gb(storage, "storage"),
        keyword=_clean_text(keyword, "keyword"),
    )
    if (
        filters.max_price is not None
        and filters.min_price is not None
        and filters.min_price > filters.max_price
    ):
        raise InvalidArgumentError("'min_price' cannot be greater than 'max_price'.")
    return filters


def product_matches(product: dict[str, Any], filters: SearchFilters) -> bool:
    """True if one product dict satisfies every filter. RAM/storage are minimums."""
    specs = product.get("specifications", {})
    if filters.category and product["category"] != filters.category:
        return False
    if filters.max_price is not None and product["price"] > filters.max_price:
        return False
    if filters.min_price is not None and product["price"] < filters.min_price:
        return False
    if filters.brand and filters.brand.lower() not in product["brand"].lower():
        return False
    if filters.min_ram_gb is not None:
        product_ram = parse_size_gb(specs.get("ram"), "ram") if specs.get("ram") else None
        if product_ram is None or product_ram < filters.min_ram_gb:
            return False
    if filters.min_storage_gb is not None:
        product_storage = (
            parse_size_gb(specs.get("storage"), "storage") if specs.get("storage") else None
        )
        if product_storage is None or product_storage < filters.min_storage_gb:
            return False
    if filters.keyword:
        haystack = " ".join(
            [product["name"], product["brand"], product.get("description", "")]
            + [str(v) for v in specs.values()]
        ).lower()
        if not all(word in haystack for word in filters.keyword.lower().split()):
            return False
    return True


def build_search_result(
    matches: list[dict[str, Any]], filters: SearchFilters, limit: int = 10
) -> dict[str, Any]:
    """Turn already-filtered product dicts into the search_product tool payload."""
    summaries = sorted((_summary(p) for p in matches), key=lambda p: p["price"])
    result: dict[str, Any] = {
        "filters_applied": filters.applied(),
        "count": len(summaries),
        "products": summaries[:limit],
    }
    if not summaries:
        result["message"] = "No products matched these filters."
    return result


def search_products(
    products: list[dict[str, Any]],
    category: Any = None,
    max_price: Any = None,
    min_price: Any = None,
    brand: Any = None,
    ram: Any = None,
    storage: Any = None,
    keyword: Any = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Filter products. RAM and storage are treated as *minimum* requirements."""
    filters = parse_search_filters(
        category=category, max_price=max_price, min_price=min_price, brand=brand,
        ram=ram, storage=storage, keyword=keyword,
    )
    return build_search_result([p for p in products if product_matches(p, filters)], filters, limit)


def find_product(products: list[dict[str, Any]], product_id: Any) -> dict[str, Any] | None:
    pid = validate_product_id(product_id)
    return next((p for p in products if p["id"].upper() == pid), None)


def compare_prices(products: list[dict[str, Any]], product_id: Any) -> dict[str, Any]:
    pid = validate_product_id(product_id)
    product = find_product(products, pid)
    if product is None:
        return {"error": f"Product '{pid}' was not found.", "product_id": pid}

    stores = [
        _with_display_prices(s) for s in sorted(product.get("stores", []), key=lambda s: s["price"])
    ]
    in_stock = [s for s in stores if s.get("in_stock", True)]
    best = (in_stock or stores)[0] if stores else None
    highest = stores[-1]["price"] if stores else None
    return {
        "product_id": product["id"],
        "name": product["name"],
        "currency": product.get("currency", "INR"),
        "list_price": product["price"],
        "list_price_formatted": format_inr(product["price"]),
        "stores": stores,
        "best_deal": best,
        "max_saving": (highest - best["price"]) if best and highest is not None else 0,
    }


def get_product_details(products: list[dict[str, Any]], product_id: Any) -> dict[str, Any]:
    pid = validate_product_id(product_id)
    product = find_product(products, pid)
    if product is None:
        return {"error": f"Product '{pid}' was not found.", "product_id": pid}
    details = dict(product)
    details["price_formatted"] = format_inr(product["price"])
    details["stores"] = [_with_display_prices(s) for s in product.get("stores", [])]
    return details
