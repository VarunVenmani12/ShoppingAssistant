"""
Shopping Assistant MCP Server.

Exposes three MCP tools over the stdio transport:

    search_product  - filter the product catalogue
    compare_price   - list store prices for one product
    get_details     - full specifications for one product

This server contains NO LLM logic. It only validates arguments and reads
product data through the provider layer (``catalog.py`` -> ``providers``).
The default provider is the local ``products.json``; see SHOPPING_PROVIDERS. Clients talk to it exclusively
through the Model Context Protocol (JSON-RPC over stdin/stdout).

Run standalone (it will wait for an MCP client on stdin/stdout):
    python -m server.mcp_server
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from server.catalog import ProductCatalog
from server.product_store import InvalidArgumentError, ProductDataError
from server.providers.base import ProviderError

# --------------------------------------------------------------------------- #
# Logging: stdout is reserved for the MCP protocol, so logs go to stderr.
# --------------------------------------------------------------------------- #
try:  # make "₹" printable on Windows consoles
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

LOG_LEVEL = os.getenv("SERVER_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(message)s")
log = logging.getLogger("shopping.server")
log.setLevel(LOG_LEVEL)

mcp = FastMCP(
    name="shopping-assistant",
    instructions=(
        "Product catalogue tools for a demo electronics store. Prices are in INR. "
        "Use search_product to find products, compare_price to compare store prices "
        "for a product ID, and get_details for full specifications."
    ),
    log_level="WARNING",
)


_catalog: ProductCatalog | None = None


def _get_catalog() -> ProductCatalog:
    """Build the catalogue from SHOPPING_PROVIDERS once per server process."""
    global _catalog
    if _catalog is None:
        _catalog = ProductCatalog.from_env()
    return _catalog


def _run(tool: str, operation: str, **kwargs) -> dict[str, Any]:
    """Execute a data query and convert failures into friendly error payloads."""
    shown = {k: v for k, v in kwargs.items() if v is not None}
    log.info("[SERVER] %s called with %s", tool, shown)
    try:
        result = getattr(_get_catalog(), operation)(**kwargs)
    except InvalidArgumentError as exc:
        log.info("[SERVER] %s rejected invalid arguments: %s", tool, exc)
        return {"error": f"Invalid argument: {exc}"}
    except ProductDataError as exc:
        log.error("[SERVER] product data problem: %s", exc)
        return {"error": f"Product catalogue is unavailable: {exc}"}
    except ProviderError as exc:
        log.info("[SERVER] %s provider problem: %s", tool, exc)
        return {"error": str(exc)}
    except Exception:  # never leak stack traces to the client
        log.exception("[SERVER] unexpected error in %s", tool)
        return {"error": "Internal server error while processing the request."}

    if "error" in result:
        log.info("[SERVER] %s -> %s", tool, result["error"])
    elif "count" in result:
        log.info("[SERVER] %s -> %d product(s) found", tool, result["count"])
    else:
        log.info("[SERVER] %s -> OK", tool)
    return result


# --------------------------------------------------------------------------- #
# MCP tools
# --------------------------------------------------------------------------- #
@mcp.tool()
def search_product(
    category: str | None = None,
    max_price: float | None = None,
    min_price: float | None = None,
    brand: str | None = None,
    ram: str | None = None,
    storage: str | None = None,
    keyword: str | None = None,
) -> dict[str, Any]:
    """Search the product catalogue. All parameters are optional filters.

    Args:
        category: One of laptop, smartphone, monitor, keyboard, mouse, headphones.
        max_price: Maximum price in INR (e.g. 50000).
        min_price: Minimum price in INR.
        brand: Brand name, e.g. "Lenovo", "Samsung".
        ram: Minimum RAM, e.g. "16GB".
        storage: Minimum storage, e.g. "512GB" or "1TB".
        keyword: Free-text keyword matched against name, description and specs.
    """
    return _run(
        "search_product", "search",
        category=category, max_price=max_price, min_price=min_price, brand=brand,
        ram=ram, storage=storage, keyword=keyword,
    )


@mcp.tool()
def compare_price(product_id: str) -> dict[str, Any]:
    """Compare the prices of one product across different stores.

    Args:
        product_id: Product ID such as "L001" or "S002".
    """
    return _run("compare_price", "compare", product_id=product_id)


@mcp.tool()
def get_details(product_id: str) -> dict[str, Any]:
    """Get the complete specifications and information for one product.

    Args:
        product_id: Product ID such as "L001" or "S002".
    """
    return _run("get_details", "details", product_id=product_id)


def main() -> None:
    log.info("[SERVER] Shopping Assistant MCP server starting (stdio transport)")
    try:
        names = [p.name for p in _get_catalog().providers]
        log.info("[SERVER] Data providers: %s (primary: %s)", ", ".join(names), names[0])
    except Exception:  # never block start-up; tool calls report the problem
        log.exception("[SERVER] could not initialise data providers")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
