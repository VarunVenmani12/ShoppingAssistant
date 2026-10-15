"""Tests for the provider layer: selection, normalization, placeholders, catalogue.

No credentials and no network are needed. The marketplace providers are
placeholders; the tests block socket connections to prove they never call out.

Run:  python -m pytest tests/test_providers.py -v
"""

import asyncio
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from client.mcp_client import ShoppingMCPClient
from server.catalog import ProductCatalog
from server.product_store import (
    InvalidArgumentError,
    ProductDataError,
    compare_prices,
    get_product_details,
    load_products,
    parse_search_filters,
    search_products,
)
from server.providers import (
    AmazonProvider,
    FlipkartProvider,
    LocalJSONProvider,
    ProductProvider,
    ProviderNotConfiguredError,
    ProviderStatus,
    UnsupportedOperationError,
    build_providers,
    load_provider_env,
    normalize_product,
    parse_provider_config,
)

AMAZON_CREDS = {
    "AMAZON_CREATORS_CLIENT_ID": "id-123",
    "AMAZON_CREATORS_CLIENT_SECRET": "secret-xyz",
    "AMAZON_PARTNER_TAG": "tag-21",
}
FLIPKART_CREDS = {"FLIPKART_AFFILIATE_ID": "aff", "FLIPKART_AFFILIATE_TOKEN": "token-abc"}


def no_network():
    """Patch socket connections so any network attempt fails the test."""
    def refuse(*args, **kwargs):
        raise AssertionError("provider attempted a network connection")
    return mock.patch.multiple(socket.socket, connect=refuse, connect_ex=refuse)


class ProviderSelectionTests(unittest.TestCase):
    def test_default_is_local(self):
        self.assertEqual(parse_provider_config({}).enabled, ("local",))
        self.assertEqual(parse_provider_config({"SHOPPING_PROVIDERS": "  , "}).enabled, ("local",))
        self.assertEqual([p.name for p in build_providers({})], ["local"])

    def test_order_case_and_duplicates(self):
        config = parse_provider_config({"SHOPPING_PROVIDERS": " Local , FLIPKART,amazon,local "})
        self.assertEqual(config.enabled, ("local", "flipkart", "amazon"))
        self.assertEqual(config.unknown, ())

    def test_providers_enabled_independently(self):
        self.assertEqual(
            [p.name for p in build_providers({"SHOPPING_PROVIDERS": "local,flipkart"})],
            ["local", "flipkart"],
        )
        types = [type(p) for p in build_providers({"SHOPPING_PROVIDERS": "amazon"})]
        self.assertEqual(types, [AmazonProvider])

    def test_unknown_providers_ignored(self):
        config = parse_provider_config({"SHOPPING_PROVIDERS": "local,ebay,ebay"})
        self.assertEqual(config.enabled, ("local",))
        self.assertEqual(config.unknown, ("ebay",))
        self.assertEqual(parse_provider_config({"SHOPPING_PROVIDERS": "ebay"}).enabled, ("local",))

    def test_env_overrides_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "# comment\nSHOPPING_PROVIDERS=local,amazon\nFLIPKART_AFFILIATE_ID='x'\n"
                "OLLAMA_MODEL=qwen3:8b\n",
                encoding="utf-8",
            )
            from_file = load_provider_env({}, env_file)
            self.assertEqual(from_file["SHOPPING_PROVIDERS"], "local,amazon")
            self.assertEqual(from_file["FLIPKART_AFFILIATE_ID"], "x")
            self.assertNotIn("OLLAMA_MODEL", from_file)  # only provider keys

            overridden = load_provider_env({"SHOPPING_PROVIDERS": "local"}, env_file)
            self.assertEqual(overridden["SHOPPING_PROVIDERS"], "local")

    def test_missing_env_file(self):
        self.assertEqual(load_provider_env({}, Path(tempfile.gettempdir()) / "no-such.env"), {})


class NormalizationTests(unittest.TestCase):
    def test_local_catalogue_round_trips_unchanged(self):
        for raw in load_products():
            self.assertEqual(normalize_product(raw).to_dict(), raw, raw["id"])

    def test_normalizes_values(self):
        product = normalize_product(
            {
                "id": "A1", "name": " Phone X ", "category": "Mobiles", "price": "₹12,999",
                "rating": "great", "offers": [{"store": "Shop", "price": "12,499", "url": "https://x"}],
                "warranty": "1 year",
            },
            source="amazon",
        )
        self.assertEqual(product.name, "Phone X")
        self.assertEqual(product.category, "smartphone")
        self.assertEqual(product.price, 12999)
        self.assertIsInstance(product.price, int)
        self.assertIsNone(product.rating)  # invalid rating dropped, not guessed
        self.assertEqual(product.source, "amazon")
        self.assertEqual(product.offers[0].price, 12499)
        self.assertTrue(product.offers[0].in_stock)
        self.assertEqual(product.to_dict()["stores"][0]["url"], "https://x")
        self.assertEqual(product.extra, {"warranty": "1 year"})

    def test_missing_optional_fields_get_empty_defaults(self):
        product = normalize_product({"id": "B1", "name": "Bare", "price": 100})
        self.assertEqual(
            (product.category, product.brand, product.description, product.currency),
            ("", "", "", "INR"),
        )
        self.assertIsNone(product.rating)
        self.assertEqual(product.specifications, {})
        self.assertEqual(product.offers, ())
        data = product.to_dict()
        self.assertNotIn("rating", data)
        self.assertNotIn("description", data)

    def test_price_from_lowest_offer_when_missing(self):
        product = normalize_product(
            {"id": "C1", "name": "X", "stores": [{"store": "A", "price": 500}, {"store": "B", "price": 450}]}
        )
        self.assertEqual(product.price, 450)

    def test_invalid_offers_are_skipped(self):
        product = normalize_product(
            {"id": "D1", "name": "X", "price": 10, "stores": [
                {"store": "", "price": 5}, {"store": "A"}, {"store": "B", "price": "free"},
                "junk", {"store": "C", "price": 9, "in_stock": "maybe"},
            ]}
        )
        self.assertEqual([o.store for o in product.offers], ["C"])
        self.assertTrue(product.offers[0].in_stock)

    def test_missing_required_fields_rejected(self):
        for raw in (
            {"name": "No id", "price": 1},
            {"id": "E1", "price": 1},
            {"id": "E2", "name": "No price, no offers"},
            {"id": "E3", "name": "Bad price", "price": "cheap"},
            {"id": "E4", "name": "Negative", "price": -1},
            "not a dict",
        ):
            with self.assertRaises(ProductDataError, msg=raw):
                normalize_product(raw)


class UnsupportedOperationTests(unittest.TestCase):
    class StatusOnly(ProductProvider):
        name = "minimal"

        def status(self):
            return ProviderStatus(self.name, available=True, live=False, message="ok")

    def test_default_operations_raise_unsupported(self):
        provider = self.StatusOnly()
        with self.assertRaises(UnsupportedOperationError):
            provider.search(parse_search_filters())
        with self.assertRaises(UnsupportedOperationError):
            provider.get_product("L001")
        with self.assertRaises(UnsupportedOperationError):
            provider.get_offers("L001")

    def test_catalog_reports_unsupported_primary_as_error(self):
        result = ProductCatalog([self.StatusOnly()]).search(category="laptop")
        self.assertIn("does not support search", result["error"])


class UnconfiguredProviderTests(unittest.TestCase):
    def test_status_without_credentials(self):
        for provider in (AmazonProvider({}), FlipkartProvider({})):
            status = provider.status()
            self.assertFalse(status.available)
            self.assertFalse(status.live)
            self.assertFalse(status.credentials_present)
            self.assertIn("not configured", status.message)

    def test_credentials_present_is_still_not_live(self):
        for provider in (AmazonProvider(AMAZON_CREDS), FlipkartProvider(FLIPKART_CREDS)):
            status = provider.status()
            self.assertTrue(status.credentials_present)
            self.assertFalse(status.available)
            self.assertFalse(status.live)
            self.assertIn("not implemented", status.message)

    def test_partial_credentials_count_as_missing(self):
        partial = dict(AMAZON_CREDS, AMAZON_CREATORS_CLIENT_SECRET="  ")
        self.assertFalse(AmazonProvider(partial).credentials_present)

    def test_secrets_never_exposed(self):
        for provider, creds in ((AmazonProvider(AMAZON_CREDS), AMAZON_CREDS),
                                (FlipkartProvider(FLIPKART_CREDS), FLIPKART_CREDS)):
            text = json.dumps(provider.status().to_dict())
            try:
                provider.search(parse_search_filters())
            except ProviderNotConfiguredError as exc:
                text += str(exc)
            for value in creds.values():
                self.assertNotIn(value, text)

    def test_operations_raise_without_network(self):
        filters = parse_search_filters(category="laptop")
        with no_network():
            for provider in (AmazonProvider(AMAZON_CREDS), FlipkartProvider({})):
                with self.assertRaises(ProviderNotConfiguredError):
                    provider.search(filters)
                with self.assertRaises(ProviderNotConfiguredError):
                    provider.get_product("L001")
                with self.assertRaises(ProviderNotConfiguredError):
                    provider.get_offers("L001")


class LocalProviderTests(unittest.TestCase):
    def test_status_and_queries(self):
        provider = LocalJSONProvider()
        self.assertTrue(provider.status().available)
        self.assertFalse(provider.status().live)  # demo data is never "live"
        laptops = provider.search(parse_search_filters(category="laptop", max_price=50000))
        self.assertTrue(laptops and all(p.category == "laptop" for p in laptops))
        self.assertEqual(provider.get_product("L001").name, "Lenovo IdeaPad Slim 3")
        self.assertIsNone(provider.get_product("L999"))
        self.assertGreaterEqual(len(provider.get_offers("L001")), 2)

    def test_missing_file(self):
        provider = LocalJSONProvider(Path(tempfile.gettempdir()) / "missing-products-1234.json")
        self.assertFalse(provider.status().available)
        with self.assertRaises(ProductDataError):
            provider.search(parse_search_filters())


class CatalogCompatibilityTests(unittest.TestCase):
    """The default catalogue must return exactly what the original functions return."""

    SEARCHES = [
        {}, {"category": "laptop", "max_price": 50000}, {"category": "phone", "max_price": "30k"},
        {"category": "laptop", "ram": "16GB", "storage": "1TB"}, {"brand": "logitech", "keyword": "wireless"},
        {"category": "laptop", "max_price": 1000},
    ]

    def setUp(self):
        self.catalog = ProductCatalog.from_env({})
        self.products = load_products()

    def test_search_matches_legacy(self):
        for args in self.SEARCHES:
            self.assertEqual(self.catalog.search(**args), search_products(self.products, **args), args)

    def test_compare_and_details_match_legacy(self):
        for pid in ("L001", "s002", "L999"):
            self.assertEqual(self.catalog.compare(pid), compare_prices(self.products, pid))
            self.assertEqual(self.catalog.details(pid), get_product_details(self.products, pid))

    def test_invalid_arguments_still_raise(self):
        with self.assertRaises(InvalidArgumentError):
            self.catalog.search(category="spaceship")
        with self.assertRaises(InvalidArgumentError):
            self.catalog.compare("../etc/passwd")

    def test_no_providers_key_by_default(self):
        self.assertNotIn("providers", self.catalog.search(category="laptop"))


class MultiProviderCatalogTests(unittest.TestCase):
    def test_secondary_status_reported_and_results_unchanged(self):
        catalog = ProductCatalog.from_env({"SHOPPING_PROVIDERS": "local,amazon,flipkart"})
        with no_network():
            result = catalog.search(category="laptop", max_price=50000)
            compare = catalog.compare("L001")
        legacy = search_products(load_products(), category="laptop", max_price=50000)
        self.assertEqual(result["products"], legacy["products"])
        notes = {n["provider"]: n for n in result["providers"]}
        self.assertEqual(notes["local"]["role"], "primary")
        for name in ("amazon", "flipkart"):
            self.assertEqual(notes[name]["role"], "secondary")
            self.assertFalse(notes[name]["live"])
            self.assertFalse(notes[name]["available"])
            self.assertIn("not configured", notes[name]["message"])
        self.assertEqual(len(compare["stores"]), 3)  # no fabricated marketplace offers
        self.assertEqual(len(compare["providers"]), 3)

    def test_unconfigured_primary_returns_error_not_data(self):
        catalog = ProductCatalog.from_env({"SHOPPING_PROVIDERS": "flipkart"})
        with no_network():
            for result in (catalog.search(category="laptop"), catalog.details("L001")):
                self.assertIn("not configured", result["error"])
                self.assertNotIn("products", result)

    def test_secondary_that_responds_is_reported_but_not_merged(self):
        class FakeLive(ProductProvider):
            name = "fake"

            def status(self):
                return ProviderStatus(self.name, available=True, live=True, message="ok")

            def search(self, filters):
                return [normalize_product({"id": "F001", "name": "Fake", "price": 1})]

        catalog = ProductCatalog([LocalJSONProvider(), FakeLive()])
        result = catalog.search(category="laptop")
        self.assertNotIn("F001", [p["id"] for p in result["products"]])
        self.assertIn("not enabled yet", result["providers"][1]["message"])

    def test_empty_provider_list_rejected(self):
        with self.assertRaises(ValueError):
            ProductCatalog([])


class MCPProviderConfigTests(unittest.TestCase):
    def test_client_forwards_provider_env_to_server(self):
        extra = {"SHOPPING_PROVIDERS": "local,amazon", "FLIPKART_AFFILIATE_ID": "x"}
        with mock.patch.dict(os.environ, extra):
            env = ShoppingMCPClient()._server_parameters().env
        for key, value in extra.items():
            self.assertEqual(env[key], value)

    def test_enabled_placeholder_through_mcp(self):
        async def body():
            async with ShoppingMCPClient(server_log_level="WARNING") as client:
                return await client.call_tool("search_product", {"category": "laptop"})

        with mock.patch.dict(os.environ, {"SHOPPING_PROVIDERS": "local,amazon"}):
            result = asyncio.run(body())
        data = json.loads(result.text)
        self.assertFalse(result.is_error)
        self.assertGreater(data["count"], 0)
        amazon = next(n for n in data["providers"] if n["provider"] == "amazon")
        self.assertFalse(amazon["live"])
        self.assertIn("not configured", amazon["message"])


if __name__ == "__main__":
    unittest.main()
