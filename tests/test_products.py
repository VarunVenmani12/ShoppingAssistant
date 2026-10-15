"""Unit tests for product loading and the data functions behind the MCP tools.

Run:  python -m unittest discover -s tests -v     (or: python -m pytest)
"""

import json
import tempfile
import unittest
from pathlib import Path

from server.product_store import (
    DEFAULT_PRODUCTS_FILE,
    InvalidArgumentError,
    ProductDataError,
    compare_prices,
    format_inr,
    get_product_details,
    load_products,
    parse_price,
    parse_size_gb,
    search_products,
)

CATEGORIES = {"laptop", "smartphone", "monitor", "keyboard", "mouse", "headphones"}


class ProductLoadingTests(unittest.TestCase):
    def test_loads_at_least_15_products(self):
        products = load_products()
        self.assertGreaterEqual(len(products), 15)

    def test_all_categories_present(self):
        categories = {p["category"] for p in load_products()}
        self.assertTrue(CATEGORIES.issubset(categories))

    def test_products_have_required_fields_and_inr(self):
        ids = set()
        for p in load_products():
            for key in ("id", "name", "category", "brand", "price", "specifications", "stores"):
                self.assertIn(key, p)
            self.assertEqual(p["currency"], "INR")
            self.assertGreaterEqual(len(p["stores"]), 2)
            self.assertNotIn(p["id"], ids, "duplicate product id")
            ids.add(p["id"])

    def test_missing_file_raises_friendly_error(self):
        with self.assertRaises(ProductDataError):
            load_products(Path(tempfile.gettempdir()) / "does-not-exist-12345.json")

    def test_invalid_json_raises_friendly_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            fh.write("{not valid json")
        try:
            with self.assertRaises(ProductDataError):
                load_products(fh.name)
        finally:
            Path(fh.name).unlink()

    def test_default_file_is_project_products_json(self):
        self.assertEqual(DEFAULT_PRODUCTS_FILE.name, "products.json")
        json.loads(DEFAULT_PRODUCTS_FILE.read_text(encoding="utf-8"))


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.products = load_products()

    def test_laptops_under_50000(self):
        result = search_products(self.products, category="laptop", max_price=50000)
        self.assertGreater(result["count"], 0)
        for p in result["products"]:
            self.assertEqual(p["category"], "laptop")
            self.assertLessEqual(p["price"], 50000)

    def test_laptops_with_16gb_ram(self):
        result = search_products(self.products, category="laptops", ram="16GB")
        self.assertGreater(result["count"], 0)
        for p in result["products"]:
            self.assertGreaterEqual(parse_size_gb(p["ram"], "ram"), 16)

    def test_laptop_16gb_1tb_under_50000(self):
        result = search_products(
            self.products, category="laptop", max_price="₹50,000", ram="16GB", storage="1TB SSD"
        )
        self.assertEqual([p["id"] for p in result["products"]], ["L002"])

    def test_smartphones_under_30000(self):
        result = search_products(self.products, category="phone", max_price=30000)
        self.assertGreater(result["count"], 0)
        self.assertTrue(all(p["category"] == "smartphone" for p in result["products"]))

    def test_brand_and_keyword(self):
        result = search_products(self.products, brand="logitech", keyword="wireless")
        self.assertGreaterEqual(result["count"], 2)
        self.assertTrue(all(p["brand"] == "Logitech" for p in result["products"]))

    def test_results_sorted_by_price(self):
        prices = [p["price"] for p in search_products(self.products)["products"]]
        self.assertEqual(prices, sorted(prices))

    def test_no_matching_products(self):
        result = search_products(self.products, category="laptop", max_price=1000)
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["products"], [])
        self.assertIn("message", result)

    def test_invalid_category(self):
        with self.assertRaises(InvalidArgumentError):
            search_products(self.products, category="spaceship")

    def test_invalid_price(self):
        with self.assertRaises(InvalidArgumentError):
            search_products(self.products, max_price="cheap")
        with self.assertRaises(InvalidArgumentError):
            search_products(self.products, max_price=-5)

    def test_min_greater_than_max(self):
        with self.assertRaises(InvalidArgumentError):
            search_products(self.products, min_price=60000, max_price=50000)


class ComparePriceTests(unittest.TestCase):
    def setUp(self):
        self.products = load_products()

    def test_compare_l001(self):
        result = compare_prices(self.products, "L001")
        self.assertEqual(result["product_id"], "L001")
        self.assertGreaterEqual(len(result["stores"]), 2)
        prices = [s["price"] for s in result["stores"]]
        self.assertEqual(prices, sorted(prices))
        self.assertEqual(result["best_deal"]["price"], min(prices))

    def test_lowercase_id_accepted(self):
        self.assertEqual(compare_prices(self.products, "l001")["product_id"], "L001")

    def test_unknown_product_id(self):
        result = compare_prices(self.products, "L999")
        self.assertIn("error", result)

    def test_malformed_product_id(self):
        for bad in ("", "../etc/passwd", "L001; rm -rf /", "12345"):
            with self.assertRaises(InvalidArgumentError):
                compare_prices(self.products, bad)


class DetailsTests(unittest.TestCase):
    def setUp(self):
        self.products = load_products()

    def test_details_l001(self):
        result = get_product_details(self.products, "L001")
        self.assertEqual(result["id"], "L001")
        self.assertEqual(result["specifications"]["ram"], "16GB")
        self.assertIn("processor", result["specifications"])

    def test_details_invalid_id(self):
        self.assertIn("error", get_product_details(self.products, "Z999"))


class HelperTests(unittest.TestCase):
    def test_parse_price(self):
        self.assertEqual(parse_price("₹50,000", "p"), 50000)
        self.assertEqual(parse_price("50k", "p"), 50000)
        self.assertIsNone(parse_price(None, "p"))

    def test_parse_size(self):
        self.assertEqual(parse_size_gb("1TB SSD", "s"), 1024)
        self.assertEqual(parse_size_gb("16 GB", "r"), 16)
        self.assertEqual(parse_size_gb(8, "r"), 8)

    def test_format_inr(self):
        self.assertEqual(format_inr(48999), "₹48,999")
        self.assertEqual(format_inr(1234567), "₹12,34,567")
        self.assertEqual(format_inr(999), "₹999")


if __name__ == "__main__":
    unittest.main()
