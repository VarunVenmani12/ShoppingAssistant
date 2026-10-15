"""Tests that talk to the MCP server THROUGH THE MCP PROTOCOL.

The server is started as a subprocess and every call is a real JSON-RPC
'tools/list' or 'tools/call' request over stdio - exactly what the
Shopping Assistant does. No server functions are imported here.
"""

import asyncio
import json
import unittest

from client.mcp_client import ShoppingMCPClient, sanitize_arguments


def run_with_client(test_body):
    """Start the MCP server, run `test_body(client)`, shut the server down."""

    async def runner():
        async with ShoppingMCPClient(server_log_level="WARNING") as client:
            return await test_body(client)

    return asyncio.run(runner())


class MCPServerProtocolTests(unittest.TestCase):
    def test_tools_are_discovered(self):
        async def body(client):
            return client.tool_names(), client.ollama_tool_definitions()

        names, definitions = run_with_client(body)
        self.assertEqual(set(names), {"search_product", "compare_price", "get_details"})
        search = next(d for d in definitions if d["function"]["name"] == "search_product")
        params = search["function"]["parameters"]["properties"]
        for key in ("category", "max_price", "min_price", "brand", "ram", "storage", "keyword"):
            self.assertIn(key, params)
        self.assertEqual(params["max_price"]["type"], "number")

    def test_search_compare_details_and_errors(self):
        async def body(client):
            return {
                "search": await client.call_tool(
                    "search_product", {"category": "laptop", "max_price": "₹50,000", "ram": 16}
                ),
                "none": await client.call_tool(
                    "search_product", {"category": "laptop", "max_price": 100}
                ),
                "compare": await client.call_tool("compare_price", {"product_id": "L001"}),
                "details": await client.call_tool("get_details", {"product_id": "L001"}),
                "bad_id": await client.call_tool("get_details", {"product_id": "X999"}),
                "evil_id": await client.call_tool("compare_price", {"product_id": "../../secrets"}),
                "bad_cat": await client.call_tool("search_product", {"category": "rocket"}),
                "missing": await client.call_tool("compare_price", {}),
                "unknown_tool": await client.call_tool("delete_everything", {}),
                "bad_json": await client.call_tool("get_details", "{not json"),
            }

        r = run_with_client(body)

        search = json.loads(r["search"].text)
        self.assertFalse(r["search"].is_error)
        self.assertGreater(search["count"], 0)
        self.assertTrue(all(p["price"] <= 50000 for p in search["products"]))

        none = json.loads(r["none"].text)
        self.assertEqual(none["count"], 0)

        compare = json.loads(r["compare"].text)
        self.assertEqual(compare["product_id"], "L001")
        self.assertGreaterEqual(len(compare["stores"]), 2)

        details = json.loads(r["details"].text)
        self.assertEqual(details["specifications"]["ram"], "16GB")

        for key in ("bad_id", "evil_id", "bad_cat", "missing", "unknown_tool", "bad_json"):
            self.assertTrue(r[key].is_error, key)
            self.assertIn("error", json.loads(r[key].text), key)


class ArgumentSanitizerTests(unittest.TestCase):
    SCHEMA = {
        "properties": {
            "max_price": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "ram": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "product_id": {"type": "string"},
        },
        "required": [],
    }

    def test_converts_and_drops(self):
        clean = sanitize_arguments(
            self.SCHEMA, {"max_price": "₹50,000", "ram": 16, "extra": "x", "product_id": None}
        )
        self.assertEqual(clean, {"max_price": 50000.0, "ram": "16GB"})

    def test_json_string_arguments(self):
        self.assertEqual(sanitize_arguments(self.SCHEMA, '{"max_price": 30000}'), {"max_price": 30000})

    def test_invalid(self):
        with self.assertRaises(ValueError):
            sanitize_arguments(self.SCHEMA, "{bad json")
        with self.assertRaises(ValueError):
            sanitize_arguments(self.SCHEMA, {"max_price": "cheap"})
        with self.assertRaises(ValueError):
            sanitize_arguments({"properties": {"product_id": {"type": "string"}},
                                "required": ["product_id"]}, {})


if __name__ == "__main__":
    unittest.main()
