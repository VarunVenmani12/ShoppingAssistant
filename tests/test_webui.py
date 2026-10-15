"""Tests for the web UI server (no Ollama needed).

Ollama is pointed at an unused port, so these tests check that the page,
status endpoint, MCP connection, input validation and the friendly
"Ollama unavailable" error all work.
"""

import json
import unittest
from dataclasses import replace

from starlette.testclient import TestClient

from client.config import load_settings
from webui.app import create_app


def make_client() -> TestClient:
    settings = replace(load_settings(), ollama_host="http://127.0.0.1:11999",
                       server_log_level="WARNING")
    return TestClient(create_app(settings), base_url="http://127.0.0.1")


class WebUITests(unittest.TestCase):
    def test_page_status_and_chat_errors(self):
        with make_client() as client:
            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn("AI Shopping Assistant", page.text)

            status = client.get("/api/status").json()
            self.assertFalse(status["ollama"]["ok"])
            self.assertTrue(status["mcp"]["ok"])
            self.assertEqual(
                {t["name"] for t in status["mcp"]["tools"]},
                {"search_product", "compare_price", "get_details"},
            )

            bad = client.post("/api/chat", json={"session_id": "x", "message": "hi"})
            self.assertEqual(bad.status_code, 400)
            empty = client.post("/api/chat", json={"session_id": "abcdefgh-1", "message": "  "})
            self.assertEqual(empty.status_code, 400)
            not_json = client.post("/api/chat", content=b"{oops",
                                   headers={"content-type": "application/json"})
            self.assertEqual(not_json.status_code, 400)

            stream = client.post("/api/chat", json={"session_id": "abcdefgh-1", "message": "hi"})
            events = [json.loads(line) for line in stream.text.splitlines() if line.strip()]
            self.assertEqual(events[-1]["type"], "error")
            self.assertTrue(
                "Ollama" in events[-1]["text"] or "ollama" in events[-1]["text"]
            )

            self.assertEqual(client.post("/api/reset", json={"session_id": "abcdefgh-1"}).json(),
                             {"ok": True})

    def test_rejects_foreign_host_header(self):
        with make_client() as client:
            response = client.get("/api/status", headers={"host": "evil.example.com"})
            self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
