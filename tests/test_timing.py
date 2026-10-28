"""Tests for the latency instrumentation (no Ollama needed).

The Ollama client is replaced by a fake, so these check that timing metadata
is recorded and logged, that no message content leaks into timing logs, and
that the request sent to Ollama is unchanged (same model, think flag and
options; no context-window or keep-alive overrides).
"""

import asyncio
import json
import unittest
from types import SimpleNamespace

from client.llm_client import LLMError, LLMReply, OllamaLLM, ToolCall, format_timing, ollama_timing
from client.mcp_client import ShoppingMCPClient
from client.shopping_assistant import ShoppingAssistant

SECRET_QUESTION = "my-private-question-42 laptops under 50000"


def fake_response(content="", tool_calls=None, thinking=None):
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls, thinking=thinking),
        total_duration=7_900_000_000, load_duration=12_000_000,
        prompt_eval_count=793, prompt_eval_duration=1_100_000_000,
        eval_count=32, eval_duration=6_600_000_000,
    )


class FakeOllamaClient:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.requests = response, error, []

    async def chat(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def make_llm(client) -> OllamaLLM:
    llm = OllamaLLM(host="http://127.0.0.1:11999", model="qwen3:8b", think=False, temperature=0.2)
    llm._client = client
    return llm


class OllamaTimingTests(unittest.TestCase):
    def test_extracts_metadata_in_seconds(self):
        timing = ollama_timing(fake_response(thinking="hmm"), 25.678, 3)
        self.assertEqual(timing, {
            "elapsed_s": 25.68, "messages": 3, "total_s": 7.9, "load_s": 0.01,
            "prompt_eval_s": 1.1, "eval_s": 6.6, "prompt_eval_count": 793, "eval_count": 32,
            "thinking_chars": 3,
        })

    def test_missing_fields_are_skipped(self):
        response = SimpleNamespace(message=SimpleNamespace(content="x"), eval_count=None)
        timing = ollama_timing(response, 1.0, 2)
        self.assertEqual(timing, {"elapsed_s": 1.0, "messages": 2, "thinking_chars": 0})
        self.assertIn("prompt ? tok", format_timing(timing))

    def test_chat_records_and_logs_timing_without_content(self):
        client = FakeOllamaClient(fake_response(content="Here are laptops"))
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": SECRET_QUESTION}]
        with self.assertLogs("shopping.llm", "INFO") as logs:
            reply = asyncio.run(make_llm(client).chat(messages))
        self.assertEqual(reply.content, "Here are laptops")
        self.assertEqual(reply.timing["prompt_eval_count"], 793)
        self.assertEqual(reply.timing["messages"], 2)
        line = next(l for l in logs.output if "[TIMING]" in l)
        self.assertIn("prompt 793 tok/1.1s", line)
        self.assertNotIn(SECRET_QUESTION, "\n".join(logs.output))

    def test_request_configuration_is_unchanged(self):
        """Baseline phase: no new options may reach Ollama."""
        client = FakeOllamaClient(fake_response())
        tools = [{"type": "function", "function": {"name": "search_product", "parameters": {}}}]
        asyncio.run(make_llm(client).chat([{"role": "user", "content": "hi"}], tools))
        request = client.requests[0]
        self.assertEqual(set(request), {"model", "messages", "options", "tools", "think"})
        self.assertEqual(request["model"], "qwen3:8b")
        self.assertIs(request["think"], False)
        self.assertEqual(request["options"], {"temperature": 0.2})
        self.assertEqual(request["tools"], tools)

    def test_tool_calls_still_parsed(self):
        call = SimpleNamespace(function=SimpleNamespace(name="search_product", arguments={"max_price": 50000}))
        reply = asyncio.run(make_llm(FakeOllamaClient(fake_response(tool_calls=[call]))).chat([]))
        self.assertEqual([(c.name, c.arguments) for c in reply.tool_calls],
                         [("search_product", {"max_price": 50000})])

    def test_failure_logs_elapsed_and_still_raises(self):
        client = FakeOllamaClient(error=ConnectionRefusedError("refused"))
        with self.assertLogs("shopping.llm", "INFO") as logs, self.assertRaises(LLMError):
            asyncio.run(make_llm(client).chat([{"role": "user", "content": "hi"}]))
        self.assertTrue(any("request failed after" in l for l in logs.output))


class TimedLLM:
    """Scripted LLM whose replies carry timing like the real client."""

    def __init__(self, script):
        self.script = list(script)

    async def chat(self, messages, tools=None):
        reply = self.script.pop(0)
        reply.timing = {"elapsed_s": 0.01, "messages": len(messages), "prompt_eval_count": 10 * len(messages)}
        return reply


class AssistantTimingTests(unittest.TestCase):
    def run_turns(self, *turns):
        async def runner():
            async with ShoppingMCPClient(server_log_level="WARNING") as client:
                assistant = ShoppingAssistant(TimedLLM([r for t in turns for r in t[1]]), client)
                timings = []
                for question, _ in turns:
                    await assistant.ask(question)
                    timings.append(assistant.last_timing)
                return assistant, timings

        return asyncio.run(runner())

    def test_phases_recorded_in_order(self):
        script = [LLMReply("", [ToolCall("search_product", {"category": "laptop", "max_price": 50000})]),
                  LLMReply("Found laptops.")]
        with self.assertLogs("shopping.app", "INFO") as logs:
            _, (timing,) = self.run_turns((SECRET_QUESTION, script))

        self.assertEqual([s["step"] for s in timing["steps"]],
                         ["tool-selection", "mcp", "response-generation"])
        mcp = timing["steps"][1]
        self.assertEqual((mcp["tool"], mcp["is_error"]), ("search_product", False))
        self.assertGreaterEqual(timing["total_s"], sum(s["elapsed_s"] for s in timing["steps"]) - 0.01)
        self.assertEqual(timing["steps"][0]["prompt_eval_count"], 20)  # Ollama metadata carried over

        timing_lines = [l for l in logs.output if "[TIMING]" in l]
        self.assertTrue(any("Query total" in l and "MCP 1 call(s)" in l for l in timing_lines))
        self.assertNotIn(SECRET_QUESTION, "\n".join(timing_lines))

    def test_context_growth_visible_across_turns(self):
        first = [LLMReply("", [ToolCall("search_product", {"category": "laptop"})]), LLMReply("Found.")]
        second = [LLMReply("", [ToolCall("get_details", {"product_id": "L001"})]), LLMReply("Details.")]
        _, (t1, t2) = self.run_turns(("Find laptops", first), ("Details of the first", second))
        self.assertGreater(t2["steps"][0]["messages"], t1["steps"][0]["messages"])

    def test_empty_input_records_nothing(self):
        async def runner():
            async with ShoppingMCPClient(server_log_level="WARNING") as client:
                assistant = ShoppingAssistant(TimedLLM([]), client)
                await assistant.ask("   ")
                return assistant.last_timing

        timing = asyncio.run(runner())
        self.assertEqual(timing["steps"], [])
        self.assertIsNotNone(timing["total_s"])


if __name__ == "__main__":
    unittest.main()
