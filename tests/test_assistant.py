"""End-to-end test of the assistant loop WITHOUT Ollama.

A scripted fake LLM stands in for qwen3:8b, but the MCP part is real:
the MCP server is started and every tool call goes over the protocol.
This proves the full User -> LLM -> MCP client -> MCP server -> LLM flow,
including multiple tool calls in one turn.
"""

import asyncio
import json
import unittest

from client.llm_client import LLMReply, ToolCall, clean_content, parse_text_tool_calls
from client.mcp_client import ShoppingMCPClient
from client.shopping_assistant import ShoppingAssistant


class ScriptedLLM:
    """Returns pre-defined replies; records what it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def chat(self, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        step = self.script.pop(0)
        return step(messages) if callable(step) else step


def run_assistant(script, question):
    async def runner():
        async with ShoppingMCPClient(server_log_level="WARNING") as client:
            llm = ScriptedLLM(script)
            assistant = ShoppingAssistant(llm, client)
            answer = await assistant.ask(question)
            return answer, assistant, llm

    return asyncio.run(runner())


class AssistantFlowTests(unittest.TestCase):
    def test_single_tool_flow(self):
        def final(messages):
            data = json.loads(messages[-1]["content"])
            return LLMReply(content=f"Found {data['count']} laptops.")

        answer, assistant, llm = run_assistant(
            [LLMReply("", [ToolCall("search_product", {"category": "laptop", "max_price": 50000})]),
             final],
            "Find laptops under ₹50000",
        )
        self.assertEqual(assistant.tool_calls_made, ["search_product"])
        self.assertEqual(answer, "Found 3 laptops.")
        self.assertEqual(llm.calls[1][-1]["role"], "tool")

    def test_multiple_tool_calls_search_then_compare(self):
        def compare_each(messages):
            found = json.loads(messages[-1]["content"])["products"]
            return LLMReply("", [ToolCall("compare_price", {"product_id": p["id"]}) for p in found])

        def final(messages):
            tool_msgs = [m for m in messages if m["role"] == "tool"]
            return LLMReply(f"Compared {len(tool_msgs) - 1} products.")

        answer, assistant, _ = run_assistant(
            [LLMReply("", [ToolCall("search_product", {"category": "laptop", "max_price": 50000})]),
             compare_each, final],
            "Find a laptop under ₹50000 and compare the prices of the matching products",
        )
        self.assertEqual(assistant.tool_calls_made[0], "search_product")
        self.assertEqual(assistant.tool_calls_made.count("compare_price"), 3)
        self.assertEqual(answer, "Compared 3 products.")

    def test_invalid_product_id_is_reported_to_llm(self):
        def final(messages):
            self.assertIn("not found", messages[-1]["content"])
            return LLMReply("That product ID does not exist.")

        answer, _, _ = run_assistant(
            [LLMReply("", [ToolCall("get_details", {"product_id": "L999"})]), final],
            "Specs of L999",
        )
        self.assertEqual(answer, "That product ID does not exist.")

    def test_empty_input(self):
        async def runner():
            async with ShoppingMCPClient(server_log_level="WARNING") as client:
                return await ShoppingAssistant(ScriptedLLM([]), client).ask("   ")

        self.assertIn("Please type a question", asyncio.run(runner()))


class LLMHelperTests(unittest.TestCase):
    def test_think_block_removed(self):
        self.assertEqual(clean_content("<think>hmm</think>\nHello"), "Hello")

    def test_text_tool_call_fallback(self):
        calls = parse_text_tool_calls(
            '<tool_call>{"name": "get_details", "arguments": {"product_id": "L001"}}</tool_call>'
        )
        self.assertEqual(calls[0].name, "get_details")
        self.assertEqual(calls[0].arguments, {"product_id": "L001"})


if __name__ == "__main__":
    unittest.main()
