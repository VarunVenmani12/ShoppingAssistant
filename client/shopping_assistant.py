"""
AI Shopping Assistant - the application that ties everything together.

    User -> ShoppingAssistant -> Qwen3 (Ollama) -> MCP client -> MCP server
         -> products.json -> MCP server -> MCP client -> Qwen3 -> User

Run:
    python -m client.shopping_assistant            # interactive CLI
    python -m client.shopping_assistant --demo     # run the sample questions
    python -m client.shopping_assistant --quiet    # hide the MCP flow logs
    python -m client.shopping_assistant --ask "Compare prices for L001"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import threading
import time
from typing import Any, Awaitable, Callable, Protocol

from client.config import Settings, load_settings
from client.llm_client import LLMError, LLMReply, OllamaLLM
from client.mcp_client import MCPConnectionError, ShoppingMCPClient

log = logging.getLogger("shopping.app")

SYSTEM_PROMPT = """You are a friendly shopping assistant for a demo electronics store in India.
All prices are in Indian Rupees (INR, shown as ₹).

You do NOT know the product catalogue yourself. You MUST use the tools to get
product information. Never invent products, prices, specifications or stores.

Tools:
- search_product: find products. Only pass the filters the user asked for.
  category must be one of: laptop, smartphone, monitor, keyboard, mouse, headphones.
  max_price / min_price are plain numbers (₹50,000 -> 50000).
  ram and storage are minimums written like "16GB" or "1TB".
- compare_price(product_id): prices of ONE product at different stores.
- get_details(product_id): full specifications of ONE product.

If the user wants prices compared for search results, first call search_product,
then call compare_price for every matching product ID.

Answer format:
- Short intro sentence, then a numbered list.
- For each product: name and ID, price (use the price_formatted values, e.g. ₹48,999),
  and the most relevant specs (RAM, storage, processor, display...).
- For price comparisons list each store with its price and point out the best deal.
- If nothing matches, say so clearly and suggest relaxing a filter.
- If a tool returns an error (e.g. unknown product ID), explain it simply.
- If a result has a "providers" list, a provider with "available": false has no data.
  Say briefly that live prices from it (e.g. Amazon, Flipkart) are not available;
  never guess them.
- Be concise. Do not mention tools, JSON or internal details."""

EXIT_COMMANDS = {"exit", "quit"}
MAX_HISTORY_MESSAGES = 24
# Product-ID-like tokens in user text (same shape the server accepts: L001, MS002, X999).
PRODUCT_ID_IN_TEXT = re.compile(r"\b[A-Za-z]{1,3}\d{3}\b")
DEMO_QUESTIONS = [
    "Find laptops under ₹50000",
    "Find laptops with 16GB RAM",
    "Compare prices for L001",
    "Show details for L001",
    "Find smartphones under ₹30000",
]


class ChatModel(Protocol):
    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMReply: ...


EventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


class ShoppingAssistant:
    """Orchestrates the conversation between the LLM and the MCP client.

    ``ask()`` accepts an optional ``on_event`` callback that receives one
    dict per step of the flow (user, llm, mcp_call, mcp_result, answer).
    The CLI ignores it; the web UI streams these events to the MCP trace panel.
    """

    def __init__(self, llm: ChatModel, mcp_client: ShoppingMCPClient, max_tool_rounds: int = 5):
        self.llm = llm
        self.mcp = mcp_client
        self.max_tool_rounds = max_tool_rounds
        self.history: list[dict[str, Any]] = []
        self.tool_calls_made: list[str] = []  # for tests / demo output
        # Timing of the last ask(): {"total_s": float, "steps": [...]}. No message content.
        self.last_timing: dict[str, Any] = {"total_s": None, "steps": []}

    def reset(self) -> None:
        self.history.clear()

    async def ask(self, user_text: str, on_event: EventCallback | None = None) -> str:
        started = time.perf_counter()
        self.last_timing = {"total_s": None, "steps": []}
        try:
            return await self._ask(user_text, on_event, started)
        finally:
            self._log_query_timing(time.perf_counter() - started)

    async def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> LLMReply:
        """One timed LLM request, labelled by what the model did with it."""
        t0 = time.perf_counter()
        reply = await self.llm.chat(messages, tools)
        step = "tool-selection" if reply.tool_calls else "response-generation"
        record = {"step": step, "elapsed_s": round(time.perf_counter() - t0, 2)}
        record.update({k: v for k, v in (getattr(reply, "timing", None) or {}).items() if k != "elapsed_s"})
        self.last_timing["steps"].append(record)
        log.info("[TIMING] LLM %s: %.2fs", step, record["elapsed_s"])
        return reply

    async def _call_tool(self, call: Any) -> Any:
        """One timed MCP tool execution (arguments are not logged here)."""
        t0 = time.perf_counter()
        result = await self.mcp.call_tool(call.name, call.arguments)
        elapsed = round(time.perf_counter() - t0, 3)
        self.last_timing["steps"].append(
            {"step": "mcp", "tool": call.name, "elapsed_s": elapsed, "is_error": result.is_error}
        )
        log.info("[TIMING] MCP %s: %.3fs", call.name, elapsed)
        return result

    def _log_query_timing(self, total: float) -> None:
        steps = self.last_timing["steps"]
        self.last_timing["total_s"] = round(total, 2)
        if not steps:
            return
        llm = [s for s in steps if s["step"] != "mcp"]
        mcp = [s for s in steps if s["step"] == "mcp"]
        log.info(
            "[TIMING] Query total %.2fs | LLM %d call(s) %.2fs | MCP %d call(s) %.3fs | "
            "history %d msgs",
            total, len(llm), sum(s["elapsed_s"] for s in llm),
            len(mcp), sum(s["elapsed_s"] for s in mcp), len(self.history),
        )

    async def _ask(self, user_text: str, on_event: EventCallback | None, started: float) -> str:

        async def emit(kind: str, **data: Any) -> None:
            if on_event is None:
                return
            event = {"type": kind, "t": round(time.perf_counter() - started, 2), **data}
            outcome = on_event(event)
            if asyncio.iscoroutine(outcome):
                await outcome

        user_text = (user_text or "").strip()
        if not user_text:
            return "Please type a question, for example: 'Find laptops under ₹50000'."

        log.info("[USER] %s", user_text)
        await emit("user", text=user_text)
        self.tool_calls_made = []
        tools = self.mcp.ollama_tool_definitions()
        messages: list[dict[str, Any]] = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + self.history
            + [{"role": "user", "content": user_text}]
        )
        turn_start = len(messages) - 1

        log.info("[LLM] Determining required tool...")
        await emit("llm", text="Determining required tool...")
        reply = await self._chat(messages, tools)

        unchecked = self._unchecked_product_ids(user_text)
        if unchecked and tools and not reply.tool_calls:
            # The model answered about a product ID nothing has looked up yet. Small
            # models sometimes guess ("does not exist") late in a conversation; ask
            # once more to check with a tool. The nudge is not stored in history.
            ids = ", ".join(unchecked)
            log.info("[LLM] Answer about unchecked product ID(s) %s without a tool call; "
                     "asking the model to check with a tool", ids)
            await emit("llm", text=f"Asking the model to check {ids} with a tool first...")
            nudge = {"role": "user", "content": (
                f"Before answering, call get_details or compare_price for {ids}. "
                "Do not say whether a product exists without a tool result.")}
            reply = await self._chat(messages + [nudge], tools)

        rounds = 0
        while reply.tool_calls and rounds < self.max_tool_rounds:
            rounds += 1
            log.info(
                "[LLM] Model requested %d tool call(s): %s",
                len(reply.tool_calls), ", ".join(c.name for c in reply.tool_calls),
            )
            await emit(
                "llm",
                text=f"Model requested {len(reply.tool_calls)} tool call(s): "
                + ", ".join(c.name for c in reply.tool_calls),
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": reply.content,
                    "tool_calls": [
                        {"function": {"name": c.name, "arguments": _as_dict(c.arguments)}}
                        for c in reply.tool_calls
                    ],
                }
            )
            for call in reply.tool_calls:
                await emit("mcp_call", tool=call.name, arguments=_as_dict(call.arguments))
                result = await self._call_tool(call)
                self.tool_calls_made.append(call.name)
                await emit(
                    "mcp_result", tool=call.name, arguments=result.arguments,
                    is_error=result.is_error, result=result.text,
                )
                messages.append({"role": "tool", "tool_name": call.name, "content": result.text})

            log.info("[LLM] Generating final response...")
            await emit("llm", text="Generating final response...")
            reply = await self._chat(messages, tools)

        if reply.tool_calls:  # tool budget exhausted: force a text answer
            log.info("[LLM] Tool-call limit reached; asking for a final answer")
            reply = await self._chat(messages, None)

        answer = reply.content.strip() or (
            "Sorry, I couldn't produce an answer. Please try rephrasing your question."
        )
        if not self.tool_calls_made:
            log.info("[LLM] Answered without calling an MCP tool")
            await emit("llm", text="Answered without calling an MCP tool")
        await emit("answer", text=answer, tools_used=list(self.tool_calls_made))
        messages.append({"role": "assistant", "content": answer})
        self._remember(messages[turn_start:])
        return answer

    def _unchecked_product_ids(self, user_text: str) -> list[str]:
        """Product IDs in the user's message that no tool result in history mentions yet."""
        ids = {m.upper() for m in PRODUCT_ID_IN_TEXT.findall(user_text)}
        seen = " ".join(m.get("content", "") for m in self.history if m.get("role") == "tool")
        return sorted(i for i in ids if f'"{i}"' not in seen)

    def _remember(self, turn: list[dict[str, Any]]) -> None:
        self.history.extend(turn)
        if len(self.history) > MAX_HISTORY_MESSAGES:
            trimmed = self.history[-MAX_HISTORY_MESSAGES:]
            # always start history on a user message
            while trimmed and trimmed[0]["role"] != "user":
                trimmed.pop(0)
            self.history = trimmed


def _as_dict(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    try:
        value = json.loads(arguments)
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _banner(settings: Settings) -> str:
    line = "=" * 50
    return (
        f"{line}\n        AI SHOPPING ASSISTANT\n{line}\n\n"
        f"Using:\nLLM       : Ollama ({settings.ollama_host})\n"
        f"Model     : {settings.ollama_model}\nProtocol  : MCP (stdio transport)\n"
        "Data      : products.json demo catalogue (sample prices, not live marketplace prices)\n\n"
        "Type your question, 'tools' to list MCP tools, 'clear' to reset the\n"
        "conversation, or 'exit' / 'quit' to leave.\n"
    )


async def _read_line(prompt: str) -> str | None:
    """Read input on a daemon thread so Ctrl+C never hangs the app (Windows-safe)."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str | None] = loop.create_future()

    def worker() -> None:
        try:
            value: str | None = input(prompt)
        except (EOFError, KeyboardInterrupt):
            value = None
        loop.call_soon_threadsafe(lambda: future.done() or future.set_result(value))

    threading.Thread(target=worker, daemon=True).start()
    return await future


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
    for name in ("shopping.app", "shopping.mcp", "shopping.llm"):
        logging.getLogger(name).setLevel(level)
    for noisy in ("httpx", "httpcore", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def run(demo: bool = False, quiet: bool = False, questions_to_ask: list[str] | None = None) -> int:
    settings = load_settings()
    _setup_logging("WARNING" if quiet else settings.log_level)
    print(_banner(settings))

    try:
        llm = OllamaLLM(
            host=settings.ollama_host,
            model=settings.ollama_model,
            think=settings.ollama_think,
            temperature=settings.ollama_temperature,
            timeout=settings.ollama_timeout,
        )
        await llm.verify()
    except LLMError as error:
        print(f"Error: {error}")
        return 1

    try:
        async with ShoppingMCPClient(
            server_log_level="WARNING" if quiet else settings.server_log_level
        ) as mcp_client:
            assistant = ShoppingAssistant(llm, mcp_client, settings.max_tool_rounds)
            print(f"Connected. MCP tools available: {', '.join(mcp_client.tool_names())}\n")

            scripted = list(questions_to_ask or []) or (DEMO_QUESTIONS if demo else [])
            demo = bool(scripted)
            questions = iter(scripted) if scripted else None
            while True:
                if questions is not None:
                    text = next(questions, None)
                    if text is None:
                        break
                    print(f"You: {text}")
                else:
                    text = await _read_line("You: ")
                    if text is None:
                        print()
                        break
                command = text.strip().lower()
                if command in EXIT_COMMANDS:
                    break
                if command == "tools":
                    for tool in mcp_client.tools:
                        print(f"  - {tool.name}: {(tool.description or '').splitlines()[0]}")
                    print()
                    continue
                if command == "clear":
                    assistant.reset()
                    print("Conversation cleared.\n")
                    continue
                if not command:
                    print("Please type a question (or 'exit' to quit).\n")
                    continue

                try:
                    answer = await assistant.ask(text)
                except LLMError as error:
                    answer = f"Sorry, the language model is unavailable: {error}"
                print(f"\nAssistant:\n{answer}\n")
                if demo:
                    print(f"(MCP tools used: {', '.join(assistant.tool_calls_made) or 'none'})\n")
                    print("-" * 50)
    except MCPConnectionError as error:
        print(f"Error: {error}")
        return 1

    print("Goodbye!")
    return 0


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Local AI Shopping Assistant (Ollama + MCP)")
    parser.add_argument("--demo", action="store_true", help="run the sample questions and exit")
    parser.add_argument("--quiet", action="store_true", help="hide MCP flow logging")
    parser.add_argument(
        "--ask", action="append", metavar="QUESTION",
        help="ask one question non-interactively (can be repeated) and exit",
    )
    args = parser.parse_args()
    try:
        code = asyncio.run(run(demo=args.demo, quiet=args.quiet, questions_to_ask=args.ask))
    except KeyboardInterrupt:
        print("\nGoodbye!")
        code = 0
    except Exception as error:  # last-resort guard: no stack traces for CLI users
        logging.getLogger("shopping.app").debug("Unexpected error", exc_info=True)
        print(f"Unexpected error: {type(error).__name__}: {error}")
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
