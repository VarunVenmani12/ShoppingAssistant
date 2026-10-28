"""
Thin wrapper around the Ollama Python library.

The LLM (qwen3:8b by default) is responsible for:
    * understanding the user's shopping request
    * deciding which MCP tool(s) to call and with which arguments
    * writing the final natural-language answer

This module knows nothing about MCP. It just sends messages + tool
definitions to Ollama and returns the reply.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("shopping.llm")

try:
    import httpx
except ImportError:  # pragma: no cover - httpx is a dependency of ollama
    httpx = None  # type: ignore[assignment]


class LLMError(RuntimeError):
    """User-friendly error raised for any Ollama problem."""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] | str


@dataclass
class LLMReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)


_NS = 1e9
# Ollama response fields copied into LLMReply.timing (durations are nanoseconds).
_OLLAMA_DURATIONS = ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")
_OLLAMA_COUNTS = ("prompt_eval_count", "eval_count")


def ollama_timing(response: Any, elapsed_s: float, message_count: int) -> dict[str, Any]:
    """Timing metadata from an Ollama chat response. Contains no message content."""
    timing: dict[str, Any] = {"elapsed_s": round(elapsed_s, 2), "messages": message_count}
    for name in _OLLAMA_DURATIONS:
        value = getattr(response, name, None)
        if isinstance(value, (int, float)):
            timing[name.replace("_duration", "_s")] = round(value / _NS, 2)
    for name in _OLLAMA_COUNTS:
        value = getattr(response, name, None)
        if isinstance(value, int):
            timing[name] = value
    message = getattr(response, "message", None)
    timing["thinking_chars"] = len(getattr(message, "thinking", None) or "")
    return timing


def format_timing(timing: dict[str, Any]) -> str:
    """One-line summary, e.g. '25.7s (ollama 7.9s, load 0.0s, prompt 793 tok/1.1s, ...)'."""
    def g(key: str) -> Any:
        return timing.get(key, "?")

    return (
        f"{g('elapsed_s')}s (ollama total {g('total_s')}s, load {g('load_s')}s, "
        f"prompt {g('prompt_eval_count')} tok/{g('prompt_eval_s')}s, "
        f"output {g('eval_count')} tok/{g('eval_s')}s, "
        f"thinking {g('thinking_chars')} chars, {g('messages')} msgs)"
    )


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TEXT_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def clean_content(text: str | None) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks from visible output."""
    if not text:
        return ""
    return _THINK_BLOCK.sub("", text).strip()


def parse_text_tool_calls(text: str) -> list[ToolCall]:
    """Fallback: some models print '<tool_call>{json}</tool_call>' as plain text."""
    calls = []
    for match in _TEXT_TOOL_CALL.finditer(text or ""):
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "name" in data:
            calls.append(ToolCall(data["name"], data.get("arguments") or {}))
    return calls


def _is_connection_error(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    if httpx is not None and isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return True
    return "connect" in str(exc).lower() and "refused" in str(exc).lower()


class OllamaLLM:
    """Async chat client for a local Ollama model."""

    def __init__(
        self,
        host: str,
        model: str,
        think: bool = False,
        temperature: float = 0.2,
        timeout: float = 300,
    ) -> None:
        try:
            import ollama  # imported lazily so tests can run without it
        except ImportError as exc:
            raise LLMError(
                "The 'ollama' Python package is not installed. Run: pip install -r requirements.txt"
            ) from exc
        self._ollama = ollama
        self.host = host
        self.model = model
        self.think = think
        self.options = {"temperature": temperature}
        self._client = ollama.AsyncClient(host=host, timeout=timeout)

    # ------------------------------------------------------------------ #
    async def verify(self) -> None:
        """Check that Ollama is reachable and the configured model is installed."""
        log.info("[LLM] Checking Ollama at %s ...", self.host)
        try:
            response = await self._client.list()
        except Exception as exc:
            if _is_connection_error(exc):
                raise LLMError(
                    f"Unable to connect to Ollama at {self.host}. "
                    "Make sure Ollama is running (open the Ollama app or run 'ollama serve')."
                ) from exc
            raise LLMError(f"Ollama returned an unexpected error: {exc}") from exc

        names = set()
        for item in getattr(response, "models", None) or []:
            name = getattr(item, "model", None) or getattr(item, "name", None)
            if name:
                names.add(name)
                if name.endswith(":latest"):
                    names.add(name[: -len(":latest")])
        wanted = {self.model, f"{self.model}:latest"}
        if not names & wanted:
            available = ", ".join(sorted(names)) or "none"
            raise LLMError(
                f"Model '{self.model}' is not available in Ollama (installed: {available}). "
                f"Install it with 'ollama pull {self.model}' or change OLLAMA_MODEL in .env."
            )
        log.info("[LLM] Ollama is reachable and model '%s' is available", self.model)

    # ------------------------------------------------------------------ #
    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMReply:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": self.options,
        }
        if tools:
            kwargs["tools"] = tools
        if self.think is not None:
            kwargs["think"] = self.think

        started = time.perf_counter()
        try:
            try:
                response = await self._client.chat(**kwargs)
            except TypeError:
                # Very old ollama-python without the 'think' argument.
                kwargs.pop("think", None)
                response = await self._chat_or_raise(kwargs)
            except Exception as exc:
                raise self._translate(exc) from exc
        except LLMError:
            log.info("[TIMING] Ollama request failed after %.2fs", time.perf_counter() - started)
            raise
        timing = ollama_timing(response, time.perf_counter() - started, len(messages))
        log.info("[TIMING] Ollama request: %s", format_timing(timing))

        message = response.message
        raw_content = message.content or ""
        calls = [
            ToolCall(tc.function.name, tc.function.arguments or {})
            for tc in (message.tool_calls or [])
        ]
        if not calls:
            calls = parse_text_tool_calls(raw_content)
        return LLMReply(
            content=clean_content(_TEXT_TOOL_CALL.sub("", raw_content)),
            tool_calls=calls,
            timing=timing,
        )

    async def _chat_or_raise(self, kwargs: dict[str, Any]) -> Any:
        try:
            return await self._client.chat(**kwargs)
        except Exception as exc:
            raise self._translate(exc) from exc

    def _translate(self, exc: Exception) -> LLMError:
        if _is_connection_error(exc):
            return LLMError(f"Unable to connect to Ollama at {self.host}.")
        status = getattr(exc, "status_code", None)
        if status == 404:
            return LLMError(
                f"Model '{self.model}' was not found in Ollama. Run 'ollama list' to check."
            )
        if httpx is not None and isinstance(exc, httpx.TimeoutException):
            return LLMError("Ollama took too long to respond. Try again or raise OLLAMA_TIMEOUT.")
        return LLMError(f"Ollama error: {getattr(exc, 'error', None) or exc}")
