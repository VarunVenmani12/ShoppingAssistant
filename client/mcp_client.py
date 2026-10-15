"""
MCP client for the Shopping Assistant.

Responsibilities (MCP only - no LLM logic here):
    * start the MCP server as a subprocess and connect over stdio
    * perform the MCP handshake (initialize)
    * discover tools (tools/list)
    * execute tools (tools/call) and return their results

The client never imports the server's Python functions. Everything goes
through the Model Context Protocol (JSON-RPC 2.0 messages over stdin/stdout).

Quick self-test (lists tools and calls each one through MCP):
    python -m client.mcp_client
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

from client.config import PROJECT_ROOT

log = logging.getLogger("shopping.mcp")

CONNECT_TIMEOUT_SECONDS = 20
TOOL_TIMEOUT_SECONDS = 30
# Must match PROVIDER_ENV_PREFIXES in server/providers/registry.py.
PROVIDER_ENV_PREFIXES = ("SHOPPING_", "AMAZON_", "FLIPKART_")


class MCPConnectionError(RuntimeError):
    """Raised when the MCP server cannot be started or reached."""


@dataclass
class ToolResult:
    tool: str
    arguments: dict[str, Any]
    text: str
    is_error: bool


class ShoppingMCPClient:
    """Async context manager wrapping one MCP client session."""

    def __init__(self, server_log_level: str = "INFO") -> None:
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self.tools: list[Any] = []  # mcp.types.Tool objects
        self.server_name = "unknown"
        self._server_log_level = server_log_level

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #
    def _server_parameters(self) -> StdioServerParameters:
        env = get_default_environment()
        env.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
                "SERVER_LOG_LEVEL": self._server_log_level,
                "PYTHONPATH": str(PROJECT_ROOT),
            }
        )
        # Pass through virtual-env info so the same interpreter/packages are used.
        for key in ("VIRTUAL_ENV", "SYSTEMROOT", "TEMP", "TMP"):
            if key in os.environ:
                env[key] = os.environ[key]
        # Data-provider settings (SHOPPING_PROVIDERS, AMAZON_*, FLIPKART_*). The SDK's
        # default environment drops them; the server also reads the same keys from .env.
        for key, value in os.environ.items():
            if key.startswith(PROVIDER_ENV_PREFIXES):
                env[key] = value
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", "server.mcp_server"],
            cwd=str(PROJECT_ROOT),
            env=env,
        )

    async def connect(self) -> None:
        log.info("[MCP] Starting MCP server (stdio transport)...")
        try:
            read, write = await self._stack.enter_async_context(
                stdio_client(self._server_parameters())
            )
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            init = await asyncio.wait_for(self._session.initialize(), CONNECT_TIMEOUT_SECONDS)
            self.server_name = init.serverInfo.name
            listed = await asyncio.wait_for(self._session.list_tools(), CONNECT_TIMEOUT_SECONDS)
            self.tools = list(listed.tools)
        except Exception as exc:  # includes timeouts and process start failures
            await self.close()
            raise MCPConnectionError(
                "Unable to start or connect to the MCP server "
                f"({type(exc).__name__}). Check that 'server/mcp_server.py' exists and "
                "that dependencies are installed (pip install -r requirements.txt)."
            ) from exc

        log.info("[MCP] Connected to server '%s'", self.server_name)
        log.info("[MCP] Discovered tools: %s", ", ".join(t.name for t in self.tools))

    async def close(self) -> None:
        try:
            await self._stack.aclose()
        except Exception:  # pragma: no cover - best effort shutdown
            pass
        self._session = None

    async def __aenter__(self) -> "ShoppingMCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    def get_tool(self, name: str) -> Any | None:
        return next((t for t in self.tools if t.name == name), None)

    def ollama_tool_definitions(self) -> list[dict[str, Any]]:
        """Translate MCP tool schemas into the function-calling format Ollama expects."""
        definitions = []
        for tool in self.tools:
            schema = tool.inputSchema or {"type": "object", "properties": {}}
            properties = {
                name: _simplify_property(prop)
                for name, prop in schema.get("properties", {}).items()
            }
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": (tool.description or "").strip(),
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": schema.get("required", []),
                        },
                    },
                }
            )
        return definitions

    async def call_tool(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        """Execute an MCP tool and return its (text) result.

        Unknown tools and invalid arguments are reported as error results
        instead of exceptions, so the LLM can recover.
        """
        if self._session is None:
            raise MCPConnectionError("MCP session is not connected.")

        tool = self.get_tool(name)
        if tool is None:
            return ToolResult(name, arguments or {}, json.dumps(
                {"error": f"Unknown tool '{name}'. Available: {', '.join(self.tool_names())}"}
            ), True)

        try:
            clean_args = sanitize_arguments(tool.inputSchema or {}, arguments)
        except ValueError as exc:
            return ToolResult(name, arguments or {}, json.dumps({"error": str(exc)}), True)

        log.info("[MCP] Calling %s(%s)", name, _format_args(clean_args))
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, clean_args), TOOL_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            return ToolResult(name, clean_args, json.dumps({"error": "Tool call timed out."}), True)
        except Exception as exc:
            raise MCPConnectionError(f"MCP server stopped responding ({type(exc).__name__}).") from exc

        text = "\n".join(getattr(c, "text", "") for c in result.content if hasattr(c, "text"))
        is_error = bool(result.isError) or _payload_has_error(text)
        log.info("[MCP] Tool %s completed%s", name, " (returned an error)" if is_error else "")
        return ToolResult(name, clean_args, text or "{}", is_error)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _simplify_property(prop: dict[str, Any]) -> dict[str, Any]:
    """Collapse 'anyOf: [{type: X}, {type: null}]' into a plain type for the LLM."""
    result: dict[str, Any] = {}
    prop_type = prop.get("type")
    if prop_type is None and "anyOf" in prop:
        types = [p.get("type") for p in prop["anyOf"] if p.get("type") not in (None, "null")]
        prop_type = types[0] if types else "string"
    result["type"] = prop_type or "string"
    if "description" in prop:
        result["description"] = prop["description"]
    return result


def _property_type(prop: dict[str, Any]) -> str:
    return _simplify_property(prop)["type"]


def sanitize_arguments(schema: dict[str, Any], arguments: Any) -> dict[str, Any]:
    """Validate and normalise LLM-produced arguments against the tool's JSON schema.

    * accepts a dict or a JSON string
    * drops unknown keys and empty/null values
    * converts "₹50,000" -> 50000.0 for numeric fields and 16 -> "16" for strings
    * checks required fields are present
    """
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError("Tool arguments were not valid JSON.") from exc
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    properties: dict[str, Any] = schema.get("properties", {})
    clean: dict[str, Any] = {}
    for key, value in arguments.items():
        if key not in properties:
            log.debug("[MCP] Dropping unknown argument %r", key)
            continue
        if value is None or (isinstance(value, str) and value.strip().lower() in {"", "null", "none", "any"}):
            continue
        expected = _property_type(properties[key])
        if expected in ("number", "integer"):
            if isinstance(value, bool):
                raise ValueError(f"'{key}' must be a number.")
            if isinstance(value, str):
                digits = value.lower().replace("₹", "").replace("rs.", "").replace("inr", "")
                digits = digits.replace(",", "").replace(" ", "")
                multiplier = 1000 if digits.endswith("k") else 1
                digits = digits.rstrip("k")
                try:
                    value = float(digits) * multiplier
                except ValueError as exc:
                    raise ValueError(f"'{key}' must be a number, got {value!r}.") from exc
            if not isinstance(value, (int, float)):
                raise ValueError(f"'{key}' must be a number.")
            if expected == "integer":
                value = int(value)
        elif expected == "string":
            if isinstance(value, (dict, list)):
                raise ValueError(f"'{key}' must be text.")
            if isinstance(value, (int, float)) and key in ("ram", "storage"):
                value = f"{value:g}GB"
            value = str(value).strip()
        clean[key] = value

    missing = [r for r in schema.get("required", []) if r not in clean]
    if missing:
        raise ValueError(f"Missing required argument(s): {', '.join(missing)}")
    return clean


def _payload_has_error(text: str) -> bool:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(data, dict) and "error" in data


def _format_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


# ---------------------------------------------------------------------- #
# Stand-alone MCP smoke test (no LLM involved)
# ---------------------------------------------------------------------- #
async def _self_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    async with ShoppingMCPClient() as client:
        print("\nTools exposed by the MCP server:")
        for tool in client.tools:
            print(f"  - {tool.name}: {(tool.description or '').splitlines()[0]}")
        for name, args in [
            ("search_product", {"category": "laptop", "max_price": 50000}),
            ("compare_price", {"product_id": "L001"}),
            ("get_details", {"product_id": "L001"}),
            ("get_details", {"product_id": "X999"}),
        ]:
            result = await client.call_tool(name, args)
            preview = result.text if len(result.text) < 400 else result.text[:400] + " ..."
            print(f"\n{name}({args}) ->\n{preview}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        asyncio.run(_self_test())
    except MCPConnectionError as error:
        print(f"Error: {error}")
        sys.exit(1)
