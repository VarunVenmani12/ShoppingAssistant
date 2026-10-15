"""
Web UI for the AI Shopping Assistant.

A small local web server (Starlette + uvicorn, both already installed with the
``mcp`` package) that serves a chat page and streams every step of the
MCP flow to the browser, so the trace panel shows

    USER -> LLM -> MCP call -> MCP result -> LLM -> answer

as it happens. The architecture is unchanged: this module only replaces the
CLI. It uses the same ShoppingAssistant, OllamaLLM and ShoppingMCPClient
classes, and the client still talks to the MCP server only through MCP.

Run:
    python web_app.py                 (opens http://127.0.0.1:8000)
    python web_app.py --no-browser --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import threading
import webbrowser
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from client.config import Settings, load_settings
from client.llm_client import LLMError, OllamaLLM
from client.mcp_client import MCPConnectionError, ShoppingMCPClient
from client.shopping_assistant import DEMO_QUESTIONS, ShoppingAssistant, _setup_logging

log = logging.getLogger("shopping.app")

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{8,64}$")
MAX_MESSAGE_LENGTH = 1000
MAX_SESSIONS = 50


@dataclass
class AppState:
    settings: Settings
    llm: OllamaLLM | None = None
    llm_error: str | None = None
    mcp: ShoppingMCPClient | None = None
    mcp_error: str | None = None
    sessions: "OrderedDict[str, ShoppingAssistant]" = field(default_factory=OrderedDict)
    # One question at a time: a local 8B model can only answer one request efficiently.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def ensure_llm(self) -> None:
        """Verify Ollama; re-checks on every request until it succeeds once."""
        if self.llm is not None and self.llm_error is None:
            return
        s = self.settings
        try:
            if self.llm is None:
                self.llm = OllamaLLM(
                    host=s.ollama_host, model=s.ollama_model, think=s.ollama_think,
                    temperature=s.ollama_temperature, timeout=s.ollama_timeout,
                )
            await self.llm.verify()
            self.llm_error = None
        except LLMError as error:
            self.llm_error = str(error)
            raise

    def assistant_for(self, session_id: str) -> ShoppingAssistant:
        assert self.llm is not None and self.mcp is not None
        assistant = self.sessions.get(session_id)
        if assistant is None:
            assistant = ShoppingAssistant(self.llm, self.mcp, self.settings.max_tool_rounds)
            self.sessions[session_id] = assistant
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.popitem(last=False)
        self.sessions.move_to_end(session_id)
        return assistant


def _state(request: Request) -> AppState:
    return request.app.state.shop


# --------------------------------------------------------------------------- #
# Lifespan: connect to Ollama and the MCP server once, for the whole app
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: Starlette):
    settings: Settings = app.state.settings
    state = AppState(settings)
    app.state.shop = state

    try:
        await state.ensure_llm()
    except LLMError as error:
        log.warning("[LLM] %s", error)

    mcp_client = ShoppingMCPClient(server_log_level=settings.server_log_level)
    try:
        await mcp_client.connect()
        state.mcp = mcp_client
    except MCPConnectionError as error:
        state.mcp_error = str(error)
        log.error("[MCP] %s", error)

    try:
        yield
    finally:
        if state.mcp is not None:
            await state.mcp.close()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
async def index(request: Request) -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


async def status(request: Request) -> JSONResponse:
    state = _state(request)
    if state.llm_error:
        try:  # lets the page turn green once Ollama is started
            await state.ensure_llm()
        except LLMError:
            pass
    tools = []
    if state.mcp is not None:
        for tool in state.mcp.tools:
            props = (tool.inputSchema or {}).get("properties", {})
            tools.append(
                {
                    "name": tool.name,
                    "description": (tool.description or "").strip().splitlines()[0],
                    "parameters": list(props.keys()),
                }
            )
    s = state.settings
    return JSONResponse(
        {
            "model": s.ollama_model,
            "ollama_host": s.ollama_host,
            "protocol": "MCP (stdio transport)",
            "ollama": {"ok": state.llm_error is None and state.llm is not None,
                       "error": state.llm_error},
            "mcp": {"ok": state.mcp is not None, "error": state.mcp_error,
                    "server": state.mcp.server_name if state.mcp else None, "tools": tools},
            "examples": DEMO_QUESTIONS + [
                "Find laptops under ₹50000 with 16GB RAM and 1TB SSD",
                "Find a laptop under ₹50000 and compare the prices of the matching products",
            ],
        }
    )


async def _read_json(request: Request) -> dict[str, Any] | None:
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def _valid_session(value: Any) -> str | None:
    return value if isinstance(value, str) and SESSION_ID_PATTERN.match(value) else None


async def chat(request: Request) -> JSONResponse | StreamingResponse:
    state = _state(request)
    body = await _read_json(request)
    if body is None:
        return JSONResponse({"error": "Request body must be JSON."}, status_code=400)
    session_id = _valid_session(body.get("session_id"))
    message = body.get("message")
    if session_id is None:
        return JSONResponse({"error": "Missing or invalid session_id."}, status_code=400)
    if not isinstance(message, str) or not message.strip():
        return JSONResponse({"error": "Please type a question."}, status_code=400)
    if len(message) > MAX_MESSAGE_LENGTH:
        return JSONResponse(
            {"error": f"Message is too long (max {MAX_MESSAGE_LENGTH} characters)."},
            status_code=400,
        )

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def worker() -> None:
        try:
            if state.lock.locked():
                await queue.put({"type": "info", "text": "Waiting for the previous question to finish..."})
            async with state.lock:
                if state.mcp is None:
                    raise MCPConnectionError(state.mcp_error or "MCP server is not connected.")
                await state.ensure_llm()
                assistant = state.assistant_for(session_id)
                await assistant.ask(message, on_event=queue.put)
        except LLMError as error:
            await queue.put({"type": "error", "text": str(error)})
        except MCPConnectionError as error:
            await queue.put({"type": "error", "text": str(error)})
        except Exception:  # never send stack traces to the browser
            log.exception("Unexpected error while answering")
            await queue.put({"type": "error", "text": "Unexpected error. See the server console."})
        finally:
            await queue.put(None)

    task = asyncio.create_task(worker())

    async def stream():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"
        await task

    return StreamingResponse(
        stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"}
    )


async def reset(request: Request) -> JSONResponse:
    body = await _read_json(request) or {}
    session_id = _valid_session(body.get("session_id"))
    if session_id:
        _state(request).sessions.pop(session_id, None)
    return JSONResponse({"ok": True})


def create_app(settings: Settings) -> Starlette:
    allowed_hosts = ["127.0.0.1", "localhost", settings.web_host]
    app = Starlette(
        routes=[
            Route("/", index),
            Route("/api/status", status),
            Route("/api/chat", chat, methods=["POST"]),
            Route("/api/reset", reset, methods=["POST"]),
        ],
        middleware=[Middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)],
        lifespan=lifespan,
    )
    app.state.settings = settings
    return app


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="AI Shopping Assistant - web UI")
    parser.add_argument("--port", type=int, help="port to listen on (default from .env, 8000)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = parser.parse_args()

    settings = load_settings()
    if args.port:
        settings = Settings(**{**settings.__dict__, "web_port": args.port})
    _setup_logging(settings.log_level)

    url = f"http://{'127.0.0.1' if settings.web_host in ('0.0.0.0', '') else settings.web_host}:{settings.web_port}"
    line = "=" * 50
    print(f"{line}\n     AI SHOPPING ASSISTANT - WEB UI\n{line}")
    print(f"Model     : {settings.ollama_model} via Ollama ({settings.ollama_host})")
    print("Protocol  : MCP (stdio transport)")
    print(f"Open      : {url}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    config = uvicorn.Config(
        create_app(settings), host=settings.web_host, port=settings.web_port,
        log_level="warning", access_log=False,
    )
    try:
        asyncio.run(uvicorn.Server(config).serve())
    except KeyboardInterrupt:
        pass
    except OSError as error:
        print(f"Error: could not start the web server on port {settings.web_port} ({error}). "
              "Try another port: python web_app.py --port 8080")
        sys.exit(1)
    print("Goodbye!")


if __name__ == "__main__":
    main()
