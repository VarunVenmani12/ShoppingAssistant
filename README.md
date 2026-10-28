# AI Shopping Assistant (Ollama + Qwen3 + MCP)

A small, fully local shopping assistant that shows how the **Model Context Protocol (MCP)**
connects a local LLM to tools and data.

* **LLM:** `qwen3:8b` running in [Ollama](https://ollama.com) on your machine
* **Protocol:** MCP (official `mcp` Python SDK, stdio transport)
* **Data:** a local demo catalogue, `products.json` (19 products, prices in ₹ INR)
* **No cloud LLM, no API keys, no web scraping, no LangChain / RAG / vector DB**
* **Pluggable data providers:** local JSON by default; Amazon/Flipkart placeholders for later phases

> **Demo data:** all products, prices and store names are fictional sample values
> from `products.json`. They are **not** live Amazon, Flipkart or other marketplace prices.

### Quick start (Windows)

1. Install Python 3.10+ and [Ollama](https://ollama.com), then run `ollama pull qwen3:8b`.
2. Double-click `setup_and_verify.bat` once (creates `.venv`, installs requirements, runs the tests).
3. Optional: copy `.env.example` to `.env` to change settings. The default model is
   `qwen3:8b`; if you have a `.env`, make sure it says `OLLAMA_MODEL=qwen3:8b`.
4. Double-click `run_web.bat` (browser UI at http://127.0.0.1:8000) or `run_assistant.bat` (CLI).
   Keep the laptop on AC power; on battery the CPU is throttled and answers get much slower.

### Supported features

* Product search with filters: category, price range, brand, minimum RAM/storage, keyword.
* Product details (full specifications) and price comparison across the demo stores,
  including the best in-stock deal.
* Follow-up questions in the same conversation ("details for the first one", "compare its prices").
* Browser UI with a live MCP trace panel, and a CLI with demo and one-shot modes.
* Friendly errors for unknown product IDs, invalid filters, Ollama not running or a missing model;
  every MCP tool call is limited to 30 seconds and returns an error instead of hanging.
* `[TIMING]` logs for each LLM request, MCP call and question (no message text logged).

---

## 1. Project overview

You type a shopping question such as *"Find laptops under ₹50000 with 16GB RAM"*.

1. **Qwen3** works out what you want and which tool it needs, e.g.
   `search_product(category="laptop", max_price=50000, ram="16GB")`.
2. The **MCP client** sends that tool call to the **MCP server** using the MCP protocol.
3. The **MCP server** runs the tool, which reads `products.json`, and returns the result over MCP.
4. **Qwen3** turns the result into a friendly answer.

Every step is logged, so you can see MCP at work:

```text
[USER] Find laptops under ₹50000
[LLM] Determining required tool...
[LLM] Model requested 1 tool call(s): search_product
[MCP] Calling search_product(category='laptop', max_price=50000.0)
[SERVER] search_product called with {'category': 'laptop', 'max_price': 50000.0}
[SERVER] search_product -> 3 product(s) found
[MCP] Tool search_product completed
[LLM] Generating final response...
```

---

## 2. Architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                         Shopping Assistant (CLI)                         │
│                      client/shopping_assistant.py                        │
│                                                                          │
│   ┌──────────────────────┐              ┌──────────────────────────┐     │
│   │  LLM client          │              │  MCP client              │     │
│   │  client/llm_client.py│              │  client/mcp_client.py    │     │
│   └──────────┬───────────┘              └────────────┬─────────────┘     │
└──────────────┼───────────────────────────────────────┼───────────────────┘
               │ HTTP (localhost:11434)                │ MCP = JSON-RPC 2.0
               ▼                                       │ over stdin/stdout
     ┌───────────────────┐                             ▼
     │  Ollama            │               ┌─────────────────────────────┐
     │  model: qwen3:8b   │               │  MCP server (subprocess)    │
     │  • understands     │               │  server/mcp_server.py       │
     │  • picks tools     │               │  tools:                     │
     │  • writes answer   │               │   • search_product          │
     └───────────────────┘               │   • compare_price           │
                                          │   • get_details             │
                                          └──────────────┬──────────────┘
                                                         │ read-only
                                                         ▼
                                                  products.json
```

### Conceptual flow

```text
User
 ↓
Shopping Assistant
 ↓
Qwen3:8b
 ↓
MCP Client
 ↓
MCP Protocol
 ↓
MCP Server
 ↓
search_product()
 ↓
products.json
 ↓
MCP Server
 ↓
MCP Client
 ↓
Qwen3:8b
 ↓
User
```

---

## 3. What is MCP?

The **Model Context Protocol** is an open standard for connecting AI applications
to tools and data. Instead of hard-wiring functions into an app, a **server**
*advertises* tools (name, description, JSON schema for the arguments) and a
**client** can discover and call them using standard JSON-RPC messages:

| MCP message   | Used for                                         |
|---------------|--------------------------------------------------|
| `initialize`  | handshake: protocol version, server name, capabilities |
| `tools/list`  | discover which tools exist and their argument schemas |
| `tools/call`  | run one tool with arguments and get the result   |

Because the contract is the protocol, the same server could be used by any
MCP-capable app (Claude Desktop, VS Code, MCP Inspector, ...), and the same
client could talk to any MCP server.

**Transport:** this project uses **stdio**. The client starts the server as a
child process and they exchange MCP messages over its stdin/stdout. It is the
simplest local transport, needs no ports, and works on Windows.

## 4. The MCP client (`client/mcp_client.py`)

* Starts `python -m server.mcp_server` as a subprocess and connects over stdio.
* Performs the MCP handshake (`initialize`).
* Discovers tools with `tools/list` and converts their JSON schemas into the
  function-calling format Ollama understands (so the LLM sees exactly what
  the server advertises; nothing is hard-coded).
* Validates/normalises the LLM's arguments against each tool's schema
  (drops unknown keys, turns `"₹50,000"` into `50000`, rejects bad JSON).
* Executes tools with `tools/call` and returns the result text.
* **Never imports the server's Python code.** Everything goes through MCP.

## 5. The MCP server (`server/mcp_server.py`)

Built with `FastMCP` from the official SDK. It exposes three tools:

| Tool             | Parameters                                                             | Returns |
|------------------|------------------------------------------------------------------------|---------|
| `search_product` | `category`, `max_price`, `min_price`, `brand`, `ram`, `storage`, `keyword` (all optional) | matching products, cheapest first |
| `compare_price`  | `product_id`                                                           | price at each store, best deal, max saving |
| `get_details`    | `product_id`                                                           | complete product record and specifications |

Notes:

* `ram` and `storage` are **minimum** requirements (`16GB` matches 16GB and 32GB).
* Categories accept friendly aliases (`laptops`, `phone`, `mobile`, `mice`, ...).
* The server has **no LLM logic**. Tool calls go through a small data layer:
  `server/catalog.py` asks the enabled **providers** (`server/providers/`) and
  formats their products with the helpers in `server/product_store.py`.
  The default provider reads `products.json`; see [9b. Data providers](#9b-data-providers).
* Errors (unknown ID, bad category, bad price...) are returned as
  `{"error": "..."}` so the LLM can explain them; stack traces are never sent.

## 6. Role of Ollama / Qwen3:8b (`client/llm_client.py`)

Qwen3 is the "brain":

1. **Understands** the request ("under ₹50000 with 16GB RAM and 1TB SSD").
2. **Decides** which MCP tool(s) to call and fills in the arguments
   (native Ollama tool calling).
3. **Writes** the final answer from the tool results.

The assistant loop (`client/shopping_assistant.py`) lets the model call tools
several times per question (up to `MAX_TOOL_ROUNDS`), so a request like
*"find laptops under ₹50000 and compare their prices"* becomes
`search_product` → `compare_price` × N → answer.

Qwen3's "thinking" mode is off by default (`OLLAMA_THINK=false`) for faster
replies; any `<think>` text is stripped from the output.

---

## 7. Project structure

```text
ShoppingAssistant
│
├── README.md              this file
├── requirements.txt       Python dependencies (mcp, ollama, pytest)
├── products.json          demo product catalogue (INR)
├── .env.example           configuration template
├── .env                   your local configuration (git-ignored)
├── .gitignore
├── main.py                shortcut: python main.py
├── web_app.py             shortcut: python web_app.py (browser UI)
├── setup_and_verify.bat   one-click: venv + install + checks + tests + demo (Windows)
├── run_assistant.bat      one-click: start the CLI chat (Windows)
├── run_web.bat            one-click: start the web UI (Windows)
│
├── server
│   ├── __init__.py
│   ├── mcp_server.py      MCP server + tool definitions (stdio)
│   ├── catalog.py         tool operations on top of the enabled providers
│   ├── product_store.py   JSON loading, validation, search/compare/details helpers
│   └── providers
│       ├── base.py        Product / Offer model, normalization, provider interface
│       ├── local_json.py  LocalJSONProvider (products.json, default)
│       ├── marketplace.py Amazon / Flipkart placeholders (no live calls, no data)
│       ├── registry.py    SHOPPING_PROVIDERS parsing and provider creation
│       └── __main__.py    python -m server.providers  (print provider status)
│
├── client
│   ├── __init__.py
│   ├── config.py          reads .env / environment variables
│   ├── llm_client.py      Ollama wrapper (connectivity check, chat, tool calls)
│   ├── mcp_client.py      MCP client (connect, list tools, call tools)
│   └── shopping_assistant.py   assistant loop + CLI
│
├── webui
│   ├── __init__.py
│   ├── app.py             local web server (Starlette + uvicorn), streams the MCP trace
│   └── static
│       └── index.html     chat page + live MCP trace panel (no external resources)
│
└── tests
    ├── __init__.py
    ├── test_products.py   data loading, search, compare, details, invalid IDs
    ├── test_server.py     real MCP protocol tests against the server
    ├── test_assistant.py  full loop with a scripted fake LLM + real MCP
    ├── test_providers.py  provider selection, normalization, placeholders, compatibility
    └── test_webui.py      web server: page, status, validation, Ollama-down error
```

---

## 8. Installation (Windows)

Requirements: **Python 3.10+**, **Ollama** running, and the model `qwen3:8b`.

**Quickest way:** double-click `setup_and_verify.bat`. It creates `.venv`,
installs the requirements, runs `ollama list`, checks Ollama connectivity,
runs all tests and the sample questions, and writes everything to
`verification_log.txt`. Afterwards double-click `run_assistant.bat` to chat.

Manual steps:

```powershell
cd D:\Varun\AI\ShoppingAssistant

# 1. create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
#   (cmd.exe:  .venv\Scripts\activate.bat)
#   If PowerShell blocks the script:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 2. install dependencies
pip install -r requirements.txt

# 3. check Ollama and the model (no download needed if it is already listed)
ollama list          # should show qwen3:8b
```

## 9. Configuration

Settings live in `.env` (copy of `.env.example`). Environment variables override the file.

| Variable            | Default                  | Meaning |
|---------------------|--------------------------|---------|
| `OLLAMA_MODEL`      | `qwen3:8b`               | Ollama model to use |
| `OLLAMA_HOST`       | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_THINK`      | `false`                  | Qwen3 thinking mode |
| `OLLAMA_TEMPERATURE`| `0.2`                    | lower = more deterministic |
| `OLLAMA_TIMEOUT`    | `300`                    | seconds to wait for Ollama |
| `MAX_TOOL_ROUNDS`   | `5`                      | max tool-call rounds per question |
| `LOG_LEVEL`         | `INFO`                   | `INFO` shows the MCP flow, `WARNING` hides it |
| `WEB_HOST`          | `127.0.0.1`              | address the web UI listens on (keep local) |
| `WEB_PORT`          | `8000`                   | port of the web UI |
| `SERVER_LOG_LEVEL`  | `INFO`                   | logging of the MCP server (`[SERVER]` lines) |
| `SHOPPING_PROVIDERS`| `local`                  | data providers, comma-separated (see 9b) |

The model name is only defined here; the default setup needs no secrets.

## 9b. Data providers

The MCP tools get their data from **providers**. Every provider implements the
same interface (`server/providers/base.py`): `status()`, `search()`,
`get_product()` and `get_offers()`, returning a common `Product` / `Offer` model.

| Provider   | What it does today | Credentials (future) |
|------------|--------------------|----------------------|
| `local`    | Reads `products.json`. **Default.** Demo data, never reported as live. | none |
| `amazon`   | Placeholder. Makes no network calls and returns no data; reports *not configured*. | `AMAZON_CREATORS_CLIENT_ID`, `AMAZON_CREATORS_CLIENT_SECRET`, `AMAZON_PARTNER_TAG` |
| `flipkart` | Placeholder. Makes no network calls and returns no data; reports *not configured*. | `FLIPKART_AFFILIATE_ID`, `FLIPKART_AFFILIATE_TOKEN` |

`SHOPPING_PROVIDERS` turns providers on one at a time. The **first** one
answers the tools. Every other provider is asked as well, and a `providers`
list with each one's status (`available`, `live`, `message`) is added to the
result. Their results are **not** merged in yet. With the default (`local`
only) the tool output is the same as before. Unknown names are ignored with a
warning. Even with credentials filled in, the placeholders say *"credentials
were found, but live integration is not implemented yet"* and never claim to be live.

Settings are read from environment variables first, then `.env`. The MCP client
passes `SHOPPING_*`, `AMAZON_*` and `FLIPKART_*` variables on to the server subprocess.

**Windows: check or change providers**

```powershell
cd D:\Varun\AI\ShoppingAssistant
.\.venv\Scripts\Activate.ps1

python -m server.providers                      # show enabled providers and status

# try the placeholders for this terminal session only (PowerShell)
$env:SHOPPING_PROVIDERS = "local,amazon,flipkart"
python -m server.providers
python main.py --ask "Find laptops under ₹50000"
Remove-Item Env:SHOPPING_PROVIDERS              # back to the default (local)
```

In `cmd.exe` use `set SHOPPING_PROVIDERS=local,amazon` and `set SHOPPING_PROVIDERS=`
to clear it. To make the setting permanent, put it in `.env` instead.
`run_web.bat` and `run_assistant.bat` pick up `.env` automatically.

Never put real credentials in `.env.example` or commit them. `.env` is git-ignored.

## 10. Starting / testing the MCP server

You normally **don't start the server yourself**: with the stdio transport the
client launches it automatically as a subprocess.

To test the server on its own through MCP (no LLM involved):

```powershell
python -m client.mcp_client
```

This starts the server, performs the MCP handshake, lists the tools and calls
each one, printing the raw MCP results.

You can also run it directly (it then waits silently for an MCP client on
stdin; press Ctrl+C to stop):

```powershell
python -m server.mcp_server
```

Optional: inspect it with the official MCP Inspector (needs Node.js):

```powershell
npx @modelcontextprotocol/inspector python -m server.mcp_server
```

## 11. Starting the Shopping Assistant

```powershell
python main.py              # interactive chat
python main.py --demo       # runs the 5 sample questions automatically
python main.py --quiet      # chat without the [LLM]/[MCP] flow logs
python main.py --ask "Compare prices for L001"   # one-shot question(s), repeatable
```

(`python -m client.shopping_assistant` works too.)

In the chat you can type:

* any shopping question
* `tools` - list the tools discovered over MCP
* `clear` - forget the conversation so far
* `exit` or `quit` - leave

The first question can take a while because Ollama loads the model into memory.

**Timing logs.** With `LOG_LEVEL=INFO`, every question also prints `[TIMING]` lines:
each Ollama request (wall-clock time plus Ollama's own `total`, `load`, prompt and
output token counts and durations, thinking characters and message count), each
LLM step (`tool-selection` or `response-generation`), each MCP tool call, and the
query total. They contain no message text or credentials. A large gap between the
wall-clock time and Ollama's prompt + output durations usually means the model is
loading or the PC is short of RAM.

## 11b. Web UI (browser)

```powershell
python web_app.py                  # opens http://127.0.0.1:8000 in your browser
python web_app.py --port 8080      # use another port
python web_app.py --no-browser     # don't open the browser automatically
```

Or double-click `run_web.bat`. Press **Ctrl+C** in the terminal to stop it.

The page has two panels:

* **Chat** (left): type a question or click an example. Answers render
  lists, tables and ₹ prices; under each answer you see which MCP tools were used.
* **MCP trace** (right): every step streams in live while the question runs:
  `USER` → `LLM` (tool decision) → `MCP` (`tools/call` with arguments) →
  `SERVER` (result summary, expandable raw JSON) → `LLM` → `ANSWER`.

The status pills at the top show whether Ollama/qwen3:8b and the MCP server
are connected (hover for details). If Ollama is not running, the page says so
and turns green by itself once you start Ollama.

How it works: `webui/app.py` starts once, checks Ollama and connects to the MCP
server (same `OllamaLLM`, `ShoppingMCPClient` and `ShoppingAssistant` classes the
CLI uses). Each question is sent to `POST /api/chat`, which streams one JSON
line per step (`application/x-ndjson`) to the page. The web server listens on
`127.0.0.1` only, rejects other Host headers, and answers one question at a
time. Starlette and uvicorn are installed automatically with `mcp`, so no new
dependencies are needed. `WEB_HOST` and `WEB_PORT` can be set in `.env`.

```text
Browser ──HTTP──► webui/app.py ──► ShoppingAssistant ──► Ollama (qwen3:8b)
   ▲                                   │
   └──── streamed trace events ◄───────┴──► MCP client ══MCP══► MCP server ──► products.json
```

## 12. Example conversations

```text
==================================================
        AI SHOPPING ASSISTANT
==================================================

Using:
LLM       : Ollama (http://localhost:11434)
Model     : qwen3:8b
Protocol  : MCP (stdio transport)

You: Find me a laptop under ₹50000 with 16GB RAM

Assistant:
Here are laptops under ₹50,000 with at least 16GB RAM:

1. HP 15s Ryzen 5 (L002)
   Price: ₹45,999
   RAM: 16GB | Storage: 1TB SSD | Processor: AMD Ryzen 5 7520U

2. Lenovo IdeaPad Slim 3 (L001)
   Price: ₹48,999
   RAM: 16GB | Storage: 512GB SSD | Processor: AMD Ryzen 7 7730U
```

| You type | Tool(s) the model calls |
|----------|-------------------------|
| Find laptops under ₹50000 | `search_product(category="laptop", max_price=50000)` |
| Find laptops under ₹50000 with 16GB RAM and 1TB SSD | `search_product(category="laptop", max_price=50000, ram="16GB", storage="1TB")` |
| Compare the prices of Laptop L001 | `compare_price(product_id="L001")` |
| Give me the specifications of L001 | `get_details(product_id="L001")` |
| Find smartphones under ₹30000 | `search_product(category="smartphone", max_price=30000)` |
| Find a laptop under ₹50000 and compare the prices of the matching products | `search_product(...)` then `compare_price` for each match |
| Show details for X999 | `get_details(product_id="X999")` → friendly "not found" |

Follow-up questions work in the same session, e.g. *"compare prices for the second one"*.

## 13. Running the tests

```powershell
python -m pytest -v
# or, with only the standard library:
python -m unittest discover -v
```

* `test_products.py` - product loading, search, price comparison, details,
  invalid product IDs, no matching products, invalid JSON / missing file.
* `test_server.py` - starts the real MCP server and uses `tools/list` and
  `tools/call` (valid calls, unknown IDs, path-like IDs, bad JSON, unknown tool).
* `test_assistant.py` - the whole assistant loop with a scripted fake LLM
  (so it runs without Ollama) and the real MCP server, including a
  multi-tool "search then compare" turn.
* `test_providers.py` - provider selection from `SHOPPING_PROVIDERS` and `.env`,
  product normalization and missing fields, unsupported operations, the
  unconfigured Amazon/Flipkart placeholders (socket connections are blocked to
  prove they make no network calls), and checks that the default catalogue returns
  exactly what the original functions return. No credentials are needed.

Run one file on Windows: `.\.venv\Scripts\python.exe -m pytest tests\test_providers.py -v`

### End-to-end test procedure (with Ollama)

1. `ollama list` → confirm `qwen3:8b` is listed.
2. `python -m client.mcp_client` → confirm 3 tools are listed and the calls return JSON.
3. `python main.py --demo` → the 5 sample questions run; each shows `[MCP] Calling ...`
   and `(MCP tools used: ...)`.
4. `python main.py` and try:
   * `Find laptops under ₹50000 with 16GB RAM and 1TB SSD` → only L002
   * `Find a laptop under ₹50000 and compare the prices of the matching products`
     → one `search_product` + several `compare_price` calls
   * `Show details for X999` → friendly not-found message
   * press Enter on an empty line → prompt to type a question
   * `exit`
5. Error check: stop Ollama (or set `OLLAMA_HOST=http://localhost:11999`) and run
   `python main.py` → `Unable to connect to Ollama at ...`.
6. Model check: `set OLLAMA_MODEL=not-a-model` then `python main.py` → model-not-available message.

---

## 14. Troubleshooting

| Problem | Fix |
|---------|-----|
| `Unable to connect to Ollama at http://localhost:11434` | Start Ollama (open the Ollama app, or run `ollama serve`). Check `OLLAMA_HOST` in `.env`. |
| `Model 'qwen3:8b' is not available` | Run `ollama list`. Use the exact name shown there in `OLLAMA_MODEL`. |
| `Unable to start or connect to the MCP server` | Activate the virtual environment and run `pip install -r requirements.txt`. Run from the project folder. Try `python -m client.mcp_client`. |
| `ModuleNotFoundError: No module named 'client'` | Run commands from the project root (`D:\Varun\AI\ShoppingAssistant`). |
| First answer is very slow | Normal: Ollama is loading the model. Later answers are faster. Keep `OLLAMA_THINK=false`. |
| Timeout from Ollama | Increase `OLLAMA_TIMEOUT` in `.env`. |
| "Ollama took too long" after the laptop slept, or answers suddenly very slow | Sleep/standby or battery power interrupts or throttles the CPU. Keep the laptop plugged in and awake during a demo, then ask the question again; the conversation continues normally. |
| `₹` shows as `?` or garbled | Use Windows Terminal, or run `chcp 65001` first. Typing `Rs 50000` or `50000` also works. |
| The model answers without using a tool | Rephrase more specifically (e.g. include "laptop", a price or an ID) or `clear` the conversation. Small local models occasionally skip tools. |
| Web UI: `could not start the web server on port 8000` | Another program uses the port. Run `python web_app.py --port 8080`. |
| Web UI shows "Cannot reach the local web server" | The terminal running `web_app.py` was closed. Start it again. |
| PowerShell won't activate the venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## 15. The complete request flow, step by step

Example: **"Find laptops under ₹50000 with 16GB RAM"**

1. **User → Shopping Assistant.** The CLI reads the line and calls `ShoppingAssistant.ask()`.
2. **Tool discovery (done once at start-up).** The MCP client started the server
   subprocess, sent `initialize` and `tools/list`, and received the three tool
   schemas. They were converted to Ollama's tool format.
3. **Shopping Assistant → Qwen3.** The system prompt, the conversation so far,
   the user message and the tool list are sent to Ollama (`/api/chat`).
4. **Qwen3 decides.** It replies with a tool call instead of text:
   `search_product({"category": "laptop", "max_price": 50000, "ram": "16GB"})`.
5. **Qwen3 → MCP Client.** The client validates the arguments against the tool's
   schema and sends an MCP `tools/call` request (JSON-RPC over the server's stdin).
6. **MCP Server → tool → products.json.** FastMCP dispatches to `search_product`,
   which validates the filters, reads `products.json`, filters and sorts the
   products and returns JSON.
7. **MCP Server → MCP Client.** The result travels back as an MCP response on stdout.
8. **MCP Client → Qwen3.** The result is added to the conversation as a `tool`
   message and sent to Ollama again.
9. **Qwen3 either calls more tools or answers.** If it needs more data
   (e.g. `compare_price` for each match) steps 4-8 repeat. Otherwise it writes
   the final answer.
10. **Qwen3 → User.** The CLI prints the answer and waits for the next question.
    The turn is kept in the history so follow-up questions have context.

---

## Security notes

* The MCP tools can only read `products.json` (fixed path); no file paths,
  shell commands or code are accepted as input. The marketplace providers are
  placeholders and make no network calls.
* Provider credentials are only checked for presence; their values are never
  logged or returned.
* Keep credentials in `.env` only. `.gitignore` excludes `.env`, `.env.*`
  (except `.env.example`), key files (`*.pem`, `*.key`, `*.pfx`, `*.p12`) and `secrets/`.
* All tool arguments are validated twice: against the JSON schema in the
  client and again in the server (types, lengths, price ranges, product-ID
  pattern `^[A-Z]{1,3}\d{3}$`, known categories).
* Errors are returned as short messages; stack traces are never shown to the user.
* Everything runs locally; no secrets are stored in the code.

## Limitations (first version)

* Demo data only: 19 products with fictional store names and prices. The web UI
  shows a permanent "Demo data · sample prices, not live" label and the CLI banner
  says the same; nothing is presented as a live marketplace price.
* Speed on a CPU-only laptop: Ollama runs qwen3:8b on the CPU when no supported GPU
  is found (`ollama ps` shows `100% CPU`). Measured on a 4-core i5-1135G7 on AC
  power: first model load about 15-20 s, short tool-selection steps 6-37 s, final
  answers 40-75 s, a warm 3-question conversation about 200 s. On battery Windows
  throttles the CPU and answers can become several times slower; plug in for demos.
* Follow-up questions re-read part of the conversation because of how the stock
  qwen3 template handles `think=false`. An optional local `qwen3:8b-nothink` model
  tag (same weights, adjusted template) measured about 25% faster per conversation,
  but it is an experiment that is not part of this repository; the supported
  configuration is `OLLAMA_MODEL=qwen3:8b`.
* Amazon and Flipkart are placeholders: no live prices yet. Product IDs must
  still match the local pattern (`L001`), so marketplace IDs (ASINs, Flipkart
  IDs) are not accepted yet. Results from secondary providers are reported but
  not merged.
* The web UI status panel does not show provider status yet; use
  `python -m server.providers`.
* Answer quality depends on the local 8B model; it occasionally formats
  numbers differently or skips a tool call. Low temperature and a strict
  system prompt keep this rare.
* stdio transport only (one client per server process). The SDK also supports
  Streamable HTTP if you want a network server later.
* The conversation history is in memory and is lost when you exit.
