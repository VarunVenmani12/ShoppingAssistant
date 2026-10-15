@echo off
REM ==========================================================================
REM  Shopping Assistant - one-click setup and verification (Windows)
REM  Creates .venv, installs requirements, checks Ollama/qwen3:8b, runs the
REM  tests and the sample questions. Everything is also written to
REM  verification_log.txt in this folder.
REM ==========================================================================
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set LOG=verification_log.txt

echo Running setup and verification. This can take a few minutes...
echo (Progress is written to %LOG%)

echo ===== Shopping Assistant verification  %date% %time% ===== > %LOG%

set PY=
where python >nul 2>&1 && set PY=python
if not defined PY (where py >nul 2>&1 && set PY=py -3)
if not defined PY (
  echo ERROR: Python was not found on PATH. >> %LOG%
  echo ERROR: Python was not found on PATH. Install Python 3.10+ from python.org.
  goto end
)
echo ----- [1] Python ----- >> %LOG%
%PY% --version >> %LOG% 2>&1

echo ----- [2] Virtual environment + dependencies ----- >> %LOG%
if not exist .venv\Scripts\python.exe (
  %PY% -m venv .venv >> %LOG% 2>&1
)
set VPY=.venv\Scripts\python.exe
%VPY% -m pip install --disable-pip-version-check -q -r requirements.txt >> %LOG% 2>&1
%VPY% -m pip list --disable-pip-version-check 2>nul | findstr /i "mcp ollama pytest pydantic httpx" >> %LOG%

echo ----- [3] ollama list ----- >> %LOG%
ollama list >> %LOG% 2>&1

echo ----- [4] Ollama connectivity (GET /api/version) ----- >> %LOG%
curl -s -m 10 http://localhost:11434/api/version >> %LOG% 2>&1
echo. >> %LOG%

echo ----- [5] Unit + MCP protocol tests ----- >> %LOG%
%VPY% -m pytest -v -p no:cacheprovider >> %LOG% 2>&1

echo ----- [6] MCP server self-test (client -^> MCP -^> server, no LLM) ----- >> %LOG%
%VPY% -m client.mcp_client >> %LOG% 2>&1

echo ----- [7] Required sample questions with qwen3:8b ----- >> %LOG%
%VPY% main.py --demo >> %LOG% 2>&1

echo ----- [8] Extra examples (filters + multiple tool calls + invalid id) ----- >> %LOG%
%VPY% main.py --ask "Find laptops under ₹50000 with 16GB RAM and 1TB SSD" --ask "Find a laptop under ₹50000 and compare the prices of the matching products" --ask "Give me the specifications of X999" >> %LOG% 2>&1

echo ----- [9] Error handling: Ollama unreachable ----- >> %LOG%
set OLLAMA_HOST=http://localhost:11999
%VPY% main.py --ask "hi" >> %LOG% 2>&1
set OLLAMA_HOST=

echo ----- [10] Error handling: model not installed ----- >> %LOG%
set OLLAMA_MODEL=not-a-real-model:1b
%VPY% main.py --ask "hi" >> %LOG% 2>&1
set OLLAMA_MODEL=

:end
echo ===== DONE ===== >> %LOG%
echo.
echo Finished. See %LOG% for the results.
pause
