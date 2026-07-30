#!/usr/bin/env bash
# Git Bash launcher — starts Ollama if needed, then the copilot REPL.
OLLAMA="$LOCALAPPDATA/Programs/Ollama/ollama.exe"
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "starting Ollama..."
  ("$OLLAMA" serve >/dev/null 2>&1 &)
  sleep 4
fi
exec ./.venv/Scripts/python -m copilot.chat "$@"
