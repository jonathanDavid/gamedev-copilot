@echo off
REM One-command launcher: starts Ollama if needed, then the copilot REPL.
set OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I "ollama.exe" >NUL
if errorlevel 1 (
  echo starting Ollama...
  start "" /B "%OLLAMA%" serve
  timeout /t 4 /nobreak >NUL
)
.venv\Scripts\python -m copilot.chat
