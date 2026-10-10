@echo off
rem Start the GLM-5.3-Flash server (OpenAI / Anthropic compatible) from strata-glm5.json next to this file (2026-10-10).
rem   START-GLM5.bat            - port 8097 (or the "port" in strata-glm5.json)
rem First: build-glm5.bat, then python tools\glm5_ref\make_glm5_config.py --gguf <shard 1 .gguf>  (writes strata-glm5.json)
rem Log: build-engine\server-glm5.log (the server) and the "log" path in strata-glm5.json (the engine).
setlocal
cd /d "%~dp0"
if not exist "strata-glm5.json" echo strata-glm5.json not found: run  python tools\glm5_ref\make_glm5_config.py --gguf ^<shard 1 .gguf^> && exit /b 1
set PY=python
if exist "%~dp0.venv\Scripts\python.exe" set PY=%~dp0.venv\Scripts\python.exe
if not exist "%~dp0.venv\Scripts\python.exe" if exist "%~dp0..\strata\.venv\Scripts\python.exe" set PY=%~dp0..\strata\.venv\Scripts\python.exe
set PORT=
for /f "usebackq delims=" %%p in (`"%PY%" -c "import json;print(json.load(open('strata-glm5.json',encoding='utf-8')).get('port',8097))"`) do set PORT=%%p
if "%PORT%"=="" set PORT=8097
netstat -ano | findstr ":%PORT%" | findstr LISTENING >nul
if not errorlevel 1 echo a server already listens on %PORT% && exit /b 0
if not exist "build-engine" mkdir build-engine
echo starting GLM-5.3-Flash on http://127.0.0.1:%PORT%/v1 (the first start pins the RAM tier: about a minute)
"%PY%" "%~dp0serve\server.py" --engine strata --config "%~dp0strata-glm5.json" --port %PORT% > "%~dp0build-engine\server-glm5.log" 2>&1
