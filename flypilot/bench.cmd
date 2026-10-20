@echo off
rem Replay a recorded flight through an avoidance policy and report the decision cost:
rem   bench.cmd recordings\blocked.jsonl [--policy geometric] [--repeat 20]
setlocal
set "WSL=wsl.exe"
if defined FLYPILOT_DISTRO set "WSL=wsl.exe -d %FLYPILOT_DISTRO%"
rem WSL runs the command through the Linux shell, which eats backslashes: pass paths with / instead.
set "ARGS=%*"
if defined ARGS set "ARGS=%ARGS:\=/%"
%WSL% --cd "%~dp0." -- bash ./py.sh bench.py %ARGS%
