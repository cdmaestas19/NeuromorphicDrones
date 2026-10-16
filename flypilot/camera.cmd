@echo off
rem Live AI-deck camera view (start run_sim.cmd first). q or Esc quits, s saves a snapshot.
setlocal
set "WSL=wsl.exe"
if defined FLYPILOT_DISTRO set "WSL=wsl.exe -d %FLYPILOT_DISTRO%"
%WSL% --cd "%~dp0." -- bash ./py.sh camera_view.py %*
