@echo off
rem Kill every CrazySim process inside WSL (firmware, simulator, camera bridge).
setlocal
set "WSL=wsl.exe"
if defined FLYPILOT_DISTRO set "WSL=wsl.exe -d %FLYPILOT_DISTRO%"
%WSL% --cd "%~dp0." -- bash ./stop_sim.sh
