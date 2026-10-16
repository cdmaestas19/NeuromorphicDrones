@echo off
rem Start CrazySim (MuJoCo window + AI-deck camera stream + obstacle scene) inside WSL.
rem Close the MuJoCo window or press Ctrl-C to stop. Extra args go to run_sim.sh.
setlocal
set "WSL=wsl.exe"
if defined FLYPILOT_DISTRO set "WSL=wsl.exe -d %FLYPILOT_DISTRO%"
%WSL% --cd "%~dp0." -- bash ./run_sim.sh %*
