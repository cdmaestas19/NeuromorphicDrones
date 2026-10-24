@echo off
rem Train the camera's person / vehicle classifier on the recorded flights in flypilot\recordings
rem (see flypilot/README.md, "Person or vehicle"):
rem   camera_train.cmd [--test PREFIX,...] [--epochs 15] [--rebuild]
setlocal
set "WSL=wsl.exe"
if defined FLYPILOT_DISTRO set "WSL=wsl.exe -d %FLYPILOT_DISTRO%"
%WSL% --cd "%~dp0" -- bash ./camera_train.sh %*
