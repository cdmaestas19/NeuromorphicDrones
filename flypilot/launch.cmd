@echo off
rem One command for everything: sim + camera window, optionally a flight.
rem   launch.cmd             sim + camera
rem   launch.cmd waypoints   sim + camera, then fly the waypoint route (Ctrl-C lands)
setlocal
set "WSL=wsl.exe"
if defined FLYPILOT_DISTRO set "WSL=wsl.exe -d %FLYPILOT_DISTRO%"
%WSL% --cd "%~dp0." -- bash ./py.sh launch.py %*
