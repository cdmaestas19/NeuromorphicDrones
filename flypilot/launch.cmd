@echo off
rem One command for everything, all shown in the MuJoCo window:
rem   launch.cmd             sim (drone camera inset top-left)
rem   launch.cmd waypoints   sim, then fly the waypoint route (Ctrl-C lands)
rem   launch.cmd avoid       sim + avoidance dashboard, then the avoidance demo
setlocal
set "WSL=wsl.exe"
if defined FLYPILOT_DISTRO set "WSL=wsl.exe -d %FLYPILOT_DISTRO%"
rem WSL runs the command through the Linux shell, which eats backslashes: pass paths with / instead.
set "ARGS=%*"
if defined ARGS set "ARGS=%ARGS:\=/%"
%WSL% --cd "%~dp0." -- bash ./py.sh launch.py %ARGS%
