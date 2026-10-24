@echo off
rem Check the fly-brain model on this PC against published results (see flybrain/README.md):
rem   brain_reproduce.cmd [sugar] [escape] [--trials 10]
setlocal
set "WSL=wsl.exe"
if defined FLYPILOT_DISTRO set "WSL=wsl.exe -d %FLYPILOT_DISTRO%"
%WSL% --cd "%~dp0..\flybrain" -- bash ./py.sh reproduce.py %*
