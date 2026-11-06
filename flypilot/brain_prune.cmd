@echo off
rem Activity-based pruning of the fly brain (see flybrain/README.md): run the whole brain on many looming
rem tests, keep the neurons that spike, and check the result on held-out tests:
rem   brain_prune.cmd                     check, run, build, evaluate
rem   brain_prune.cmd evaluate            one step (check / run / build / evaluate)
rem   brain_prune.cmd --procs 6           fewer processes at once (the PC stays responsive)
setlocal
set "WSL=wsl.exe"
if defined FLYPILOT_DISTRO set "WSL=wsl.exe -d %FLYPILOT_DISTRO%"
%WSL% --cd "%~dp0..\flybrain" -- bash ./py.sh prune.py %*
