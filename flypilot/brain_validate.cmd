@echo off
rem Check the fly brain's escape circuit against the whole brain (run brain_reproduce.cmd escape first;
rem see flybrain/README.md):
rem   brain_validate.cmd [--hops 4] [--min-synapses 5] [--trials 10]
setlocal
set "WSL=wsl.exe"
if defined FLYPILOT_DISTRO set "WSL=wsl.exe -d %FLYPILOT_DISTRO%"
%WSL% --cd "%~dp0..\flybrain" -- bash ./py.sh validate.py %*
