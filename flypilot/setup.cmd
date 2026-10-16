@echo off
rem One-time setup from Windows. Installs WSL system packages as root (no sudo prompt),
rem then CrazySim, the Python venv and the SITL firmware inside WSL. Safe to re-run.
rem Uses your default WSL distro; set FLYPILOT_DISTRO to pick another.
setlocal
set "WSL=wsl.exe"
if defined FLYPILOT_DISTRO set "WSL=wsl.exe -d %FLYPILOT_DISTRO%"

%WSL% -u root --cd "%~dp0." -- bash ./system_deps.sh || goto :fail
%WSL% --cd "%~dp0." -- bash ./setup.sh || goto :fail
echo.
echo Setup finished. Next: flypilot\launch.cmd (tests to run are in flypilot\run.txt)
exit /b 0

:fail
echo.
echo Setup FAILED - see the messages above.
exit /b 1
