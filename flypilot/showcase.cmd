@echo off
rem The showcase route, filmed: the 3D view follows the drone from behind and a little above, and the window
rem (with the drone's camera and the dashboard) is recorded to flypilot\videos\showcase_<date_time>.mp4. The
rem video is drawn after the flight, when you close the 3D window or press Ctrl-C (the terminal shows progress).
rem   showcase.cmd                       the fly brain with receptive fields (flybrain-rf)
rem   showcase.cmd --policy flybrain     any other policy; any launch.cmd option (e.g. --headless)
setlocal
set "POLICY=--policy flybrain-rf"
echo %* | findstr /c:"--policy" >nul && set "POLICY="
call "%~dp0launch.cmd" avoid --route showcase %POLICY% --chase --video videos/showcase_{time}.mp4 %*
