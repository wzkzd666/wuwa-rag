@echo off
REM ============================================================
REM  Wuwa-RAG one-command launcher
REM    dev.bat                 start all (docker + backend + frontend)
REM    dev.bat stop            stop all
REM    dev.bat [start] <args>  pass args to start.ps1
REM                            e.g. dev.bat -NoDocker -Open
REM                                 dev.bat start -NoFront
REM  Real logic: scripts\start.ps1 / scripts\stop.ps1
REM  NOTE: keep this file pure ASCII (cmd codepage may not be UTF-8)
REM ============================================================

setlocal
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

if /I "%~1"=="stop" (
    powershell.exe -NoProfile -File "%ROOT%\scripts\stop.ps1"
    set "RC=%ERRORLEVEL%"
    goto :end
)

REM Batch note: shift does NOT change %* (it always holds all original args),
REM so strip a leading "start" keyword by substring instead.
set "ARGS=%*"
if /I "%~1"=="start" set "ARGS=%ARGS:~6%"

powershell.exe -NoProfile -File "%ROOT%\scripts\start.ps1" %ARGS%
set "RC=%ERRORLEVEL%"

:end
endlocal & exit /b %RC%
