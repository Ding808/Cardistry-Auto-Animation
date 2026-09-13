@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Setup.ps1" %*
set "setupResult=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %setupResult%
