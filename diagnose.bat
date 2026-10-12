@echo off
title MARIN-LIVE Diagnostics

echo.
echo  ==============================================
echo   MARIN-LIVE Diagnostics
echo  ==============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo  [ERROR] .venv is missing. Run setup.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" tools\diagnose_live2d.py
if errorlevel 1 (
    echo.
    echo  Diagnostic script failed to run.
    pause
    exit /b 1
)

echo.
echo  Done. The results were saved to:
echo      diagnose_out.txt
echo.
echo  Please send that file to the developer.
echo.
pause
