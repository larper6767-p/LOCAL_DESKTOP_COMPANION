@echo off
title MARIN-LIVE

rem ---- Locate the venv python -------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo  [ERROR] .venv is missing.
    echo  Please run  setup.bat  first.
    echo.
    pause
    exit /b 1
)

rem ---- First-run: copy .env.example if no .env exists ---------------
if not exist ".env" (
    echo  No .env found - copying .env.example. You'll be asked for keys.
    copy /y ".env.example" ".env" >nul
)

echo  Starting MARIN-LIVE...
echo  (The app will ask for your API keys on first launch.)
echo.

".venv\Scripts\python.exe" src\app.py
if errorlevel 1 (
    echo.
    echo  MARIN-LIVE exited with an error.
    echo  Run  diagnose.bat  and send diagnose_out.txt to the developer.
    echo.
    pause
)
