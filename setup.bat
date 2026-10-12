@echo off
setlocal enabledelayedexpansion
title MARIN-LIVE Setup

echo.
echo  ==============================================
echo   MARIN-LIVE - One-time setup
echo  ==============================================
echo.

rem ---- Find Python ------------------------------------------------
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (
    where py >nul 2>nul && set "PY=py -3"
)
if not defined PY (
    echo  [ERROR] Python was not found.
    echo.
    echo  Please install Python 3.11 to 3.14 from https://www.python.org/downloads/
    echo  and make sure "Add Python to PATH" is ticked during install.
    echo.
    pause
    exit /b 1
)

echo  Using: %PY%

rem ---- Check version ----------------------------------------------
for /f "delims=" %%v in ('%PY% --version 2^>^&1') do set "VER=%%v"
echo  Detected: !VER!
set "OK="
for /f "tokens=2 delims= " %%v in ("!VER!") do (
    set "VN=%%v"
    for /f "tokens=1 delims=." %%a in ("%%v") do (
        if %%a GEQ 3 set "OK=1"
    )
)
if not defined OK (
    echo.
    echo  [ERROR] Python 3.x is required. Found: !VER!
    pause
    exit /b 1
)

rem ---- Create venv ------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo  Creating virtual environment (.venv) ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo  [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo  .venv already exists - skipping creation.
)

echo.
echo  Installing packages from requirements.lock (this can take a few minutes) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.lock
if errorlevel 1 (
    echo.
    echo  [ERROR] Package installation failed. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo  ==============================================
echo   Setup complete!
echo.
echo   Next steps:
echo    1. Copy .env.example to .env and add your own API keys
echo       (Fish Audio + Cerebras), or set them on first launch.
echo    2. Run  run.bat  to start the app.
echo    3. If anything looks wrong, run  diagnose.bat  and send
echo       the resulting diagnose_out.txt to the developer.
echo  ==============================================
echo.
pause
endlocal
