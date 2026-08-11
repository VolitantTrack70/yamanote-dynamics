@echo off
REM Yamanote Line dynamics model — double-click to launch.
REM
REM Starts the Streamlit server on a free port and opens the app. Uses the
REM project's virtual environment, so it does not matter what Python is on PATH.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   No virtual environment found at .venv
    echo.
    echo   Set it up first:
    echo       python -m venv .venv
    echo       .venv\Scripts\python.exe -m pip install -e ".[viz]"
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" launcher.py %*

if errorlevel 1 (
    echo.
    echo   The launcher exited with an error. See the output above.
    echo.
    pause
)
