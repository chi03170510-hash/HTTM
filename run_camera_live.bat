@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "PY_EXE="

if exist "%ROOT_DIR%.venv\Scripts\python.exe" (
    set "PY_EXE=%ROOT_DIR%.venv\Scripts\python.exe"
) else if exist "%ROOT_DIR%.venv-hybrid\Scripts\python.exe" (
    set "PY_EXE=%ROOT_DIR%.venv-hybrid\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)

echo [HTTM Camera Live] Using Python: %PY_EXE%
"%PY_EXE%" "%ROOT_DIR%ai_training\integration\camera_coordinator_live.py" %*

if errorlevel 1 (
    echo.
    echo [HTTM Camera Live] Process exited with error code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)
