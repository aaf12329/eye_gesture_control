@echo off
chcp 65001 >nul
setlocal
set "TASK_PYTHON=D:\Anaconda_envs\mul_pro\python.exe"
cd /d "%~dp0"
if not exist "%TASK_PYTHON%" (
    echo Python not found: %TASK_PYTHON%
    pause
    exit /b 1
)
"%TASK_PYTHON%" -u "%~dp0main_control.py" %*
if errorlevel 1 pause
endlocal
