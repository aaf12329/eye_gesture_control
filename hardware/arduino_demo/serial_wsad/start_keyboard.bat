@echo off
chcp 65001 >nul
setlocal
set "TASK_PYTHON=D:\Anaconda_envs\mul_pro\python.exe"
if not exist "%TASK_PYTHON%" (
    echo Python not found: %TASK_PYTHON%
    pause
    exit /b 1
)
"%TASK_PYTHON%" -u "%~dp0keyboard_serial.py" %*
if errorlevel 1 pause
endlocal
