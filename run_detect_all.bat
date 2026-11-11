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
"%TASK_PYTHON%" -u "%~dp0detect_all.py" --pic-dir test/ --gesture-model best.onnx --landmark-model shape_predictor_68_face_landmarks.dat --serial-port COM5 %*
if errorlevel 1 pause
endlocal
