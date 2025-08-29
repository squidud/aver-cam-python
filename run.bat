@echo off
echo AVer CAM520 Pro Controller - HTTP Edition
echo ==========================================
echo.

REM Try to use Python 3.12 first, then 3.11, then default python
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.12
    echo Using Python 3.12
    goto :install_deps
)

py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.11
    echo Using Python 3.11
    goto :install_deps
)

python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    echo Using default Python
    goto :install_deps
)

echo ERROR: No compatible Python version found
echo Please install Python 3.11 or 3.12 from https://www.python.org/downloads/
pause
exit /b 1

:install_deps
echo Installing/updating dependencies...
echo.
%PYTHON_CMD% -m pip install --upgrade pip
%PYTHON_CMD% -m pip install opencv-python numpy PyQt6 mediapipe pyvirtualcam requests cv2-enumerate-cameras

if errorlevel 1 (
    echo ERROR: Failed to install some dependencies
    echo Try running as administrator or check your Python installation
    pause
    exit /b 1
)

echo ✓ All dependencies installed successfully
echo.

REM Run the application
echo Launching AVer CAM520 Pro Controller...
echo.
echo IMPORTANT: Make sure your camera software is running at localhost:36680
echo You can change the port in the app if needed.
echo.

%PYTHON_CMD% main.py

if errorlevel 1 (
    echo.
    echo ERROR: Application crashed
    pause
)