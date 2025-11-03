@echo off
echo ========================================
echo   Traffic Sign Recognition Setup
echo   Version 2.0 - Advanced Features
echo ========================================
echo.

echo [1/4] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    echo Please make sure Python is installed and in PATH
    pause
    exit /b 1
)
echo ✓ Virtual environment created
echo.

echo [2/4] Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment activated
echo.

echo [3/4] Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed
echo.

echo [4/4] Creating necessary directories...
if not exist "captured_images" mkdir captured_images
if not exist "recorded_videos" mkdir recorded_videos
if not exist "output" mkdir output
echo ✓ Directories created
echo.

echo ========================================
echo   Setup Complete! ✓
echo ========================================
echo.
echo To run the application:
echo   1. Activate venv: venv\Scripts\activate
echo   2. Run app: streamlit run app.py
echo   3. Open browser: http://localhost:8501
echo.
pause

