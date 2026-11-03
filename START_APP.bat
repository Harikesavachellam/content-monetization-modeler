@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    py -3.12 -m venv .venv
) else (
    python -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1
    if errorlevel 1 goto python_missing
    python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" goto setup_failed
:install
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto setup_failed
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1
if errorlevel 1 goto app_failed
exit /b 0
:python_missing
echo Install 64-bit Python 3.12 from python.org, then run this file again.
pause
exit /b 1
:setup_failed
echo Setup did not complete. Check your Internet connection and the message above.
pause
exit /b 1
:app_failed
echo The app stopped. Read README.md for troubleshooting and alternative port commands.
pause
exit /b 1
