@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating ScreenThai environment. Python 3.11 x64 is required.
    py -3.11 -m venv .venv
    if errorlevel 1 goto :fail
)
if not exist ".venv\screen-thai-installed" (
    echo Installing dependencies. First run needs an internet connection.
    ".venv\Scripts\python.exe" -m pip install -e .
    if errorlevel 1 goto :fail
    echo installed> ".venv\screen-thai-installed"
)
".venv\Scripts\python.exe" -m screen_thai.app
if errorlevel 1 goto :fail
exit /b 0
:fail
 echo.
 echo ScreenThai could not start. See the error above.
 echo Install Python 3.11 x64 and Microsoft Visual C++ 2015-2022 x64 Redistributable.
 pause
 exit /b 1
