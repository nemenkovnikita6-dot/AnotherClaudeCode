@echo off
rem Builds a fully standalone AnotherCC-standalone.exe (no Python needed to run it)
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Install it from https://python.org and tick "Add Python to PATH".
  pause
  exit /b 1
)
python -m pip install --upgrade pyinstaller
if errorlevel 1 (
  echo Could not install PyInstaller. Check your internet connection.
  pause
  exit /b 1
)
python -m PyInstaller --onefile --console --clean --name AnotherCC src\anothercc.py
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)
copy /y dist\AnotherCC.exe AnotherCC-standalone.exe >nul
echo.
echo Done: AnotherCC-standalone.exe
pause
