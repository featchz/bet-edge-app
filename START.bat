@echo off
title Bet Edge App
echo.
echo  ========================================
echo   BET EDGE -- Starting up...
echo  ========================================
echo.

:: API Keys
SET ODDS_API_KEY=2177b2ec282ca158c9bc5d61b2977f3d
SET BALLDONTLIE_KEY=dfef48ff-15a7-45d4-86de-27461fa3e31f

:: Find Python
SET PYTHON=
IF EXIST "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" SET PYTHON=%LOCALAPPDATA%\Programs\Python\Python314\python.exe
IF EXIST "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" SET PYTHON=%LOCALAPPDATA%\Programs\Python\Python313\python.exe
IF EXIST "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" SET PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
IF EXIST "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" SET PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
IF EXIST "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" SET PYTHON=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe
IF "%PYTHON%"=="" SET PYTHON=python

echo  Python: %PYTHON%
echo  Installing dependencies...
"%PYTHON%" -m pip install flask requests --quiet

echo.
echo  Dashboard: http://127.0.0.1:5000
echo  Press CTRL+C to stop.
echo.

start http://127.0.0.1:5000
"%PYTHON%" app.py
pause
