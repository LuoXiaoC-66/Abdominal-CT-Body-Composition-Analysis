@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem ============================================================
rem  If python is not in PATH, edit the line below to your python.
rem  e.g.  set "PY=D:\Install\Anaconda\python.exe"
rem ============================================================
set "PY=python"
where "%PY%" >nul 2>nul
if errorlevel 1 (
    if exist "D:\Install\Anaconda\python.exe" set "PY=D:\Install\Anaconda\python.exe"
)
"%PY%" main.py
if errorlevel 1 pause
