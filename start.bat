@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
if exist "C:\application\Miniconda3\envs\py312\python.exe" (
  "C:\application\Miniconda3\envs\py312\python.exe" app.py
) else (
  call conda run --no-capture-output -n py312 python app.py
)
pause
