@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist index.json (
    echo 首次运行，正在构建向量索引...
    .venv\Scripts\python ingest.py
    if errorlevel 1 pause & exit /b 1
)
.venv\Scripts\python query.py
pause