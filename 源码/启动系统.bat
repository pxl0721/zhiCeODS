@echo off
chcp 65001 >nul
title 智策 · 开发服务器

:: 切换到脚本所在目录
cd /d "%~dp0"

echo ========================================
echo     智策 · 企业运营决策支持系统
echo           (开发调试模式)
echo ========================================
echo.

:: 检查 Python 环境
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.11
    pause
    exit /b
)

:: 激活 Conda 环境（如果存在）
where conda >nul 2>&1
if %errorlevel% equ 0 (
    echo [1/4] 正在激活 Conda 环境 ops_env ...
    call conda activate ops_env
    if errorlevel 1 (
        echo [警告] Conda 环境 ops_env 不存在，尝试创建...
        call conda create -n ops_env python=3.11 -y
        call conda activate ops_env
    )
) else (
    echo [提示] 未检测到 Conda，将使用系统 Python。
)

:: 检查并安装依赖（仅首次或 requirements 变更时）
echo [2/4] 检查依赖包...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络连接。
    pause
    exit /b
)

:: 检查 .env 文件
if not exist ".env" (
    echo [警告] 未找到 .env 文件，请创建并配置 DASHSCOPE_API_KEY。
    echo 按任意键继续（将无法调用 AI 功能）...
    pause >nul
)

echo [3/4] 启动 Flask 开发服务器...
echo.
echo ========================================
echo   服务已启动！请在浏览器中访问：
echo   http://127.0.0.1:5000
echo   按 Ctrl+C 停止服务
echo ========================================
echo.

python app.py

pause