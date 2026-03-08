@echo off
chcp 65001 >nul
:: ============================================================
:: agent-learning 项目一键初始化脚本（Windows CMD）
::
:: 使用方式：在项目根目录双击运行，或在 CMD 中执行 setup.bat
:: ============================================================

echo.
echo ==================================================
echo  agent-learning 项目初始化
echo ==================================================
echo.

:: ── Step 1：检查 uv ──────────────────────────────────────────
where uv >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 uv，请先在 PowerShell 中运行：
    echo   irm https://astral.sh/uv/install.ps1 ^| iex
    echo 安装后重新运行此脚本。
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('uv --version') do echo [OK] 找到 %%i

:: ── Step 2：安装 Python 3.12 ─────────────────────────────────
echo.
echo [1/3] 安装 Python 3.12（已安装则跳过）...
uv python install 3.12
echo [OK] Python 3.12 就绪

:: ── Step 3：创建虚拟环境 ─────────────────────────────────────
echo.
echo [2/3] 创建虚拟环境 .venv ...
uv venv .venv --python 3.12
echo [OK] .venv 创建完成

:: ── Step 4：安装依赖 ─────────────────────────────────────────
echo.
echo [3/3] 安装依赖（首次约需 1-3 分钟）...
uv pip install -r requirements.txt --python .venv\Scripts\python.exe
echo [OK] 依赖安装完成

:: ── Step 5：提示配置 ─────────────────────────────────────────
echo.
echo ==================================================
echo  初始化完成！
echo ==================================================
echo.
if not exist ".env" (
    echo [提示] 还需配置 API Key：
    echo   1. 复制 .env.example 为 .env
    echo   2. 编辑 .env，填入 DEEPSEEK_API_KEY 等
    echo.
)
echo 在 VSCode 中：
echo   直接打开任意 .py 文件运行即可（VSCode 会自动使用 .venv）
echo.
pause
