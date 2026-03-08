#!/usr/bin/env bash
# ============================================================
# agent-learning 项目一键初始化脚本
#
# 使用方式（在项目根目录执行）：
#   bash setup.sh
#
# 前提：已安装 uv（https://docs.astral.sh/uv/）
# 脚本会自动完成：
#   1. 安装 Python 3.12（通过 uv，不影响系统 Python）
#   2. 在项目根目录创建 .venv 虚拟环境
#   3. 安装 requirements.txt 中的所有依赖
# ============================================================

set -e  # 任何命令失败立即退出

# ── 颜色输出 ────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "=================================================="
echo " agent-learning 项目初始化"
echo "=================================================="
echo ""

# ── Step 1：检查 uv ──────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo -e "${RED}[错误] 未找到 uv，请先安装：${NC}"
    echo "  Windows (PowerShell): irm https://astral.sh/uv/install.ps1 | iex"
    echo "  Mac/Linux:            curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo -e "${GREEN}[✓] 找到 uv：$(uv --version)${NC}"

# ── Step 2：安装 Python 3.12 ─────────────────────────────────
echo ""
echo "[1/3] 安装 Python 3.12（已安装则跳过）..."
uv python install 3.12
echo -e "${GREEN}[✓] Python 3.12 就绪${NC}"

# ── Step 3：创建虚拟环境 ─────────────────────────────────────
echo ""
echo "[2/3] 创建虚拟环境 .venv ..."
uv venv .venv --python 3.12
echo -e "${GREEN}[✓] .venv 创建完成${NC}"

# ── Step 4：安装依赖 ─────────────────────────────────────────
echo ""
echo "[3/3] 安装依赖（首次约需 1-3 分钟）..."
uv pip install -r requirements.txt --python .venv/Scripts/python.exe 2>/dev/null \
    || uv pip install -r requirements.txt --python .venv/bin/python
echo -e "${GREEN}[✓] 依赖安装完成${NC}"

# ── Step 5：提示配置 API Key ─────────────────────────────────
echo ""
echo "=================================================="
echo -e "${GREEN} 初始化完成！${NC}"
echo "=================================================="
echo ""

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}[提示] 还需配置 API Key：${NC}"
    echo "  复制 .env.example 为 .env，并填入你的 Key："
    echo "    cp .env.example .env"
    echo "  然后编辑 .env 填写 DEEPSEEK_API_KEY 等"
else
    echo -e "${GREEN}[✓] 检测到 .env 文件${NC}"
fi

echo ""
echo "在 VSCode 中使用："
echo "  1. 重新打开项目（VSCode 会自动检测 .venv）"
echo "  2. 直接运行任意 .py 文件即可"
echo ""
