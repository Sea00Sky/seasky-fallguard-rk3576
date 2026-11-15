#!/bin/bash
# ============================================================
# RKNNLite 安装脚本 v2（基于 Kickpi 官方文档）
# 用法：chmod +x install_rknnlite_v2.sh && ./install_rknnlite_v2.sh
# ============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "============================================================"
echo "  RKNNLite 安装脚本 v2（Kickpi 官方方式）"
echo "============================================================"
echo ""

# 1. 检查 Python 版本
echo_info "步骤 1/4: 检查 Python 版本..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo_info "Python 版本: $PYTHON_VERSION"

# 2. 检查是否已安装
echo_info "步骤 2/4: 检查 RKNNLite 是否已安装..."
if python3 -c "from rknnlite.api import RKNNLite" 2>/dev/null; then
    echo_info "RKNNLite 已安装，无需重复安装"
    exit 0
fi

# 3. 尝试从 PyPI 安装
echo_info "步骤 3/4: 尝试从 PyPI 安装..."
if pip3 install --break-system-packages rknn-toolkit-lite2; then
    echo_info "PyPI 安装成功"
else
    echo_warn "PyPI 安装失败，尝试下载 whl 文件..."

    # 4. 下载 whl 文件
    echo_info "步骤 4/4: 下载 whl 文件..."

    # 根据 Python 版本选择 whl
    case $PYTHON_VERSION in
        3.8)
            WHL_NAME="rknn_toolkit_lite2-2.0.0b0-cp38-cp38-linux_aarch64.whl"
            ;;
        3.10)
            WHL_NAME="rknn_toolkit_lite2-2.0.0b0-cp310-cp310-linux_aarch64.whl"
            ;;
        3.12)
            WHL_NAME="rknn_toolkit_lite2-2.0.0b0-cp312-cp312-linux_aarch64.whl"
            ;;
        *)
            echo_error "不支持的 Python 版本: $PYTHON_VERSION"
            exit 1
            ;;
    esac

    echo_info "下载 $WHL_NAME..."

    # 尝试从 GitHub 下载
    cd /tmp
    if wget "https://github.com/airockchip/rknn-toolkit2/raw/master/rknn-toolkit-lite2/packages/$WHL_NAME"; then
        echo_info "下载成功"
        pip3 install --break-system-packages "$WHL_NAME"
        echo_info "安装成功"
    else
        echo_error "下载失败，请手动下载："
        echo_error "https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages"
        exit 1
    fi
fi

# 验证安装
echo ""
echo "------------------------------------------------------------"
echo "  验证安装"
echo "------------------------------------------------------------"

if python3 -c "from rknnlite.api import RKNNLite; print('RKNNLite OK')" 2>/dev/null; then
    echo_info "RKNNLite 安装成功！"
else
    echo_error "RKNNLite 安装失败"
    exit 1
fi

# 检查 NPU
echo_info "NPU 检查:"
if dmesg | grep -q "rknpu"; then
    echo_info "  NPU 驱动: 已加载"
else
    echo_warn "  NPU 驱动: 未检测到"
fi

if [ -f /usr/lib/librknnrt.so ]; then
    echo_info "  librknnrt.so: 存在"
    strings /usr/lib/librknnrt.so | grep -i "librknnrt version" | head -1 || true
else
    echo_warn "  librknnrt.so: 不存在"
fi

echo ""
echo "============================================================"
echo "  RKNNLite 安装完成！"
echo "============================================================"
echo ""
echo_info "下一步："
echo "  cd ~/rk3576_deployment"
echo "  python3 class_display_demo.py --cam 0"
echo ""
