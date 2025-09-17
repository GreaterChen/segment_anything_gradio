#!/bin/bash

# SAM抠图工具安装脚本

echo "📦 安装SAM抠图工具依赖..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未找到，请先安装Python3"
    exit 1
fi

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 未找到，请先安装pip3"
    exit 1
fi

# 进入sam目录
cd "$(dirname "$0")"

# 升级pip
echo "⬆️  升级pip..."
pip3 install --upgrade pip

# 安装依赖
echo "📦 安装Python依赖..."
pip3 install -r requirements.txt

# 检查PyTorch安装
echo "🔍 检查PyTorch安装..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

# 检查segment-anything
echo "🔍 检查Segment Anything安装..."
python3 -c "import segment_anything; print('Segment Anything安装成功')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Segment Anything安装可能失败，尝试重新安装..."
    pip3 install git+https://github.com/facebookresearch/segment-anything.git
fi

# 创建必要目录
echo "📁 创建目录..."
mkdir -p models uploads

echo ""
echo "✅ 安装完成！"
echo "📥 接下来可以运行 ./download_models.sh 下载模型"
echo "🚀 或直接运行 ./start.sh 启动服务"
