#!/bin/bash

# SAM抠图工具启动脚本

echo "🚀 启动SAM抠图工具..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未找到，请先安装Python3"
    exit 1
fi

# 进入sam目录
cd "$(dirname "$0")"

# 检查requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt 文件不存在"
    exit 1
fi

# 检查是否需要安装依赖（覆盖 requirements.txt 中的关键库）
echo "📦 检查依赖..."
python3 -c "import gradio, torch, segment_anything, numpy, PIL, cv2, transformers, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  检测到缺少依赖，正在安装..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ 依赖安装失败，请手动执行: pip3 install -r requirements.txt"
        exit 1
    fi
fi

# 检查models目录
if [ ! -d "models" ]; then
    echo "📁 创建models目录..."
    mkdir -p models
fi

# 检查是否有模型文件
MODEL_EXISTS=false
for model in models/sam_vit_*.pth; do
    if [ -f "$model" ]; then
        MODEL_EXISTS=true
        break
    fi
done

if [ "$MODEL_EXISTS" = false ]; then
    echo "⚠️  未找到SAM模型文件"
    echo "📥 您可以运行 ./download_models.sh 下载模型"
    echo "   或手动下载并放入 models/ 目录"
    echo ""
fi

# 启动服务
echo "🌐 启动服务器，端口: 8001"
echo "🔗 访问地址: http://localhost:8001"
echo "⏹️  按 Ctrl+C 停止服务"
echo ""

python3 sam_app.py
