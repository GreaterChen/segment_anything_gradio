#!/bin/bash

# SAM模型下载脚本

echo "📥 SAM模型下载工具"

# 进入sam目录
cd "$(dirname "$0")"

# 创建models目录
mkdir -p models

# 模型信息
declare -A models
models["vit_b"]="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
models["vit_l"]="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
models["vit_h"]="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

declare -A sizes
sizes["vit_b"]="约358MB"
sizes["vit_l"]="约1.2GB"
sizes["vit_h"]="约2.4GB"

echo ""
echo "可用模型:"
echo "1. vit_b - ${sizes["vit_b"]} (推荐，速度快)"
echo "2. vit_l - ${sizes["vit_l"]} (平衡)"
echo "3. vit_h - ${sizes["vit_h"]} (精度高，速度慢)"
echo "4. 全部下载"
echo ""

read -p "请选择要下载的模型 (1-4): " choice

download_model() {
    local model_name=$1
    local url=${models[$model_name]}
    local file_path="models/sam_${model_name}.pth"
    
    if [ -f "$file_path" ]; then
        echo "✅ $model_name 模型已存在: $file_path"
        return 0
    fi
    
    echo "📥 下载 $model_name 模型 (${sizes[$model_name]})..."
    echo "🔗 URL: $url"
    
    # 使用wget或curl下载
    if command -v wget &> /dev/null; then
        wget -O "$file_path" "$url"
    elif command -v curl &> /dev/null; then
        curl -L -o "$file_path" "$url"
    else
        echo "❌ 未找到wget或curl，请手动下载:"
        echo "   $url"
        echo "   保存为: $file_path"
        return 1
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ $model_name 模型下载完成"
    else
        echo "❌ $model_name 模型下载失败"
        rm -f "$file_path"
        return 1
    fi
}

case $choice in
    1)
        download_model "vit_b"
        ;;
    2)
        download_model "vit_l"
        ;;
    3)
        download_model "vit_h"
        ;;
    4)
        echo "📥 下载所有模型..."
        download_model "vit_b"
        download_model "vit_l"
        download_model "vit_h"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "📂 当前models目录内容:"
ls -lh models/

echo ""
echo "✅ 下载完成！现在可以运行 ./start.sh 启动服务"
