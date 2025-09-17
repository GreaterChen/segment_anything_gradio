# SAM智能抠图工具

基于Meta的Segment Anything Model (SAM) 和 Gradio 构建的智能抠图工具，支持交互式前端操作。

## 🌟 特性

- 🎯 基于SAM模型的高精度图像分割
- 🖱️ 交互式点击选择分割区域  
- 🌐 Web界面，支持本地部署
- 🚀 GPU加速支持
- 📱 响应式设计，支持移动端
- 🎨 透明背景输出，完美抠图效果

## 📋 系统要求

- Python 3.7+
- 4GB+ 内存
- 推荐：NVIDIA GPU（可选，用于加速）

## 🚀 快速开始

### 1. 安装依赖

```bash
chmod +x install.sh
./install.sh
```

### 2. 下载模型

```bash
chmod +x download_models.sh
./download_models.sh
```

推荐选择 `vit_b` 模型（约358MB），速度较快。

### 3. 启动服务

```bash
chmod +x start.sh
./start.sh
```

服务将在 http://localhost:8001 启动

## 📖 使用说明

1. **加载模型**: 首次使用需要在界面中选择并加载SAM模型
2. **上传图像**: 上传要处理的图片
3. **交互分割**: 在图像上点击要分割的区域
4. **查看结果**: 查看抠图结果和分割掩码
5. **下载保存**: 下载透明背景的抠图结果

## 🔧 配置说明

### 模型选择

- **vit_b**: 轻量级，速度快，推荐日常使用
- **vit_l**: 平衡性能和精度
- **vit_h**: 最高精度，需要更多计算资源

### 端口配置

默认端口为8001，可在 `sam_app.py` 中修改：

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=8001,  # 修改此处
    share=False
)
```

## 📁 目录结构

```
sam/
├── sam_app.py          # 主应用程序
├── requirements.txt    # Python依赖
├── install.sh         # 安装脚本
├── download_models.sh  # 模型下载脚本
├── start.sh           # 启动脚本
├── README.md          # 说明文档
├── models/            # 模型文件目录
└── uploads/           # 上传文件临时目录
```

## 🐛 常见问题

### Q: 模型加载失败
A: 确保模型文件正确下载到 `models/` 目录，文件名格式为 `sam_vit_*.pth`

### Q: GPU不可用
A: 检查CUDA安装，或使用CPU模式（速度较慢但功能正常）

### Q: 依赖安装失败
A: 确保Python版本>=3.7，升级pip后重试

### Q: 分割效果不佳
A: 尝试点击目标物体的中心区域，或使用更高精度的模型

## 🔗 相关链接

- [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [Gradio](https://gradio.app/)
- [模型下载地址](https://github.com/facebookresearch/segment-anything#model-checkpoints)

## 📄 许可证

本项目遵循相关开源许可证。SAM模型遵循其原始许可证。
