"""
SAM抠图工具 - 基于Gradio的交互式界面
使用Segment Anything Model实现智能抠图
"""

import gradio as gr
import numpy as np
from PIL import Image
import torch
import cv2
import os

# 尝试导入segment_anything，如果失败则提供友好的错误信息
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("警告: segment_anything 库未安装，请运行: pip install segment-anything")

class SAMProcessor:
    """SAM模型处理器"""
    
    def __init__(self):
        self.predictor = None
        self.auto_mask_generator = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_image = None
    
    def _normalize_image_input(self, image):
        """将任意输入规范化为 HxWx3 的 uint8 连续内存 RGB 图像"""
        try:
            # 路径 → PIL
            if isinstance(image, str):
                image = Image.open(image)
            
            # PIL → np.uint8 RGB
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_array = np.array(image)
            else:
                # 列表/np数组 → np数组
                image_array = np.array(image)
                
                # 处理dtype
                if image_array.dtype != np.uint8:
                    if np.issubdtype(image_array.dtype, np.floating):
                        image_array = np.clip(image_array * (255.0 if image_array.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
                    else:
                        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            # 处理灰度 HxW → HxWx3
            if image_array.ndim == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            
            # 处理CHW → HWC（如 3xHxW 或 4xHxW）
            if image_array.ndim == 3 and image_array.shape[0] in (1, 3, 4) and image_array.shape[0] < image_array.shape[-1]:
                # 如果 channels 在第0维且远小于空间维度，视为CHW
                # 再次判断避免已经是HWC的情况
                if image_array.shape[-1] not in (1, 3, 4):
                    image_array = np.transpose(image_array, (1, 2, 0))
            
            # 去掉alpha，或扩展到3通道
            if image_array.ndim == 3:
                if image_array.shape[2] == 4:
                    image_array = image_array[:, :, :3]
                elif image_array.shape[2] == 1:
                    image_array = np.repeat(image_array, 3, axis=2)
                elif image_array.shape[2] > 4:
                    image_array = image_array[:, :, :3]
            
            # 确保连续内存
            image_array = np.ascontiguousarray(image_array, dtype=np.uint8)
            return image_array
        except Exception as e:
            raise RuntimeError(f"图像标准化失败: {str(e)}")
        
    def load_model(self, model_type="vit_b"):
        """加载SAM模型"""
        if not SAM_AVAILABLE:
            return False, "segment_anything 库未安装"
            
        try:
            # 模型文件路径
            model_path = f"models/sam_{model_type}.pth"
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                download_urls = {
                    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
                    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                }
                
                error_msg = f"""
模型文件不存在: {model_path}

请下载模型文件:
1. 创建 models/ 目录: mkdir -p models
2. 下载模型: wget {download_urls[model_type]} -O {model_path}

或者手动下载后放入 models/ 目录
"""
                return False, error_msg
                
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            # 自动分割生成器（按需使用）
            try:
                self.auto_mask_generator = SamAutomaticMaskGenerator(
                    sam,
                    points_per_side=32,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=0
                )
            except Exception as e:
                self.auto_mask_generator = None
            self.model_loaded = True
            return True, f"模型 {model_type} 加载成功!"
            
        except Exception as e:
            return False, f"模型加载失败: {str(e)}"
    
    def set_image(self, image):
        """设置当前图像"""
        if not self.model_loaded:
            return False, "模型未加载"
            
        try:
            image_array = self._normalize_image_input(image)
            self.predictor.set_image(image_array)
            self.current_image = image_array
            return True, "图像设置成功"
            
        except Exception as e:
            return False, f"图像设置失败: {str(e)}"
    
    def predict_with_points(self, points):
        """使用点击点进行预测"""
        if not self.model_loaded or self.current_image is None:
            return None, None, "模型未加载或图像未设置"
            
        try:
            if not points or len(points) == 0:
                return None, None, "请先在图像上点击选择区域"
                
            # 转换点格式
            input_points = np.array([[p[0], p[1]] for p in points])
            input_labels = np.ones(len(points))  # 1表示前景点
            
            # 预测
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            
            # 返回所有掩码和分数
            return masks, scores, f"分割完成，生成{len(masks)}个候选结果"
            
        except Exception as e:
            return None, None, f"预测失败: {str(e)}"

    def auto_segment(self, image):
        """无点击点时的自动分割，返回最大区域掩码"""
        if not self.model_loaded:
            return None, None, "模型未加载"
        try:
            # 生成所有候选掩码
            if self.auto_mask_generator is None:
                return None, None, "自动分割不可用"
            masks = self.auto_mask_generator.generate(image)
            if not masks:
                return None, None, "未生成任何掩码"
            # 选择面积最大的掩码
            best = max(masks, key=lambda m: m.get('area', 0))
            best_mask = best.get('segmentation')
            score = best.get('stability_score', 0.0)
            return best_mask, score, f"自动分割完成，置信度: {score:.3f}"
        except Exception as e:
            return None, None, f"自动分割失败: {str(e)}"

def apply_mask_to_image(image, mask, background_transparent=True):
    """将掩码应用到图像"""
    if image is None or mask is None:
        return None
        
    try:
        result = image.copy()
        
        if background_transparent:
            # 创建带透明背景的图像
            if len(result.shape) == 3:
                result_rgba = np.zeros((result.shape[0], result.shape[1], 4), dtype=np.uint8)
                result_rgba[:, :, :3] = result
                result_rgba[:, :, 3] = (mask * 255).astype(np.uint8)
                return result_rgba
        else:
            # 白色背景
            background = np.ones_like(result) * 255
            result = np.where(mask[..., None], result, background)
            
        return result.astype(np.uint8)
        
    except Exception as e:
        return None

# 全局SAM处理器
sam_processor = SAMProcessor()

def load_sam_model(model_type):
    """加载SAM模型的包装函数"""
    success, message = sam_processor.load_model(model_type)
    return message


def segment_with_points(display_image, click_coords_text):
    """使用指定坐标点进行分割"""
    global clicked_points, original_image
    
    # 使用原始图像进行分割，而不是带标记点的显示图像
    if original_image is None:
        return None, None, "请先上传图像"
    
    # 优先使用全局点击点，如果没有则解析坐标文本
    click_points = []
    
    if clicked_points:
        # 使用全局点击点
        click_points = clicked_points.copy()
    elif click_coords_text and click_coords_text.strip():
        # 解析坐标文本作为备用
        try:
            coords_text = click_coords_text.strip()
            
            if coords_text.startswith('[') and coords_text.endswith(']'):
                # 多点格式: [(100,200), (300,400)]
                import ast
                coords_list = ast.literal_eval(coords_text)
                for coord in coords_list:
                    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                        click_points.append([float(coord[0]), float(coord[1])])
            elif '(' in coords_text:
                # 单点格式: (100, 200)
                coords_text = coords_text.replace('(', '').replace(')', '')
                parts = coords_text.split(',')
                if len(parts) >= 2:
                    click_points.append([float(parts[0].strip()), float(parts[1].strip())])
            else:
                # 简单格式: 100,200
                parts = coords_text.split(',')
                if len(parts) >= 2:
                    click_points.append([float(parts[0].strip()), float(parts[1].strip())])
                
        except Exception as e:
            return None, None, f"坐标格式错误: {e}"
    
    # 使用原始图像（无标记点）进行分割
    success, message = sam_processor.set_image(original_image)
    if not success:
        return None, None, f"图像设置失败: {message}"
    
    # 如果有点击点，优先点引导分割；否则自动分割
    if click_points:
        masks, scores, message = sam_processor.predict_with_points(click_points)
        coords_info = f"使用坐标点: {click_points}" if len(click_points) <= 3 else f"使用{len(click_points)}个坐标点"
        message = f"{message} | {coords_info}"
    else:
        # 自动分割只返回一个结果，转换为多结果格式
        mask, score, auto_message = sam_processor.auto_segment(sam_processor.current_image)
        if mask is not None:
            masks = [mask]
            scores = [score]
            message = f"{auto_message} | 使用自动分割"
        else:
            return None, None, None, None, f"分割失败: {auto_message}"
    
    if masks is not None and len(masks) > 0:
        # 生成多个分割结果
        result_images = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            result_img = apply_mask_to_image(sam_processor.current_image, mask, True)
            result_images.append(result_img)
        
        # 创建叠加图像
        overlay_image = create_overlay_image(sam_processor.current_image, masks, scores)
        
        # 确保返回3个结果，不足的用None填充
        while len(result_images) < 3:
            result_images.append(None)
        
        status_msg = f"分割完成! 生成{len(masks)}个候选结果 + 1个并集结果"
        if click_points:
            status_msg += f" | {coords_info}"
        
        return result_images[0], result_images[1], result_images[2], overlay_image, status_msg
    else:
        return None, None, None, None, f"分割失败: {message}"

def create_overlay_image(original_image, masks, scores):
    """创建所有掩码的并集（union）分割结果"""
    if masks is None or len(masks) == 0:
        return original_image
    
    # 创建并集掩码：所有mask的逻辑或运算
    union_mask = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        union_mask = union_mask | mask
    
    # 应用并集掩码到原始图像
    result_image = apply_mask_to_image(original_image, union_mask, True)
    
    return result_image

# 全局变量存储点击点和原始图像
clicked_points = []
original_image = None

def handle_image_click(image, evt: gr.SelectData):
    """处理图像点击事件，累积点击坐标并在图像上显示"""
    global clicked_points, original_image
    
    if image is None:
        return image, ""
    
    # 保存原始图像（无标记点版本）
    if original_image is None or len(clicked_points) == 0:
        original_image = image.copy()
    
    x, y = evt.index[0], evt.index[1]
    clicked_points.append([x, y])
    
    # 在原始图像上绘制所有点击点
    image_with_points = draw_points_on_image(original_image, clicked_points)
    
    # 格式化坐标文本
    if len(clicked_points) == 1:
        coords_text = f"({clicked_points[0][0]}, {clicked_points[0][1]})"
    else:
        coords_text = str(clicked_points)
    
    return image_with_points, coords_text

def draw_points_on_image(image, points):
    """在图像上绘制点击点"""
    if image is None or not points:
        return image
    
    # 复制图像避免修改原图
    image_copy = image.copy()
    
    # 绘制每个点击点
    for i, (x, y) in enumerate(points):
        # 绘制红色圆点
        cv2.circle(image_copy, (int(x), int(y)), 8, (255, 0, 0), -1)  # 红色实心圆
        cv2.circle(image_copy, (int(x), int(y)), 10, (255, 255, 255), 2)  # 白色边框
        
        # 添加点编号
        cv2.putText(image_copy, str(i+1), (int(x)+12, int(y)-12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image_copy

def clear_points_and_restore_image():
    """清除所有点击点并恢复原始图像"""
    global clicked_points, original_image
    clicked_points = []
    restored_image = original_image.copy() if original_image is not None else None
    return restored_image, "", "已清除所有点击点"

def reset_image_to_original(new_image):
    """重置图像到原始状态（无点击点标记）"""
    global clicked_points, original_image
    clicked_points = []
    original_image = new_image.copy() if new_image is not None else None
    return new_image, ""

def create_sam_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="SAM智能抠图工具", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎯 SAM智能抠图工具
        
        基于Meta的Segment Anything Model实现的智能抠图工具
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔧 模型设置")
                
                model_type = gr.Dropdown(
                    choices=["vit_b", "vit_l", "vit_h"],
                    value="vit_b",
                    label="模型类型",
                    info="vit_b: 快速, vit_l: 平衡, vit_h: 高精度"
                )
                
                load_btn = gr.Button("加载模型", variant="primary")
                model_status = gr.Textbox(
                    label="模型状态",
                    value="模型未加载",
                    interactive=False
                )
                
                gr.Markdown("### 💡 使用方法")
                gr.Markdown("""
                1. 加载SAM模型
                2. 上传图像并点击要分割的区域
                3. 点击"执行分割"查看3个候选结果
                
                **特色功能：**
                - 🎯 显示3个候选分割结果
                - 🔗 自动生成所有结果的并集
                - 📊 显示每个结果的置信度
                """)
                
            with gr.Column(scale=2):
                gr.Markdown("### 🖱️ 交互式分割")
                
                # 统一的图像组件，既能上传又能点击
                interactive_image = gr.Image(
                    label="上传图像并点击选择要分割的区域",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    interactive=True
                )
                
                with gr.Row():
                    clear_btn = gr.Button("清除点击点", variant="secondary")
                    segment_btn = gr.Button("执行分割", variant="primary")
                
                # 坐标显示和编辑
                click_coords = gr.Textbox(
                    label="点击坐标",
                    placeholder="点击图像后会显示坐标，也可手动输入",
                    interactive=True,
                    lines=2
                )
                
                process_status = gr.Textbox(
                    label="处理状态",
                    interactive=False
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎨 候选结果 1")
                result_image_1 = gr.Image(
                    label="候选结果 1（最佳）",
                    type="numpy"
                )
                
            with gr.Column():
                gr.Markdown("### 🎨 候选结果 2") 
                result_image_2 = gr.Image(
                    label="候选结果 2",
                    type="numpy"
                )
                
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎨 候选结果 3")
                result_image_3 = gr.Image(
                    label="候选结果 3",
                    type="numpy"
                )
                
            with gr.Column():
                gr.Markdown("### 🔗 合并结果")
                overlay_image = gr.Image(
                    label="所有候选结果的并集（Union）",
                    type="numpy"
                )
        
        with gr.Row():
            gr.Markdown("💡 首次使用需要下载模型文件（约几百MB），支持GPU加速")
        
        # 事件绑定
        load_btn.click(
            fn=load_sam_model,
            inputs=[model_type],
            outputs=[model_status]
        )
        
        # 处理图像点击事件 - 累积点击并实时显示
        interactive_image.select(
            fn=handle_image_click,
            inputs=[interactive_image],
            outputs=[interactive_image, click_coords]
        )
        
        # 清除点击点并恢复原始图像
        clear_btn.click(
            fn=clear_points_and_restore_image,
            outputs=[interactive_image, click_coords, process_status]
        )
        
        # 当上传新图像时，重置点击点
        interactive_image.upload(
            fn=reset_image_to_original,
            inputs=[interactive_image],
            outputs=[interactive_image, click_coords]
        )
        
        # 执行分割
        segment_btn.click(
            fn=segment_with_points,
            inputs=[interactive_image, click_coords],
            outputs=[result_image_1, result_image_2, result_image_3, overlay_image, process_status]
        )
    
    return demo

def main():
    """主函数"""
    # 确保必要目录存在
    os.makedirs("models", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    # 创建界面
    demo = create_sam_interface()
    
    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=8002,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
