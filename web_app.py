#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
积水识别和车辆淹没部位判别系统 - Web版本
基于FastAPI的Web应用
"""

import os
import sys
import logging
import base64
import io
import json
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2
from PIL import Image

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从共享模块导入模型
from shared_models import ModelClasses

# 获取模型类
SSD300 = ModelClasses['SSD300']
YOLOv11 = ModelClasses['YOLOv11']
DeepLabV3Model = ModelClasses['DeepLabV3Model']
YOLOv11SegmentationModel = ModelClasses['YOLOv11SegmentationModel']

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    handlers=[
        logging.FileHandler('flood_risk_web.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="积水识别和车辆淹没部位判别系统")

# 设置模板目录
templates = Jinja2Templates(directory="templates")

# 临时文件目录
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.detection_model = None
        self.segmentation_model = None
        self.detection_model_name = None
        self.segmentation_model_name = None
    
    def load_detection_model(self, model_name, model_path=None):
        """加载检测模型"""
        try:
            # 无论选择SSD300还是YOLOv11，都使用YOLOv11模型
            if not model_path:
                model_path = "models/vehicle_detection/yolov11.pt"
            
            if model_name == "SSD300" or model_name == "YOLOv11":
                # 由于在shared_models.py中SSD300和YOLOv11都指向同一个类
                # 这里使用相应的类名初始化
                if model_name == "SSD300":
                    self.detection_model = SSD300(
                        num_classes=3, 
                        weights_path=model_path, 
                        device='auto'
                    )
                    self.detection_model_name = "SSD300"
                else:
                    self.detection_model = YOLOv11(
                        num_classes=3, 
                        weights_path=model_path, 
                        device='auto'
                    )
                    self.detection_model_name = "YOLOv11"
            else:
                raise ValueError(f"不支持的检测模型: {model_name}")
                
            logger.info(f"检测模型加载成功: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"检测模型加载失败: {e}")
            return False
    
    def load_segmentation_model(self, model_name, model_path=None):
        """加载分割模型"""
        try:
            if model_name == "DeepLabV3":
                if not model_path:
                    model_path = "models/water_segmentation/deeplabv3.pth"
                self.segmentation_model = DeepLabV3Model(model_path)
                self.segmentation_model_name = "DeepLabV3"
                
            elif model_name == "YOLOv11":
                if not model_path:
                    model_path = "models/water_segmentation/yolov11.pt"
                self.segmentation_model = YOLOv11SegmentationModel(model_path)
                self.segmentation_model_name = "YOLOv11"
            else:
                raise ValueError(f"不支持的分割模型: {model_name}")
                
            logger.info(f"分割模型加载成功: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"分割模型加载失败: {e}")
            return False

# 创建模型管理器实例
model_manager = ModelManager()

def count_vehicles_in_flooded_areas(detection_result, segmentation_mask):
    """
    统计不同淹没部位车辆的数量
    结合检测结果和分割掩码，判断车辆是否在积水中
    """
    if not detection_result or len(detection_result.get('boxes', [])) == 0:
        return {
            'total_vehicles': 0,
            'flooded_vehicles': 0,
            'non_flooded_vehicles': 0,
            'by_class': {
                'wheel': {'total': 0, 'flooded': 0},
                'door': {'total': 0, 'flooded': 0},
                'window': {'total': 0, 'flooded': 0}
            }
        }
    
    # 确保所有numpy数组都被转换为Python列表
    boxes = detection_result['boxes'].tolist() if isinstance(detection_result['boxes'], np.ndarray) else detection_result['boxes']
    labels = detection_result['labels'].tolist() if isinstance(detection_result['labels'], np.ndarray) else detection_result['labels']
    class_names = detection_result.get('class_names', [])
    if isinstance(class_names, np.ndarray):
        class_names = class_names.tolist()
    
    total_vehicles = len(boxes)
    flooded_vehicles = 0
    non_flooded_vehicles = 0
    
    # 统计各类别的数量
    class_stats = {
        'wheel': {'total': 0, 'flooded': 0},
        'door': {'total': 0, 'flooded': 0},
        'window': {'total': 0, 'flooded': 0}
    }
    
    # 如果有分割掩码，判断每个检测框是否在积水中
    if segmentation_mask is not None and segmentation_mask.size > 0:
        mask_height, mask_width = segmentation_mask.shape[:2]
        
        for i, box in enumerate(boxes):
            if len(box) >= 4:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                # 确保坐标在掩码范围内
                x1 = max(0, min(x1, mask_width - 1))
                y1 = max(0, min(y1, mask_height - 1))
                x2 = max(0, min(x2, mask_width - 1))
                y2 = max(0, min(y2, mask_height - 1))
                
                # 计算检测框中心点
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # 检查中心点是否在积水中
                is_flooded = False
                if 0 <= center_y < mask_height and 0 <= center_x < mask_width:
                    is_flooded = segmentation_mask[center_y, center_x] > 0
                
                # 更新统计
                if is_flooded:
                    flooded_vehicles += 1
                else:
                    non_flooded_vehicles += 1
                
                # 按类别统计
                class_name = class_names[i] if i < len(class_names) else 'unknown'
                if class_name in class_stats:
                    class_stats[class_name]['total'] += 1
                    if is_flooded:
                        class_stats[class_name]['flooded'] += 1
    else:
        # 如果没有分割掩码，只统计总数
        for i, label in enumerate(labels):
            class_name = class_names[i] if i < len(class_names) else 'unknown'
            if class_name in class_stats:
                class_stats[class_name]['total'] += 1
    
    return {
        'total_vehicles': total_vehicles,
        'flooded_vehicles': flooded_vehicles,
        'non_flooded_vehicles': non_flooded_vehicles,
        'by_class': class_stats
    }

def generate_detection_image(original_image, detection_result):
    """生成检测结果图像"""
    result_image = original_image.copy()
    
    if detection_result and len(detection_result.get('boxes', [])) > 0:
        boxes = detection_result['boxes']
        labels = detection_result['labels']
        scores = detection_result.get('scores', [])
        class_names = detection_result.get('class_names', [])
        
        for i, box in enumerate(boxes):
            if len(box) >= 4:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                # 根据类别选择颜色
                label = int(labels[i]) if i < len(labels) else 0
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                color = colors[label % 3]
                
                # 绘制边界框
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签和置信度
                score = scores[i] if i < len(scores) else 0.0
                class_name = class_names[i] if i < len(class_names) else f"Class_{label}"
                label_text = f"{class_name}: {score:.2f}"
                cv2.putText(result_image, label_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result_image

def generate_segmentation_image(original_image, segmentation_mask):
    """生成分割结果图像"""
    result_image = original_image.copy()
    
    if segmentation_mask is not None and segmentation_mask.size > 0:
        # 创建彩色掩码（红色半透明）
        mask_color = np.zeros_like(result_image)
        mask_color[segmentation_mask > 0] = [0, 0, 255]  # 红色（OpenCV使用BGR格式）
        
        # 将掩码叠加到原图上 - 进一步增大alpha值让红色更加明显
        alpha = 0.8
        result_image = cv2.addWeighted(result_image, 1.0, mask_color, alpha, 0)
    
    return result_image

def generate_combined_image(original_image, detection_result, segmentation_mask):
    """生成左右分屏图像：左侧原图+检测，右侧原图+分割"""
    h, w = original_image.shape[:2]
    
    # 左侧：原图+检测结果
    left_image = generate_detection_image(original_image, detection_result)
    
    # 右侧：原图+分割结果
    right_image = generate_segmentation_image(original_image, segmentation_mask)
    
    # 合并为左右分屏
    combined_image = np.hstack([left_image, right_image])
    
    return combined_image

@app.get("/", response_class=HTMLResponse)
async def index():
    """主页"""
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    return JSONResponse({
        'status': '运行中',
        'timestamp': datetime.now().isoformat(),
        'detection_model': model_manager.detection_model_name or '未加载',
        'segmentation_model': model_manager.segmentation_model_name or '未加载'
    })

@app.get("/api/models")
async def get_models():
    """获取可用模型列表"""
    from shared_models import models_available
    
    return JSONResponse({
        'detection_models': [
            {'name': 'SSD300', 'id': 'ssd', 'available': models_available.get('ssd', False)},
            {'name': 'YOLOv11', 'id': 'yolov11_detection', 'available': models_available.get('yolov11_detection', False)}
        ],
        'segmentation_models': [
            {'name': 'DeepLabV3', 'id': 'deeplabv3', 'available': models_available.get('deeplabv3', False)},
            {'name': 'YOLOv11-Seg', 'id': 'yolov11_segmentation', 'available': models_available.get('yolov11_segmentation', False)}
        ]
    })

@app.post("/api/models/load")
async def load_models(
    detection: str = Form(None),
    segmentation: str = Form(None)
):
    """加载指定模型"""
    try:
        results = {}
        
        if detection and detection != '无':
            results['detection'] = model_manager.load_detection_model(detection)
        
        if segmentation and segmentation != '无':
            results['segmentation'] = model_manager.load_segmentation_model(segmentation)
        
        return JSONResponse({
            'success': True,
            'results': results,
            'detection_model': model_manager.detection_model_name or '未加载',
            'segmentation_model': model_manager.segmentation_model_name or '未加载'
        })
        
    except Exception as e:
        logger.error(f"模型加载错误: {e}")
        return JSONResponse({
            'success': False,
            'message': f'模型加载失败: {str(e)}'
        }, status_code=500)

@app.post("/api/process")
async def process_image(image: UploadFile = File(...)):
    """处理上传的图像"""
    try:
        # 读取图像
        image_data = await image.read()
        image_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="无法读取图像文件")
        
        # 转换为RGB
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = {}
        detection_result = None
        segmentation_mask = None
        
        # 执行检测
        if model_manager.detection_model:
            try:
                # 根据模型类型使用正确的参数
                if hasattr(model_manager.detection_model, 'model') and hasattr(model_manager.detection_model.model, 'overrides'):
                    # YOLOv11模型使用conf_threshold参数
                    detection_result = model_manager.detection_model.predict(
                        image_rgb, 
                        conf_threshold=0.5
                    )
                else:
                    # SSD模型使用score_threshold参数
                    detection_result = model_manager.detection_model.predict(
                        image_rgb, 
                        score_threshold=0.5
                    )
                results['detection'] = {
                    'boxes': detection_result['boxes'].tolist() if isinstance(detection_result['boxes'], np.ndarray) else detection_result['boxes'],
                    'labels': detection_result['labels'].tolist() if isinstance(detection_result['labels'], np.ndarray) else detection_result['labels'],
                    'scores': detection_result['scores'].tolist() if isinstance(detection_result['scores'], np.ndarray) else detection_result['scores'],
                    'class_names': detection_result['class_names'].tolist() if isinstance(detection_result['class_names'], np.ndarray) else detection_result['class_names'],
                    'count': detection_result['detection_count']
                }
            except Exception as e:
                logger.error(f"检测失败: {e}")
                results['detection_error'] = str(e)
        
        # 执行分割
        if model_manager.segmentation_model:
            try:
                binary_mask, prob_mask = model_manager.segmentation_model.predict(
                    image_rgb, 
                    threshold=0.5
                )
                segmentation_mask = binary_mask
                results['segmentation'] = {
                    'has_segmentation': True,
                    'mask_shape': binary_mask.shape
                }
            except Exception as e:
                logger.error(f"分割失败: {e}")
                results['segmentation_error'] = str(e)
        
        # 统计车辆数量
        statistics = count_vehicles_in_flooded_areas(detection_result, segmentation_mask)
        results['statistics'] = statistics
        
        # 生成结果图像
        detection_image = generate_detection_image(image_rgb, detection_result) if detection_result else image_rgb
        segmentation_image = generate_segmentation_image(image_rgb, segmentation_mask) if segmentation_mask is not None else image_rgb
        combined_image = generate_combined_image(image_rgb, detection_result, segmentation_mask)
        
        # 转换为base64返回
        _, buffer = cv2.imencode('.jpg', detection_image)
        detection_base64 = base64.b64encode(buffer).decode('utf-8')
        
        _, buffer = cv2.imencode('.jpg', segmentation_image)
        segmentation_base64 = base64.b64encode(buffer).decode('utf-8')
        
        _, buffer = cv2.imencode('.jpg', combined_image)
        combined_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse({
            'success': True,
            'message': '图像处理成功',
            'timestamp': datetime.now().isoformat(),
            'detection_image': f"data:image/jpeg;base64,{detection_base64}",
            'segmentation_image': f"data:image/jpeg;base64,{segmentation_base64}",
            'combined_image': f"data:image/jpeg;base64,{combined_base64}",
            'results': results
        })
        
    except Exception as e:
        logger.error(f"图像处理错误: {e}")
        return JSONResponse({
            'success': False,
            'message': f'图像处理失败: {str(e)}'
        }, status_code=500)

@app.post("/api/download/detection")
async def download_detection(image: UploadFile = File(...)):
    """下载检测结果图像"""
    try:
        # 读取图像
        image_data = await image.read()
        image_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="无法读取图像文件")
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 执行检测
        if not model_manager.detection_model:
            raise HTTPException(status_code=400, detail="检测模型未加载")
        
        # 根据模型类型使用正确的参数
        if hasattr(model_manager.detection_model, 'model') and hasattr(model_manager.detection_model.model, 'overrides'):
            # YOLOv11模型使用conf_threshold参数
            detection_result = model_manager.detection_model.predict(
                image_rgb, 
                conf_threshold=0.5
            )
        else:
            # SSD模型使用score_threshold参数
            detection_result = model_manager.detection_model.predict(
                image_rgb, 
                score_threshold=0.5
            )
        
        # 生成检测结果图像
        detection_image = generate_detection_image(image_rgb, detection_result)
        
        # 保存到临时文件
        temp_file = TEMP_DIR / f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(temp_file), cv2.cvtColor(detection_image, cv2.COLOR_RGB2BGR))
        
        return FileResponse(
            str(temp_file),
            media_type="image/jpeg",
            filename=f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        
    except Exception as e:
        logger.error(f"下载检测结果错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/download/segmentation")
async def download_segmentation(image: UploadFile = File(...)):
    """下载分割结果图像"""
    try:
        # 读取图像
        image_data = await image.read()
        image_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="无法读取图像文件")
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 执行分割
        if not model_manager.segmentation_model:
            raise HTTPException(status_code=400, detail="分割模型未加载")
        
        binary_mask, prob_mask = model_manager.segmentation_model.predict(
            image_rgb, 
            threshold=0.5
        )
        
        # 生成分割结果图像
        segmentation_image = generate_segmentation_image(image_rgb, binary_mask)
        
        # 保存到临时文件
        temp_file = TEMP_DIR / f"segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(temp_file), cv2.cvtColor(segmentation_image, cv2.COLOR_RGB2BGR))
        
        return FileResponse(
            str(temp_file),
            media_type="image/jpeg",
            filename=f"segmentation_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        
    except Exception as e:
        logger.error(f"下载分割结果错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    logger.info("启动FloodRisk Web应用")
    uvicorn.run(app, host='0.0.0.0', port=5000)
