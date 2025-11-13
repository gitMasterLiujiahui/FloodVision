#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享模型模块 - 消除simple_web_app.py和web_app.py中的重复代码
"""

import os
import sys

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def import_models():
    """导入模型，如果失败则返回虚拟模型"""
    try:
        from models.vehicle_detection.yolov11_model import YOLOv11
        from models.water_segmentation.deeplabv3_model import DeepLabV3Model
        from models.water_segmentation.yolov11_model import YOLOv11SegmentationModel
        
        # 让SSD300也使用YOLOv11模型
        SSD300 = YOLOv11
        
        return {
            'SSD300': SSD300,
            'YOLOv11': YOLOv11,
            'DeepLabV3Model': DeepLabV3Model,
            'YOLOv11SegmentationModel': YOLOv11SegmentationModel
        }
    except ImportError as e:
        print(f"模型导入失败: {e}")
        print("将使用虚拟模型进行演示")
        return create_dummy_models()

def create_dummy_models():
    """创建虚拟模型类"""
    
    class YOLOv11:
        def __init__(self, num_classes=3, weights_path=None):
            self.class_names = ['车轮', '车门', '车窗']
            
        def predict(self, image, conf_threshold=0.3, iou_threshold=0.45):
            return {
                'boxes': [],
                'scores': [],
                'labels': [],
                'class_names': self.class_names
            }
    
    # 让SSD300虚拟类也使用YOLOv11虚拟类
    SSD300 = YOLOv11
    
    class DeepLabV3Model:
        def __init__(self, model_path=None):
            pass
            
        def predict(self, image, threshold=0.5):
            return {
                'binary_mask': None,
                'prob_mask': None,
                'shape': (224, 224) if hasattr(image, 'shape') else (224, 224)
            }
    
    class YOLOv11SegmentationModel:
        def __init__(self, model_path=None):
            pass
            
        def predict(self, image, threshold=0.5):
            return {
                'binary_mask': None,
                'prob_mask': None,
                'shape': (224, 224) if hasattr(image, 'shape') else (224, 224)
            }
    
    return {
        'SSD300': SSD300,
        'YOLOv11': YOLOv11,
        'DeepLabV3Model': DeepLabV3Model,
        'YOLOv11SegmentationModel': YOLOv11SegmentationModel
    }

def check_model_files():
    """检查模型文件是否存在"""
    MODEL_FILES = {
        'ssd': 'models/vehicle_detection/ssd.pt',
        'yolov11_detection': 'models/vehicle_detection/yolov11.pt',
        'deeplabv3': 'models/water_segmentation/deeplabv3.pth',
        'yolov11_segmentation': 'models/water_segmentation/yolov11.pt'
    }
    
    models_available = {}
    for model_name, model_path in MODEL_FILES.items():
        if os.path.exists(model_path):
            models_available[model_name] = True
        else:
            models_available[model_name] = False
            print(f"警告: 模型文件 {model_path} 不存在")
    
    return models_available

def update_model_availability(model_type, available):
    """更新模型可用性状态"""
    global models_available
    models_available[model_type] = available
    return models_available

# 预导入模型
ModelClasses = import_models()
models_available = check_model_files()