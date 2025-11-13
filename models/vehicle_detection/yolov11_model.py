import torch 
import torch.nn as nn
import logging
import numpy as np
import cv2
from PIL import Image
import time
import os
from typing import List, Dict, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class YOLOv11(nn.Module):
    '''
    YOLOv11模型
    该模型基于YOLOv11n架构，用于车辆检测任务。
    参考yolo_prediction.py的YOLOv11Predictor类重构，提供专业的预测接口
    '''
    def __init__(self, num_classes=3, weights_path=None, device='auto'):
        super().__init__()
        self.weights_path = weights_path
        self.num_classes = num_classes
        self.device = self._setup_device(device)
        
        # 类别配置（与训练目录中的data.yaml一致）
        self.class_names = {
            0: "wheel",   # 车轮淹没部位
            1: "door",    # 车门淹没部位
            2: "window"   # 车窗淹没部位
        }
        
        # 类别颜色（用于可视化）
        self.class_colors = {
            0: (255, 0, 0),      # 红色 - 车轮
            1: (0, 255, 0),      # 绿色 - 车门
            2: (0, 0, 255)       # 蓝色 - 车窗
        }
        
        # 默认图像尺寸
        self.input_size = 640
        
        # 加载模型
        self.model = self.load_pretrained_yolo(self.weights_path)

    def _setup_device(self, device: str) -> str:
        """设置计算设备"""
        if device == 'auto':
            device = '0' if torch.cuda.is_available() else 'cpu'
        
        if device == '0' and not torch.cuda.is_available():
            logger.warning("CUDA不可用，使用CPU")
            device = 'cpu'
        
        logger.info(f"使用设备: {device}")
        return device

    def forward(self, x):
        '''
        前向传播
        '''
        return self.model(x)
    
    def predict(self, image, conf_threshold=0.3, iou_threshold=0.45):
        '''
        对单张图像进行预测（与训练目录中的实现保持一致）
        
        Args:
            image: 图像路径、numpy数组或PIL图像
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            
        Returns:
            预测结果字典
        '''
        # 设置预测参数
        self.model.overrides['conf'] = conf_threshold
        self.model.overrides['iou'] = iou_threshold
        
        # 执行预测
        if isinstance(image, str):
            # 从文件路径预测
            results = self.model.predict(image, verbose=False)
        else:
            # 从numpy数组或PIL图像预测
            results = self.model.predict(image, verbose=False)
        
        # 解析预测结果
        if len(results) == 0:
            return self._create_empty_result(image)
        
        result = results[0]  # 获取第一个结果
        
        # 获取检测框信息
        boxes = []
        scores = []
        labels = []
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            # 获取检测框数据
            boxes_data = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores_data = result.boxes.conf.cpu().numpy()  # 置信度
            labels_data = result.boxes.cls.cpu().numpy().astype(int)  # 类别索引
            
            for i in range(len(boxes_data)):
                boxes.append(boxes_data[i].tolist())
                scores.append(float(scores_data[i]))
                labels.append(int(labels_data[i]))
        
        # 获取原始图像尺寸
        if hasattr(result, 'orig_shape'):
            orig_size = (result.orig_shape[1], result.orig_shape[0])  # (width, height)
        else:
            # 如果无法获取原始尺寸，使用默认值
            orig_size = (640, 640)
        
        # 构建结果字典
        result_dict = {
            'image_path': image if isinstance(image, str) else 'numpy_array',
            'original_size': orig_size,
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'class_names': [self.class_names.get(label, f"Class_{label}") for label in labels],
            'detection_count': len(boxes),
            'timestamp': time.time(),
            'model_name': 'YOLOv11'
        }
        
        return result_dict
    
    def _create_empty_result(self, image):
        """创建空结果字典"""
        return {
            'image_path': image if isinstance(image, str) else 'numpy_array',
            'original_size': (640, 640),
            'boxes': [],
            'scores': [],
            'labels': [],
            'class_names': [],
            'detection_count': 0,
            'timestamp': time.time(),
            'model_name': 'YOLOv11'
        }
    
    def predict_batch(self, image_paths: List[str], 
                     conf_threshold: float = 0.3,
                     iou_threshold: float = 0.45,
                     batch_size: int = 8) -> List[Dict]:
        """
        批量预测
        
        Args:
            image_paths: 图像路径列表
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            batch_size: 批次大小
            
        Returns:
            预测结果列表
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            logger.info(f"处理批次 {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")
            
            # 设置预测参数
            self.model.overrides['conf'] = conf_threshold
            self.model.overrides['iou'] = iou_threshold
            
            # 批量预测
            batch_results = self.model.predict(batch_paths, verbose=False)
            
            for j, result in enumerate(batch_results):
                image_path = batch_paths[j]
                
                # 获取检测框信息
                boxes = []
                scores = []
                labels = []
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes_data = result.boxes.xyxy.cpu().numpy()
                    scores_data = result.boxes.conf.cpu().numpy()
                    labels_data = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for k in range(len(boxes_data)):
                        boxes.append(boxes_data[k].tolist())
                        scores.append(float(scores_data[k]))
                        labels.append(int(labels_data[k]))
                
                # 获取原始图像尺寸
                if hasattr(result, 'orig_shape'):
                    orig_size = (result.orig_shape[1], result.orig_shape[0])
                else:
                    orig_size = (640, 640)
                
                # 构建结果字典
                result_dict = {
                    'image_path': image_path,
                    'original_size': orig_size,
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels,
                    'class_names': [self.class_names.get(label, f"Class_{label}") for label in labels],
                    'detection_count': len(boxes),
                    'timestamp': time.time(),
                    'model_name': 'YOLOv11'
                }
                
                results.append(result_dict)
        
        return results
       

    def load_pretrained_yolo(self, weights_path):
        '''
        加载预训练的YOLOv11模型权重
        '''
        try:
            from ultralytics import YOLO
            
            # 检查权重文件是否存在
            if not weights_path or not os.path.exists(weights_path):
                raise FileNotFoundError(f"模型文件不存在: {weights_path}")
            
            logger.info(f"加载模型: {weights_path}")
            
            # 加载YOLOv11模型
            yolov11_model = YOLO(weights_path)
            
            # 设置模型参数
            yolov11_model.overrides['conf'] = 0.25  # 默认置信度阈值
            yolov11_model.overrides['iou'] = 0.45   # 默认IoU阈值
            
            logger.info("YOLOv11模型加载成功")
            
            # 显示模型信息
            self._show_model_info(yolov11_model)
            
            return yolov11_model

        except Exception as e:
            logger.error(f"加载预训练权重失败: {e}")
            # 重新抛出异常以便调用方处理
            raise
    
    def _show_model_info(self, model):
        """显示模型信息"""
        try:
            # 获取模型信息
            model_info = model.info()
            logger.info(f"模型架构: {model.__class__.__name__}")
            logger.info(f"输入尺寸: {self.input_size}x{self.input_size}")
            logger.info(f"类别数量: {self.num_classes}")
            
            # 显示类别名称
            logger.info("类别名称:")
            for class_id, class_name in self.class_names.items():
                logger.info(f"  {class_id}: {class_name}")
                
        except Exception as e:
            logger.warning(f"获取模型信息失败: {e}")
    
    def _infer_num_classes_from_weights(self):
        """从权重文件推断类别数"""
        try:
            from ultralytics import YOLO
            
            if not self.weights_path:
                return 3  # 默认值
                
            # 加载模型但不修改类别数
            temp_model = YOLO(self.weights_path)
            
            # 获取实际的类别数
            if hasattr(temp_model, 'names') and temp_model.names:
                num_classes = len(temp_model.names)
                logger.info(f"从权重文件推断类别数: {num_classes}")
                return num_classes
            else:
                return 3  # 默认值
                
        except Exception as e:
            logger.warning(f"无法从权重文件推断类别数: {e}, 使用默认值3")
            return 3
    
    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        获取预测统计信息
        
        Args:
            results: 预测结果列表
            
        Returns:
            统计信息字典
        """
        total_images = len(results)
        total_detections = 0
        class_counts = {class_id: 0 for class_id in self.class_names.keys()}
        
        for result in results:
            total_detections += result['detection_count']
            
            for label in result['labels']:
                if label in class_counts:
                    class_counts[label] += 1
        
        statistics = {
            'total_images': total_images,
            'total_detections': total_detections,
            'average_detections_per_image': total_detections / total_images if total_images > 0 else 0,
            'class_distribution': {self.class_names[class_id]: count for class_id, count in class_counts.items()},
            'detection_rate': total_detections / total_images if total_images > 0 else 0
        }
        
        return statistics
   