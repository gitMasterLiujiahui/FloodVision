import torch
import torch.nn as nn
import cv2
import numpy as np
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

class YOLOv11SegmentationModel:
    def __init__(self, model_path, device='auto'):
        """
        初始化YOLOv11分割模型
        Args:
            model_path: 模型权重文件路径
            device: 设备类型 ('auto', 'cpu', 'cuda')
        """
        self.device = self._setup_device(device)
        self.model_path = model_path
        
        # 加载YOLO模型
        self.model = YOLO(model_path)
        
        # 设置设备
        self.model.to(self.device)
        
        # 设置预测参数
        self.model.overrides['conf'] = 0.25  # 默认置信度阈值
        self.model.overrides['iou'] = 0.45   # 默认IoU阈值
        
        logger.info(f"YOLOv11分割模型加载成功: {model_path}, 设备: {self.device}")
    
    def _setup_device(self, device):
        """设置计算设备"""
        if device == 'auto':
            device = '0' if torch.cuda.is_available() else 'cpu'
        
        if device == '0' and not torch.cuda.is_available():
            logger.warning("CUDA不可用，使用CPU")
            device = 'cpu'
        
        logger.info(f"使用设备: {device}")
        return device
    
    def eval(self):
        """设置模型为评估模式"""
        self.model.eval()
    
    def preprocess_image(self, image):
        """预处理图像"""
        # 保存原始尺寸用于后处理
        if isinstance(image, str):
            # 如果是文件路径，加载图像
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_size = image.shape[:2]  # (H, W)
        
        return image, original_size
    
    def predict(self, image, threshold=0.5):
        """
        预测水面分割结果
        Args:
            image: 输入图像（文件路径、numpy数组或PIL图像）
            threshold: 分割阈值
        Returns:
            binary_mask: 二值分割掩码 [H, W]
            prob_mask: 概率分割掩码 [H, W]
        """
        try:
            # 预处理图像
            processed_image, original_size = self.preprocess_image(image)
            
            # 使用YOLO进行预测
            results = self.model(processed_image, verbose=False)
            
            # 获取分割结果
            if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                # 获取分割掩码
                prob_mask = results[0].masks.data.cpu().numpy()
                
                # 处理掩码形状
                if len(prob_mask.shape) == 3:
                    prob_mask = prob_mask[0]  # 取第一个掩码
                
                # 调整到原始尺寸
                prob_mask_resized = cv2.resize(prob_mask, (original_size[1], original_size[0]))
                
                # 应用阈值
                binary_mask = (prob_mask_resized > threshold).astype(np.uint8)
                
                logger.info(f"YOLOv11分割成功，检测到分割区域")
                return binary_mask, prob_mask_resized
            else:
                # 如果没有检测到分割结果，返回全零掩码
                logger.warning("YOLOv11未检测到分割结果")
                binary_mask = np.zeros(original_size, dtype=np.uint8)
                prob_mask = np.zeros(original_size, dtype=np.float32)
                return binary_mask, prob_mask
                
        except Exception as e:
            logger.error(f"YOLOv11分割预测失败: {e}")
            # 返回错误掩码
            if 'original_size' in locals():
                binary_mask = np.zeros(original_size, dtype=np.uint8)
                prob_mask = np.zeros(original_size, dtype=np.float32)
            else:
                binary_mask = np.zeros((640, 640), dtype=np.uint8)
                prob_mask = np.zeros((640, 640), dtype=np.float32)
            return binary_mask, prob_mask
    
    def __call__(self, image):
        """调用接口，返回与DeepLabV3相同的格式"""
        binary_mask, prob_mask = self.predict(image)
        return binary_mask, prob_mask