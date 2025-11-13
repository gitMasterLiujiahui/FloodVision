import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepLabV3Model:
    def __init__(self, model_path, device='cpu'):
        """
        初始化DeepLabV3模型
        Args:
            model_path: 模型权重文件路径
            device: 设备类型
        """
        self.device = device
        self.model_path = model_path
        
        # 创建模型结构
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet18",
            encoder_weights=None,  # 不加载预训练权重
            in_channels=3,
            classes=1,
            activation=None
        )
        
        # 图像预处理
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        # 加载权重
        self.load_weights()
        
        # 设置为评估模式
        self.model.eval()
        self.model.to(device)
    
    def load_weights(self):
        """加载模型权重"""
        try:
            # 尝试不同的加载方式
            try:
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=True)
            except:
                # 如果weights_only失败，尝试不使用weights_only
                checkpoint = torch.load(self.model_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"DeepLabV3模型权重加载成功: {self.model_path}")
        except Exception as e:
            print(f"DeepLabV3模型权重加载失败: {e}")
            # 如果加载失败，创建一个随机初始化的模型
            print("使用随机初始化的模型")
            pass
    
    def eval(self):
        """设置模型为评估模式"""
        self.model.eval()
    
    def preprocess_image(self, image):
        """预处理图像"""
        # 保存原始尺寸用于后处理
        original_size = image.shape[:2]  # (H, W)
        
        # 应用预处理
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # 添加批次维度
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, original_size
    
    def predict(self, image, threshold=0.5):
        """
        预测水面分割结果
        Args:
            image: 输入图像 (numpy数组)
            threshold: 分割阈值
        Returns:
            binary_mask: 二值分割掩码 [H, W]
            prob_mask: 概率分割掩码 [H, W]
        """
        with torch.no_grad():
            # 预处理
            image_tensor, original_size = self.preprocess_image(image)
            
            # 预测
            output = self.model(image_tensor)
            pred_mask = torch.sigmoid(output)
        
        # 后处理
        pred_mask_np = pred_mask.cpu().numpy()[0, 0]  # (H, W)
        
        # 调整到原始尺寸
        pred_mask_resized = cv2.resize(pred_mask_np, (original_size[1], original_size[0]))
        
        # 应用阈值
        binary_mask = (pred_mask_resized > threshold).astype(np.uint8)
        
        return binary_mask, pred_mask_resized
    
    def __call__(self, image):
        return self.predict(image)