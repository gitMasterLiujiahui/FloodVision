import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
import torchvision.ops
import logging
import numpy as np
import cv2
from PIL import Image
import time
import os
from typing import List, Dict, Tuple, Optional, Union

logger = logging.getLogger(__name__)

def create_ssd_model(num_classes=4, freeze_backbone=True):
    """
    创建SSD300模型的便捷函数
    
    Args:
        num_classes: 类别数量（包含背景）
        freeze_backbone: 是否冻结骨干网络
    
    Returns:
        SSD300模型实例
    """
    return SSD300(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone
    )

class SSD300(nn.Module):
    def __init__(self, num_classes=3, weights_path="models/vehicle_detection/ssd.pt", freeze_backbone=True, device='auto'):
        """
        SSD300模型，支持预训练权重加载
        
        Args:
            num_classes: 类别数量，默认为3（车辆淹没检测）
            weights_path: 权重文件路径，默认为训练好的ssd.pt权重文件
            freeze_backbone: 是否冻结骨干网络，默认为True
            device: 设备类型 ('auto', 'cpu', 'cuda')
        """
        super().__init__()
        # 根据权重文件分析，实际类别数可能为4（含背景）或6（含背景）
        # 使用更灵活的类别数处理
        self.num_classes = 4  # 强制设置为4个类别（3个目标+1个背景）
        self.weights_path = weights_path
        self.freeze_backbone = freeze_backbone
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
        self.input_size = (300, 300)
        
        # 加载模型
        self.model = self._load_pretrained_model()
        
        # 修改分类头以适应自定义类别数
        self._modify_classification_head()
        
        # 冻结骨干网络参数（可选）
        if self.freeze_backbone:
            self._freeze_backbone()
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"SSD300模型初始化完成: 类别数(含背景)={self.num_classes}, 冻结骨干={freeze_backbone}, 设备={self.device}")
    
    def _setup_device(self, device):
        """设置设备"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device}")
        return torch.device(device)
    
    def _load_pretrained_model(self):
        """加载预训练模型"""
        try:
            # 优先使用训练好的权重文件
            if self.weights_path and os.path.exists(self.weights_path):
                logger.info(f"从指定路径加载训练好的权重: {self.weights_path}")
                
                # 创建基础模型
                model = ssd300_vgg16(weights=None, num_classes=self.num_classes)
                
                # 加载权重
                try:
                    state_dict = torch.load(self.weights_path, map_location='cpu', weights_only=True)
                except:
                    state_dict = torch.load(self.weights_path, map_location='cpu')
                
                # 处理不同的权重文件格式
                if 'model' in state_dict:
                    model.load_state_dict(state_dict['model'], strict=False)
                elif 'state_dict' in state_dict:
                    model.load_state_dict(state_dict['state_dict'], strict=False)
                else:
                    model.load_state_dict(state_dict, strict=False)
                
                logger.info(f"从训练好的权重文件加载成功: {self.weights_path}")
            else:
                logger.info("加载torchvision预训练权重作为备选方案")
                
                # 加载torchvision预训练权重
                model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT, num_classes=91)
                
                # 修改分类头以适应我们的类别数
                self._modify_classification_head_after_loading(model)
                
                logger.info("使用torchvision预训练权重初始化模型")
            
            return model
            
        except Exception as e:
            logger.error(f"加载预训练模型失败: {e}")
            # 创建基础模型作为备选
            model = ssd300_vgg16(weights=None, num_classes=self.num_classes)
            return model
    
    def _modify_classification_head_after_loading(self, model):
        """在加载预训练权重后修改分类头和回归头以适应自定义类别数量"""
        # 根据torchvision SSD300的默认配置，每个检测头的先验框数量
        priors_per_head = [4, 6, 6, 6, 4, 4]
        
        # 修改分类头（类别数包含背景）
        self._modify_head(model.head.classification_head, priors_per_head, self.num_classes, "分类头")
        
        # 修改回归头（保持4个坐标不变）
        self._modify_head(model.head.regression_head, priors_per_head, 4, "回归头")
        
        # 关键修复：更新分类头的num_columns参数
        model.head.classification_head.num_columns = self.num_classes
        
        logger.info(f"分类头和回归头已修改为适应 {self.num_classes} 个类别(含背景)")
        logger.info(f"更新分类头num_columns参数: 91 -> {self.num_classes}")
    
    def _modify_head(self, head, priors_per_head, num_outputs_per_prior, head_name):
        """修改指定的头（分类头或回归头）"""
        module_list = head.module_list
        
        for i, module in enumerate(module_list):
            if isinstance(module, nn.Conv2d):
                # 获取当前卷积层的输出通道数
                current_out_channels = module.out_channels
                
                # 计算每个先验框的原始输出数
                outputs_per_prior = current_out_channels // priors_per_head[i]
                
                # 计算新的输出通道数
                new_out_channels = priors_per_head[i] * num_outputs_per_prior
                
                # 创建新的卷积层
                new_conv = nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=new_out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None
                )
                
                # 初始化新卷积层的权重
                if module.weight is not None:
                    # 权重形状: [out_channels, in_channels, k, k]
                    # 初始化新权重
                    new_weight = torch.zeros(new_out_channels, module.in_channels, 
                                           module.kernel_size[0], module.kernel_size[1])
                    
                    # 复制权重
                    for j in range(priors_per_head[i]):
                        start_old = j * outputs_per_prior
                        start_new = j * num_outputs_per_prior
                        
                        # 复制权重
                        copy_length = min(num_outputs_per_prior, outputs_per_prior)
                        if copy_length > 0:
                            new_weight[start_new:start_new + copy_length] = \
                                module.weight.data[start_old:start_old + copy_length]
                        
                        # 如果新输出数大于原始输出数，随机初始化剩余部分
                        if num_outputs_per_prior > outputs_per_prior:
                            remaining = num_outputs_per_prior - outputs_per_prior
                            new_weight[start_new + outputs_per_prior:start_new + num_outputs_per_prior] = \
                                torch.randn(remaining, module.in_channels, 
                                          module.kernel_size[0], module.kernel_size[1]) * 0.01
                    
                    new_conv.weight.data = new_weight
                
                # 复制偏置
                if module.bias is not None:
                    new_bias = torch.zeros(new_out_channels)
                    
                    for j in range(priors_per_head[i]):
                        start_old = j * outputs_per_prior
                        start_new = j * num_outputs_per_prior
                        
                        # 复制偏置
                        copy_length = min(num_outputs_per_prior, outputs_per_prior)
                        if copy_length > 0:
                            new_bias[start_new:start_new + copy_length] = \
                                module.bias.data[start_old:start_old + copy_length]
                        
                        # 如果新输出数大于原始输出数，零初始化剩余部分
                        if num_outputs_per_prior > outputs_per_prior:
                            remaining = num_outputs_per_prior - outputs_per_prior
                            new_bias[start_new + outputs_per_prior:start_new + num_outputs_per_prior] = 0
                    
                    new_conv.bias.data = new_bias
                
                # 替换模块
                module_list[i] = new_conv
    
    def _modify_classification_head(self):
        """修改分类头以适应自定义类别数量（占位方法，保持兼容性）"""
        # 此方法在构造函数中调用，但实际修改逻辑在_load_pretrained_model中完成
        # 保持空实现以确保兼容性
        pass
    
    def _freeze_backbone(self):
        """冻结部分骨干：只冻结前半段 VGG16 特征层，解冻高层特征与头部"""
        # VGG16 backbone 是一个 nn.Sequential(features)
        features = self.model.backbone
        
        # 正确冻结骨干网络：基于模块层数而非参数数量
        # VGG16 features 包含多个卷积层和池化层
        layers = list(features.children())
        num_layers = len(layers)
        cutoff = int(num_layers * 0.66)  # 冻结前2/3的层
        
        for i, layer in enumerate(layers):
            for param in layer.parameters():
                param.requires_grad = i >= cutoff  # 只解冻后1/3的层
        
        frozen_params = sum(p.numel() for p in features.parameters() if not p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"部分冻结骨干参数: {frozen_params:,}/{total_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    def forward(self, x, targets=None):
        """
        前向传播
        
        Args:
            x: 输入图像张量 [batch_size, 3, 300, 300]
            targets: 训练时的目标（可选）
            
        Returns:
            如果是训练模式：返回损失字典
            如果是评估模式：返回预测结果列表
        """
        return self.model(x, targets)
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            "模型名称": "SSD300 with VGG16 Backbone",
            "类别数量": self.num_classes,
            "总参数": f"{total_params:,}",
            "可训练参数": f"{trainable_params:,}",
            "冻结参数": f"{total_params - trainable_params:,}",
            "冻结比例": f"{(total_params - trainable_params) / total_params * 100:.1f}%"
        }
        
        return info
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int], Image.Image]:
        """
        预处理图像（参考ssd_prediction.py的实现）
        
        Args:
            image: 图像路径、numpy数组或PIL图像
            
        Returns:
            预处理后的图像张量、原始尺寸、原始图像
        """
        # 加载图像
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError("不支持的图像格式")
        
        # 记录原始尺寸
        orig_size = pil_image.size  # (width, height)
        
        # 调整尺寸
        resized_image = pil_image.resize(self.input_size)
        
        # 转换为numpy数组并归一化
        image_array = np.array(resized_image).transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # 转换为张量
        image_tensor = torch.from_numpy(image_array)
        
        return image_tensor, orig_size, pil_image
    
    def predict(self, image: Union[str, np.ndarray, Image.Image], 
                score_threshold: float = 0.3) -> Dict:
        """
        对单张图像进行预测（参考ssd_prediction.py的实现）
        
        Args:
            image: 图像路径、numpy数组或PIL图像
            score_threshold: 置信度阈值
            
        Returns:
            预测结果字典
        """
        # 预处理图像
        image_tensor, orig_size, orig_image = self.preprocess_image(image)
        
        # 添加批次维度
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # 解析预测结果
        pred_boxes = predictions[0]['boxes'].cpu().numpy()
        pred_scores = predictions[0]['scores'].cpu().numpy()
        pred_labels = predictions[0]['labels'].cpu().numpy()
        
        # 过滤低置信度检测
        valid_indices = pred_scores >= score_threshold
        filtered_boxes = pred_boxes[valid_indices]
        filtered_scores = pred_scores[valid_indices]
        filtered_labels = pred_labels[valid_indices]
        
        # 应用非极大值抑制（NMS）去除重叠检测框
        if len(filtered_boxes) > 0:
            # 转换为torch张量进行NMS
            boxes_tensor = torch.from_numpy(filtered_boxes)
            scores_tensor = torch.from_numpy(filtered_scores)
            
            # 按类别分别应用NMS
            unique_labels = np.unique(filtered_labels)
            nms_boxes = []
            nms_scores = []
            nms_labels = []
            
            for label in unique_labels:
                label_mask = filtered_labels == label
                label_boxes = boxes_tensor[label_mask]
                label_scores = scores_tensor[label_mask]
                
                if len(label_boxes) > 0:
                    # 应用NMS，IoU阈值为0.5
                    keep_indices = torchvision.ops.nms(label_boxes, label_scores, 0.5)
                    
                    nms_boxes.extend(label_boxes[keep_indices].numpy())
                    nms_scores.extend(label_scores[keep_indices].numpy())
                    nms_labels.extend([label] * len(keep_indices))
            
            # 更新过滤后的结果
            if len(nms_boxes) > 0:
                filtered_boxes = np.array(nms_boxes)
                filtered_scores = np.array(nms_scores)
                filtered_labels = np.array(nms_labels)
            else:
                # 如果NMS后没有检测框，使用空数组
                filtered_boxes = np.array([]).reshape(0, 4)
                filtered_scores = np.array([])
                filtered_labels = np.array([])
        
        # 过滤背景类（背景类为0）
        valid_foreground = filtered_labels > 0
        filtered_boxes = filtered_boxes[valid_foreground]
        filtered_scores = filtered_scores[valid_foreground]
        filtered_labels = filtered_labels[valid_foreground]
        
        # 修复标签映射逻辑：根据实际类别数进行正确映射
        if len(filtered_labels) > 0:
            # 检查标签范围，确定正确的映射方式
            min_label = np.min(filtered_labels)
            max_label = np.max(filtered_labels)
            
            # 如果标签范围是1-3，则映射到0-2（对应wheel, door, window）
            if min_label == 1 and max_label <= 3:
                filtered_labels = filtered_labels - 1
            # 如果标签范围是0-2，则已经是正确的映射
            elif min_label >= 0 and max_label <= 2:
                # 已经是正确的映射，无需调整
                pass
            else:
                # 其他情况，使用模运算确保标签在有效范围内
                filtered_labels = filtered_labels % 3
                logger.warning(f"检测到异常标签范围{min_label}-{max_label}，使用模运算修正")
        
        # 将边界框坐标缩放回原始图像尺寸
        scale_x = orig_size[0] / self.input_size[0]
        scale_y = orig_size[1] / self.input_size[1]
        
        scaled_boxes = []
        for box in filtered_boxes:
            x1, y1, x2, y2 = box
            scaled_boxes.append([
                x1 * scale_x,  # x1
                y1 * scale_y,  # y1
                x2 * scale_x,  # x2
                y2 * scale_y   # y2
            ])
        
        # 构建结果字典
        result = {
            'image_path': image if isinstance(image, str) else 'numpy_array',
            'original_size': orig_size,
            'boxes': np.array(scaled_boxes),
            'scores': filtered_scores,
            'labels': filtered_labels,
            'class_names': [self.class_names.get(label, f"Class_{label}") for label in filtered_labels],
            'detection_count': len(filtered_boxes),
            'timestamp': time.time(),
            'model_name': 'SSD300'
        }
        
        return result
    
    def predict_batch(self, image_paths: List[str], 
                     score_threshold: float = 0.3,
                     batch_size: int = 4) -> List[Dict]:
        """
        批量预测
        
        Args:
            image_paths: 图像路径列表
            score_threshold: 置信度阈值
            batch_size: 批次大小
            
        Returns:
            预测结果列表
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            logger.info(f"处理批次 {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")
            
            for image_path in batch_paths:
                try:
                    result = self.predict(image_path, score_threshold)
                    results.append(result)
                except Exception as e:
                    logger.error(f"预测图像失败 {image_path}: {e}")
                    results.append({
                        'image_path': image_path,
                        'error': str(e),
                        'detection_count': 0
                    })
        
        return results
    
    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        获取预测统计信息
        
        Args:
            results: 预测结果列表
            
        Returns:
            统计信息字典
        """
        total_images = len(results)
        successful_predictions = 0
        total_detections = 0
        class_counts = {class_id: 0 for class_id in self.class_names.keys()}
        
        for result in results:
            if 'error' not in result:
                successful_predictions += 1
                total_detections += result['detection_count']
                
                for label in result['labels']:
                    if label in class_counts:
                        class_counts[label] += 1
        
        statistics = {
            'total_images': total_images,
            'successful_predictions': successful_predictions,
            'failed_predictions': total_images - successful_predictions,
            'success_rate': successful_predictions / total_images if total_images > 0 else 0,
            'total_detections': total_detections,
            'average_detections_per_image': total_detections / successful_predictions if successful_predictions > 0 else 0,
            'class_distribution': {self.class_names[class_id]: count for class_id, count in class_counts.items()},
            'detection_rate': total_detections / total_images if total_images > 0 else 0
        }
        
        return statistics


def create_ssd_model(num_classes=3, weights_path=None, freeze_backbone=True):
    """
    创建SSD300模型的便捷函数
    
    Args:
        num_classes: 类别数量
        weights_path: 权重文件路径
        freeze_backbone: 是否冻结骨干网络
        
    Returns:
        SSD300模型实例
    """
    
    return SSD300(
        num_classes=num_classes,
        weights_path=weights_path,
        freeze_backbone=freeze_backbone
    )


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("SSD300 模型测试")
    print("=" * 60)
    
    # 测试模型加载
    try:
        model_path = "models/vehicle_detection/ssd.pt"
        if not os.path.exists(model_path):
            model_path = "models/vehicle_detection/ssd.pt"
        
        ssd_model = create_ssd_model(num_classes=3, weights_path=model_path, freeze_backbone=True)
        
        print(f"模型加载成功: {ssd_model.__class__.__name__}")
        print(f"类别数量: {ssd_model.num_classes - 1}")  # 减去背景类
        print(f"类别名称: {list(ssd_model.class_names.values())}")
        
    except Exception as e:
        print(f"模型加载失败: {e}")
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)