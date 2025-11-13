# 积水识别和车辆淹没部位判别系统

## 系统简介

本系统是一个基于深度学习的积水识别和车辆淹没部位判别系统，支持图像上传、目标检测、图像分割、结果统计和模型管理等功能。

## 主要功能

### 1. 图像处理
- 支持多种图像格式上传（JPG、PNG、BMP、TIFF）
- 支持拖拽上传图像
- 图像预览和信息显示

### 2. 目标检测
- 支持SSD300和YOLOv11检测模型
- 检测车辆淹没部位（未淹没、部分淹没、完全淹没）
- 显示检测框和置信度

### 3. 图像分割
- 支持DeepLabV3和YOLOv11分割模型
- 水面区域分割
- 分割结果可视化

### 4. 结果展示
- 左右分屏显示原图和结果
- 支持检测结果和分割结果切换
- 结果图像保存功能

### 5. 统计分析
- 不同淹没部位车辆数量统计
- 详细统计信息显示
- 统计结果导出功能

### 6. 模型管理
- 支持多种模型选择
- 模型状态监控
- 模型测试功能

## 系统要求

- Python 3.7+
- Windows 10/11
- 内存: 4GB+
- 显卡: 支持CUDA的NVIDIA显卡（可选，用于GPU加速）

## 安装说明

### 1. 克隆项目
```bash
git clone <repository-url>
cd FloodRisk
```

### 2. 创建虚拟环境
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 准备模型文件
确保以下模型文件存在：
```
models/
├── vehicle_detection/
│   ├── ssd.pt
│   └── yolov11.pt
└── water_segmentation/
    ├── deeplabv3.pth
    └── yolov11.pt
```

## 使用方法

### 启动系统
```bash
python main.py
```

### 基本操作流程

1. **上传图像**
   - 点击"选择图像"按钮或拖拽图像到上传区域
   - 系统会显示图像预览和基本信息

2. **选择模型**
   - 在左侧面板选择检测模型和分割模型
   - 系统会自动加载选中的模型

3. **开始处理**
   - 点击"开始处理"按钮
   - 系统会显示处理进度

4. **查看结果**
   - 右侧面板显示处理结果
   - 可以切换查看检测结果和分割结果
   - 统计信息会显示在下方

5. **保存结果**
   - 点击"保存结果"按钮保存当前显示的结果
   - 或使用菜单栏的"保存结果"功能

## 模型说明

### 检测模型
- **SSD300**: 基于VGG16骨干网络的单阶段检测器
- **YOLOv11**: 最新的YOLO检测模型，速度快精度高

### 分割模型
- **DeepLabV3**: 基于ResNet18的语义分割模型
- **YOLOv11**: 支持实例分割的YOLO模型

## 配置说明

系统支持通过设置对话框进行配置：

- **模型设置**: 默认模型选择、置信度阈值等
- **显示设置**: 图像质量、显示选项等
- **系统设置**: 日志级别、保存路径等

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件是否存在
   - 确认模型文件格式正确
   - 查看日志文件获取详细错误信息

2. **内存不足**
   - 减少图像尺寸
   - 关闭其他应用程序
   - 调整系统设置中的内存限制

3. **处理速度慢**
   - 使用GPU加速（需要CUDA支持）
   - 减少图像分辨率
   - 选择更快的模型

### 日志文件
系统运行日志保存在 `flood_risk_system.log` 文件中，可以查看详细的错误信息。

## 开发说明

### 项目结构
```
FloodRisk/
├── main.py                 # 主程序入口
├── gui/                    # GUI组件
│   ├── main_window.py      # 主窗口
│   ├── image_upload_widget.py
│   ├── model_selection_widget.py
│   ├── image_display_widget.py
│   ├── statistics_widget.py
│   ├── processing_worker.py
│   ├── model_management_dialog.py
│   └── settings_dialog.py
├── models/                 # 模型文件
│   ├── vehicle_detection/
│   └── water_segmentation/
├── requirements.txt        # 依赖包
└── README.md              # 说明文档
```

### 扩展开发
- 添加新的检测模型：在 `models/vehicle_detection/` 目录添加模型文件
- 添加新的分割模型：在 `models/water_segmentation/` 目录添加模型文件
- 修改界面：编辑对应的GUI组件文件

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请联系开发团队。
