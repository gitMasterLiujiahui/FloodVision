# FloodRisk - 积水识别和车辆淹没部位判别系统

## 项目概述

FloodRisk 是一个基于深度学习的智能积水识别和车辆淹没部位判别系统。该系统能够自动检测图像中的积水区域，并识别车辆的不同淹没部位（车轮、车门、车窗），为城市内涝监测和车辆安全评估提供技术支持。

## 主要功能

### 🎯 核心功能
- **智能积水识别**：基于深度学习模型自动识别图像中的积水区域
- **车辆淹没部位检测**：检测车辆在积水中的不同淹没部位（车轮、车门、车窗）
- **多模型支持**：支持SSD300、YOLOv11等多种检测模型和DeepLabV3、YOLOv11等分割模型
- **实时处理**：支持图像上传和实时处理分析

### 📊 分析功能
- **统计报告**：自动生成不同淹没部位车辆数量统计
- **可视化展示**：左右分屏对比显示原图与处理结果
- **结果导出**：支持检测结果和分割结果的图像导出

### 🔧 系统管理
- **模型管理**：动态加载和切换不同模型
- **状态监控**：实时监控系统运行状态和模型加载情况
- **日志记录**：详细的运行日志记录和错误追踪

## 技术栈

### 后端技术
- **框架**：FastAPI + Uvicorn
- **深度学习**：PyTorch + OpenCV + Pillow
- **图像处理**：NumPy + OpenCV-Python
- **Web服务**：FastAPI + Jinja2模板

### 前端技术
- **界面框架**：Bootstrap 5.3
- **图标库**：Font Awesome 6.0
- **交互技术**：原生JavaScript + Fetch API
- **响应式设计**：支持桌面和移动端访问

### 模型架构
- **检测模型**：SSD300、YOLOv11（车辆淹没部位检测）
- **分割模型**：DeepLabV3、YOLOv11（积水区域分割）
- **模型格式**：PyTorch (.pt/.pth)

## 系统要求

### 硬件要求
- **操作系统**：Windows 10/11、Linux、macOS
- **内存**：4GB RAM（推荐8GB）
- **存储空间**：至少2GB可用空间
- **显卡**：支持CUDA的NVIDIA显卡（可选，用于GPU加速）

### 软件要求
- **Python版本**：3.7+
- **包管理器**：pip
- **虚拟环境**：venv（推荐）

## 安装与配置

### 1. 获取项目代码
```bash
git clone <repository-url>
cd FloodRisk
```

### 2. 创建虚拟环境（推荐）
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖包
```bash
pip install -r requirements.txt
```

### 4. 准备模型文件
确保以下模型文件存在于相应目录：
```
models/
├── vehicle_detection/
│   ├── ssd.pt          # SSD300检测模型
│   └── yolov11.pt      # YOLOv11检测模型
└── water_segmentation/
    ├── deeplabv3.pth   # DeepLabV3分割模型
    └── yolov11.pt      # YOLOv11分割模型
```

> **注意**：如果模型文件不存在，系统将使用虚拟模型进行演示，但功能会受限。

### 5. 启动系统
```bash
# 方式一：使用启动脚本（推荐）
python run_web.py

# 方式二：直接运行Web应用
python web_app.py
```

系统启动后，默认在 http://localhost:5000 提供服务。

## 使用方法

### 基本操作流程

1. **访问系统**
   - 打开浏览器访问 http://localhost:5000
   - 系统将显示加载界面，完成后进入主界面

2. **配置模型**
   - 在"模型配置"区域选择需要的检测模型和分割模型
   - 点击"加载模型"按钮加载选中的模型
   - 查看模型状态确认加载成功

3. **上传图像**
   - 点击上传区域或拖拽图像文件到指定区域
   - 支持JPG、PNG、BMP等常见图像格式

4. **开始处理**
   - 点击"开始处理"按钮启动分析
   - 系统将显示处理进度和实时结果

5. **查看结果**
   - **左侧面板**：显示检测结果（车辆淹没部位识别）
   - **右侧面板**：显示分割结果（积水区域识别）
   - **统计信息**：显示不同淹没部位车辆数量统计

6. **下载结果**
   - 点击"下载检测图"保存检测结果
   - 点击"下载分割图"保存分割结果

### 高级功能

#### 批量处理
系统支持批量处理多张图像，可通过API接口进行调用：
```python
import requests

# 批量处理示例
files = {'image': open('test_image.jpg', 'rb')}
response = requests.post('http://localhost:5000/api/process', files=files)
result = response.json()
```

#### API接口
系统提供完整的RESTful API接口：
- `GET /api/status` - 获取系统状态
- `GET /api/models` - 获取可用模型列表
- `POST /api/models/load` - 加载指定模型
- `POST /api/process` - 处理单张图像
- `POST /api/download/detection` - 下载检测结果
- `POST /api/download/segmentation` - 下载分割结果

## 目录结构说明

```
FloodRisk/
├── .venv/                          # Python虚拟环境
├── __pycache__/                    # Python字节码缓存
├── models/                         # 模型文件目录
│   ├── vehicle_detection/          # 车辆检测模型
│   │   ├── ssd.pt                  # SSD300模型权重
│   │   ├── yolov11.pt             # YOLOv11模型权重
│   │   ├── ssd_model.py            # SSD模型实现
│   │   └── yolov11_model.py        # YOLOv11模型实现
│   └── water_segmentation/         # 积水分割模型
│       ├── deeplabv3.pth           # DeepLabV3模型权重
│       ├── yolov11.pt              # YOLOv11模型权重
│       ├── deeplabv3_model.py      # DeepLabV3模型实现
│       └── yolov11_model.py        # YOLOv11模型实现
├── templates/                      # HTML模板文件
│   └── index.html                  # 主界面模板
├── temp/                           # 临时文件目录
├── test_img/                       # 测试图像目录
├── web_app.py                      # FastAPI Web应用主文件
├── shared_models.py                # 共享模型管理模块
├── run_web.py                      # Web应用启动脚本
├── requirements.txt                # Python依赖包列表
├── flood_risk_web.log              # 系统运行日志
├── PROJECT_SUMMARY.md              # 项目概要文档
└── README.md                       # 项目说明文档
```

## 模型说明

### 检测模型（车辆淹没部位识别）

#### SSD300
- **架构**：基于VGG16骨干网络的单阶段检测器
- **特点**：检测速度快，适合实时应用
- **输出**：车辆淹没部位（车轮、车门、车窗）的边界框和置信度

#### YOLOv11
- **架构**：最新的YOLO检测架构
- **特点**：高精度，支持多种分辨率
- **优势**：在复杂场景下表现优异

### 分割模型（积水区域识别）

#### DeepLabV3
- **架构**：基于ResNet18的语义分割模型
- **特点**：精确的边界分割能力
- **应用**：积水区域的像素级识别

#### YOLOv11分割模型
- **架构**：支持实例分割的YOLO变体
- **特点**：结合检测和分割的优势
- **优势**：一体化处理检测和分割任务

## 开发指南

### 项目架构

系统采用模块化设计，主要包含以下核心模块：

1. **Web应用模块** (`web_app.py`)
   - FastAPI应用配置和路由定义
   - 图像处理流程控制
   - API接口实现

2. **模型管理模块** (`shared_models.py`)
   - 统一模型加载和管理
   - 模型可用性检查
   - 虚拟模型支持

3. **检测模型模块** (`models/vehicle_detection/`)
   - SSD300模型实现
   - YOLOv11模型实现
   - 检测结果后处理

4. **分割模型模块** (`models/water_segmentation/`)
   - DeepLabV3模型实现
   - YOLOv11分割模型实现
   - 分割结果可视化

### 扩展开发

#### 添加新模型
1. 在相应模型目录创建新的模型实现文件
2. 在`shared_models.py`中注册新模型
3. 在Web界面中添加模型选择选项

#### 修改处理流程
1. 编辑`web_app.py`中的处理函数
2. 调整模型调用参数和结果处理逻辑
3. 更新前端界面显示逻辑

#### 自定义界面
1. 修改`templates/index.html`模板文件
2. 调整CSS样式和JavaScript交互逻辑
3. 添加新的功能组件

## 故障排除

### 常见问题

#### 1. 模型加载失败
**症状**：控制台显示模型文件不存在警告
**解决方案**：
```bash
# 检查模型文件是否存在
ls models/vehicle_detection/
ls models/water_segmentation/

# 如果文件缺失，需要下载相应模型文件
```

#### 2. 端口被占用
**症状**：启动时提示端口被占用
**解决方案**：
```bash
# 使用不同端口启动
python run_web.py
# 系统会自动尝试5000-5004端口

# 或手动指定端口
uvicorn web_app:app --host 0.0.0.0 --port 5005
```

#### 3. 内存不足
**症状**：处理大图像时出现内存错误
**解决方案**：
- 减小输入图像尺寸
- 关闭其他占用内存的应用程序
- 使用GPU加速（如果可用）

#### 4. 依赖包冲突
**症状**：导入错误或运行时异常
**解决方案**：
```bash
# 重新创建虚拟环境
python -m venv .venv --clear
source .venv/bin/activate  # 或 .venv\Scripts\activate
pip install -r requirements.txt
```

### 日志分析
系统运行日志保存在`flood_risk_web.log`文件中，包含详细的错误信息和调试信息。遇到问题时可以查看日志文件获取更多信息。

## 贡献指南

### 代码贡献
欢迎提交Pull Request来改进项目。请遵循以下规范：

1. **代码风格**：遵循PEP 8 Python编码规范
2. **文档更新**：修改代码时同步更新相关文档
3. **测试验证**：确保修改不会破坏现有功能

### 问题报告
发现bug或有改进建议时，请通过GitHub Issues提交，包含：
- 问题描述
- 复现步骤
- 期望行为
- 环境信息

### 功能请求
如果有新的功能需求，欢迎提出建议，包括：
- 功能描述
- 使用场景
- 预期效果

## 许可证信息

本项目采用MIT许可证，允许自由使用、修改和分发。详见LICENSE文件。

## 联系方式

- **项目主页**：GitHub Repository
- **问题反馈**：GitHub Issues
- **技术支持**：项目维护团队

## 更新日志

### v1.0.0 (当前版本)
- 初始版本发布
- 支持SSD300和YOLOv11检测模型
- 支持DeepLabV3和YOLOv11分割模型
- 提供Web界面和API接口
- 完整的文档和安装指南

---

**注意**：本系统为研究用途，在实际应用前请进行充分的测试和验证。