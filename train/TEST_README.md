# 模型测试说明

这个文档介绍如何使用训练好的SlowFast模型来测试视频文件和RTSP流。

## 文件说明

- `test_model.py` - 主要的测试脚本，支持单个视频文件和RTSP流测试
- `batch_test.py` - 批量测试脚本，用于测试多个视频文件
- `TEST_README.md` - 本说明文档

## 环境要求

确保已安装所有必要的依赖：
```bash
pip install torch torchvision pytorchvideo opencv-python pyyaml numpy
```

## 使用方法

### 1. 测试单个视频文件

```bash
# 基本用法
python test_model.py --video /path/to/your/video.mp4

# 指定配置文件和模型checkpoint
python test_model.py --video /path/to/your/video.mp4 --config configs/config.yaml --checkpoint checkpoints/best_model.pth

# 保存带检测结果的输出视频
python test_model.py --video /path/to/your/video.mp4 --output /path/to/output_video.mp4
```

### 2. 测试RTSP流

```bash
# 测试RTSP流（默认60秒）
python test_model.py --rtsp rtsp://your.camera.ip:554/stream

# 指定测试时长（秒）
python test_model.py --rtsp rtsp://your.camera.ip:554/stream --duration 120

# 示例RTSP地址格式
python test_model.py --rtsp rtsp://admin:password@192.168.1.100:554/h264/ch1/main/av_stream
```

### 3. 批量测试多个视频

```bash
# 测试目录下所有视频文件
python batch_test.py /path/to/video/directory

# 指定输出目录保存检测结果视频
python batch_test.py /path/to/video/directory --output-dir /path/to/output/directory

# 指定特定的视频格式
python batch_test.py /path/to/video/directory --extensions "*.mp4" "*.avi"
```

## 参数说明

### test_model.py 参数

- `--config`: 配置文件路径 (默认: `configs/config.yaml`)
- `--checkpoint`: 模型checkpoint路径 (默认: `checkpoints/best_model.pth`)
- `--video`: 要测试的视频文件路径
- `--rtsp`: RTSP流URL
- `--output`: 输出视频路径（仅用于视频文件测试）
- `--duration`: RTSP流测试时长，单位秒 (默认: 60)

### batch_test.py 参数

- `video_dir`: 包含视频文件的目录路径（必需）
- `--config`: 配置文件路径 (默认: `configs/config.yaml`)
- `--checkpoint`: 模型checkpoint路径 (默认: `checkpoints/best_model.pth`)
- `--output-dir`: 输出目录路径（可选）
- `--extensions`: 要处理的视频文件扩展名 (默认: `*.mp4 *.avi *.mov *.mkv *.flv *.wmv`)

## 输出说明

### 控制台输出

测试过程中会在控制台显示：
- 模型加载信息
- 视频/流的基本信息（分辨率、帧率等）
- 实时检测结果（当检测到盗窃行为时）
- 处理进度
- 最终检测摘要

### 检测结果格式

当检测到盗窃行为时，会输出如下信息：
```
THEFT DETECTED at 15.23s (frame 456), confidence: 0.847
```

### 最终摘要

测试完成后会显示检测摘要：
```
=== Detection Summary ===
Total frames processed: 1500
Theft detections: 3

Detection details:
  - Time: 15.23s, Frame: 456, Confidence: 0.847
  - Time: 28.67s, Frame: 860, Confidence: 0.756
  - Time: 45.12s, Frame: 1354, Confidence: 0.923
```

## 输出视频

如果指定了输出路径，程序会生成包含检测结果的视频文件，采用**外部可视化面板**设计：

### 可视化界面说明

**布局特点**：
- **左侧**：完整的原始视频画面，不被任何信息遮挡
- **右侧**：专用信息面板（400像素宽），显示所有检测信息
- **扩展分辨率**：输出视频宽度 = 原视频宽度 + 400像素

**信息面板内容**：
- **标题区域**：显示"THEFT DETECTION"标题
- **主要结果**：当前预测类别和平滑后的置信度
  - 绿色：正常行为 (Normal)  
  - 红色：盗窃行为 (Theft)
- **置信度可视化**：
  - 数值显示：精确到小数点后3位
  - 进度条：直观的条形图显示
  - 百分比：置信度百分比形式
- **原始预测**：未经平滑处理的模型原始输出
- **类别概率**：两个类别的详细概率分布
  - Normal概率（绿色显示）
  - Theft概率（红色显示）
- **视频信息**：
  - 当前时间戳
  - 当前帧数/总帧数
  - 处理进度条（黄色）
- **盗窃警报**：检测到盗窃时显示红色警告横幅

**RTSP流特殊优化**：
- **稳定显示**：所有帧都显示最新预测结果，消除闪烁
- **实时更新**：信息面板每0.5秒更新一次预测结果
- **流畅体验**：中间帧显示上次预测结果，保持连续性

### 可视化演示
可以运行以下命令查看可视化效果演示：
```bash
# 基本可视化演示
python demo_visualization.py

# 测试外部可视化面板
python test_external_viz.py

# 测试稳定的RTSP流可视化
python test_rtsp_stable.py

# 测试快速响应功能（推荐）
python test_fast_response.py

# 使用真实RTSP流测试快速响应
python test_fast_response.py --rtsp rtsp://your.camera.ip:554/stream --duration 60

# 测试指定视频的快速响应
python test_fast_response.py --video ../videos/your_video.mp4
```

## 性能优化

### 🚀 快速响应优化（v2.1.0新增）

**响应时间大幅改进**：
- **检测响应时间**: 从10秒优化到1-2秒
- **预测频率**: 从每0.5秒提升到每0.33秒
- **最小检测时间**: 从2.5秒减少到0.66秒

**智能判定机制**：
- **高置信度快速检测**: 2次高置信度(>0.7)预测即可确认
- **连续预测确认**: 最近一次高置信度(>0.8) + 历史盗窃预测
- **持续行为检测**: 连续中等置信度(>0.6)预测

**使用方法**：
```bash
# 测试快速响应功能
python test_fast_response.py

# 对比测试（查看改进效果）
python test_fast_response.py --video ../videos/your_video.mp4
```

### GPU使用

- 如果有NVIDIA GPU，确保安装了CUDA版本的PyTorch
- 程序会自动检测并使用GPU加速推理

### 内存优化

- 视频处理使用帧缓冲区，避免加载整个视频到内存
- 对于长视频，程序会逐帧处理而不是一次性加载

### 处理速度

- **优化后**: 每0.33秒进行一次预测（更快响应）
- **原版本**: 每0.5秒进行一次预测
- RTSP流处理包含实时显示窗口（按'q'键退出）
- 批量处理会显示详细进度信息

## 故障排除

### 常见错误

1. **模型加载错误**
   ```
   Error: Cannot load checkpoint
   ```
   - 检查checkpoint文件路径是否正确
   - 确保checkpoint文件完整且未损坏

2. **视频文件错误**
   ```
   Error: Cannot open video file
   ```
   - 检查视频文件路径是否正确
   - 确保视频格式被OpenCV支持
   - 检查文件权限

3. **RTSP连接错误**
   ```
   Error: Cannot connect to RTSP stream
   ```
   - 检查RTSP URL格式是否正确
   - 确认网络连接正常
   - 验证摄像头用户名和密码

4. **内存不足**
   ```
   CUDA out of memory
   ```
   - 减小batch_size（修改配置文件）
   - 使用CPU推理（修改配置文件中的device为"cpu"）

### 调试模式

如果遇到问题，可以在代码中添加调试信息：
- 修改`test_model.py`中的日志级别
- 检查视频预处理步骤
- 验证模型输出维度

## 自定义配置

可以通过修改`configs/config.yaml`来调整：
- 输入视频的分辨率和帧数
- 检测阈值和平滑参数
- 设备设置（CPU/GPU）

## 示例用法

```bash
# 测试单个视频文件并保存结果
python test_model.py --video ../videos/test_video.mp4 --output results/test_video_detected.mp4

# 测试RTSP摄像头流2分钟
python test_model.py --rtsp rtsp://admin:123456@192.168.1.100:554/h264/ch1/main/av_stream --duration 120

# 批量测试所有MP4文件并保存结果
python batch_test.py ../videos --output-dir results --extensions "*.mp4"
``` 