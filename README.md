# 视频对齐系统

这是一个用于自动对齐两个视频的系统，特别适合处理红外和热成像视频的对齐。本系统使用先进的特征提取和匹配算法，能够处理不同帧率、不同分辨率的视频，实现精确的时间和空间对齐。

## 功能特点

- 支持不同帧率视频的自动同步
- 智能空间对齐和几何变换
- 实时预览对齐效果
- 用户友好的图形界面
- 支持暂停/继续处理
- 自动优化处理性能

## 系统要求

- Python 3.8 或更高版本
- 操作系统：Windows/Linux/MacOS
- NVIDIA GPU（可选，用于加速处理）
- CUDA 11.0+（使用GPU加速时需要）

## 安装说明

1. 克隆或下载本项目到本地

2. 创建并激活虚拟环境（可选但推荐）:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/MacOS
source venv/bin/activate
```

3. 安装依赖:
```bash
pip install -r requirements.txt
```

4. GPU加速配置（可选）:
```bash
# 安装GPU版本的pytorch和相关库
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 设置环境变量启用GPU
# Windows
set CUDA_VISIBLE_DEVICES=0
# Linux/MacOS
export CUDA_VISIBLE_DEVICES=0
```

## 使用方法

1. 运行程序:
```bash
python main.py
```

2. 在程序界面中：
   - 点击"选择视频1"按钮选择第一个视频文件
   - 点击"选择视频2"按钮选择第二个视频文件
   - 点击"开始对齐"按钮开始处理
   - 使用"暂停"按钮可以暂停/继续处理
   - 使用"停止对齐"按钮停止处理

## 技术实现

### 1. 特征提取
- 使用ORB算法进行特征点提取
- 多尺度特征选择
- 边缘特征和全局特征的结合

### 2. 时间对齐
- 动态规划算法进行序列对齐
- 滑动窗口优化
- 自适应帧率处理

### 3. 空间对齐
- RANSAC算法进行几何变换估计
- 卡尔曼滤波平滑处理
- 仿射变换和透视变换支持

## 注意事项

- 处理大文件时可能需要较长时间
- 建议使用清晰度较好的视频以获得更好的对齐效果
- 对齐过程中避免关闭程序
- 如果对齐效果不理想，可以尝试调整视频画面的对比度和亮度
- 默认情况下，系统会尝试使用GPU加速。如果没有检测到兼容的GPU，将自动切换到CPU模式

## 错误排除

1. 如果出现"无法加载视频文件"错误：
   - 检查视频文件格式是否支持
   - 确认视频文件是否完整
   - 检查视频编解码器是否已安装

2. 如果对齐效果不理想：
   - 确保两个视频内容相关
   - 检查视频质量是否足够好
   - 可以尝试预处理视频（如提高对比度）

3. 如果程序运行缓慢：
   - 检查电脑配置是否满足要求
   - 可以尝试处理较低分辨率的视频
   - 关闭其他占用资源的程序

4. 如果系统没有使用GPU：
   - 确认是否安装了兼容的NVIDIA驱动和CUDA
   - 检查是否正确安装了GPU版本的PyTorch和相关库
   - 运行以下代码检查GPU是否可用：
   ```python
   import torch
   print("CUDA是否可用:", torch.cuda.is_available())
   print("可用的GPU数量:", torch.cuda.device_count())
   print("当前GPU:", torch.cuda.current_device())
   print("GPU名称:", torch.cuda.get_device_name(0))
   ```
   - 确保没有其他程序占用全部GPU内存
   - 尝试在命令行中使用`--force-gpu`参数启动程序：`python main.py --force-gpu`

## 许可证

MIT License
