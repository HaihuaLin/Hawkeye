# Hawkeye A10 服务器快速测试指南

本指南专为在 **NVIDIA A10 (24GB 显存)** 服务器上一键测试任意视频而设计。

---

## 1. 上传文件到服务器

请确保服务器上的目录结构如下（保持相对路径一致）：
```text
your_workspace/
├── Hawkeye-main/            # 代码目录
│   ├── infer_video.py       # 一键测试脚本
│   ├── llava/
│   └── ...
├── Model Zoo/               # 下载的作者微调权重
│   ├── adapter_model.safetensors
│   ├── non_lora_trainables.bin
│   ├── config.json
│   └── adapter_config.json
└── Anomaly-Videos-Part-1/   # 测试视频目录
    ├── Abuse/
    ├── Arrest/
    ├── Arson/
    └── Assault/
```

---

## 2. 在 A10 服务器上配置环境

在服务器终端中执行以下命令（基于 Python 3.10 与 PyTorch 2.1）：

```bash
# 1. 创建并激活环境
conda create -n hawkeye python=3.10 -y
conda activate hawkeye

# 2. 安装 PyTorch (适配 CUDA 12.1 或 11.8)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 PyTorch Geometric (GTN 场景图网络依赖)
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

# 4. 安装大模型与多模态核心依赖
pip install transformers==4.31.0 peft==0.4.0 accelerate==0.21.0
pip install opencv-python decord einops open-clip-torch av pandas tqdm
```

> **说明**：A10 拥有 24GB 显存，无需开启 4-bit 量化，全精度 FP16 运行显存占用约 15GB，速度极快。

---

## 3. 一键测试视频

进入 `Hawkeye-main` 目录运行：

### (1) 时序扫描模式（推荐，自动按秒切片并定位异常区间）
```bash
python infer_video.py --video ../Anomaly-Videos-Part-1/Abuse/Abuse001_x264.mp4
```

**输出示例**：
```text
============================================================
 Hawkeye Anomaly Detection Inference
 Target Video : ../Anomaly-Videos-Part-1/Abuse/Abuse001_x264.mp4
 Model Path   : ../Model Zoo
 Base Model   : lmsys/vicuna-7b-v1.5
 Device       : cuda
============================================================

[1/3] Loading Hawkeye model and LanguageBind video tower...
Model loaded successfully!

Video Info: FPS=25.00, Total Frames=1250, Duration=50.00s (00:50)

[2/3] Running Inference (Mode: scan)...
Split video into 25 segments (segment length=2.0s, stride=2.0s):
Testing segments: 100%|██████████████████████████████████████| 25/25

[3/3] ===================== Detection Timeline =====================
[00:00 - 00:02]  [  Normal (0)  ]  (output: 0)
[00:02 - 00:04]  [  Normal (0)  ]  (output: 0)
...
[00:16 - 00:18]  [! ANOMALOUS (1) !]  (output: 1)
[00:18 - 00:20]  [! ANOMALOUS (1) !]  (output: 1)
[00:20 - 00:22]  [! ANOMALOUS (1) !]  (output: 1)
[00:22 - 00:24]  [  Normal (0)  ]  (output: 0)

============================================================
 SUMMARY REPORT:
 - Total segments evaluated : 25
 - Anomalous segments count : 3 (12.0%)
 - Detected Anomaly Time Windows:
   * 00:16 --> 00:22
============================================================
```

### (2) 整段视频快速判断模式（耗时仅数秒）
如果你只想快速看整个视频是否存在异常情绪倾向，无需切片：
```bash
python infer_video.py --video ../Anomaly-Videos-Part-1/Abuse/Abuse001_x264.mp4 --mode overall
```

---

## 4. 常用参数说明

| 参数 | 默认值 | 作用说明 |
| :--- | :--- | :--- |
| `--video` | 必需 | 待测试的 `.mp4` 视频路径 |
| `--model_path` | `../Model Zoo` | 微调权重目录路径 |
| `--model_base` | `lmsys/vicuna-7b-v1.5` | 基础模型名称或本地路径 |
| `--mode` | `scan` | `scan` 为时序切片扫描定位；`overall` 为整段全局检测 |
| `--segment_sec` | `2.0` | 切片窗口长度（秒） |
| `--stride_sec` | `2.0` | 切片步长（秒） |
| `--max_segments` | `100` | 最多测试的切片数量（防止过长视频耗费太多时间） |
