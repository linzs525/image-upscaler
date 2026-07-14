# 图片高清化工具 — Image Upscaler

基于 **Real-ESRGAN** 深度学习模型的图片超分辨率桌面应用。
支持将低分辨率图片高清放大 2~8 倍，同时保持画质细节。

---

## 功能特性

- 🚀 **深度学习超分** — 基于 Real-ESRGAN，效果远超传统插值
- 🖥️ **图形界面** — PySide6 桌面应用，支持拖拽、预览、批量处理
- ⌨️ **命令行模式** — 也支持 CLI 批量脚本调用
- 🎯 **多种模型** — 通用 / 动画 / 轻量 四种模型可选
- 🔧 **灵活配置** — 放大倍数、输出格式、推理设备自由切换
- 📦 **自动下载** — 模型权重首次使用自动下载，后续本地缓存

## 项目结构

```
image-upscaler/
├── main.py                 # 入口（GUI + CLI）
├── requirements.txt        # Python 依赖
├── README.md               # 本文件
├── src/
│   ├── __init__.py
│   ├── config.py           # 全局配置常量
│   ├── utils.py            # 图片 I/O 工具函数
│   ├── upscaler.py         # 核心超分引擎（封装 Real-ESRGAN）
│   └── gui.py              # PySide6 图形界面
├── models/                 # 模型权重存放目录（可选）
└── output/                 # 默认输出目录
```

## 快速开始

### 1. 环境要求

- **Python 3.10+**
- **CUDA 兼容 GPU**（推荐，也可 CPU 运行）
- 约 **2 GB** 磁盘空间（模型权重）

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS / Linux

# 安装 PyTorch（根据 CUDA 版本选择，详见 pytorch.org）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其余依赖
pip install -r requirements.txt
```

### 3. 启动应用

```bash
# GUI 模式（默认）
python main.py

# CLI 模式 — 处理单张
python main.py --cli -i photo.jpg -o output/

# CLI 模式 — 批量处理
python main.py --cli -i input_folder/ -o output/ -m realesrgan-x4plus -s 4
```

## 使用说明

### GUI 界面

1. 启动后拖拽图片到左侧区域（或点击 [选择图片]）
2. 在底部设置栏选择模型、倍数、格式
3. 点击 [开始处理]，等待进度条完成
4. 右侧预览区可查看原图/结果对比
5. 输出文件默认保存在 `output/` 目录

### CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cli` | 启用命令行模式 | — |
| `-i / --input` | 输入文件或目录 | (必填) |
| `-o / --output` | 输出路径 | `output/` |
| `-m / --model` | 模型名称 | `realesr-animevideov3` |
| `-s / --scale` | 放大倍数 (2/3/4/8) | `4` |
| `-f / --format` | 输出格式 | `.png` |
| `-d / --device` | 推理设备 (auto/cuda/cpu) | `auto` |
| `-t / --tile` | 分块大小 (0=不分块) | `0` |

### 可用模型

| 模型 key | 说明 | 适用场景 |
|----------|------|----------|
| `realesr-animevideov3` | 动画视频模型 v3 | **动漫/二次元** 图片 |
| `realesrgan-x4plus` | 通用超分 x4+ | **真实照片**（推荐） |
| `realesrgan-x4plus-anime` | 动画超分 x4+ | 动漫图片 |
| `realesrnet-x4plus` | 轻量超分 x4+ | 速度优先场景 |

## 常见问题

**Q: 提示显存不足？**
设置分块大小：`--tile 400`（CLI）或在代码中设置 `tile_size=400`。

**Q: 处理速度慢？**
- 确保安装了 CUDA 版 PyTorch 并有 NVIDIA 显卡
- 可尝试 `realesrnet-x4plus` 轻量模型
- CPU 模式下处理较慢，属正常现象

**Q: 首次运行卡住？**
首次使用会自动从 GitHub 下载模型权重（约 60~300 MB），请耐心等待。
如果下载失败，可手动下载 `.pth` 文件放入 `models/` 目录。

## 技术栈

- **超分引擎**: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **深度学习**: PyTorch
- **图像处理**: OpenCV, Pillow
- **GUI**: PySide6 (Qt for Python)

## License

本项目仅供学习研究使用。Real-ESRGAN 模型版权归原作者所有。
