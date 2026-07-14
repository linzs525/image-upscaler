"""
config.py — 全局配置常量
"""

import sys
from pathlib import Path

# ---------- 项目路径 ----------
if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ---------- 支持的图片格式 ----------
SUPPORTED_INPUT_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")
SUPPORTED_OUTPUT_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")

# ---------- 超分参数 ----------
DEFAULT_SCALE = 4                       # 默认放大倍数
AVAILABLE_SCALES = [2, 3, 4, 8]         # 可选放大倍数

# ---------- Real-ESRGAN 模型 ----------
#   key  → 显示名称, value → RealESRGANer 模型名
MODELS = {
    "realesr-animevideov3":  "动画视频模型 (AnimeVideo v3)",
    "realesrgan-x4plus":     "通用模型 x4plus",
    "realesrgan-x4plus-anime": "动画模型 x4plus",
    "realesrnet-x4plus":     "轻量通用模型 x4plus",
}

DEFAULT_MODEL = "realesr-animevideov3"

# ---------- 设备 ----------
# "auto" → 自动选择 CUDA > MPS > CPU
DEFAULT_DEVICE = "auto"

# ---------- 输出后缀 ----------
OUTPUT_SUFFIX = "_upscaled"

# ---------- GUI ----------
WINDOW_TITLE = "图片高清化工具 — Image Upscaler"
WINDOW_MIN_WIDTH = 960
WINDOW_MIN_HEIGHT = 680
PREVIEW_MAX_SIZE = (512, 512)  # 预览区域最大显示尺寸
