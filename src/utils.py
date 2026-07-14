"""
utils.py — 图像加载、保存、格式校验等工具函数
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

from .config import SUPPORTED_INPUT_FORMATS, SUPPORTED_OUTPUT_FORMATS, OUTPUT_SUFFIX


def is_supported_input(filepath: str | Path) -> bool:
    """检查是否为支持的输入图片格式"""
    ext = Path(filepath).suffix.lower()
    return ext in SUPPORTED_INPUT_FORMATS


def is_supported_output(filepath: str | Path) -> bool:
    """检查输出格式是否支持"""
    ext = Path(filepath).suffix.lower()
    return ext in SUPPORTED_OUTPUT_FORMATS


def load_image(filepath: str | Path) -> np.ndarray:
    """
    加载图片为 BGR numpy 数组（OpenCV 格式）。
    支持含中文路径。
    """
    filepath = str(filepath)
    # 使用 PIL 读取以支持更多格式和中文路径，再转为 OpenCV BGR
    pil_img = ImageOps.exif_transpose(Image.open(filepath)).convert("RGB")
    rgb = np.array(pil_img)
    return rgb[:, :, ::-1].copy()


def save_image(image: np.ndarray, filepath: str | Path) -> bool:
    """
    保存 BGR numpy 数组到文件。
    返回是否保存成功。
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 转为 RGB 再用 PIL 保存（支持中文路径 + 更多格式）
    rgb = bgr_to_rgb(image)
    pil_img = Image.fromarray(rgb)
    pil_img.save(str(filepath))
    return True


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """将 BGR/BGRA/灰度 numpy 图像转换为 PIL 友好的 RGB/RGBA/灰度数组。"""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return image[:, :, [2, 1, 0, 3]].copy()
    return image[:, :, ::-1].copy()


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """将 RGB/RGBA/灰度 numpy 图像转换为 BGR/BGRA 数组。"""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return image[:, :, [2, 1, 0, 3]].copy()
    return image[:, :, ::-1].copy()


def ensure_bgr_3ch(image: np.ndarray) -> np.ndarray:
    """确保图像是 3 通道 BGR，便于 Real-ESRGAN 推理。"""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def resize_bgr(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """使用高质量 Lanczos 算法缩放 BGR 图像，size 为 (width, height)。"""
    pil_img = Image.fromarray(bgr_to_rgb(image))
    resized = pil_img.resize(size, Image.Resampling.LANCZOS)
    return rgb_to_bgr(np.array(resized))


def get_image_info(filepath: str | Path) -> dict:
    """获取图片基本信息"""
    filepath = Path(filepath)
    pil_img = ImageOps.exif_transpose(Image.open(filepath))
    size_bytes = filepath.stat().st_size
    return {
        "path": str(filepath),
        "filename": filepath.name,
        "width": pil_img.width,
        "height": pil_img.height,
        "mode": pil_img.mode,
        "size_kb": round(size_bytes / 1024, 1),
        "size_mb": round(size_bytes / (1024 * 1024), 2),
    }


def generate_output_path(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    suffix: str | None = None,
    output_format: str = ".png",
) -> Path:
    """
    根据输入路径生成输出路径。
    - 若指定 output_dir，输出到该目录，文件名 = 原名 + suffix + 格式
    - 否则输出到源文件同目录
    """
    input_path = Path(input_path)
    suffix = suffix if suffix is not None else OUTPUT_SUFFIX
    stem = input_path.stem
    if not output_format.startswith("."):
        output_format = f".{output_format}"

    out_name = f"{stem}{suffix}{output_format}"

    if output_dir:
        return Path(output_dir) / out_name
    else:
        return input_path.with_name(out_name)


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
