"""
upscaler.py — 核心超分引擎，封装 Real-ESRGAN
"""

import threading
from pathlib import Path

import numpy as np

from .config import (
    MODELS,
    DEFAULT_MODEL,
    DEFAULT_SCALE,
    DEFAULT_DEVICE,
    MODELS_DIR,
)
from .utils import (
    ensure_bgr_3ch,
    generate_output_path,
    is_supported_input,
    load_image,
    resize_bgr,
    save_image,
)


class Upscaler:
    """
    图片超分辨率引擎。

    使用方法:
        upscaler = Upscaler(model_name="realesr-animevideov3", scale=4)
        upscaler.upscale("input.jpg", "output.png")

    也支持信号式进度回调，适合 GUI 集成。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        scale: int = DEFAULT_SCALE,
        device: str = DEFAULT_DEVICE,
        tile_size: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        fp32: bool = False,
        backend: str = "auto",
    ):
        """
        :param model_name: Real-ESRGAN 模型名
        :param scale:      放大倍数（需与模型兼容）
        :param device:     设备："auto" / "cuda" / "cpu"
        :param tile_size:  分块大小（0 = 不分块，显存不足时设 400~800）
        :param tile_pad:   分块边距
        :param pre_pad:    预处理填充
        :param fp32:       是否使用 FP32（默认 FP16，CPU 模式自动切换）
        :param backend:    "auto" / "realesrgan" / "pillow"
        """
        if backend not in {"auto", "realesrgan", "pillow"}:
            raise ValueError("backend 必须是 auto、realesrgan 或 pillow")

        self.model_name = model_name
        self.scale = scale
        self.device = self._resolve_device(device)
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.fp32 = fp32 or (self.device == "cpu")
        self.backend = backend
        self.backend_name = "未初始化"
        self.fallback_reason: str | None = None

        self._model = None
        self._lock = threading.Lock()

        # 进度回调: (current, total) → None
        self.on_progress = None

        # 加载模型；依赖不可用时允许退回到 Pillow 高质量插值，保证程序可运行。
        self._load_backend()

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def upscale(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        output_format: str = ".png",
    ) -> Path:
        """
        对单张图片执行超分。

        :param input_path:   输入路径
        :param output_path:  输出路径（None 则自动生成）
        :param output_dir:   输出目录（仅当 output_path 为 None 时生效）
        :param output_format: 输出格式（当 output_path 为 None 时生效）
        :return:             输出文件路径
        """
        if output_path is not None and output_dir is not None:
            raise ValueError("output_path 和 output_dir 不能同时指定")

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        if not is_supported_input(input_path):
            raise ValueError(f"不支持的图片格式: {input_path.suffix}")

        # 读取图片
        img_bgr = load_image(input_path)

        # 超分
        result_bgr = self.upscale_image(img_bgr)

        # 确定输出路径
        if output_path is None:
            output_path = generate_output_path(
                input_path, output_dir=output_dir, output_format=output_format
            )
        else:
            output_path = Path(output_path)

        # 保存
        save_image(result_bgr, output_path)
        return output_path

    def upscale_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        对内存中的 BGR 图像执行超分，返回 BGR 图像。
        适合 GUI 预览等场景。
        """
        with self._lock:
            if self.backend_name == "Pillow":
                return resize_bgr(
                    ensure_bgr_3ch(img_bgr),
                    (
                        int(img_bgr.shape[1] * self.scale),
                        int(img_bgr.shape[0] * self.scale),
                    ),
                )

            if self._model is None:
                raise RuntimeError("模型尚未加载")

            # 预处理：确保是 3 通道
            img_bgr = ensure_bgr_3ch(img_bgr)

            # Real-ESRGAN 需要 BGR 输入，返回 BGR
            output, _ = self._model.enhance(img_bgr, outscale=self.scale)
            return output

    def upscale_batch(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path | None = None,
        output_format: str = ".png",
    ) -> list[Path]:
        """
        批量超分。

        :return: 输出文件路径列表
        """
        results = []
        total = len(input_paths)

        for idx, path in enumerate(input_paths):
            if self.on_progress:
                self.on_progress(idx, total)

            try:
                out = self.upscale(
                    path,
                    output_dir=output_dir,
                    output_format=output_format,
                )
                results.append(out)
            except Exception as e:
                print(f"[警告] 处理失败: {path} — {e}")

        if self.on_progress:
            self.on_progress(total, total)

        return results

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    _MODEL_SPECS = {
        "realesr-animevideov3": {
            "filename": "realesr-animevideov3.pth",
            "url": (
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
                "realesr-animevideov3.pth"
            ),
            "arch": "srvgg",
            "netscale": 4,
            "num_conv": 16,
        },
        "realesrgan-x4plus": {
            "filename": "RealESRGAN_x4plus.pth",
            "url": (
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
                "RealESRGAN_x4plus.pth"
            ),
            "arch": "rrdb",
            "netscale": 4,
            "num_block": 23,
        },
        "realesrgan-x4plus-anime": {
            "filename": "RealESRGAN_x4plus_anime_6B.pth",
            "url": (
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
                "RealESRGAN_x4plus_anime_6B.pth"
            ),
            "arch": "rrdb",
            "netscale": 4,
            "num_block": 6,
        },
        "realesrnet-x4plus": {
            "filename": "RealESRNet_x4plus.pth",
            "url": (
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/"
                "RealESRNet_x4plus.pth"
            ),
            "arch": "rrdb",
            "netscale": 4,
            "num_block": 23,
        },
    }

    def _load_backend(self):
        if self.backend == "pillow":
            self._enable_pillow_backend("手动选择 Pillow 后端")
            return

        try:
            self._load_model()
        except (ModuleNotFoundError, ImportError) as e:
            if self.backend == "realesrgan":
                raise RuntimeError(
                    "Real-ESRGAN 依赖未安装或版本不兼容。请先安装 requirements.txt，"
                    "并按 PyTorch 官网选择匹配 CUDA/CPU 的 torch 与 torchvision。"
                ) from e
            self._enable_pillow_backend(str(e))

    def _enable_pillow_backend(self, reason: str):
        self._model = None
        self.backend_name = "Pillow"
        self.fallback_reason = reason

    def _load_model(self):
        """加载 Real-ESRGAN 模型（首次使用自动下载权重，后续使用本地缓存）"""
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        from basicsr.archs.rrdbnet_arch import RRDBNet

        if self.model_name not in MODELS:
            raise ValueError(
                f"未知模型: {self.model_name}\n可用: {list(MODELS.keys())}"
            )

        # 根据模型选择网络架构
        config = self._MODEL_SPECS[self.model_name]
        if config["arch"] == "srvgg":
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=config["num_conv"],
                upscale=config["netscale"],
                act_type="prelu",
            )
        else:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=config["num_block"],
                num_grow_ch=32,
                scale=config["netscale"],
            )

        model_path = self._resolve_model_path(config)

        self._model = RealESRGANer(
            scale=config["netscale"],
            model_path=model_path,
            model=model,
            tile=self.tile_size,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=not self.fp32,
            device=self.device,
        )
        self.backend_name = "Real-ESRGAN"
        self.fallback_reason = None

    @staticmethod
    def _resolve_model_path(config: dict) -> str:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        local_path = MODELS_DIR / config["filename"]
        if local_path.exists() and local_path.stat().st_size > 0:
            return str(local_path)
        return config["url"]

    @staticmethod
    def _resolve_device(device: str) -> str:
        """解析设备字符串"""
        if device != "auto":
            return device
        try:
            import torch
        except ModuleNotFoundError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def is_ready(self) -> bool:
        """模型是否已加载就绪"""
        return self._model is not None

    @property
    def model_display_name(self) -> str:
        """获取模型的显示名称"""
        return MODELS.get(self.model_name, self.model_name)
