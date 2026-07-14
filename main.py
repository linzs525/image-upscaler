"""
main.py — 图片高清化工具入口

用法:
    python main.py                # 启动 GUI
    python main.py --help         # 查看帮助
"""

import sys
import argparse
from pathlib import Path

# 确保项目根目录在 sys.path 中，方便直接运行
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main_cli():
    """命令行模式"""
    parser = argparse.ArgumentParser(
        description="图片高清化工具 — 基于 Real-ESRGAN 深度学习模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py -i photo.jpg                    # GUI 模式（默认）
    python main.py --cli -i photo.jpg              # CLI 处理单张图片
    python main.py --cli -i input/ -o output/      # CLI 批量处理目录
    python main.py --cli -i a.jpg b.png -m realesrgan-x4plus -s 4
        """,
    )
    parser.add_argument(
        "--cli", action="store_true",
        help="使用命令行模式（默认启动 GUI）"
    )
    parser.add_argument(
        "-i", "--input", nargs="+",
        help="输入文件或目录路径"
    )
    parser.add_argument(
        "-o", "--output",
        help="输出目录或文件路径"
    )
    parser.add_argument(
        "-m", "--model", default="realesr-animevideov3",
        choices=[
            "realesr-animevideov3", "realesrgan-x4plus",
            "realesrgan-x4plus-anime", "realesrnet-x4plus",
        ],
        help="超分模型 (默认: realesr-animevideov3)"
    )
    parser.add_argument(
        "-s", "--scale", type=int, default=4,
        choices=[2, 3, 4, 8],
        help="放大倍数 (默认: 4)"
    )
    parser.add_argument(
        "-f", "--format", default=".png",
        choices=[".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"],
        help="输出格式 (默认: .png)"
    )
    parser.add_argument(
        "-d", "--device", default="auto",
        choices=["auto", "cuda", "cpu"],
        help="推理设备 (默认: auto)"
    )
    parser.add_argument(
        "-t", "--tile", type=int, default=0,
        help="分块大小，0=不分块 (显存不足时建议 400~800)"
    )

    args = parser.parse_args()

    # 既没有 --cli 也没有 -i → 启动 GUI
    if not args.cli and not args.input:
        from src.gui import launch
        launch()
        return

    # ---- CLI 模式 ----
    if not args.input:
        parser.print_help()
        sys.exit(1)

    from src.config import OUTPUT_DIR
    from src.upscaler import Upscaler
    from src.utils import is_supported_input, ensure_dir, generate_output_path

    # 收集输入文件
    input_files = []
    for p in args.input:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.is_file() and is_supported_input(f):
                    input_files.append(f)
        elif p.is_file():
            input_files.append(p)
        else:
            print(f"[警告] 跳过不存在的路径: {p}")

    if not input_files:
        print("没有找到可处理的图片文件。")
        sys.exit(1)

    # 初始化引擎
    print(f"加载模型: {args.model}  设备: {args.device}  倍数: {args.scale}x")
    try:
        upscaler = Upscaler(
            model_name=args.model,
            scale=args.scale,
            device=args.device,
            tile_size=args.tile,
        )
    except Exception as e:
        print(f"模型初始化失败: {e}")
        sys.exit(1)

    print(f"模型就绪，设备: {upscaler.device.upper()}，后端: {upscaler.backend_name}")
    if upscaler.fallback_reason:
        print("提示: 未能启用 Real-ESRGAN，已使用 Pillow 高质量插值后端。")

    # 处理
    output_dest = Path(args.output) if args.output else OUTPUT_DIR
    # 判断输出目标是目录还是单文件
    if output_dest:
        if output_dest.suffix:
            # 有扩展名 → 单文件模式（仅限单输入）
            if len(input_files) > 1:
                print("输出路径是单个文件时，只能处理一张输入图片。")
                sys.exit(1)
        else:
            ensure_dir(output_dest)

    print(f"\n共 {len(input_files)} 张图片待处理:\n")

    success = 0
    for i, f in enumerate(input_files, 1):
        print(f"  [{i}/{len(input_files)}] {f.name} ...", end=" ", flush=True)
        try:
            # 确定输出路径
            if output_dest and not output_dest.suffix:
                # 输出到指定目录
                out_path = generate_output_path(f, output_dir=output_dest,
                                                output_format=args.format)
            elif output_dest and len(input_files) == 1:
                out_path = output_dest  # 单文件直接使用
            else:
                out_path = None  # 自动生成

            out = upscaler.upscale(
                f,
                output_path=out_path,
                output_format=args.format,
            )
            print(f"[OK] -> {out.name}")
            success += 1
        except Exception as e:
            print(f"[失败] {e}")

    print(f"\n完成: {success}/{len(input_files)} 张成功")


def main():
    """自动判断 CLI / GUI"""
    # 如果传了命令行参数（除了脚本名），走 CLI
    if len(sys.argv) > 1:
        main_cli()
    else:
        from src.gui import launch
        launch()


if __name__ == "__main__":
    main()
