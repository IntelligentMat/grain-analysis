"""
main.py — CLI 入口

用法：
  python main.py --input /path/to/image.jpg
  python main.py --input /path/to/folder/ --output ./data
"""

import sys
import click
from tqdm import tqdm

from src import io_utils, pipeline


@click.command()
@click.option("--input", "-i", "input_path", required=True,
              help="输入图像文件路径或文件夹路径")
@click.option("--output", "-o", "output_dir", default="./data", show_default=True,
              help="输出根目录")
# 预处理参数
@click.option("--smooth-mode", default="gaussian", show_default=True,
              type=click.Choice(["gaussian", "bilateral", "anisotropic"]),
              help="平滑模式：gaussian（默认）/ bilateral（双边）/ anisotropic（各向异性扩散）")
@click.option("--gaussian-sigma", default=None, show_default=True,
              type=float, help="预处理高斯滤波标准差（不设则自动估计）")
@click.option("--median-kernel", default=3, show_default=True,
              type=int, help="中值滤波核大小（奇数）")
@click.option("--clahe-clip", default=2.0, show_default=True,
              type=float, help="CLAHE 对比度限制")
# 分割参数
@click.option("--min-distance", default=None, show_default=True,
              type=int, help="Watershed marker 最小间距（像素），不设则按图像尺寸自动估算")
@click.option("--closing-disk", default=2, show_default=True,
              type=int, help="形态学闭运算核半径（增大可连接断裂晶界）")
@click.option("--opening-disk", default=1, show_default=True,
              type=int, help="形态学开运算核半径（增大可消除划痕，如 2~4）")
@click.option("--min-grain-area", default=None, type=int,
              help="最小晶粒面积（像素²），不设则默认 10")
@click.option("--remove-border/--keep-border", default=False, show_default=True,
              help="是否移除接触边界的晶粒")
# 截线法参数
@click.option("--pixels-per-micron", default=1.0, show_default=True,
              type=float, help="像素/微米换算系数（如 500X 不锈钢 2.25 px/μm）")
# 异常检测参数
@click.option("--rule-a-threshold", default=3.0, show_default=True,
              type=float, help="规则A：d_max/d_avg 阈值")
@click.option("--rule-b-top-pct", default=5.0, show_default=True,
              type=float, help="规则B：检测前 X% 大晶粒")
@click.option("--rule-b-area-frac", default=0.30, show_default=True,
              type=float, help="规则B：面积占比阈值")
def main(input_path, output_dir, smooth_mode, gaussian_sigma, median_kernel, clahe_clip,
         min_distance, closing_disk, opening_disk, min_grain_area, remove_border,
         pixels_per_micron,
         rule_a_threshold, rule_b_top_pct, rule_b_area_frac):
    """晶粒自动化分析系统 — ASTM E112 面积法 + 截线法"""

    # 收集图像文件
    try:
        image_files = io_utils.collect_images(input_path)
    except ValueError as e:
        click.echo(f"[ERROR] {e}", err=True)
        sys.exit(1)

    if not image_files:
        click.echo("[WARNING] 未找到任何支持的图像文件。", err=True)
        sys.exit(1)

    click.echo(f"共发现 {len(image_files)} 张图像，输出目录：{output_dir}")

    # 批量处理
    failed = []
    for img_path in tqdm(image_files, desc="Processing", unit="img"):
        try:
            result = pipeline.run(
                image_path=img_path,
                output_dir=output_dir,
                smooth_mode=smooth_mode,
                gaussian_sigma=gaussian_sigma,
                median_kernel=median_kernel,
                clahe_clip_limit=clahe_clip,
                min_distance=min_distance,
                closing_disk_size=closing_disk,
                opening_disk_size=opening_disk,
                min_grain_area=min_grain_area,
                remove_border=remove_border,
                pixels_per_micron=pixels_per_micron,
                rule_a_threshold=rule_a_threshold,
                rule_b_top_pct=rule_b_top_pct,
                rule_b_area_frac=rule_b_area_frac,
            )
            tqdm.write(
                f"  {result['image_name']:30s} | "
                f"grains={result['total_grains']:4d} | "
                f"G(area)={result['astm_g_area']:5.2f} | "
                f"G(intercept)={result['astm_g_intercept']:5.2f} | "
                f"anomaly={'YES' if result['has_anomaly'] else 'no ':3s}"
            )
        except Exception as exc:
            tqdm.write(f"  [FAIL] {img_path}: {exc}")
            failed.append(img_path)

    click.echo(f"\n完成。失败 {len(failed)}/{len(image_files)} 张。")
    if failed:
        for f in failed:
            click.echo(f"  {f}")


if __name__ == "__main__":
    main()
