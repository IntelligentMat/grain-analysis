"""
main.py — CLI 入口

用法：
  python main.py --input /path/to/image.jpg
  python main.py --input /path/to/folder/ --output ./data
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from tqdm import tqdm

from src import config as app_config
from src import io_utils, pipeline, visualization


def _explicit_configurable_params(ctx: click.Context) -> set[str]:
    explicit: set[str] = set()
    get_source = getattr(ctx, "get_parameter_source", None)
    raw_args = sys.argv[1:]

    for spec in app_config.OPTION_SPECS:
        if callable(get_source):
            source = get_source(spec.param_name)
            if getattr(source, "name", "") == "COMMANDLINE":
                explicit.add(spec.param_name)
                continue

        if any(app_config.was_option_explicit(arg, spec.flags) for arg in raw_args):
            explicit.add(spec.param_name)
    return explicit


@click.command()
@click.option("--input", "-i", "input_path", help="输入图像文件或文件夹路径")
@click.option(
    "--render-from-results",
    "render_results_path",
    default=None,
    help="根据已有 results.json 重绘可视化",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    help="YAML 配置文件路径（仅分析模式可用）",
)
@click.option(
    "--output", "-o", "output_dir", default=None, help="输出根目录；重绘时不传则写回原结果目录"
)
@click.option(
    "--smooth-mode",
    default="gaussian",
    show_default=True,
    type=click.Choice(["gaussian", "bilateral", "anisotropic"]),
    help="平滑模式：gaussian / bilateral / anisotropic",
)
@click.option(
    "--gaussian-sigma",
    default=None,
    show_default=True,
    type=float,
    help="预处理高斯滤波标准差（不设则自动估计）",
)
@click.option(
    "--median-kernel", default=3, show_default=True, type=int, help="中值滤波核大小（奇数）"
)
@click.option("--clahe-clip", default=2.0, show_default=True, type=float, help="CLAHE 对比度限制")
@click.option(
    "--segmentation-backend",
    default="optical",
    show_default=True,
    type=click.Choice(["optical", "sam3"]),
    help="分割后端：传统 optical 或 SAM3 prompt-based 后端",
)
@click.option(
    "--min-distance",
    default=None,
    show_default=True,
    type=int,
    help="Watershed marker 最小间距（像素）",
)
@click.option("--closing-disk", default=2, show_default=True, type=int, help="形态学闭运算核半径")
@click.option("--opening-disk", default=1, show_default=True, type=int, help="形态学开运算核半径")
@click.option(
    "--min-grain-area", default=None, type=int, help="最小晶粒面积（像素²），不设则自动估算"
)
@click.option(
    "--remove-border/--keep-border", default=False, show_default=True, help="是否移除接触边界的晶粒"
)
@click.option(
    "--pixels-per-micron", default=1.0, show_default=True, type=float, help="像素/微米换算系数"
)
@click.option(
    "--min-intercept-px", default=3, show_default=True, type=int, help="最小有效截段长度（px）"
)
@click.option(
    "--rule-a-threshold", default=3.0, show_default=True, type=float, help="规则A：d_max/d_avg 阈值"
)
@click.option(
    "--rule-b-top-pct", default=5.0, show_default=True, type=float, help="规则B：检测前 X% 大晶粒"
)
@click.option(
    "--rule-b-area-frac", default=0.30, show_default=True, type=float, help="规则B：面积占比阈值"
)
@click.option(
    "--sam3-model-id", default="facebook/sam3", show_default=True, help="Hugging Face SAM3 model id"
)
@click.option(
    "--sam3-device",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    help="SAM3 device",
)
@click.option(
    "--sam3-score-threshold",
    default=0.5,
    show_default=True,
    type=float,
    help="SAM3 instance score threshold",
)
@click.option(
    "--sam3-mask-threshold",
    default=0.5,
    show_default=True,
    type=float,
    help="SAM3 binary mask threshold",
)
@click.option(
    "--sam3-opening-disk",
    default=1,
    show_default=True,
    type=int,
    help="SAM3 每个 mask 的开运算核半径",
)
@click.option(
    "--sam3-closing-disk",
    default=2,
    show_default=True,
    type=int,
    help="SAM3 每个 mask 的闭运算核半径",
)
@click.option(
    "--sam3-prompt-top-ratio",
    default=0.05,
    show_default=True,
    type=float,
    help="从 optical labels 中按面积选前多少比例晶粒做 prompts",
)
@click.pass_context
def main(
    ctx: click.Context,
    input_path,
    render_results_path,
    config_path,
    output_dir,
    smooth_mode,
    gaussian_sigma,
    median_kernel,
    clahe_clip,
    segmentation_backend,
    min_distance,
    closing_disk,
    opening_disk,
    min_grain_area,
    remove_border,
    pixels_per_micron,
    min_intercept_px,
    rule_a_threshold,
    rule_b_top_pct,
    rule_b_area_frac,
    sam3_model_id,
    sam3_device,
    sam3_score_threshold,
    sam3_mask_threshold,
    sam3_opening_disk,
    sam3_closing_disk,
    sam3_prompt_top_ratio,
):
    """晶粒自动化分析系统 — optical 传统流程 + 正式 SAM3 backend。"""

    if bool(input_path) == bool(render_results_path):
        click.echo("[ERROR] 请二选一提供 --input 或 --render-from-results。", err=True)
        sys.exit(1)

    if render_results_path and config_path:
        click.echo("[ERROR] --config 仅分析模式可用，不能和 --render-from-results 同时使用。", err=True)
        sys.exit(1)

    if render_results_path:
        try:
            paths = visualization.render_all_from_results(
                render_results_path, output_dir=output_dir
            )
        except Exception as exc:
            click.echo(f"[ERROR] 重绘失败: {exc}", err=True)
            sys.exit(1)
        click.echo(f"重绘完成，结果目录：{Path(paths['json']).parent}")
        return

    explicit_params = _explicit_configurable_params(ctx)
    try:
        resolved = app_config.build_resolved_config(
            config_path=config_path,
            cli_values=ctx.params,
            explicit_param_names=explicit_params,
        )
    except Exception as exc:
        click.echo(f"[ERROR] 配置解析失败: {exc}", err=True)
        sys.exit(1)

    actual_output_dir = resolved.runtime_values["output_dir"]
    config_info = {
        "source_path": resolved.source_path,
        "effective": resolved.effective,
        "cli_overrides": app_config.prune_empty_override_groups(resolved.cli_overrides),
    }

    try:
        image_files = io_utils.collect_images(input_path)
    except ValueError as exc:
        click.echo(f"[ERROR] {exc}", err=True)
        sys.exit(1)

    if not image_files:
        click.echo("[WARNING] 未找到任何支持的图像文件。", err=True)
        sys.exit(1)

    click.echo(f"共发现 {len(image_files)} 张图像，输出目录：{actual_output_dir}")

    failed = []
    for img_path in tqdm(image_files, desc="Processing", unit="img"):
        try:
            result = pipeline.run(
                image_path=img_path,
                config_info=config_info,
                **resolved.pipeline_kwargs,
            )
            tqdm.write(
                f"  {result['image_name']:30s} | "
                f"backend={result['segmentation_backend']:9s} | "
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
        for failed_path in failed:
            click.echo(f"  {failed_path}")


if __name__ == "__main__":
    main()
