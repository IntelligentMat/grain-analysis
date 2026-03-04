"""
pipeline.py — 端到端流程编排模块

将预处理、分割、分析、异常检测、可视化、输出整合为单张图像处理流程。
"""

import numpy as np
from pathlib import Path

from src import preprocessing, segmentation, analysis, anomaly, visualization, io_utils


def run(image_path: str,
        pixels_per_micron: float = 2.25,
        output_dir: str = "./data",
        # 预处理参数
        gaussian_sigma: float = 3.0,
        median_kernel: int = 3,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple = (8, 8),
        # 分割参数
        seg_gaussian_sigma: float = 3.0,
        min_distance: int = 50,
        closing_disk_size: int = 2,
        min_grain_area: int | None = None,
        remove_border: bool = False,
        # 截线法参数
        n_lines_h: int = 5,
        n_lines_v: int = 5,
        # 异常检测参数
        rule_a_threshold: float = 3.0,
        rule_b_top_pct: float = 5.0,
        rule_b_area_frac: float = 0.30) -> dict:
    """
    对单张图像执行完整分析流程。

    Args:
        image_path: 图像文件路径
        pixels_per_micron: 像素/微米换算比（数据集默认 2.25）
        output_dir: 输出根目录
        其余参数参见各子模块说明

    Returns:
        包含所有结果的字典（与 results.json 内容对应）
    """
    image_name = Path(image_path).stem

    # ── 1. 读取图像 ───────────────────────────────────────────────
    image = io_utils.load_image(image_path)

    # ── 2. 预处理 ─────────────────────────────────────────────────
    enhanced = preprocessing.preprocess(
        image,
        gaussian_sigma=gaussian_sigma,
        median_kernel=median_kernel,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
    )

    # ── 3. 晶粒分割 ───────────────────────────────────────────────
    labels = segmentation.segment(
        enhanced,
        pixels_per_micron=pixels_per_micron,
        gaussian_sigma=seg_gaussian_sigma,
        min_distance=min_distance,
        closing_disk_size=closing_disk_size,
        min_grain_area=min_grain_area,
        remove_border=remove_border,
    )
    total_grains = int(labels.max())

    # ── 4. 特征提取 ───────────────────────────────────────────────
    grain_props = analysis.extract_grain_props(labels, pixels_per_micron)
    stats = analysis.compute_grain_statistics(grain_props)

    # ── 5a. 面积法 ────────────────────────────────────────────────
    area_result = analysis.area_method(labels, pixels_per_micron)

    # ── 5b. 截线法 ────────────────────────────────────────────────
    intercept_result = analysis.intercept_method(
        labels, pixels_per_micron,
        n_lines_h=n_lines_h, n_lines_v=n_lines_v,
    )

    # ── 6. 异常检测 ───────────────────────────────────────────────
    anomaly_result = anomaly.detect_anomalies(
        grain_props, stats,
        rule_a_threshold=rule_a_threshold,
        rule_b_top_pct=rule_b_top_pct,
        rule_b_area_frac_threshold=rule_b_area_frac,
    )

    # ── 7. 输出目录与路径 ─────────────────────────────────────────
    out_dir = io_utils.make_output_dir(output_dir, image_name)
    paths = io_utils.output_paths(out_dir, image_name)

    # ── 8. 可视化 ─────────────────────────────────────────────────
    visualization.save_original(image, paths["original"])
    visualization.save_segmented(image, labels, paths["segmented"])
    visualization.save_area_method(image, labels, area_result, paths["area"])
    visualization.save_intercept_method(image, intercept_result, paths["intercept"])
    visualization.save_anomaly(image, labels, anomaly_result, paths["anomaly"])
    visualization.save_distribution(stats, paths["distribution"])

    # ── 9. JSON 结果 ──────────────────────────────────────────────
    seg_params = {
        "gaussian_sigma": seg_gaussian_sigma,
        "min_distance": min_distance,
        "closing_disk_size": closing_disk_size,
        "min_grain_area": (min_grain_area if min_grain_area is not None
                           else segmentation._auto_min_grain_area(pixels_per_micron)),
    }
    io_utils.save_results_json(
        output_path=paths["json"],
        image_name=image_name,
        image_path=str(image_path),
        pixels_per_micron=pixels_per_micron,
        segmentation_params=seg_params,
        total_grains=total_grains,
        stats=stats,
        area_result=area_result,
        intercept_result=intercept_result,
        anomaly_result=anomaly_result,
    )

    return {
        "image_name": image_name,
        "total_grains": total_grains,
        "astm_g_area": area_result.astm_g_value,
        "astm_g_intercept": intercept_result.astm_g_value,
        "has_anomaly": anomaly_result.has_anomaly,
        "output_dir": str(out_dir),
        "paths": paths,
        "stats": stats,
        "area_result": area_result,
        "intercept_result": intercept_result,
        "anomaly_result": anomaly_result,
    }
