"""
pipeline.py — 端到端流程编排模块

将预处理、分割、分析、异常检测、可视化、输出整合为单张图像处理流程。
"""

from pathlib import Path

from src import preprocessing, segmentation, analysis, anomaly, visualization, io_utils


def run(image_path: str,
        output_dir: str = "./data",
        # 预处理参数
        smooth_mode: str = "gaussian",
        gaussian_sigma: float | None = None,
        median_kernel: int = 3,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple = (8, 8),
        # 分割参数
        min_distance: int | None = None,
        closing_disk_size: int = 2,
        opening_disk_size: int = 1,
        min_grain_area: int | None = None,
        remove_border: bool = False,
        # 截线法参数
        pixels_per_micron: float = 1.0,
        min_intercept_px: int = 3,
        # 异常检测参数
        rule_a_threshold: float = 3.0,
        rule_b_top_pct: float = 5.0,
        rule_b_area_frac: float = 0.30) -> dict:
    """
    对单张图像执行完整分析流程。

    Args:
        image_path: 图像文件路径
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
        smooth_mode=smooth_mode,
        gaussian_sigma=gaussian_sigma,
        median_kernel=median_kernel,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
    )

    # ── 3. 晶粒分割 ───────────────────────────────────────────────
    labels = segmentation.segment(
        enhanced,
        min_distance=min_distance,
        closing_disk_size=closing_disk_size,
        opening_disk_size=opening_disk_size,
        min_grain_area=min_grain_area,
        remove_border=remove_border,
    )
    total_grains = int(labels.max())

    # ── 4. 特征提取 ───────────────────────────────────────────────
    grain_props = analysis.extract_grain_props(labels)
    stats = analysis.compute_grain_statistics(grain_props)

    # ── 5a. 面积法 ────────────────────────────────────────────────
    area_result = analysis.area_method(labels, pixels_per_micron=pixels_per_micron)

    # ── 5b. 截线法 ────────────────────────────────────────────────
    intercept_result = analysis.intercept_method(
        labels,
        pixels_per_micron=pixels_per_micron,
        min_intercept_px=min_intercept_px,
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
    effective_min_distance = segmentation._auto_min_distance(enhanced.shape, min_distance)
    effective_min_grain_area = (
        min_grain_area
        if min_grain_area is not None
        else segmentation._auto_min_grain_area(effective_min_distance)
    )
    seg_params = {
        "gaussian_sigma": None,
        "min_distance": effective_min_distance,
        "closing_disk_size": closing_disk_size,
        "min_grain_area": effective_min_grain_area,
    }
    io_utils.save_results_json(
        output_path=paths["json"],
        image_name=image_name,
        image_path=str(image_path),
        segmentation_params=seg_params,
        total_grains=total_grains,
        stats=stats,
        area_result=area_result,
        intercept_result=intercept_result,
        anomaly_result=anomaly_result,
    )

    return {
        "image_name": image_name,           # 图像文件名（不含扩展名），如 "RG36_2_1"
        "total_grains": total_grains,       # 分割得到的晶粒总数（整数）
        "astm_g_area": area_result.astm_g_value,          # 面积法（Jeffries Planimetric）计算的 ASTM G 值
        "astm_g_intercept": intercept_result.astm_g_value, # 截线法（Heyn Intercept）计算的 ASTM G 值
        "has_anomaly": anomaly_result.has_anomaly,         # 是否存在异常晶粒（规则 A/B/C 任一触发则为 True）
        "output_dir": str(out_dir),         # 本次分析结果的输出目录路径
        "paths": paths,                     # 各输出文件路径字典（original/segmented/area/intercept/anomaly/distribution/json）
        "stats": stats,                     # 晶粒统计量（GrainStatistics dataclass：均值/标准差/最大最小直径等）
        "area_result": area_result,         # 面积法完整结果（AreaResult dataclass：N_inside/N_intersect/N_A/G值/平均直径等）
        "intercept_result": intercept_result, # 截线法完整结果（InterceptResult dataclass：交点数/线段总长/N_L/平均截距/G值等）
        "anomaly_result": anomaly_result,   # 异常检测完整结果（AnomalyResult dataclass：各规则触发状态/异常晶粒ID列表等）
    }
