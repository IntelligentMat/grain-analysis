"""
pipeline.py — 端到端流程编排模块

将预处理、分割、分析、异常检测、可视化、输出整合为单张图像处理流程。
"""

from __future__ import annotations

from pathlib import Path

from src import preprocessing, segmentation, analysis, anomaly, visualization, io_utils


def _resolve_output_root(output_dir: str, segmentation_backend: str) -> str:
    if segmentation_backend == "watershed":
        return output_dir
    return str(Path(output_dir) / segmentation_backend)


def _default_sam3_model_dir() -> str:
    return str((Path(__file__).resolve().parents[2] / "sam3").resolve())


def run(image_path: str,
        output_dir: str = "./data",
        # 预处理参数
        smooth_mode: str = "gaussian",
        gaussian_sigma: float | None = None,
        median_kernel: int = 3,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple = (8, 8),
        # 分割参数
        segmentation_backend: str = "watershed",
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
    """对单张图像执行完整分析流程。"""
    if segmentation_backend not in {"watershed", "sam3"}:
        raise ValueError("segmentation_backend must be 'watershed' or 'sam3'")

    image_name = Path(image_path).stem
    out_dir = io_utils.make_output_dir(_resolve_output_root(output_dir, segmentation_backend), image_name)
    paths = io_utils.output_paths(out_dir, image_name)

    image = io_utils.load_image(image_path)
    enhanced = preprocessing.preprocess(
        image,
        smooth_mode=smooth_mode,
        gaussian_sigma=gaussian_sigma,
        median_kernel=median_kernel,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
    )

    effective_min_distance = segmentation._auto_min_distance(enhanced.shape, min_distance)
    effective_min_grain_area = (
        min_grain_area
        if min_grain_area is not None
        else segmentation._auto_min_grain_area(effective_min_distance)
    )

    extra_artifacts: dict[str, str] = {}
    segmentation_details: dict[str, object] = {}
    labels = segmentation.segment(
        enhanced,
        min_distance=min_distance,
        closing_disk_size=closing_disk_size,
        opening_disk_size=opening_disk_size,
        min_grain_area=min_grain_area,
        remove_border=remove_border,
    )
    labels_artifact_path = paths["labels"]
    io_utils.save_labels(labels_artifact_path, labels)
    segmentation_params = {
        "gaussian_sigma": gaussian_sigma,
        "min_distance": effective_min_distance,
        "closing_disk_size": closing_disk_size,
        "opening_disk_size": opening_disk_size,
        "min_grain_area": effective_min_grain_area,
        "remove_border": remove_border,
    }
    total_grains = int(labels.max())
    grain_props = analysis.extract_grain_props(labels)
    stats = analysis.compute_grain_statistics(grain_props)
    area_result = analysis.area_method(labels, pixels_per_micron=pixels_per_micron)
    intercept_result = analysis.intercept_method(
        labels,
        pixels_per_micron=pixels_per_micron,
        min_intercept_px=min_intercept_px,
    )
    anomaly_result = anomaly.detect_anomalies(
        grain_props,
        stats,
        rule_a_threshold=rule_a_threshold,
        rule_b_top_pct=rule_b_top_pct,
        rule_b_area_frac_threshold=rule_b_area_frac,
    )

    io_utils.save_results_json(
        output_path=paths["json"],
        labels_path=str(Path(labels_artifact_path).resolve()),
        image_name=image_name,
        image_path=str(Path(image_path).resolve()),
        image_shape=image.shape,
        segmentation_method=segmentation_backend,
        segmentation_params=segmentation_params,
        total_grains=total_grains,
        stats=stats,
        area_result=area_result,
        intercept_result=intercept_result,
        anomaly_result=anomaly_result,
        extra_artifacts=extra_artifacts,
        segmentation_details=segmentation_details,
    )

    visualization.render_all_from_results(paths["json"])

    return {
        "image_name": image_name,
        "total_grains": total_grains,
        "astm_g_area": area_result.astm_g_value,
        "astm_g_intercept": intercept_result.astm_g_value,
        "has_anomaly": anomaly_result.has_anomaly,
        "segmentation_backend": segmentation_backend,
        "output_dir": str(out_dir),
        "paths": paths,
        "stats": stats,
        "area_result": area_result,
        "intercept_result": intercept_result,
        "anomaly_result": anomaly_result,
    }
