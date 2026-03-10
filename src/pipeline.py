"""
pipeline.py — 端到端流程编排模块

将预处理、分割、分析、异常检测、可视化、输出整合为单张图像处理流程。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src import analysis, anomaly, io_utils, preprocessing, segmentation, visualization
from src.sam3_backend import run_prompted_sam3


def _normalize_backend(segmentation_backend: str) -> str:
    if segmentation_backend == "watershed":
        return "optical"
    if segmentation_backend not in {"optical", "sam3"}:
        raise ValueError("segmentation_backend must be 'optical' or 'sam3'")
    return segmentation_backend


def _build_optical_segmentation(
    image,
    smooth_mode: str,
    gaussian_sigma: float | None,
    median_kernel: int,
    clahe_clip_limit: float,
    clahe_tile_grid_size: tuple,
    min_distance: int | None,
    closing_disk_size: int,
    opening_disk_size: int,
    min_grain_area: int | None,
    remove_border: bool,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    labels = segmentation.segment(
        enhanced,
        min_distance=min_distance,
        closing_disk_size=closing_disk_size,
        opening_disk_size=opening_disk_size,
        min_grain_area=min_grain_area,
        remove_border=remove_border,
    )
    segmentation_params = {
        "gaussian_sigma": gaussian_sigma,
        "min_distance": effective_min_distance,
        "closing_disk_size": closing_disk_size,
        "opening_disk_size": opening_disk_size,
        "min_grain_area": effective_min_grain_area,
        "remove_border": remove_border,
    }
    return labels, segmentation_params, {}, {}


def run_analysis_from_labels(
    image_path: str,
    image,
    labels,
    output_dir: str,
    image_name: str,
    segmentation_backend: str,
    segmentation_method: str,
    segmentation_params: dict[str, Any],
    pixels_per_micron: float,
    min_intercept_px: int,
    rule_a_threshold: float,
    rule_b_top_pct: float,
    rule_b_area_frac: float,
    extra_artifacts: dict[str, Any] | None = None,
    segmentation_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = io_utils.make_output_dir(output_dir, image_name, segmentation_backend)
    paths = io_utils.output_paths(out_dir, image_name)
    io_utils.save_labels(paths["labels"], labels)

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
        labels_path=str(Path(paths["labels"]).resolve()),
        image_name=image_name,
        image_path=str(Path(image_path).resolve()),
        image_shape=image.shape,
        segmentation_backend=segmentation_backend,
        segmentation_method=segmentation_method,
        segmentation_params=segmentation_params,
        total_grains=int(labels.max()),
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
        "total_grains": int(labels.max()),
        "astm_g_area": area_result.astm_g_value,
        "astm_g_intercept": intercept_result.astm_g_value,
        "has_anomaly": anomaly_result.has_anomaly,
        "segmentation_backend": segmentation_backend,
        "segmentation_method": segmentation_method,
        "output_dir": str(out_dir),
        "paths": paths,
        "stats": stats,
        "area_result": area_result,
        "intercept_result": intercept_result,
        "anomaly_result": anomaly_result,
    }


def run(
    image_path: str,
    output_dir: str = "./data",
    smooth_mode: str = "gaussian",
    gaussian_sigma: float | None = None,
    median_kernel: int = 3,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple = (8, 8),
    segmentation_backend: str = "optical",
    min_distance: int | None = None,
    closing_disk_size: int = 2,
    opening_disk_size: int = 1,
    min_grain_area: int | None = None,
    remove_border: bool = False,
    pixels_per_micron: float = 1.0,
    min_intercept_px: int = 3,
    rule_a_threshold: float = 3.0,
    rule_b_top_pct: float = 5.0,
    rule_b_area_frac: float = 0.30,
    sam3_model_id: str = "facebook/sam3",
    sam3_device: str = "auto",
    sam3_score_threshold: float = 0.5,
    sam3_mask_threshold: float = 0.5,
    sam3_opening_disk_size: int = 1,
    sam3_closing_disk_size: int = 2,
    sam3_prompt_top_ratio: float = 0.05,
) -> dict:
    segmentation_backend = _normalize_backend(segmentation_backend)
    image_name = Path(image_path).stem
    image = io_utils.load_image(image_path)

    if segmentation_backend == "optical":
        labels, segmentation_params, segmentation_details, extra_artifacts = (
            _build_optical_segmentation(
                image=image,
                smooth_mode=smooth_mode,
                gaussian_sigma=gaussian_sigma,
                median_kernel=median_kernel,
                clahe_clip_limit=clahe_clip_limit,
                clahe_tile_grid_size=clahe_tile_grid_size,
                min_distance=min_distance,
                closing_disk_size=closing_disk_size,
                opening_disk_size=opening_disk_size,
                min_grain_area=min_grain_area,
                remove_border=remove_border,
            )
        )
        return run_analysis_from_labels(
            image_path=image_path,
            image=image,
            labels=labels,
            output_dir=output_dir,
            image_name=image_name,
            segmentation_backend="optical",
            segmentation_method="watershed",
            segmentation_params=segmentation_params,
            pixels_per_micron=pixels_per_micron,
            min_intercept_px=min_intercept_px,
            rule_a_threshold=rule_a_threshold,
            rule_b_top_pct=rule_b_top_pct,
            rule_b_area_frac=rule_b_area_frac,
            extra_artifacts=extra_artifacts,
            segmentation_details=segmentation_details,
        )

    optical_labels_path = (
        Path(io_utils.make_output_dir(output_dir, image_name, "optical"))
        / f"{image_name}_labels.npy"
    )
    if not optical_labels_path.exists():
        optical_result = run(
            image_path=image_path,
            output_dir=output_dir,
            smooth_mode=smooth_mode,
            gaussian_sigma=gaussian_sigma,
            median_kernel=median_kernel,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            segmentation_backend="optical",
            min_distance=min_distance,
            closing_disk_size=closing_disk_size,
            opening_disk_size=opening_disk_size,
            min_grain_area=min_grain_area,
            remove_border=remove_border,
            pixels_per_micron=pixels_per_micron,
            min_intercept_px=min_intercept_px,
            rule_a_threshold=rule_a_threshold,
            rule_b_top_pct=rule_b_top_pct,
            rule_b_area_frac=rule_b_area_frac,
        )
        optical_labels_path = Path(optical_result["paths"]["labels"])
        if not optical_labels_path.exists():
            raise FileNotFoundError(
                f"Optical labels not found after bootstrap run: {optical_labels_path}"
            )

    sam3_out_dir = io_utils.make_output_dir(output_dir, image_name, "sam3")
    output_prefix = sam3_out_dir / f"{image_name}_sam3_prompts"
    sam3_result = run_prompted_sam3(
        image_path=image_path,
        optical_labels_path=optical_labels_path,
        output_prefix=output_prefix,
        model_id=sam3_model_id,
        device=sam3_device,
        score_threshold=sam3_score_threshold,
        mask_threshold=sam3_mask_threshold,
        prompt_top_ratio=sam3_prompt_top_ratio,
        opening_disk_size=sam3_opening_disk_size,
        closing_disk_size=sam3_closing_disk_size,
    )

    segmentation_params = {
        "model_id": sam3_model_id,
        "device": sam3_device,
        "score_threshold": sam3_score_threshold,
        "mask_threshold": sam3_mask_threshold,
        "opening_disk_size": sam3_opening_disk_size,
        "closing_disk_size": sam3_closing_disk_size,
        "prompt_top_ratio": sam3_prompt_top_ratio,
    }
    segmentation_details = {
        "prompt_source_labels_path": str(optical_labels_path.resolve()),
        "prompt_top_ratio": float(sam3_prompt_top_ratio),
        "prompt_selected_grain_ids": sam3_result["prompt_selected_grain_ids"],
        "prompt_selected_grain_count": len(sam3_result["prompt_selected_grain_ids"]),
        "prompt_mode": "boxes_from_optical_top_area",
        "postprocess_mode": "opening_then_closing_per_mask",
        "mask_conversion": sam3_result["mask_conversion"],
        "sam3_device": sam3_result["sam3_device"],
    }
    extra_artifacts = {
        "sam3_prompt_json_path": str(sam3_result["prompt_paths"]["json"].resolve()),
        "sam3_prompt_masks_path": str(sam3_result["prompt_paths"]["masks"].resolve())
        if "masks" in sam3_result["prompt_paths"]
        else None,
        "sam3_raw_masks_path": str(sam3_result["raw_masks_path"].resolve())
        if sam3_result["raw_masks_path"]
        else None,
        "sam3_raw_json_path": str(sam3_result["raw_json_path"].resolve())
        if sam3_result["raw_json_path"]
        else None,
    }
    extra_artifacts = {key: value for key, value in extra_artifacts.items() if value is not None}

    return run_analysis_from_labels(
        image_path=image_path,
        image=image,
        labels=sam3_result["labels"],
        output_dir=output_dir,
        image_name=image_name,
        segmentation_backend="sam3",
        segmentation_method="sam3_prompt_boxes",
        segmentation_params=segmentation_params,
        pixels_per_micron=pixels_per_micron,
        min_intercept_px=min_intercept_px,
        rule_a_threshold=rule_a_threshold,
        rule_b_top_pct=rule_b_top_pct,
        rule_b_area_frac=rule_b_area_frac,
        extra_artifacts=extra_artifacts,
        segmentation_details=segmentation_details,
    )
