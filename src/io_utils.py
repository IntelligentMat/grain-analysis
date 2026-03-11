"""
io_utils.py — 输入输出与结果存储模块

负责：
  - 图像读取（单张 / 批量）
  - 输出目录创建
  - 工件与 results.json 序列化写入
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

from src.analysis import AreaMethodResult, GrainStatistics, InterceptMethodResult
from src.anomaly import AnomalyResult

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image(image_path: str) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {image_path}")
    return img


def collect_images(input_path: str) -> List[str]:
    path = Path(input_path)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        return sorted([str(file) for file in path.iterdir() if file.suffix.lower() in SUPPORTED_EXTS])
    raise ValueError(f"Input path not found: {input_path}")


def make_output_dir(base_output: str, image_name: str, backend: str | None = None) -> Path:
    out_dir = Path(base_output) / image_name
    if backend:
        out_dir = out_dir / backend
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def output_paths(out_dir: Path, name: str) -> dict[str, str]:
    return {
        "original": str(out_dir / f"{name}_original.png"),
        "segmented": str(out_dir / f"{name}_segmented.png"),
        "area": str(out_dir / f"{name}_area_method.png"),
        "intercept": str(out_dir / f"{name}_intercept_method.png"),
        "anomaly": str(out_dir / f"{name}_anomaly.png"),
        "distribution": str(out_dir / f"{name}_distribution.png"),
        "labels": str(out_dir / f"{name}_labels.npy"),
        "grain_props": str(out_dir / f"{name}_grain_props.npy"),
        "json": str(out_dir / f"{name}_results.json"),
    }


def _to_serializable(obj: Any):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    return obj


def save_labels(output_path: str, labels: np.ndarray) -> None:
    np.save(output_path, labels)


def save_grain_props(output_path: str, grain_props: np.ndarray) -> None:
    np.save(output_path, grain_props)


def load_results_json(results_json_path: str) -> dict[str, Any]:
    with open(results_json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(output_path: str, payload: dict[str, Any]) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, indent=2, ensure_ascii=False)


def save_results_json(
    output_path: str,
    labels_path: str,
    grain_props_path: str,
    image_name: str,
    image_path: str,
    image_shape: tuple[int, ...],
    segmentation_backend: str,
    segmentation_method: str,
    segmentation_params: dict,
    total_grains: int,
    pixels_per_micron: float,
    stats: GrainStatistics,
    area_result: AreaMethodResult,
    intercept_result: InterceptMethodResult,
    anomaly_result: AnomalyResult,
    extra_artifacts: dict[str, Any] | None = None,
    segmentation_details: dict[str, Any] | None = None,
    config_info: dict[str, Any] | None = None,
) -> None:
    def _rule_a(rule):
        return {
            "triggered": rule.triggered,
            "d_max_over_d_avg": rule.d_max_over_d_avg,
            "threshold": rule.threshold,
            "anomalous_grain_ids": rule.anomalous_grain_ids,
        }

    def _rule_b(rule):
        return {
            "triggered": rule.triggered,
            f"top{rule.top_pct:.0f}pct_area_fraction": rule.top_pct_area_fraction,
            "area_fraction_threshold": rule.area_fraction_threshold,
        }

    def _rule_c(rule):
        return {
            "triggered": rule.triggered,
            "anomalous_grain_ids": rule.anomalous_grain_ids,
            "threshold_um": rule.threshold_um,
        }

    result = {
        "image_name": image_name,
        "image_path": image_path,
        "pixels_per_micron": pixels_per_micron,
        "measurement_mode": "physical_um",
        "image": {
            "shape": list(image_shape),
        },
        "config": {
            "source_path": config_info.get("source_path") if config_info else None,
            "effective": config_info.get("effective", {}) if config_info else {},
            "cli_overrides": config_info.get("cli_overrides", {}) if config_info else {},
        },
        "artifacts": {
            "labels_path": labels_path,
            "grain_props_path": grain_props_path,
            **(extra_artifacts or {}),
        },
        "segmentation": {
            "backend": segmentation_backend,
            "method": segmentation_method,
            "total_grains": total_grains,
            "params": segmentation_params,
            "details": segmentation_details or {},
        },
        "grain_statistics": {
            "count": stats.count,
            "mean_diameter_um": round(stats.mean_diameter_um, 3),
            "std_diameter_um": round(stats.std_diameter_um, 3),
            "min_diameter_um": round(stats.min_diameter_um, 3),
            "max_diameter_um": round(stats.max_diameter_um, 3),
            "median_diameter_um": round(stats.median_diameter_um, 3),
            "p10_diameter_um": round(stats.p10_diameter_um, 3),
            "p90_diameter_um": round(stats.p90_diameter_um, 3),
            "mean_area_um2": round(stats.mean_area_um2, 3),
            "total_area_um2": round(stats.total_area_um2, 3),
            "mean_aspect_ratio": round(stats.mean_aspect_ratio, 4),
            "mean_circularity": round(stats.mean_circularity, 4),
            "diameters_um": [round(value, 3) for value in stats.diameters_um],
            "areas_um2": [round(value, 3) for value in stats.areas_um2],
        },
        "area_method": {
            "n_inside": area_result.n_inside,
            "n_intersect": area_result.n_intersect,
            "n_edge": area_result.n_edge,
            "n_corner": area_result.n_corner,
            "n_equivalent": area_result.n_equivalent,
            "n_a_per_mm2": round(area_result.n_a_per_mm2, 2),
            "astm_g_value": round(area_result.astm_g_value, 3),
            "mean_grain_area_mm2": area_result.mean_grain_area_mm2,
            "mean_diameter_um": round(area_result.mean_diameter_um, 3),
            "inside_grain_ids": area_result.inside_grain_ids,
            "edge_grain_ids": area_result.edge_grain_ids,
            "corner_grain_ids": area_result.corner_grain_ids,
            "intersect_grain_ids": area_result.intersect_grain_ids,
        },
        "intercept_method": {
            "n_lines": intercept_result.n_lines,
            "n_circles": intercept_result.n_circles,
            "total_intersections": intercept_result.total_intersections,
            "total_line_length_um": round(intercept_result.total_line_length_um, 3),
            "n_l_per_mm": round(intercept_result.n_l_per_mm, 6),
            "mean_intercept_length_um": round(intercept_result.mean_intercept_length_um, 3),
            "astm_g_value": round(intercept_result.astm_g_value, 3),
            "counting_basis": "grain_segments_n",
            "pattern_elements": intercept_result.pattern_elements,
            "intersection_points": intercept_result.intersection_points,
            "half_intersection_points": intercept_result.half_intersection_points,
            "intersected_grain_ids": intercept_result.intersected_grain_ids,
        },
        "anomaly_detection": {
            "has_anomaly": anomaly_result.has_anomaly,
            "rule_a": _rule_a(anomaly_result.rule_a),
            "rule_b": _rule_b(anomaly_result.rule_b),
            "rule_c": _rule_c(anomaly_result.rule_c),
            "total_anomalous_grains": anomaly_result.total_anomalous_grains,
            "anomalous_grain_ids": anomaly_result.anomalous_grain_ids,
        },
    }

    save_json(output_path, result)
