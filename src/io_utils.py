"""
io_utils.py — 输入输出与结果存储模块

负责：
  - 图像读取（单张 / 批量）
  - 输出目录创建
  - results.json 序列化写入
"""

import json
import numpy as np
import cv2
from pathlib import Path
from dataclasses import asdict
from typing import List, Optional

from src.analysis import GrainStatistics, AreaMethodResult, InterceptMethodResult
from src.anomaly import AnomalyResult


# ─────────────────────────── 图像读取 ──────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image(image_path: str) -> np.ndarray:
    """
    读取图像文件，返回 BGR uint8 数组（opencv 格式）。

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 无法解码图像
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {image_path}")
    return img


def collect_images(input_path: str) -> List[str]:
    """
    返回输入路径下所有支持格式的图像文件路径列表。

    Args:
        input_path: 文件路径或文件夹路径

    Returns:
        图像文件路径列表（已排序）
    """
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted([str(f) for f in p.iterdir()
                        if f.suffix.lower() in SUPPORTED_EXTS])
        return files
    raise ValueError(f"Input path not found: {input_path}")


# ─────────────────────────── 目录管理 ──────────────────────────────

def make_output_dir(base_output: str, image_name: str) -> Path:
    """
    创建输出子目录：{base_output}/{image_name}/

    Returns:
        输出目录 Path 对象
    """
    out_dir = Path(base_output) / image_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def output_paths(out_dir: Path, name: str) -> dict:
    """返回各输出文件的完整路径字典。"""
    return {
        "original":  str(out_dir / f"{name}_original.png"),
        "segmented": str(out_dir / f"{name}_segmented.png"),
        "area":      str(out_dir / f"{name}_area_method.png"),
        "intercept": str(out_dir / f"{name}_intercept_method.png"),
        "anomaly":   str(out_dir / f"{name}_anomaly.png"),
        "distribution": str(out_dir / f"{name}_distribution.png"),
        "json":      str(out_dir / f"{name}_results.json"),
    }


# ─────────────────────────── JSON 序列化 ───────────────────────────

def _to_serializable(obj):
    """递归将 numpy 类型转为 Python 原生类型。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(i) for i in obj]
    return obj


def save_results_json(
    output_path: str,
    image_name: str,
    image_path: str,
    pixels_per_micron: float,
    segmentation_params: dict,
    total_grains: int,
    stats: GrainStatistics,
    area_result: AreaMethodResult,
    intercept_result: InterceptMethodResult,
    anomaly_result: AnomalyResult,
) -> None:
    """将分析结果写入 JSON 文件。"""

    def _rule_a(r):
        return {
            "triggered": r.triggered,
            "d_max_over_d_avg": r.d_max_over_d_avg,
            "threshold": r.threshold,
            "anomalous_grain_ids": r.anomalous_grain_ids,
        }

    def _rule_b(r):
        return {
            "triggered": r.triggered,
            f"top{r.top_pct:.0f}pct_area_fraction": r.top_pct_area_fraction,
            "area_fraction_threshold": r.area_fraction_threshold,
        }

    def _rule_c(r):
        return {
            "triggered": r.triggered,
            "anomalous_grain_ids": r.anomalous_grain_ids,
            "threshold_um": r.threshold_um,
        }

    result = {
        "image_name": image_name,
        "image_path": image_path,
        "pixels_per_micron": pixels_per_micron,
        "segmentation": {
            "method": "watershed",
            "total_grains": total_grains,
            "params": segmentation_params,
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
        },
        "area_method": {
            "n_inside": area_result.n_inside,
            "n_intersect": area_result.n_intersect,
            "n_equivalent": area_result.n_equivalent,
            "n_a_per_mm2": round(area_result.n_a_per_mm2, 2),
            "astm_g_value": round(area_result.astm_g_value, 3),
            "mean_grain_area_mm2": area_result.mean_grain_area_mm2,
            "mean_diameter_um": round(area_result.mean_diameter_um, 3),
        },
        "intercept_method": {
            "n_lines_horizontal": intercept_result.n_lines_horizontal,
            "n_lines_vertical": intercept_result.n_lines_vertical,
            "total_intersections": intercept_result.total_intersections,
            "total_line_length_um": round(intercept_result.total_line_length_um, 2),
            "n_l_per_mm": round(intercept_result.n_l_per_mm, 3),
            "mean_intercept_length_um": round(intercept_result.mean_intercept_length_um, 3),
            "astm_g_value": round(intercept_result.astm_g_value, 3),
        },
        "anomaly_detection": {
            "has_anomaly": anomaly_result.has_anomaly,
            "rule_a": _rule_a(anomaly_result.rule_a),
            "rule_b": _rule_b(anomaly_result.rule_b),
            "rule_c": _rule_c(anomaly_result.rule_c),
            "total_anomalous_grains": anomaly_result.total_anomalous_grains,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(result), f, indent=2, ensure_ascii=False)
