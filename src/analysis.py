"""
analysis.py — 晶粒特征提取、面积法与截线法分析模块

包含：
  - regionprops 特征提取
  - Jeffries 平面测定法（面积法，ASTM E112）
  - Heyn 截距法（截线法，ASTM E112）
"""

import numpy as np
from skimage import measure
from skimage.segmentation import find_boundaries
from dataclasses import dataclass, field
from typing import List, Dict, Any


# ─────────────────────────── 数据结构 ──────────────────────────────

@dataclass
class GrainProps:
    grain_id: int
    area_px: float
    area_um2: float
    perimeter_px: float
    equivalent_diameter_um: float
    aspect_ratio: float
    circularity: float
    centroid: tuple      # (row, col)
    bbox: tuple          # (min_row, min_col, max_row, max_col)


@dataclass
class GrainStatistics:
    count: int
    mean_diameter_um: float
    std_diameter_um: float
    min_diameter_um: float
    max_diameter_um: float
    median_diameter_um: float
    p10_diameter_um: float
    p90_diameter_um: float
    mean_area_um2: float
    total_area_um2: float
    diameters: List[float] = field(default_factory=list)
    areas_um2: List[float] = field(default_factory=list)


@dataclass
class AreaMethodResult:
    n_inside: int
    n_intersect: int
    n_equivalent: float
    n_a_per_mm2: float
    astm_g_value: float
    mean_grain_area_mm2: float
    mean_diameter_um: float


@dataclass
class InterceptMethodResult:
    n_lines_horizontal: int
    n_lines_vertical: int
    total_intersections: int
    total_line_length_um: float
    n_l_per_mm: float
    mean_intercept_length_um: float
    astm_g_value: float
    line_coords: List[tuple] = field(default_factory=list)   # for visualization
    intersection_points: List[tuple] = field(default_factory=list)


# ─────────────────────────── 特征提取 ──────────────────────────────

def extract_grain_props(labels: np.ndarray,
                        pixels_per_micron: float) -> List[GrainProps]:
    """
    使用 skimage.measure.regionprops 提取每个晶粒的几何特征。

    Args:
        labels: 晶粒标签图
        pixels_per_micron: 像素/微米

    Returns:
        GrainProps 列表
    """
    props_list = []
    px2um = 1.0 / pixels_per_micron
    px2_to_um2 = px2um ** 2

    for prop in measure.regionprops(labels):
        area_um2 = prop.area * px2_to_um2
        equiv_d_um = np.sqrt(4 * area_um2 / np.pi)

        major = prop.axis_major_length
        minor = prop.axis_minor_length
        aspect_ratio = major / minor if minor > 0 else 1.0

        perim = prop.perimeter
        circularity = (4 * np.pi * prop.area / (perim ** 2)) if perim > 0 else 0.0

        props_list.append(GrainProps(
            grain_id=prop.label,
            area_px=prop.area,
            area_um2=area_um2,
            perimeter_px=perim,
            equivalent_diameter_um=equiv_d_um,
            aspect_ratio=aspect_ratio,
            circularity=circularity,
            centroid=prop.centroid,
            bbox=prop.bbox,
        ))

    return props_list


def compute_grain_statistics(grain_props: List[GrainProps]) -> GrainStatistics:
    """从 GrainProps 列表计算汇总统计量。"""
    if not grain_props:
        return GrainStatistics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    diameters = np.array([g.equivalent_diameter_um for g in grain_props])
    areas = np.array([g.area_um2 for g in grain_props])

    return GrainStatistics(
        count=len(grain_props),
        mean_diameter_um=float(np.mean(diameters)),
        std_diameter_um=float(np.std(diameters)),
        min_diameter_um=float(np.min(diameters)),
        max_diameter_um=float(np.max(diameters)),
        median_diameter_um=float(np.median(diameters)),
        p10_diameter_um=float(np.percentile(diameters, 10)),
        p90_diameter_um=float(np.percentile(diameters, 90)),
        mean_area_um2=float(np.mean(areas)),
        total_area_um2=float(np.sum(areas)),
        diameters=diameters.tolist(),
        areas_um2=areas.tolist(),
    )


# ─────────────────────────── 面积法 ────────────────────────────────

def area_method(labels: np.ndarray,
                pixels_per_micron: float) -> AreaMethodResult:
    """
    Jeffries 平面测定法（ASTM E112 面积法）。

    测量区域默认为全图。统计完全在内部的晶粒（N_inside）和
    与边界相交的晶粒（N_intersect），计算等效晶粒数与 G 值。

    Args:
        labels: 晶粒标签图
        pixels_per_micron: 像素/微米

    Returns:
        AreaMethodResult
    """
    height, width = labels.shape
    px2_to_mm2 = (1.0 / (pixels_per_micron * 1000)) ** 2  # μm → mm: ÷1000

    # 测量区域：全图矩形
    area_px = height * width
    area_mm2 = area_px * px2_to_mm2

    # 边界像素（接触图像 4 条边的晶粒 ID）
    border_ids: set = set()
    border_ids.update(np.unique(labels[0, :]))
    border_ids.update(np.unique(labels[-1, :]))
    border_ids.update(np.unique(labels[:, 0]))
    border_ids.update(np.unique(labels[:, -1]))
    border_ids.discard(0)

    all_ids = set(np.unique(labels)) - {0}

    n_intersect = len(border_ids)
    n_inside = len(all_ids - border_ids)

    n_equivalent = n_inside + 0.5 * n_intersect
    n_a = n_equivalent / area_mm2  # grains per mm²

    # ASTM G = -3.322 × log₁₀(N_A) - 2.954
    g_value = -3.322 * np.log10(n_a) - 2.954 if n_a > 0 else 0.0

    mean_grain_area_mm2 = 1.0 / n_a if n_a > 0 else 0.0
    mean_diameter_um = np.sqrt(4 * mean_grain_area_mm2 / np.pi) * 1e3  # mm → μm

    return AreaMethodResult(
        n_inside=n_inside,
        n_intersect=n_intersect,
        n_equivalent=n_equivalent,
        n_a_per_mm2=n_a,
        astm_g_value=g_value,
        mean_grain_area_mm2=mean_grain_area_mm2,
        mean_diameter_um=mean_diameter_um,
    )


# ─────────────────────────── 截线法 ────────────────────────────────

def intercept_method(labels: np.ndarray,
                     pixels_per_micron: float,
                     n_lines_h: int = 5,
                     n_lines_v: int = 5) -> InterceptMethodResult:
    """
    Heyn 截距法（ASTM E112 截线法）。

    在图像上生成等间距水平/垂直网格线，统计与晶界的交点数，
    计算平均截距长度与 G 值。

    Args:
        labels: 晶粒标签图
        pixels_per_micron: 像素/微米
        n_lines_h: 水平线数量
        n_lines_v: 垂直线数量

    Returns:
        InterceptMethodResult
    """
    height, width = labels.shape
    boundaries = find_boundaries(labels, mode='thick')

    total_intersections = 0
    total_length_px = 0
    line_coords = []       # (orientation, position) for viz
    intersection_points = []

    # 水平线
    y_positions = np.linspace(int(height * 0.1), int(height * 0.9),
                               n_lines_h, dtype=int)
    for y in y_positions:
        row = boundaries[y, :]
        crossings = _count_crossings(row)
        pts = _crossing_positions_h(boundaries[y, :], y)
        total_intersections += crossings
        total_length_px += width
        line_coords.append(('h', int(y)))
        intersection_points.extend(pts)

    # 垂直线
    x_positions = np.linspace(int(width * 0.1), int(width * 0.9),
                               n_lines_v, dtype=int)
    for x in x_positions:
        col = boundaries[:, x]
        crossings = _count_crossings(col)
        pts = _crossing_positions_v(boundaries[:, x], x)
        total_intersections += crossings
        total_length_px += height
        line_coords.append(('v', int(x)))
        intersection_points.extend(pts)

    # 单位换算：像素 → μm → mm
    total_length_um = total_length_px / pixels_per_micron
    total_length_mm = total_length_um / 1000.0

    n_l_per_mm = total_intersections / total_length_mm if total_length_mm > 0 else 0.0
    mean_intercept_mm = 1.0 / n_l_per_mm if n_l_per_mm > 0 else 0.0
    mean_intercept_um = mean_intercept_mm * 1000.0

    # ASTM G = -6.6457 × log₁₀(l_mm) - 3.298
    g_value = (-6.6457 * np.log10(mean_intercept_mm) - 3.298
               if mean_intercept_mm > 0 else 0.0)

    return InterceptMethodResult(
        n_lines_horizontal=n_lines_h,
        n_lines_vertical=n_lines_v,
        total_intersections=total_intersections,
        total_line_length_um=total_length_um,
        n_l_per_mm=n_l_per_mm,
        mean_intercept_length_um=mean_intercept_um,
        astm_g_value=g_value,
        line_coords=line_coords,
        intersection_points=intersection_points,
    )


def _count_crossings(line: np.ndarray) -> int:
    """统计 1D 边界数组中的 0→1 跳变次数（晶界穿越数）。"""
    diff = np.diff(line.astype(np.int8))
    return int(np.sum(diff > 0))


def _crossing_positions_h(line: np.ndarray, y: int) -> List[tuple]:
    """返回水平线上晶界上升沿的 (row, col) 坐标，用于可视化。"""
    diff = np.diff(line.astype(np.int8))
    xs = np.where(diff > 0)[0] + 1
    return [(y, int(x)) for x in xs]


def _crossing_positions_v(col: np.ndarray, x: int) -> List[tuple]:
    """返回垂直线上晶界上升沿的 (row, col) 坐标，用于可视化。"""
    diff = np.diff(col.astype(np.int8))
    ys = np.where(diff > 0)[0] + 1
    return [(int(y), x) for y in ys]
