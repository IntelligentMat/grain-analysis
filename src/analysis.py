"""
analysis.py — 晶粒特征提取、面积法与截线法分析模块

包含：
  - regionprops 特征提取
  - Jeffries 平面测定法（面积法，矩形测量区）
  - Heyn 截距法（截线法，按晶粒段数 N 实现）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import skimage.draw as skdraw
from skimage import measure


# ─────────────────────────── 数据结构 ──────────────────────────────


@dataclass
class GrainProps:
    grain_id: int  # 晶粒标签 ID（与 labels 中的整数标签一致）
    area_um2: float  # 晶粒面积，单位 μm²
    perimeter_um: float  # 晶粒周长，单位 μm
    equivalent_diameter_um: float  # 等效圆直径，单位 μm
    aspect_ratio: float  # 长短轴比（major_axis/minor_axis）
    circularity: float  # 圆度 4*pi*A/P^2，越接近 1 越圆
    centroid_rc_px: tuple[float, float]  # 质心坐标 (row, col)，单位 px
    bbox_rc_px: tuple[int, int, int, int]  # 外接框 (min_row, min_col, max_row, max_col)，单位 px


@dataclass
class GrainStatistics:
    count: int = 0
    mean_diameter_um: float = 0.0
    std_diameter_um: float = 0.0
    min_diameter_um: float = 0.0
    max_diameter_um: float = 0.0
    median_diameter_um: float = 0.0
    p10_diameter_um: float = 0.0
    p90_diameter_um: float = 0.0
    mean_area_um2: float = 0.0
    total_area_um2: float = 0.0
    mean_aspect_ratio: float = 0.0
    mean_circularity: float = 0.0
    diameters_um: List[float] = field(default_factory=list)
    areas_um2: List[float] = field(default_factory=list)


@dataclass
class AreaMethodResult:
    n_inside: int
    n_intersect: int
    n_edge: int
    n_corner: int
    n_equivalent: float
    n_a_per_mm2: float
    astm_g_value: float
    mean_grain_area_mm2: float
    mean_diameter_um: float
    inside_grain_ids: List[int] = field(default_factory=list)
    edge_grain_ids: List[int] = field(default_factory=list)
    corner_grain_ids: List[int] = field(default_factory=list)
    intersect_grain_ids: List[int] = field(default_factory=list)


@dataclass
class InterceptMethodResult:
    n_lines: int
    n_circles: int
    total_intersections: float
    total_line_length_um: float
    n_l_per_mm: float
    mean_intercept_length_um: float
    astm_g_value: float
    pattern_elements: List[tuple] = field(default_factory=list)
    intersection_points: List[tuple] = field(default_factory=list)
    half_intersection_points: List[tuple] = field(default_factory=list)
    intersected_grain_ids: List[int] = field(default_factory=list)


GRAIN_PROPS_DTYPE = np.dtype(
    [
        ("grain_id", np.int32),
        ("area_um2", np.float64),
        ("perimeter_um", np.float64),
        ("equivalent_diameter_um", np.float64),
        ("aspect_ratio", np.float64),
        ("circularity", np.float64),
        ("centroid_row_px", np.float64),
        ("centroid_col_px", np.float64),
        ("bbox_min_row_px", np.int32),
        ("bbox_min_col_px", np.int32),
        ("bbox_max_row_px", np.int32),
        ("bbox_max_col_px", np.int32),
    ]
)


def _validate_pixels_per_micron(pixels_per_micron: float) -> float:
    """确保像素/微米换算系数有效。"""
    value = float(pixels_per_micron)
    if value <= 0:
        raise ValueError("pixels_per_micron must be > 0")
    return value


# ─────────────────────────── 特征提取 ──────────────────────────────


def extract_grain_props(labels: np.ndarray, pixels_per_micron: float) -> List[GrainProps]:
    """
    使用 skimage.measure.regionprops 提取每个晶粒的几何特征，并换算到物理单位。

    Args:
        labels: 晶粒标签图
        pixels_per_micron: 像素/微米换算系数

    Returns:
        GrainProps 列表
    """
    pixels_per_micron = _validate_pixels_per_micron(pixels_per_micron)
    props_list: List[GrainProps] = []
    for prop in measure.regionprops(labels):
        area_px2 = float(prop.area)
        equiv_d_px = float(np.sqrt(4 * area_px2 / np.pi))
        major = float(prop.axis_major_length)
        minor = float(prop.axis_minor_length)
        aspect_ratio = major / minor if minor > 0 else 1.0
        perimeter_px = float(prop.perimeter)
        circularity = (4 * np.pi * area_px2 / (perimeter_px**2)) if perimeter_px > 0 else 0.0

        props_list.append(
            GrainProps(
                grain_id=int(prop.label),
                area_um2=area_px2 / (pixels_per_micron**2),
                perimeter_um=perimeter_px / pixels_per_micron,
                equivalent_diameter_um=equiv_d_px / pixels_per_micron,
                aspect_ratio=float(aspect_ratio),
                circularity=float(circularity),
                centroid_rc_px=(float(prop.centroid[0]), float(prop.centroid[1])),
                bbox_rc_px=(int(prop.bbox[0]), int(prop.bbox[1]), int(prop.bbox[2]), int(prop.bbox[3])),
            )
        )

    return props_list


def grain_props_to_structured_array(grain_props: List[GrainProps]) -> np.ndarray:
    array = np.zeros(len(grain_props), dtype=GRAIN_PROPS_DTYPE)
    for index, grain in enumerate(grain_props):
        min_row, min_col, max_row, max_col = grain.bbox_rc_px
        row, col = grain.centroid_rc_px
        array[index] = (
            grain.grain_id,
            grain.area_um2,
            grain.perimeter_um,
            grain.equivalent_diameter_um,
            grain.aspect_ratio,
            grain.circularity,
            row,
            col,
            min_row,
            min_col,
            max_row,
            max_col,
        )
    return array


def compute_grain_statistics(grain_props: List[GrainProps]) -> GrainStatistics:
    """从 GrainProps 列表计算汇总统计量。"""
    if not grain_props:
        return GrainStatistics()

    diameters = np.array([g.equivalent_diameter_um for g in grain_props], dtype=np.float64)
    areas = np.array([g.area_um2 for g in grain_props], dtype=np.float64)
    aspect_ratios = np.array([g.aspect_ratio for g in grain_props], dtype=np.float64)
    circularities = np.array([g.circularity for g in grain_props], dtype=np.float64)

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
        mean_aspect_ratio=float(np.mean(aspect_ratios)),
        mean_circularity=float(np.mean(circularities)),
        diameters_um=diameters.tolist(),
        areas_um2=areas.tolist(),
    )


# ─────────────────────────── 面积法 ────────────────────────────────


def area_method(labels: np.ndarray, pixels_per_micron: float = 1.0) -> AreaMethodResult:
    """
    Jeffries 平面测定法（ASTM E112 面积法）。

    测量区域默认为全图矩形。完全在内部的晶粒计 1；
    接触边界但不属于角部晶粒的计 1/2；
    同时接触两条相邻边的角部晶粒计 1/4。
    """
    pixels_per_micron = _validate_pixels_per_micron(pixels_per_micron)
    height, width = labels.shape

    area_px = height * width
    top_ids = {int(v) for v in np.unique(labels[0, :]) if v != 0}
    bottom_ids = {int(v) for v in np.unique(labels[-1, :]) if v != 0}
    left_ids = {int(v) for v in np.unique(labels[:, 0]) if v != 0}
    right_ids = {int(v) for v in np.unique(labels[:, -1]) if v != 0}

    border_ids = top_ids | bottom_ids | left_ids | right_ids
    corner_ids = (
        (top_ids & left_ids)
        | (top_ids & right_ids)
        | (bottom_ids & left_ids)
        | (bottom_ids & right_ids)
    )

    all_ids = {int(v) for v in np.unique(labels) if v != 0}
    inside_ids = sorted(all_ids - border_ids)
    corner_grain_ids = sorted(corner_ids)
    edge_grain_ids = sorted(border_ids - corner_ids)
    intersect_ids = sorted(border_ids)

    n_inside = len(inside_ids)
    n_edge = len(edge_grain_ids)
    n_corner = len(corner_grain_ids)
    n_intersect = len(intersect_ids)

    n_equivalent = n_inside + 0.5 * n_edge + 0.25 * n_corner
    area_um2 = area_px / (pixels_per_micron**2)
    area_mm2 = area_um2 / 1e6
    n_a = n_equivalent / area_mm2 if area_mm2 > 0 else 0.0
    g_value = 3.322 * np.log10(n_a) - 2.954 if n_a > 0 else 0.0

    mean_grain_area_mm2 = 1.0 / n_a if n_a > 0 else 0.0
    mean_grain_area_um2 = mean_grain_area_mm2 * 1e6
    mean_diameter_um = np.sqrt(4 * mean_grain_area_um2 / np.pi) if n_a > 0 else 0.0

    return AreaMethodResult(
        n_inside=n_inside,
        n_intersect=n_intersect,
        n_edge=n_edge,
        n_corner=n_corner,
        n_equivalent=n_equivalent,
        n_a_per_mm2=n_a,
        astm_g_value=g_value,
        mean_grain_area_mm2=mean_grain_area_mm2,
        mean_diameter_um=float(mean_diameter_um),
        inside_grain_ids=inside_ids,
        edge_grain_ids=edge_grain_ids,
        corner_grain_ids=corner_grain_ids,
        intersect_grain_ids=intersect_ids,
    )


# ─────────────────────────── 截线法 ────────────────────────────────


def intercept_method(
    labels: np.ndarray,
    pixels_per_micron: float = 1.0,
    min_intercept_px: int = 3,
    margin_ratio: float = 0.05,
) -> InterceptMethodResult:
    """
    Heyn 截距法 — 按 ASTM E112 标准图样估计平均截距长度与晶粒度等级。
    """
    pixels_per_micron = _validate_pixels_per_micron(pixels_per_micron)
    min_intercept_px = max(1, int(min_intercept_px))

    height, width = labels.shape
    size = min(height, width)
    r_c = height // 2
    c_c = width // 2
    margin = max(1, int(margin_ratio * size))

    total_intersections = 0.0
    total_length_px = 0.0
    pattern_elements: List[tuple] = []
    intersection_points: List[tuple] = []
    half_intersection_points: List[tuple] = []
    intersected_grain_ids: set[int] = set()

    line_defs = [
        (height - 1 - margin, margin, height - 1 - margin, width - 1 - margin),
        (margin, margin, height - 1 - margin, margin),
        (margin, margin, height - 1 - margin, width - 1 - margin),
        (height - 1 - margin, margin, margin, width - 1 - margin),
    ]
    for r1, c1, r2, c2 in line_defs:
        rr, cc = skdraw.line(r1, c1, r2, c2)
        mask = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[mask], cc[mask]
        line_labels = labels[rr, cc]

        intersected_grain_ids.update(
            _intercepted_grain_ids_on_path(line_labels, min_intercept_px, is_closed=False)
        )
        total_intersections += _count_intercepts(line_labels, min_intercept_px, is_closed=False)
        intersection_points += _intercept_positions_on_path(
            line_labels, rr, cc, min_intercept_px, is_closed=False
        )
        half_intersection_points += _half_intercept_positions_on_path(
            line_labels, rr, cc, min_intercept_px, is_closed=False
        )
        total_length_px += float(np.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2))
        pattern_elements.append(("line", r1, c1, r2, c2))

    astm_radii_ratios = [0.7958, 0.5305, 0.2653]
    for ratio in astm_radii_ratios:
        radius = int(round(ratio * size / 2))
        if radius <= 0:
            continue

        rr, cc = skdraw.circle_perimeter(r_c, c_c, radius, shape=labels.shape)
        rr, cc = _sort_circle_pixels(rr, cc, r_c, c_c)
        circ_labels = labels[rr, cc]
        intersected_grain_ids.update(
            _intercepted_grain_ids_on_path(circ_labels, min_intercept_px, is_closed=True)
        )
        total_intersections += _count_intercepts(circ_labels, min_intercept_px, is_closed=True)
        intersection_points += _intercept_positions_on_path(
            circ_labels, rr, cc, min_intercept_px, is_closed=True
        )
        total_length_px += 2.0 * np.pi * radius
        pattern_elements.append(("circle", r_c, c_c, radius))

    n_l_per_px = total_intersections / total_length_px if total_length_px > 0 else 0.0
    mean_intercept_px = 1.0 / n_l_per_px if n_l_per_px > 0 else 0.0
    total_line_length_um = total_length_px / pixels_per_micron
    total_line_length_mm = total_line_length_um / 1000.0
    n_l_per_mm = total_intersections / total_line_length_mm if total_line_length_mm > 0 else 0.0
    mean_intercept_length_um = mean_intercept_px / pixels_per_micron
    mean_intercept_mm = mean_intercept_length_um / 1000.0
    g_value = -6.6457 * np.log10(mean_intercept_mm) - 3.298 if mean_intercept_mm > 0 else 0.0

    return InterceptMethodResult(
        n_lines=4,
        n_circles=3,
        total_intersections=total_intersections,
        total_line_length_um=total_line_length_um,
        n_l_per_mm=n_l_per_mm,
        mean_intercept_length_um=mean_intercept_length_um,
        astm_g_value=g_value,
        pattern_elements=pattern_elements,
        intersection_points=intersection_points,
        half_intersection_points=half_intersection_points,
        intersected_grain_ids=sorted(intersected_grain_ids),
    )


def _get_grain_segments(line_labels: np.ndarray) -> list:
    """从一维标签序列中提取连续晶粒段。"""
    segments = []
    prev = 0
    seg_start = 0
    for index, label in enumerate(line_labels):
        if label == 0:
            if prev != 0:
                segments.append((seg_start, index, int(prev)))
            prev = 0
        else:
            if prev == 0:
                seg_start = index
            elif label != prev:
                segments.append((seg_start, index, int(prev)))
                seg_start = index
            prev = label
    if prev != 0:
        segments.append((seg_start, len(line_labels), int(prev)))
    return segments


def _valid_grain_segment_groups(
    line_labels: np.ndarray, min_intercept_px: int = 3, is_closed: bool = False
) -> list[list[tuple[int, int, int]]]:
    segs = _get_grain_segments(line_labels)
    groups: list[list[tuple[int, int, int]]] = [
        [seg] for seg in segs if (seg[1] - seg[0]) >= min_intercept_px
    ]
    if is_closed and len(groups) > 1 and groups[0][0][2] == groups[-1][0][2]:
        groups = [[groups[-1][0], groups[0][0]]] + groups[1:-1]
    return groups


def _count_intercepts(
    line_labels: np.ndarray, min_intercept_px: int = 3, is_closed: bool = False
) -> float:
    groups = _valid_grain_segment_groups(line_labels, min_intercept_px, is_closed=is_closed)
    path_len = len(line_labels)
    total = 0.0
    for group in groups:
        if is_closed:
            total += 1.0
            continue

        touches_start = any(seg[0] == 0 for seg in group)
        touches_end = any(seg[1] == path_len for seg in group)
        if touches_start ^ touches_end:
            total += 0.5
        else:
            total += 1.0
    return total


def _intercepted_grain_ids_on_path(
    line_labels: np.ndarray, min_intercept_px: int = 3, is_closed: bool = False
) -> set[int]:
    groups = _valid_grain_segment_groups(line_labels, min_intercept_px, is_closed=is_closed)
    return {int(seg[2]) for group in groups for seg in group}


def _sort_circle_pixels(rr: np.ndarray, cc: np.ndarray, r_center: int, c_center: int):
    angles = np.arctan2(rr - r_center, cc - c_center)
    order = np.argsort(angles)
    return rr[order], cc[order]


def _intercept_positions_on_path(
    path_labels: np.ndarray,
    path_r: np.ndarray,
    path_c: np.ndarray,
    min_intercept_px: int = 3,
    is_closed: bool = False,
) -> List[tuple]:
    groups = _valid_grain_segment_groups(path_labels, min_intercept_px, is_closed=is_closed)
    path_len = len(path_labels)
    points: List[tuple] = []
    for group in groups:
        representative = max(group, key=lambda seg: seg[1] - seg[0])
        touches_start = any(seg[0] == 0 for seg in group)
        touches_end = any(seg[1] == path_len for seg in group)

        if not is_closed and touches_start and not touches_end:
            idx = 0
        elif not is_closed and touches_end and not touches_start:
            idx = path_len - 1
        else:
            idx = representative[0] + (representative[1] - representative[0] - 1) // 2

        points.append((int(path_r[idx]), int(path_c[idx])))
    return points


def _half_intercept_positions_on_path(
    path_labels: np.ndarray,
    path_r: np.ndarray,
    path_c: np.ndarray,
    min_intercept_px: int = 3,
    is_closed: bool = False,
) -> List[tuple]:
    if is_closed:
        return []

    groups = _valid_grain_segment_groups(path_labels, min_intercept_px, is_closed=is_closed)
    path_len = len(path_labels)
    points: List[tuple] = []
    for group in groups:
        touches_start = any(seg[0] == 0 for seg in group)
        touches_end = any(seg[1] == path_len for seg in group)
        if not (touches_start ^ touches_end):
            continue

        idx = 0 if touches_start else path_len - 1
        points.append((int(path_r[idx]), int(path_c[idx])))
    return points
