"""
analysis.py — 晶粒特征提取、面积法与截线法分析模块

包含：
  - regionprops 特征提取
  - Jeffries 平面测定法（面积法，矩形测量区）
  - Heyn 截距法（截线法，按晶粒段数 N 实现）
"""

import numpy as np
from skimage import measure
import skimage.draw as skdraw
from dataclasses import dataclass, field
from typing import List, Dict, Any


# ─────────────────────────── 数据结构 ──────────────────────────────

@dataclass
class GrainProps:
    grain_id: int                 # 晶粒标签 ID（与 labels 中的整数标签一致）
    area_px: float                # 晶粒像素数（regionprops.area）
    area_px2: float               # 晶粒面积，单位 px^2（当前与 area_px 数值相同）
    perimeter_px: float           # 晶粒周长，单位 px
    equivalent_diameter_px: float # 等效圆直径，单位 px，公式 sqrt(4A/pi)
    aspect_ratio: float           # 长短轴比（major_axis/minor_axis）
    circularity: float            # 圆度 4*pi*A/P^2，越接近 1 越圆
    centroid: tuple               # 质心坐标 (row, col)
    bbox: tuple                   # 外接框 (min_row, min_col, max_row, max_col)，右开边界


@dataclass
class GrainStatistics:
    count: int                    # 晶粒总数
    mean_diameter_px: float       # 等效直径均值（px）
    std_diameter_px: float        # 等效直径标准差（px）
    min_diameter_px: float        # 等效直径最小值（px）
    max_diameter_px: float        # 等效直径最大值（px）
    median_diameter_px: float     # 等效直径中位数（px）
    p10_diameter_px: float        # 等效直径 10 分位（px）
    p90_diameter_px: float        # 等效直径 90 分位（px）
    mean_area_px2: float          # 晶粒面积均值（px^2）
    total_area_px2: float         # 晶粒面积总和（px^2）
    diameters: List[float] = field(default_factory=list)  # 所有晶粒等效直径列表（px）
    areas_px2: List[float] = field(default_factory=list)  # 所有晶粒面积列表（px^2）


@dataclass
class AreaMethodResult:
    n_inside: int                 # 完全落在测量区域内部的晶粒数
    n_intersect: int              # 与测量区域边界接触的晶粒总数（edge + corner）
    n_edge: int                   # 接触边界但不含角点的晶粒数（权重 1/2）
    n_corner: int                 # 包含任一角点像素的晶粒数（权重 1/4）
    n_equivalent: float           # 等效晶粒数 N_eq = N_inside + 0.5*N_edge + 0.25*N_corner
    n_a_per_mm2: float            # 面密度 N_A（grains/mm²）
    astm_g_value: float           # ASTM E112 对应的 G 值
    mean_grain_area_mm2: float    # 等效平均晶粒面积（mm²）
    mean_diameter_um: float       # 面积法推导的平均等效直径（μm）
    inside_grain_ids: List[int] = field(default_factory=list)
    edge_grain_ids: List[int] = field(default_factory=list)
    corner_grain_ids: List[int] = field(default_factory=list)
    intersect_grain_ids: List[int] = field(default_factory=list)


@dataclass
class InterceptMethodResult:
    n_lines: int                  # 测试线数量（ASTM E112：4）
    n_circles: int                # 同心圆数量（ASTM E112：3）
    total_intersections: float    # 总截距数/晶粒段数（开放路径端段按 0.5）
    total_line_length_px: float   # 总测试路径长度（线长 + 圆周长）（px）
    n_l_per_px: float             # 截距数密度 N_L（intercepts/px）
    mean_intercept_length_px: float  # 平均截距长度 l_bar（px）
    mean_intercept_length_um: float  # 平均截距长度 l_bar（μm）
    astm_g_value: float           # ASTM E112 对应的 G 值（基于物理单位 mm）
    pattern_elements: List[tuple] = field(default_factory=list)
    # ('line', r1,c1,r2,c2) 或 ('circle', r_c,c_c,radius)，用于可视化
    intersection_points: List[tuple] = field(default_factory=list)  # 有效晶粒段代表点（用于可视化）
    half_intersection_points: List[tuple] = field(default_factory=list)  # 开放路径端段（计 0.5）的代表点
    intersected_grain_ids: List[int] = field(default_factory=list)


def _validate_pixels_per_micron(pixels_per_micron: float) -> float:
    """确保像素/微米换算系数有效。"""
    value = float(pixels_per_micron)
    if value <= 0:
        raise ValueError("pixels_per_micron must be > 0")
    return value


# ─────────────────────────── 特征提取 ──────────────────────────────

def extract_grain_props(labels: np.ndarray) -> List[GrainProps]:
    """
    使用 skimage.measure.regionprops 提取每个晶粒的几何特征。

    Args:
        labels: 晶粒标签图

    Returns:
        GrainProps 列表
    """
    props_list = []
    for prop in measure.regionprops(labels):
        # 临时按像素尺度计算，取消物理尺寸换算
        area_px2 = float(prop.area)
        equiv_d_px = float(np.sqrt(4 * area_px2 / np.pi))

        major = prop.axis_major_length
        minor = prop.axis_minor_length
        aspect_ratio = major / minor if minor > 0 else 1.0

        perim = prop.perimeter
        circularity = (4 * np.pi * prop.area / (perim ** 2)) if perim > 0 else 0.0

        props_list.append(GrainProps(
            grain_id=prop.label,
            area_px=prop.area,
            area_px2=area_px2,
            perimeter_px=perim,
            equivalent_diameter_px=equiv_d_px,
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

    diameters = np.array([g.equivalent_diameter_px for g in grain_props])
    areas = np.array([g.area_px2 for g in grain_props])

    return GrainStatistics(
        count=len(grain_props),
        mean_diameter_px=float(np.mean(diameters)),
        std_diameter_px=float(np.std(diameters)),
        min_diameter_px=float(np.min(diameters)),
        max_diameter_px=float(np.max(diameters)),
        median_diameter_px=float(np.median(diameters)),
        p10_diameter_px=float(np.percentile(diameters, 10)),
        p90_diameter_px=float(np.percentile(diameters, 90)),
        mean_area_px2=float(np.mean(areas)),
        total_area_px2=float(np.sum(areas)),
        diameters=diameters.tolist(),
        areas_px2=areas.tolist(),
    )


# ─────────────────────────── 面积法 ────────────────────────────────

def area_method(labels: np.ndarray,
                pixels_per_micron: float = 1.0) -> AreaMethodResult:
    """
    Jeffries 平面测定法（ASTM E112 面积法）。

    测量区域默认为全图矩形。完全在内部的晶粒计 1；
    接触边界但不含角点的晶粒计 1/2；
    包含任一角点像素的晶粒计 1/4。

    Args:
        labels: 晶粒标签图
        pixels_per_micron: 像素/微米换算系数（用于面积单位转换）

    Returns:
        AreaMethodResult
    """
    pixels_per_micron = _validate_pixels_per_micron(pixels_per_micron)
    height, width = labels.shape

    # 测量区域：全图矩形（按像素）
    area_px = height * width

    # 边界像素（接触图像 4 条边的晶粒 ID）
    border_ids: set[int] = set()
    border_ids.update(int(v) for v in np.unique(labels[0, :]) if v != 0)
    border_ids.update(int(v) for v in np.unique(labels[-1, :]) if v != 0)
    border_ids.update(int(v) for v in np.unique(labels[:, 0]) if v != 0)
    border_ids.update(int(v) for v in np.unique(labels[:, -1]) if v != 0)

    corner_ids = {
        int(labels[0, 0]),
        int(labels[0, -1]),
        int(labels[-1, 0]),
        int(labels[-1, -1]),
    } - {0}

    all_ids = {int(v) for v in np.unique(labels) if v != 0}

    inside_ids = sorted(all_ids - border_ids)
    corner_grain_ids = sorted(corner_ids)
    edge_grain_ids = sorted(border_ids - corner_ids)
    intersect_ids = sorted(border_ids)

    n_intersect = len(intersect_ids)
    n_inside = len(inside_ids)
    n_edge = len(edge_grain_ids)
    n_corner = len(corner_grain_ids)

    n_equivalent = n_inside + 0.5 * n_edge + 0.25 * n_corner
    area_um2 = area_px / (pixels_per_micron ** 2)           # px² → μm²
    area_mm2 = area_um2 / 1e6                               # μm² → mm²
    n_a = n_equivalent / area_mm2 if area_mm2 > 0 else 0.0  # grains/mm²

    # ASTM G 与 N_A 正相关（N_A 越大，晶粒越细，G 越大）
    g_value = 3.322 * np.log10(n_a) - 2.954 if n_a > 0 else 0.0

    mean_grain_area_mm2 = 1.0 / n_a if n_a > 0 else 0.0  # 这是 1 / N_A 推导的等效平均面积，不是逐晶粒实测均值
    mean_grain_area_um2 = mean_grain_area_mm2 * 1e6
    mean_diameter_um = np.sqrt(4 * mean_grain_area_um2 / np.pi)

    return AreaMethodResult(
        n_inside=n_inside,
        n_intersect=n_intersect,
        n_edge=n_edge,
        n_corner=n_corner,
        n_equivalent=n_equivalent,
        n_a_per_mm2=n_a,
        astm_g_value=g_value,
        mean_grain_area_mm2=mean_grain_area_mm2,
        mean_diameter_um=mean_diameter_um,
        inside_grain_ids=inside_ids,
        edge_grain_ids=edge_grain_ids,
        corner_grain_ids=corner_grain_ids,
        intersect_grain_ids=intersect_ids,
    )


# ─────────────────────────── 截线法 ────────────────────────────────

def intercept_method(labels: np.ndarray,
                     pixels_per_micron: float = 1.0,
                     min_intercept_px: int = 3,
                     margin_ratio: float = 0.05) -> InterceptMethodResult:
    """
    Heyn 截距法 — ASTM E112 图样：4 条测试线 + 3 个同心圆。

    测试线：底部水平、左侧垂直，以及两条对角线。
    同心圆：圆心为图像中心，半径比例 0.7958 / 0.5305 / 0.2653 × min(H,W)/2，
            来自 ASTM E112 附录，使三圆周长之和等于标准测量长度。
    该实现按测试路径覆盖到的有效连续晶粒段数 N 统计截距，不直接统计晶界交点 P。

    Args:
        labels:            晶粒标签图
        pixels_per_micron: 像素/微米换算系数（如 2.25 px/μm）
        min_intercept_px:  最小有效晶粒段像素数，过滤过短伪截距
        margin_ratio:      测试线距图像边缘的留白比例（默认 5%）

    Returns:
        InterceptMethodResult
    """
    pixels_per_micron = _validate_pixels_per_micron(pixels_per_micron)
    min_intercept_px = max(1, int(min_intercept_px))
    height, width = labels.shape
    size = min(height, width)          # 以较短边为基准（保证圆不超出图像）
    r_c = height // 2
    c_c = width // 2
    margin = max(1, int(margin_ratio * size))

    total_intersections = 0
    total_length_px = 0.0
    pattern_elements: List[tuple] = []
    intersection_points: List[tuple] = []
    half_intersection_points: List[tuple] = []
    intersected_grain_ids: set[int] = set()

    # ── 4 条测试线 ────────────────────────────────────────────────────
    line_defs = [
        (height - 1 - margin, margin,   height - 1 - margin, width - 1 - margin),  # 水平（底部）
        (margin,              margin,   height - 1 - margin,  margin),              # 垂直（左侧）
        (margin,       margin,        height - 1 - margin, width - 1 - margin),     # 对角 ↘
        (height - 1 - margin, margin, margin,           width - 1 - margin),        # 对角 ↗
    ]
    for r1, c1, r2, c2 in line_defs:
        rr, cc = skdraw.line(r1, c1, r2, c2)
        mask = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[mask], cc[mask]
        line_labels = labels[rr, cc]
        intersected_grain_ids.update(
            _intercepted_grain_ids_on_path(line_labels, min_intercept_px, is_closed=False)
        )

        total_intersections += _count_intercepts(line_labels, min_intercept_px,
                                                 is_closed=False)
        intersection_points += _intercept_positions_on_path(
            line_labels, rr, cc, min_intercept_px, is_closed=False)
        half_intersection_points += _half_intercept_positions_on_path(
            line_labels, rr, cc, min_intercept_px, is_closed=False)
        total_length_px += float(np.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2))
        pattern_elements.append(('line', r1, c1, r2, c2))

    # ── 3 个同心圆（ASTM E112 标准半径比例）─────────────────────────
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

        total_intersections += _count_intercepts(circ_labels, min_intercept_px,
                                                 is_closed=True)
        intersection_points += _intercept_positions_on_path(
            circ_labels, rr, cc, min_intercept_px, is_closed=True)
        total_length_px += 2.0 * np.pi * radius
        pattern_elements.append(('circle', r_c, c_c, radius))

    # ── 计算 N_L、平均截距、G 值 ──────────────────────────────────────
    n_l_per_px = total_intersections / total_length_px if total_length_px > 0 else 0.0
    mean_intercept_px = 1.0 / n_l_per_px if n_l_per_px > 0 else 0.0
    mean_intercept_um = mean_intercept_px / pixels_per_micron
    mean_intercept_mm = mean_intercept_um / 1000.0
    g_value = (-6.6457 * np.log10(mean_intercept_mm) - 3.298
               if mean_intercept_mm > 0 else 0.0)

    return InterceptMethodResult(
        n_lines=4,
        n_circles=3,
        total_intersections=total_intersections,
        total_line_length_px=total_length_px,
        n_l_per_px=n_l_per_px,
        mean_intercept_length_px=mean_intercept_px,
        mean_intercept_length_um=mean_intercept_um,
        astm_g_value=g_value,
        pattern_elements=pattern_elements,
        intersection_points=intersection_points,
        half_intersection_points=half_intersection_points,
        intersected_grain_ids=sorted(intersected_grain_ids),
    )


def _get_grain_segments(line_labels: np.ndarray) -> list:
    """从标签序列中提取连续晶粒段。

    返回 [(start_px, end_px, grain_id), ...] 列表，end_px 为不含端点索引。
    背景像素（label=0）作为晶界分隔符，相邻不同非零标签直接视为段切换。
    """
    segments = []
    prev = 0
    seg_start = 0
    for i, lbl in enumerate(line_labels):
        if lbl == 0:
            if prev != 0:
                segments.append((seg_start, i, int(prev)))
            prev = 0
        else:
            if prev == 0:
                seg_start = i
            elif lbl != prev:
                segments.append((seg_start, i, int(prev)))
                seg_start = i
            prev = lbl
    if prev != 0:
        segments.append((seg_start, len(line_labels), int(prev)))
    return segments


def _valid_grain_segment_groups(line_labels: np.ndarray,
                                min_intercept_px: int = 3,
                                is_closed: bool = False) -> list[list[tuple[int, int, int]]]:
    """返回有效晶粒段分组；闭合路径会合并首尾同晶粒段。"""
    segs = _get_grain_segments(line_labels)
    groups: list[list[tuple[int, int, int]]] = [
        [seg] for seg in segs if (seg[1] - seg[0]) >= min_intercept_px
    ]
    if is_closed and len(groups) > 1 and groups[0][0][2] == groups[-1][0][2]:
        groups = [[groups[-1][0], groups[0][0]]] + groups[1:-1]
    return groups


def _count_intercepts(line_labels: np.ndarray,
                      min_intercept_px: int = 3,
                      is_closed: bool = False) -> float:
    """统计测试路径覆盖到的有效连续晶粒段数 N。"""
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


def _intercepted_grain_ids_on_path(line_labels: np.ndarray,
                                   min_intercept_px: int = 3,
                                   is_closed: bool = False) -> set[int]:
    """返回在该路径上形成有效晶粒段的晶粒 ID。"""
    groups = _valid_grain_segment_groups(line_labels, min_intercept_px, is_closed=is_closed)
    return {int(seg[2]) for group in groups for seg in group}


def _sort_circle_pixels(rr: np.ndarray, cc: np.ndarray,
                        r_center: int, c_center: int):
    """按角度排序圆周像素，保证沿圆周遍历的连续性。"""
    angles = np.arctan2(rr - r_center, cc - c_center)
    order = np.argsort(angles)
    return rr[order], cc[order]


def _intercept_positions_on_path(path_labels: np.ndarray,
                                 path_r: np.ndarray,
                                 path_c: np.ndarray,
                                 min_intercept_px: int = 3,
                                 is_closed: bool = False) -> List[tuple]:
    """返回任意路径上有效晶粒段的代表点 (row, col) 坐标。"""
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


def _half_intercept_positions_on_path(path_labels: np.ndarray,
                                      path_r: np.ndarray,
                                      path_c: np.ndarray,
                                      min_intercept_px: int = 3,
                                      is_closed: bool = False) -> List[tuple]:
    """返回开放路径上按 0.5 计权的端段代表点 (row, col)。"""
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
