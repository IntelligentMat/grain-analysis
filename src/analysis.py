"""
analysis.py — 晶粒特征提取、面积法与截线法分析模块

包含：
  - regionprops 特征提取
  - Jeffries 平面测定法（面积法，ASTM E112）
  - Heyn 截距法（截线法，ASTM E112）
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
    bbox: tuple                   # 外接框 (min_row, min_col, max_row, max_col)


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
    n_intersect: int              # 与测量区域边界相交的晶粒数
    n_equivalent: float           # 等效晶粒数 N_eq = N_inside + 0.5*N_intersect
    n_a_per_mm2: float            # 面密度 N_A（grains/mm²）
    astm_g_value: float           # ASTM E112 对应的 G 值
    mean_grain_area_mm2: float    # 平均晶粒面积（mm²）
    mean_diameter_um: float       # 面积法推导的平均等效直径（μm）
    inside_grain_ids: List[int] = field(default_factory=list)
    intersect_grain_ids: List[int] = field(default_factory=list)


@dataclass
class InterceptMethodResult:
    n_lines: int                  # 测试线数量（ASTM E112：4）
    n_circles: int                # 同心圆数量（ASTM E112：3）
    total_intersections: float    # 总交点数（含端点 0.5 修正）
    total_line_length_px: float   # 总测试路径长度（线长 + 圆周长）（px）
    n_l_per_px: float             # 线密度 N_L（intersections/px）
    mean_intercept_length_px: float  # 平均截距长度 l_bar（px）
    mean_intercept_length_um: float  # 平均截距长度 l_bar（μm）
    astm_g_value: float           # ASTM E112 对应的 G 值（基于物理单位 mm）
    pattern_elements: List[tuple] = field(default_factory=list)
    # ('line', r1,c1,r2,c2) 或 ('circle', r_c,c_c,radius)，用于可视化
    intersection_points: List[tuple] = field(default_factory=list)  # 交点坐标（用于可视化）
    intersected_grain_ids: List[int] = field(default_factory=list)


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

    测量区域默认为全图。统计完全在内部的晶粒（N_inside）和
    与边界相交的晶粒（N_intersect），计算等效晶粒数与 G 值。

    Args:
        labels: 晶粒标签图
        pixels_per_micron: 像素/微米换算系数（用于面积单位转换）

    Returns:
        AreaMethodResult
    """
    height, width = labels.shape

    # 测量区域：全图矩形（按像素）
    area_px = height * width

    # 边界像素（接触图像 4 条边的晶粒 ID）
    border_ids: set = set()
    border_ids.update(np.unique(labels[0, :]))
    border_ids.update(np.unique(labels[-1, :]))
    border_ids.update(np.unique(labels[:, 0]))
    border_ids.update(np.unique(labels[:, -1]))
    border_ids.discard(0)

    all_ids = set(np.unique(labels)) - {0}

    inside_ids = sorted(all_ids - border_ids)
    intersect_ids = sorted(border_ids)

    n_intersect = len(intersect_ids)
    n_inside = len(inside_ids)

    n_equivalent = n_inside + 0.5 * n_intersect
    area_um2 = area_px / (pixels_per_micron ** 2)           # px² → μm²
    area_mm2 = area_um2 / 1e6                               # μm² → mm²
    n_a = n_equivalent / area_mm2 if area_mm2 > 0 else 0.0  # grains/mm²

    # ASTM G 与 N_A 正相关（N_A 越大，晶粒越细，G 越大）
    g_value = 3.322 * np.log10(n_a) - 2.954 if n_a > 0 else 0.0

    mean_grain_area_mm2 = 1.0 / n_a if n_a > 0 else 0.0
    mean_grain_area_um2 = mean_grain_area_mm2 * 1e6
    mean_diameter_um = np.sqrt(4 * mean_grain_area_um2 / np.pi)

    return AreaMethodResult(
        n_inside=n_inside,
        n_intersect=n_intersect,
        n_equivalent=n_equivalent,
        n_a_per_mm2=n_a,
        astm_g_value=g_value,
        mean_grain_area_mm2=mean_grain_area_mm2,
        mean_diameter_um=mean_diameter_um,
        inside_grain_ids=inside_ids,
        intersect_grain_ids=intersect_ids,
    )


# ─────────────────────────── 截线法 ────────────────────────────────

def intercept_method(labels: np.ndarray,
                     pixels_per_micron: float = 1.0,
                     min_intercept_px: int = 3,
                     margin_ratio: float = 0.05) -> InterceptMethodResult:
    """
    Heyn 截距法 — ASTM E112 标准图案：4 条测试线 + 3 个同心圆。

    测试线：水平（图像中心行）、垂直（图像中心列）、左上→右下对角、右上→左下对角。
    同心圆：圆心为图像中心，半径比例 0.7958 / 0.5305 / 0.2653 × min(H,W)/2，
            来自 ASTM E112 附录，使三圆周长之和等于标准测量长度。

    Args:
        labels:            晶粒标签图
        pixels_per_micron: 像素/微米换算系数（如 2.25 px/μm）
        min_intercept_px:  最小有效截距像素数，过滤掠角产生的伪截距
        margin_ratio:      测试线距图像边缘的留白比例（默认 5%）

    Returns:
        InterceptMethodResult
    """
    height, width = labels.shape
    size = min(height, width)          # 以较短边为基准（保证圆不超出图像）
    r_c = height // 2
    c_c = width // 2
    margin = max(1, int(margin_ratio * size))

    total_intersections = 0
    total_length_px = 0.0
    pattern_elements: List[tuple] = []
    intersection_points: List[tuple] = []
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
        intersected_grain_ids.update(int(lbl) for lbl in np.unique(line_labels) if lbl != 0)

        total_intersections += _count_crossings(line_labels, min_intercept_px,
                                                 is_closed=False)
        intersection_points += _crossing_positions_on_path(
            line_labels, rr, cc, min_intercept_px)
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
        intersected_grain_ids.update(int(lbl) for lbl in np.unique(circ_labels) if lbl != 0)

        total_intersections += _count_crossings(circ_labels, min_intercept_px,
                                                 is_closed=True)
        intersection_points += _crossing_positions_on_path(
            circ_labels, rr, cc, min_intercept_px)
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


def _count_crossings(line_labels: np.ndarray,
                     min_intercept_px: int = 3,
                     is_closed: bool = False) -> float:
    """统计测试线穿越晶界的次数（含 ASTM E112 端点修正）。

    - 相邻有效段之间的晶粒切换计为 1 次交点
    - 对于开放路径（直线），若路径端点落在晶粒内部（而非背景），
      各计 0.5 次交点（ASTM E112 端点修正：N = P + 0.5 × N_ends）
    - 对于封闭路径（圆），无端点
    """
    segs = _get_grain_segments(line_labels)
    valid = [s for s in segs if (s[1] - s[0]) >= min_intercept_px]
    count = float(sum(1 for i in range(1, len(valid)) if valid[i][2] != valid[i-1][2]))

    if not is_closed and valid:
        if valid[0][0] == 0:                        # 起点落在晶粒内
            count += 0.5
        if valid[-1][1] == len(line_labels):        # 终点落在晶粒内
            count += 0.5

    return count


def _sort_circle_pixels(rr: np.ndarray, cc: np.ndarray,
                        r_center: int, c_center: int):
    """按角度排序圆周像素，保证沿圆周遍历的连续性。"""
    angles = np.arctan2(rr - r_center, cc - c_center)
    order = np.argsort(angles)
    return rr[order], cc[order]


def _crossing_positions_on_path(path_labels: np.ndarray,
                                path_r: np.ndarray,
                                path_c: np.ndarray,
                                min_intercept_px: int = 3) -> List[tuple]:
    """返回任意路径（直线或圆弧）上有效晶界穿越位置的 (row, col) 坐标。"""
    segs = _get_grain_segments(path_labels)
    valid = [s for s in segs if (s[1] - s[0]) >= min_intercept_px]
    return [(int(path_r[valid[i][0]]), int(path_c[valid[i][0]]))
            for i in range(1, len(valid))
            if valid[i][2] != valid[i-1][2]]
