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
    area_px2: float               # 晶粒面积，单位 px²
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
    Heyn 截距法 — 按 ASTM E112 标准图样估计平均截距长度与晶粒度等级。

    这段实现的核心思想，是把一组固定测试路径叠加到标签图上，
    然后统计这些路径“穿过了多少个有效晶粒段”。

    具体流程如下：
    1. 在图像上布置 4 条开放测试线；
    2. 在图像中心布置 3 个闭合同心圆；
    3. 沿每条路径读取标签序列，例如 [0, 0, 3, 3, 1, 1, 1, 0]；
    4. 将连续且长度足够的同标签区间视为一个有效晶粒段；
    5. 汇总所有路径上的有效晶粒段数 N 与总路径长度 L；
    6. 由 N/L 得到截距数密度 N_L，再反推平均截距长度和 ASTM G 值。

    这里统计的是“有效晶粒段数 N”，不是直接数晶界交点 P：
    - 对闭合路径（圆），每个有效晶粒段按 1.0 计数；
    - 对开放路径（直线），如果某个有效段只接触路径一端，说明它可能是被
      图像边界截断的端段，这类段按 0.5 计权，以减轻边界截断带来的偏差。

    测试图样说明：
    - 4 条测试线分别是：底部水平线、左侧垂直线、两条对角线；
    - 3 个同心圆的圆心取图像中心；
    - 圆半径比例 0.7958 / 0.5305 / 0.2653 × min(H, W) / 2，来自 ASTM E112
      附录，目的是让三圆组合的周长与标准图样保持一致量级。

    Args:
        labels: 晶粒标签图。0 表示背景，其余正整数表示不同晶粒。
        pixels_per_micron: 像素/微米换算系数，用于把平均截距从像素换算到物理单位。
        min_intercept_px: 最小有效晶粒段长度（像素）。短于该阈值的段会被忽略，
            以降低噪声、锯齿边界和极短掠过段对统计结果的影响。
        margin_ratio: 开放测试线距离图像边界的留白比例。设置边距可以避免测试线
            贴着图像边缘走，从而减少大量边缘截断段。

    Returns:
        InterceptMethodResult: 包含总截距数、总测试长度、平均截距、ASTM G 值，
        以及用于后续可视化的图样元素和代表点。
    """
    # 统一校验并归一化输入参数，避免出现非法物理尺度或小于 1 的段长阈值。
    pixels_per_micron = _validate_pixels_per_micron(pixels_per_micron)
    min_intercept_px = max(1, int(min_intercept_px))

    # 基础几何量：
    # - size 取图像短边，保证所有标准圆都能落在图像内部；
    # - (r_c, c_c) 是同心圆圆心；
    # - margin 控制开放测试线距离边界的留白。
    height, width = labels.shape
    size = min(height, width)          # 以较短边为基准（保证圆不超出图像）
    r_c = height // 2
    c_c = width // 2
    margin = max(1, int(margin_ratio * size))

    # 下面逐条累加：
    # - total_intersections: 有效晶粒段总数 N；
    # - total_length_px: 全部测试路径总长度 L；
    # - pattern_elements: 保存图样本身（线 / 圆），用于可视化叠加；
    # - intersection_points: 每个有效晶粒段的代表点；
    # - half_intersection_points: 开放路径中按 0.5 计权的端段代表点；
    # - intersected_grain_ids: 所有被有效路径命中的晶粒 ID。
    total_intersections = 0
    total_length_px = 0.0
    pattern_elements: List[tuple] = []
    intersection_points: List[tuple] = []
    half_intersection_points: List[tuple] = []
    intersected_grain_ids: set[int] = set()

    # ── 4 条开放测试线 ─────────────────────────────────────────────────
    # 每个元组是 (r1, c1, r2, c2)。
    # 开放路径有明确起点和终点，所以末端被截断的晶粒段可能按 0.5 计权。
    line_defs = [
        (height - 1 - margin, margin,   height - 1 - margin, width - 1 - margin),  # 水平（底部）
        (margin,              margin,   height - 1 - margin,  margin),              # 垂直（左侧）
        (margin,       margin,        height - 1 - margin, width - 1 - margin),     # 对角 ↘
        (height - 1 - margin, margin, margin,           width - 1 - margin),        # 对角 ↗
    ]
    for r1, c1, r2, c2 in line_defs:
        # 先把连续几何直线离散成像素路径，并裁掉理论上可能越界的点。
        rr, cc = skdraw.line(r1, c1, r2, c2)
        mask = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[mask], cc[mask]

        # 沿路径读取标签，得到一维标签序列；后续辅助函数会把它拆成连续晶粒段。
        line_labels = labels[rr, cc]
        intersected_grain_ids.update(
            _intercepted_grain_ids_on_path(line_labels, min_intercept_px, is_closed=False)
        )

        # _count_intercepts: 把连续晶粒段转成计权后的 N；
        # _intercept_positions_on_path: 给每个有效段选一个代表点；
        # _half_intercept_positions_on_path: 单独记录按 0.5 计权的端段位置。
        total_intersections += _count_intercepts(line_labels, min_intercept_px,
                                                 is_closed=False)
        intersection_points += _intercept_positions_on_path(
            line_labels, rr, cc, min_intercept_px, is_closed=False)
        half_intersection_points += _half_intercept_positions_on_path(
            line_labels, rr, cc, min_intercept_px, is_closed=False)

        # 长度使用两端点欧氏距离，而不是简单的像素个数，
        # 这样对角线与水平/垂直线的长度具有一致的连续几何意义。
        total_length_px += float(np.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2))
        pattern_elements.append(('line', r1, c1, r2, c2))

    # ── 3 个闭合同心圆（ASTM E112 标准半径比例）─────────────────────────
    # 圆是闭合路径，没有“起点/终点截断”的概念，因此有效晶粒段统一按 1.0 计数。
    astm_radii_ratios = [0.7958, 0.5305, 0.2653]
    for ratio in astm_radii_ratios:
        radius = int(round(ratio * size / 2))
        if radius <= 0:
            continue

        # 生成离散圆周像素后，必须按极角排序，保证沿圆周遍历时标签序列连续；
        # 否则像素顺序被打乱会导致同一个晶粒被错误拆成多个段。
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

        # 闭合圆路径长度直接使用连续几何圆周长 2πr。
        total_length_px += 2.0 * np.pi * radius
        pattern_elements.append(('circle', r_c, c_c, radius))

    # ── 从 N 和 L 推导 ASTM 结果 ───────────────────────────────────────
    # N_L = N / L：单位长度上的截距数密度；
    # 平均截距长度 l_bar = 1 / N_L；
    # 再用像素/微米系数换算到 μm 和 mm，最后代入 ASTM E112 经验公式计算 G 值。
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
    """从一维标签序列中提取连续晶粒段。

    输入通常来自某条测试路径沿途采样到的标签值，例如：
    [0, 0, 3, 3, 3, 1, 1, 0, 4, 4]

    这会被拆成：
    - (2, 5, 3)：标签 3 占据索引 [2, 5)
    - (5, 7, 1)：标签 1 占据索引 [5, 7)
    - (8, 10, 4)：标签 4 占据索引 [8, 10)

    规则是：
    - 背景 0 视为分隔符，不构成晶粒段；
    - 相邻不同的非零标签视为段切换；
    - 返回的 end_px 采用右开区间，便于后续直接用 end-start 算段长。
    """
    segments = []
    prev = 0
    seg_start = 0
    for i, lbl in enumerate(line_labels):
        if lbl == 0:
            # 遇到背景，说明前一个非零连续段到这里结束。
            if prev != 0:
                segments.append((seg_start, i, int(prev)))
            prev = 0
        else:
            if prev == 0:
                # 从背景进入非背景，开启一个新的晶粒段。
                seg_start = i
            elif lbl != prev:
                # 非零标签发生变化，说明前一个晶粒段结束、当前晶粒段开始。
                segments.append((seg_start, i, int(prev)))
                seg_start = i
            prev = lbl
    if prev != 0:
        # 序列结尾仍在某个晶粒内部时，补上最后一个段。
        segments.append((seg_start, len(line_labels), int(prev)))
    return segments


def _valid_grain_segment_groups(line_labels: np.ndarray,
                                min_intercept_px: int = 3,
                                is_closed: bool = False) -> list[list[tuple[int, int, int]]]:
    """返回参与截距统计的有效晶粒段分组。

    先用 `_get_grain_segments` 找到所有连续非零晶粒段，再做两步处理：
    1. 过滤掉长度小于 `min_intercept_px` 的短段；
    2. 若路径是闭合的（圆周），并且首尾分组属于同一晶粒，则把它们合并。

    合并首尾的原因是：闭合路径没有真正的“开头”和“结尾”，
    因此圆周序列在数组首尾被切开的同一晶粒，本质上应算作一个截段。
    """
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
    """把有效晶粒段分组转换成截距计数 N。

    计数规则：
    - 闭合路径：每个有效分组记 1.0；
    - 开放路径：
      - 若某分组只接触路径起点或终点中的一端，记 0.5；
      - 其余情况记 1.0。

    这样做是为了在直线路径上近似处理“端部截断晶粒”的情况。
    """
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
    """返回某条路径上真正参与统计的晶粒 ID。

    这里不会返回所有“碰到过”的标签，而只返回形成了有效晶粒段的标签，
    也就是通过最小段长筛选、并完成首尾合并后的那些晶粒。
    """
    groups = _valid_grain_segment_groups(line_labels, min_intercept_px, is_closed=is_closed)
    return {int(seg[2]) for group in groups for seg in group}


def _sort_circle_pixels(rr: np.ndarray, cc: np.ndarray,
                        r_center: int, c_center: int):
    """按极角对圆周像素排序，恢复沿圆周行走的顺序。

    `circle_perimeter` 返回的是一组圆周像素，但它们的顺序不一定对应
    连续的环形遍历顺序。若直接取标签，会把同一圆周上的连续晶粒段打乱。
    因此这里用相对圆心的极角排序，确保后续得到的标签序列是“沿圆周一圈”
    的顺序。
    """
    angles = np.arctan2(rr - r_center, cc - c_center)
    order = np.argsort(angles)
    return rr[order], cc[order]


def _intercept_positions_on_path(path_labels: np.ndarray,
                                 path_r: np.ndarray,
                                 path_c: np.ndarray,
                                 min_intercept_px: int = 3,
                                 is_closed: bool = False) -> List[tuple]:
    """为每个有效晶粒段选择一个代表点坐标。

    这些点主要用于可视化：在图上标出“这里存在一个被计入统计的截段”。

    选点策略：
    - 对普通分组，取该分组中最长晶粒段的中点；
    - 对开放路径上仅接触起点的端段，取路径起点；
    - 对开放路径上仅接触终点的端段，取路径终点。

    这样做可以让按 0.5 计权的端段在图上也表现出“它是一个边界端段”。
    """
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
    """返回开放路径上按 0.5 计权的端段代表点。

    这个函数是 `_intercept_positions_on_path` 的补充：
    它只提取那些“碰到路径起点或终点且只碰到一端”的分组，
    方便在可视化中单独高亮半权重截段。
    """
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
