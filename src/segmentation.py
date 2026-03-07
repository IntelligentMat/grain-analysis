"""
segmentation.py — 晶粒分割模块

主方法：Otsu 阈值 + 形态学操作 + 分水岭（Watershed）
"""

from typing import cast, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import disk, closing, opening
from skimage.segmentation import watershed, relabel_sequential
from skimage.feature import peak_local_max


def _auto_min_distance(shape: tuple[int, int], min_distance: int | None) -> int:
    """根据图像尺寸确定 marker 最小间距。"""
    if min_distance is not None:
        return max(1, int(min_distance))
    h, w = shape
    return max(10, int(min(h, w) * 0.05))


def _auto_min_grain_area(min_distance: int) -> int:
    """根据 marker 间距估算最小有效晶粒面积（像素²）。"""
    return max(20, int(np.pi * (max(min_distance, 1) * 0.35) ** 2))


def segment(enhanced: np.ndarray,
            min_distance: int | None = None,
            closing_disk_size: int = 2,
            opening_disk_size: int = 1,
            min_grain_area: int | None = None,
            boundary_mode: str = "auto",
            remove_border: bool = False) -> np.ndarray:
    """
    对预处理后的灰度图执行晶粒分割。

    Args:
        enhanced: 预处理后灰度图（uint8）
        min_distance: Watershed marker 最小间距（像素），None 时按图像尺寸自适应
        closing_disk_size: 形态学闭运算核半径，控制断裂晶界的连接力度
        opening_disk_size: 形态学开运算核半径，控制细线去除力度；增大可消除划痕（默认 1）
        min_grain_area: 最小晶粒面积（像素²），None 时自动估算
        boundary_mode: 晶界亮暗模式，可选 "auto" / "dark" / "bright"
        remove_border: 是否移除接触图像边界的晶粒

    Returns:
        labels: 整数标签图，0=背景/晶界，>0=晶粒 ID
    """
    min_distance = _auto_min_distance(enhanced.shape, min_distance)
    if min_grain_area is None:
        min_grain_area = _auto_min_grain_area(min_distance)
    if boundary_mode not in {"auto", "dark", "bright"}:
        raise ValueError("boundary_mode must be one of: auto, dark, bright")

    img_smooth = enhanced

    # 2. Otsu 自适应阈值 → 得到晶界 mask（True=晶界）
    thresh_val = threshold_otsu(img_smooth)
    boundary = img_smooth > thresh_val

    # 统一 boundary 语义为 True=晶界
    if boundary_mode == "dark":
        boundary = ~boundary
    elif boundary_mode == "auto":
        if boundary.mean() > 0.5:
            boundary = ~boundary

    # 3. 形态学操作：在晶界 mask 上填孔、去噪
    boundary = closing(boundary, disk(closing_disk_size))      # 填补断裂晶界
    boundary = opening(boundary, disk(opening_disk_size))      # 去除细线噪点/划痕

    # 晶粒区域 mask（True=晶粒内部），用于距离变换与分水岭
    grain_mask: np.ndarray = np.asarray(~boundary)

    # 4. 距离变换：在晶粒内部计算到最近晶界的距离，晶粒中心处最大
    distance: np.ndarray = ndi.distance_transform_edt(grain_mask)  # type: ignore[assignment]

    # 5. 局部极大值 → Watershed markers（峰值 = 晶粒中心）
    coords = peak_local_max(distance, min_distance=min_distance, labels=grain_mask)
    seed_mask = np.zeros(distance.shape, dtype=bool)
    if coords.size > 0:
        seed_mask[tuple(coords.T)] = True

    # 5b. 为所有没有 marker 的连通晶粒区域补种（含边缘区域和内部孤立区域）
    seed_mask = _fill_unseeded_regions(seed_mask, grain_mask, distance)

    # 从 seed_mask 构建 markers
    if seed_mask.any():
        markers: np.ndarray = np.asarray(cast(Tuple[np.ndarray, int], ndi.label(seed_mask))[0])
    else:
        # 极端情况下没有任何峰值，退化为连通域 marker，避免输出全背景
        print("[WARNING] 未检测到任何峰值，退化为连通域 marker。")
        markers = np.asarray(cast(Tuple[np.ndarray, int], ndi.label(grain_mask))[0])

    # 6. Watershed 分割（仅在晶粒区域内运行）
    labels = watershed(-distance, markers, mask=grain_mask)

    # 7. 移除过小区域
    if min_grain_area > 0:
        labels = _remove_small_grains(labels, min_grain_area)

    # 8. 可选：移除边界接触晶粒
    if remove_border:
        labels = _remove_border_grains(labels)

    # 9. 重新标记，保证 ID 连续
    labels, _, _ = relabel_sequential(labels)
    return labels.astype(np.int32)


def _fill_unseeded_regions(seed_mask: np.ndarray,
                           grain_mask: np.ndarray,
                           distance: np.ndarray) -> np.ndarray:
    """为所有没有 marker 的连通晶粒区域在距离最大处补种（含内部孤立区域和边缘区域）。"""
    result = seed_mask.copy()
    region_labels = np.asarray(cast(Tuple[np.ndarray, int], ndi.label(grain_mask))[0])
    n_regions = int(region_labels.max())
    for rid in range(1, n_regions + 1):
        region = region_labels == rid
        if result[region].any():
            continue
        dist_in_region = np.where(region, distance, -1.0)
        best_idx = np.unravel_index(int(dist_in_region.argmax()), dist_in_region.shape)
        result[best_idx] = True
    return result

def _remove_small_grains(labels: np.ndarray, min_area: int) -> np.ndarray:
    """移除面积小于 min_area 的晶粒区域。"""
    result = labels.copy()
    grain_ids, counts = np.unique(labels, return_counts=True)
    for gid, count in zip(grain_ids, counts):
        if gid == 0:
            continue
        if count < min_area:
            result[result == gid] = 0
    return result


def _remove_border_grains(labels: np.ndarray) -> np.ndarray:
    """移除接触图像边界的晶粒。"""
    result = labels.copy()
    border_ids = set()
    border_ids.update(np.unique(labels[0, :]))
    border_ids.update(np.unique(labels[-1, :]))
    border_ids.update(np.unique(labels[:, 0]))
    border_ids.update(np.unique(labels[:, -1]))
    border_ids.discard(0)
    result[np.isin(result, list(border_ids))] = 0
    return result
