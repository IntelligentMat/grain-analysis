"""
segmentation.py — 晶粒分割模块

主方法：Otsu 阈值 + 形态学操作 + 分水岭（Watershed）
"""

import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import disk, closing, opening, remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label
import math


def _auto_min_grain_area(pixels_per_micron: float) -> int:
    """
    根据像素分辨率自动估算最小有效晶粒面积（像素²）。
    依据：500X 光学显微镜最小可分辨结构约 2μm。
    """
    min_diameter_px = 2.0 * pixels_per_micron  # 2μm → 像素数
    min_area = math.pi / 4 * (min_diameter_px ** 2)
    return max(int(min_area), 10)


def segment(enhanced: np.ndarray,
            pixels_per_micron: float = 2.25,
            gaussian_sigma: float = 3.0,
            min_distance: int = 50,
            closing_disk_size: int = 2,
            min_grain_area: int | None = None,
            remove_border: bool = False) -> np.ndarray:
    """
    对预处理后的灰度图执行晶粒分割。

    Args:
        enhanced: 预处理后灰度图（uint8）
        pixels_per_micron: 像素/微米换算比
        gaussian_sigma: 分割前高斯平滑
        min_distance: Watershed marker 最小间距（像素）
        closing_disk_size: 形态学闭运算核半径
        min_grain_area: 最小晶粒面积（像素²），None 时自动估算
        remove_border: 是否移除接触图像边界的晶粒

    Returns:
        labels: 整数标签图，0=背景/晶界，>0=晶粒 ID
    """
    if min_grain_area is None:
        min_grain_area = _auto_min_grain_area(pixels_per_micron)

    # 1. 轻微高斯平滑，稳定 Otsu 阈值
    img_smooth = gaussian(enhanced.astype(np.float64), sigma=gaussian_sigma,
                          preserve_range=True)
    img_smooth = np.clip(img_smooth, 0, 255).astype(np.uint8)

    # 2. Otsu 自适应阈值 → 得到晶界 mask（True=晶界）
    thresh_val = threshold_otsu(img_smooth)
    boundary = img_smooth > thresh_val

    # 典型 316L：晶粒亮、晶界暗 → bright pixels (True) = grain interiors → >50%
    # 反转使 True=晶界（暗），适配后续形态学操作
    if boundary.sum() < boundary.size * 0.5:
        boundary = ~boundary  # 原本 True=grain，少数为晶界 → 反转取晶界

    # 3. 形态学操作：在晶界 mask 上填孔、去噪
    boundary = closing(boundary, disk(closing_disk_size))  # 填补断裂晶界
    boundary = opening(boundary, disk(1))                  # 去除小噪点

    # 晶粒区域 mask（True=晶粒内部），用于距离变换与分水岭
    grain_mask: np.ndarray = ~boundary

    # 4. 距离变换：在晶粒内部计算到最近晶界的距离，晶粒中心处最大
    distance: np.ndarray = ndi.distance_transform_edt(grain_mask)  # type: ignore[assignment]

    # 5. 局部极大值 → Watershed markers（峰值 = 晶粒中心）
    coords = peak_local_max(distance, min_distance=min_distance, labels=grain_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    labeled, _ = ndi.label(mask)  # type: ignore[misc]
    markers: np.ndarray = np.asarray(labeled)

    # 6. Watershed 分割（仅在晶粒区域内运行）
    labels = watershed(-distance, markers, mask=grain_mask)

    # 7. 移除过小区域
    if min_grain_area > 0:
        labels = _remove_small_grains(labels, min_grain_area)

    # 8. 可选：移除边界接触晶粒
    if remove_border:
        labels = _remove_border_grains(labels)

    # 9. 重新标记，保证 ID 连续
    return label(labels > 0)  # type: ignore[return-value]


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
    for gid in border_ids:
        result[result == gid] = 0
    return result
