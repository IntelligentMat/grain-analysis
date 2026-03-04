"""
preprocessing.py — 图像预处理模块

流程：灰度化 → 高斯去噪 → 中值滤波 → CLAHE 对比度增强
"""

import numpy as np
import cv2
from skimage.filters import gaussian


def preprocess(image: np.ndarray,
               gaussian_sigma: float = 3.0,
               median_kernel: int = 3,
               clahe_clip_limit: float = 2.0,
               clahe_tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    对输入图像执行预处理流程。

    Args:
        image: 输入图像，支持灰度或 BGR 彩色（uint8）
        gaussian_sigma: 高斯滤波标准差
        median_kernel: 中值滤波核大小（奇数）
        clahe_clip_limit: CLAHE 对比度限制
        clahe_tile_grid_size: CLAHE 分块大小

    Returns:
        增强后灰度图（uint8）
    """
    # 1. 灰度化
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. 高斯滤波（去随机噪声）
    blurred = gaussian(gray.astype(np.float64), sigma=gaussian_sigma,
                       preserve_range=True)
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    # 3. 中值滤波（去椒盐噪声）
    if median_kernel > 1:
        blurred = cv2.medianBlur(blurred, median_kernel)

    # 4. CLAHE 自适应直方图均衡化（增强晶界对比度）
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
                             tileGridSize=clahe_tile_grid_size)
    enhanced = clahe.apply(blurred)

    return enhanced
