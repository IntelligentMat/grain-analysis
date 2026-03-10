"""
preprocessing.py — 图像预处理模块

流程：灰度化 → 平滑去噪（高斯 / 双边 / 各向异性扩散）→ 中值滤波 → CLAHE 对比度增强
"""

import numpy as np
import cv2
from skimage.filters import gaussian
from skimage.restoration import denoise_tv_chambolle


def _estimate_noise_std(gray: np.ndarray) -> float:
    """
    估计图像噪声强度（归一化到 0~1）。
    用轻微平滑后的高频残差近似噪声。
    """
    img = gray.astype(np.float64)
    low_freq = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0, sigmaY=1.0)
    residual = img - low_freq
    return float(np.std(residual) / 255.0)


def _resolve_gaussian_sigma(
    gray: np.ndarray,
    gaussian_sigma: float | str | None,
    noise_to_sigma_k: float,
    sigma_bounds: tuple[float, float],
) -> float:
    """将用户配置转换为最终用于平滑的 sigma。"""
    if gaussian_sigma is not None and gaussian_sigma != "auto":
        return float(gaussian_sigma)

    noise_std = _estimate_noise_std(gray)
    sigma = noise_to_sigma_k * noise_std
    lo, hi = sigma_bounds
    return float(np.clip(sigma, lo, hi))


def preprocess(
    image: np.ndarray,
    smooth_mode: str = "gaussian",
    gaussian_sigma: float | str | None = "auto",
    noise_to_sigma_k: float = 18.0,
    sigma_bounds: tuple[float, float] = (0.8, 4.0),
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
    anisotropic_niter: int = 10,
    anisotropic_kappa: int = 50,
    anisotropic_gamma: float = 0.1,
    median_kernel: int = 3,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """
    对输入图像执行预处理流程。

    Args:
        image: 输入图像，支持灰度或 BGR 彩色（uint8）
        smooth_mode: 平滑方式，可选 "gaussian" / "bilateral" / "anisotropic"
            - "gaussian"    : 高斯滤波（默认，速度最快）
            - "bilateral"   : 双边滤波，保留晶界边缘，平滑晶粒内部划痕
            - "anisotropic" : Perona-Malik 各向异性扩散，扩散沿晶粒内部方向，
                              在晶界处自动停止（medpy 实现）
        gaussian_sigma: 高斯滤波标准差，"auto"/None 时按噪声自适应
        noise_to_sigma_k: 自适应系数 k（sigma = k * noise_std）
        sigma_bounds: 自适应 sigma 的上下限（像素）
        bilateral_d: 双边滤波邻域直径
        bilateral_sigma_color: 双边滤波灰度相似阈值（越大越平滑）
        bilateral_sigma_space: 双边滤波空间距离权重
        anisotropic_niter: 各向异性扩散迭代次数
        anisotropic_kappa: 边缘灵敏度（越小越保守，越大越平滑）
        anisotropic_gamma: 扩散步长（0 < gamma <= 0.25）
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

    # 2. 平滑去噪
    if smooth_mode == "bilateral":
        blurred = cv2.bilateralFilter(
            gray, d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space
        )

    elif smooth_mode == "anisotropic":
        from medpy.filter.smoothing import anisotropic_diffusion

        blurred = anisotropic_diffusion(
            gray, niter=anisotropic_niter, kappa=anisotropic_kappa, gamma=anisotropic_gamma
        )
        blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    else:  # "gaussian"（默认）
        sigma = _resolve_gaussian_sigma(
            gray,
            gaussian_sigma=gaussian_sigma,
            noise_to_sigma_k=noise_to_sigma_k,
            sigma_bounds=sigma_bounds,
        )
        blurred = gaussian(gray.astype(np.float64), sigma=sigma, preserve_range=True)
        blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    # 3. 中值滤波（去椒盐噪声）
    if median_kernel > 1:
        blurred = cv2.medianBlur(blurred, median_kernel)

    # 4. CLAHE 自适应直方图均衡化（增强晶界对比度）
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    enhanced = clahe.apply(blurred)

    return enhanced
