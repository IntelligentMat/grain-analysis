"""
visualization.py — 结果可视化与标注模块

生成：
  {name}_original.png        原始输入图像
  {name}_segmented.png       分割结果伪彩色叠加
  {name}_area_method.png     面积法标注图
  {name}_intercept_method.png 截线法标注图
  {name}_anomaly.png         异常晶粒高亮图
  {name}_distribution.png    晶粒尺寸分布直方图
"""

import numpy as np
import matplotlib
from matplotlib.figure import Figure
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
from skimage.color import label2rgb
from typing import Optional

from src.analysis import AreaMethodResult, InterceptMethodResult, GrainStatistics
from src.anomaly import AnomalyResult


def _save(fig: Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────── 原始图像 ──────────────────────────────

def save_original(image: np.ndarray, output_path: str) -> None:
    """保存原始图像（彩色或灰度）。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    if image.ndim == 3:
        ax.imshow(image[..., ::-1])   # BGR → RGB
    else:
        ax.imshow(image, cmap="gray")
    ax.set_title("Original Image")
    ax.axis("off")
    _save(fig, output_path)


# ─────────────────────────── 分割结果 ──────────────────────────────

def save_segmented(image: np.ndarray, labels: np.ndarray,
                   output_path: str) -> None:
    """将晶粒分割标签叠加到原始图像上（伪彩色）。"""
    if image.ndim == 3:
        bg = image[..., ::-1].copy()   # BGR → RGB
    else:
        bg = np.stack([image] * 3, axis=-1)

    overlay = label2rgb(labels, image=bg, alpha=0.4,
                        bg_label=0, bg_color=(0, 0, 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(bg)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Segmented ({labels.max()} grains)")
    axes[1].axis("off")

    fig.tight_layout()
    _save(fig, output_path)


# ─────────────────────────── 面积法 ────────────────────────────────

def save_area_method(image: np.ndarray, labels: np.ndarray,
                     result: AreaMethodResult, output_path: str) -> None:
    """标注面积法测量区域（全图矩形）与统计结果。"""
    if image.ndim == 3:
        rgb = image[..., ::-1].copy()
    else:
        rgb = np.stack([image] * 3, axis=-1)

    h, w = labels.shape
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(rgb)

    # 绘制测量区域边框
    rect = mpatches.Rectangle((0, 0), w - 1, h - 1,
                                linewidth=2, edgecolor="yellow",
                                facecolor="none")
    ax.add_patch(rect)

    # 统计文字
    info = (f"N_inside={result.n_inside}, N_intersect={result.n_intersect}\n"
            f"N_eq={result.n_equivalent:.1f}, N_A={result.n_a_per_mm2:.0f}/mm²\n"
            f"ASTM G = {result.astm_g_value:.2f}")
    ax.text(5, 15, info, fontsize=8, color="yellow",
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.5, pad=2))

    ax.set_title("Area Method (Planimetric / Jeffries)")
    ax.axis("off")
    _save(fig, output_path)


# ─────────────────────────── 截线法 ────────────────────────────────

def save_intercept_method(image: np.ndarray,
                          result: InterceptMethodResult,
                          output_path: str) -> None:
    """在图像上绘制网格测试线与交点标注。"""
    if image.ndim == 3:
        rgb = image[..., ::-1].copy()
    else:
        rgb = np.stack([image] * 3, axis=-1)

    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(rgb)

    # 绘制测试线
    for orientation, pos in result.line_coords:
        if orientation == 'h':
            ax.axhline(y=pos, color="cyan", linewidth=0.8, alpha=0.8)
        else:
            ax.axvline(x=pos, color="cyan", linewidth=0.8, alpha=0.8)

    # 绘制交点（红点）
    if result.intersection_points:
        pts = np.array(result.intersection_points)
        ax.scatter(pts[:, 1], pts[:, 0], s=4, c="red", zorder=5)

    info = (f"Intersections={result.total_intersections}\n"
            f"L_total={result.total_line_length_um:.0f} μm\n"
            f"l̄={result.mean_intercept_length_um:.2f} μm\n"
            f"ASTM G = {result.astm_g_value:.2f}")
    ax.text(5, 15, info, fontsize=8, color="cyan",
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.5, pad=2))

    ax.set_title("Intercept Method (Heyn)")
    ax.axis("off")
    _save(fig, output_path)


# ─────────────────────────── 异常晶粒 ──────────────────────────────

def save_anomaly(image: np.ndarray, labels: np.ndarray,
                 anomaly: AnomalyResult, output_path: str) -> None:
    """高亮标注异常晶粒（红色覆盖）。"""
    if image.ndim == 3:
        rgb = image[..., ::-1].copy()
    else:
        rgb = np.stack([image] * 3, axis=-1)

    overlay = label2rgb(labels, image=rgb, alpha=0.3,
                        bg_label=0, bg_color=(0, 0, 0))
    overlay_uint8 = (overlay * 255).astype(np.uint8)

    # 将异常晶粒像素染红
    anomaly_ids = set(anomaly.anomalous_grain_ids)
    for gid in anomaly_ids:
        mask = labels == gid
        overlay_uint8[mask, 0] = 220
        overlay_uint8[mask, 1] = 30
        overlay_uint8[mask, 2] = 30

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(overlay_uint8)

    flag = "YES" if anomaly.has_anomaly else "NO"
    info = (f"Anomaly detected: {flag}\n"
            f"Rule A: {'✓' if anomaly.rule_a.triggered else '✗'}  "
            f"Rule B: {'✓' if anomaly.rule_b.triggered else '✗'}  "
            f"Rule C: {'✓' if anomaly.rule_c.triggered else '✗'}\n"
            f"Anomalous grains: {anomaly.total_anomalous_grains}")
    ax.text(5, 15, info, fontsize=8, color="white",
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.5, pad=2))

    ax.set_title("Anomaly Detection")
    ax.axis("off")
    _save(fig, output_path)


# ─────────────────────────── 尺寸分布 ──────────────────────────────

def save_distribution(stats: GrainStatistics, output_path: str) -> None:
    """绘制晶粒等效直径分布直方图。"""
    diameters = stats.diameters
    if not diameters:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No grains detected", ha="center", va="center")
        _save(fig, output_path)
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(diameters, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(stats.mean_diameter_um, color="red", linestyle="--",
               linewidth=1.5, label=f"Mean={stats.mean_diameter_um:.1f} μm")
    ax.axvline(stats.median_diameter_um, color="orange", linestyle=":",
               linewidth=1.5, label=f"Median={stats.median_diameter_um:.1f} μm")

    # 3σ 阈值
    threshold_3sigma = stats.mean_diameter_um + 3 * stats.std_diameter_um
    ax.axvline(threshold_3sigma, color="purple", linestyle="-.",
               linewidth=1.2, label=f"μ+3σ={threshold_3sigma:.1f} μm")

    ax.set_xlabel("Equivalent Diameter (μm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Grain Size Distribution  (n={stats.count})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, output_path)
