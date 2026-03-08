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

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.figure import Figure
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from src import io_utils


def _save(fig: Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return image[..., ::-1].copy()
    return np.stack([image] * 3, axis=-1)


def _resolve_artifact_path(results_json_path: str, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    json_dir = Path(results_json_path).resolve().parent
    candidates = [
        json_dir / path,
        Path(raw_path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _label_centroids(labels: np.ndarray) -> dict[int, tuple[float, float]]:
    return {
        int(prop.label): (float(prop.centroid[0]), float(prop.centroid[1]))
        for prop in regionprops(labels)
    }


def _tint_mask(rgb: np.ndarray,
               labels: np.ndarray,
               grain_ids: list[int],
               color: tuple[float, float, float],
               alpha: float = 0.38) -> np.ndarray:
    tinted = rgb.astype(np.float32) / 255.0
    if not grain_ids:
        return tinted

    color_arr = np.array(color, dtype=np.float32)
    mask = np.isin(labels, grain_ids)
    tinted[mask] = (1 - alpha) * tinted[mask] + alpha * color_arr
    return tinted


def _overlay_boundaries(image: np.ndarray,
                        labels: np.ndarray,
                        color: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    overlay = image.copy()
    boundaries = find_boundaries(labels, mode="outer")
    overlay[boundaries] = color
    return overlay


def _annotate_grain_ids(ax, labels: np.ndarray, grain_ids: list[int], text_color: str) -> None:
    centroids = _label_centroids(labels)
    for grain_id in grain_ids:
        centroid = centroids.get(int(grain_id))
        if centroid is None:
            continue
        row, col = centroid
        ax.text(
            col,
            row,
            str(grain_id),
            color=text_color,
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.45, pad=1.2, edgecolor="none"),
            zorder=6,
        )


def save_original(image: np.ndarray, output_path: str) -> None:
    """保存原始图像（彩色或灰度）。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(_as_rgb(image))
    ax.set_title("Original Image")
    ax.axis("off")
    _save(fig, output_path)


def save_segmented(image: np.ndarray, labels: np.ndarray, output_path: str) -> None:
    """将晶粒分割标签叠加到原始图像上（伪彩色）。"""
    bg = _as_rgb(image)
    overlay = label2rgb(labels, image=bg, alpha=0.4, bg_label=0, bg_color=(0, 0, 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(bg)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Segmented ({labels.max()} grains)")
    axes[1].axis("off")

    fig.tight_layout()
    _save(fig, output_path)


def render_area_method(image: np.ndarray, labels: np.ndarray, results: dict, output_path: str) -> None:
    """根据结果工件重绘面积法可视化。"""
    rgb = _as_rgb(image)
    area_result = results["area_method"]
    inside_ids = area_result.get("inside_grain_ids", [])
    edge_ids = area_result.get("edge_grain_ids", [])
    corner_ids = area_result.get("corner_grain_ids", [])

    tinted = _tint_mask(rgb, labels, inside_ids, (0.18, 0.85, 0.35), alpha=0.35)
    tinted = _tint_mask((tinted * 255).astype(np.uint8), labels, edge_ids, (1.0, 0.6, 0.0), alpha=0.40)
    tinted = _tint_mask((tinted * 255).astype(np.uint8), labels, corner_ids, (0.95, 0.3, 0.7), alpha=0.40)
    tinted = _overlay_boundaries(tinted, labels)

    h, w = labels.shape
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(tinted)

    rect = mpatches.Rectangle((0, 0), w - 1, h - 1,
                              linewidth=2, edgecolor="yellow",
                              facecolor="none")
    ax.add_patch(rect)

    _annotate_grain_ids(ax, labels, inside_ids, text_color="#90ee90")
    _annotate_grain_ids(ax, labels, edge_ids, text_color="#ffd27f")
    _annotate_grain_ids(ax, labels, corner_ids, text_color="#ff9de1")

    info = (
        f"N_inside={area_result['n_inside']}, N_edge={area_result['n_edge']}, N_corner={area_result['n_corner']}\n"
        f"N_eq={area_result['n_equivalent']:.1f}, N_A={area_result['n_a_per_mm2']:.1f}/mm²\n"
        f"ASTM G = {area_result['astm_g_value']:.2f}"
    )
    ax.text(5, 15, info, fontsize=8, color="yellow",
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.5, pad=2))

    legend_handles = [
        mpatches.Patch(color=(0.18, 0.85, 0.35), label="Inside grains"),
        mpatches.Patch(color=(1.0, 0.6, 0.0), label="Edge grains (1/2)"),
        mpatches.Patch(color=(0.95, 0.3, 0.7), label="Corner grains (1/4)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.6)
    ax.set_title("Area Method (Planimetric / Jeffries)")
    ax.axis("off")
    _save(fig, output_path)


def render_intercept_method(image: np.ndarray, labels: np.ndarray, results: dict, output_path: str) -> None:
    """根据结果工件重绘截线法可视化。"""
    rgb = _as_rgb(image)
    intercept_result = results["intercept_method"]
    intersected_ids = intercept_result.get("intersected_grain_ids", [])

    tinted = _tint_mask(rgb, labels, intersected_ids, (0.95, 0.92, 0.25), alpha=0.32)
    tinted = _overlay_boundaries(tinted, labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(tinted)

    for elem in intercept_result.get("pattern_elements", []):
        if elem[0] == "line":
            _, r1, c1, r2, c2 = elem
            ax.plot([c1, c2], [r1, r2], color="cyan", linewidth=0.9, alpha=0.9)
        elif elem[0] == "circle":
            _, r_c, c_c, radius = elem
            circle = mpatches.Circle((c_c, r_c), radius,
                                     fill=False, edgecolor="cyan", linewidth=0.9, alpha=0.9)
            ax.add_patch(circle)

    half_points = intercept_result.get("half_intersection_points", [])
    half_point_set = {tuple(point) for point in half_points}

    points = [tuple(point) for point in intercept_result.get("intersection_points", [])
              if tuple(point) not in half_point_set]
    if points:
        pts = np.array(points)
        ax.scatter(pts[:, 1], pts[:, 0], s=15, c="red", marker="o", zorder=5)

    if half_points:
        half_pts = np.array(half_points)
        ax.scatter(half_pts[:, 1], half_pts[:, 0], s=38, c="#ff4d4f", marker="x", linewidths=1.3, zorder=6)

    _annotate_grain_ids(ax, labels, intersected_ids, text_color="#fff799")

    info = (
        f"Intercepts={intercept_result['total_intersections']}\n"
        f"L_total={intercept_result['total_line_length_px']:.0f} px\n"
        f"l̄={intercept_result['mean_intercept_length_um']:.2f} μm\n"
        f"ASTM G = {intercept_result['astm_g_value']:.2f}"
    )
    ax.text(5, 15, info, fontsize=8, color="cyan",
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.5, pad=2))

    legend_handles = [
        mpatches.Patch(color=(0.95, 0.92, 0.25), label="Intercepted grains"),
        mpatches.Patch(color="cyan", label="ASTM pattern"),
        mpatches.Patch(color="red", label="Intercept representatives"),
        plt.Line2D([0], [0], color="#ff4d4f", marker="x", linestyle="None",
                   markersize=7, markeredgewidth=1.3, label="0.5 intercepts"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.6)
    ax.set_title("Intercept Method — ASTM E112 (4 lines + 3 circles)")
    ax.axis("off")
    _save(fig, output_path)


def render_anomaly(image: np.ndarray, labels: np.ndarray, results: dict, output_path: str) -> None:
    """根据结果工件重绘异常晶粒可视化。"""
    rgb = _as_rgb(image)
    anomaly = results["anomaly_detection"]
    anomaly_ids = anomaly.get("anomalous_grain_ids", [])

    overlay = label2rgb(labels, image=rgb, alpha=0.3, bg_label=0, bg_color=(0, 0, 0))
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    for gid in anomaly_ids:
        mask = labels == gid
        overlay_uint8[mask, 0] = 220
        overlay_uint8[mask, 1] = 30
        overlay_uint8[mask, 2] = 30

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(overlay_uint8)
    _annotate_grain_ids(ax, labels, anomaly_ids, text_color="#ff8a8a")

    flag = "YES" if anomaly["has_anomaly"] else "NO"
    info = (
        f"Anomaly detected: {flag}\n"
        f"Rule A: {'✓' if anomaly['rule_a']['triggered'] else '✗'}  "
        f"Rule B: {'✓' if anomaly['rule_b']['triggered'] else '✗'}  "
        f"Rule C: {'✓' if anomaly['rule_c']['triggered'] else '✗'}\n"
        f"Anomalous grains: {anomaly['total_anomalous_grains']}"
    )
    ax.text(5, 15, info, fontsize=8, color="white",
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.5, pad=2))

    ax.set_title("Anomaly Detection")
    ax.axis("off")
    _save(fig, output_path)


def render_distribution(results: dict, output_path: str) -> None:
    """根据结果工件重绘晶粒尺寸分布直方图。"""
    stats = results["grain_statistics"]
    diameters = stats.get("diameters", [])
    if not diameters:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No grains detected", ha="center", va="center")
        _save(fig, output_path)
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(diameters, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(stats["mean_diameter_px"], color="red", linestyle="--",
               linewidth=1.5, label=f"Mean={stats['mean_diameter_px']:.1f} px")
    ax.axvline(stats["median_diameter_px"], color="orange", linestyle=":",
               linewidth=1.5, label=f"Median={stats['median_diameter_px']:.1f} px")

    threshold_3sigma = stats["mean_diameter_px"] + 3 * stats["std_diameter_px"]
    ax.axvline(threshold_3sigma, color="purple", linestyle="-.",
               linewidth=1.2, label=f"μ+3σ={threshold_3sigma:.1f} px")

    ax.set_xlabel("Equivalent Diameter (px)")
    ax.set_ylabel("Count")
    ax.set_title(f"Grain Size Distribution  (n={stats['count']})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, output_path)


def render_all_from_results(results_json_path: str, output_dir: Optional[str] = None) -> dict[str, str]:
    """从结果 JSON 与 labels.npy 重建全部可视化文件。"""
    results = io_utils.load_results_json(results_json_path)
    labels_path = _resolve_artifact_path(results_json_path, results["artifacts"]["labels_path"])
    image_path = _resolve_artifact_path(results_json_path, results["image_path"])

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels artifact not found: {labels_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for rendering: {image_path}")

    labels = np.load(labels_path)
    image = io_utils.load_image(str(image_path))

    out_dir = (
        io_utils.make_output_dir(output_dir, results["image_name"])
        if output_dir else Path(results_json_path).resolve().parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = io_utils.output_paths(out_dir, results["image_name"])

    save_original(image, paths["original"])
    save_segmented(image, labels, paths["segmented"])
    render_area_method(image, labels, results, paths["area"])
    render_intercept_method(image, labels, results, paths["intercept"])
    render_anomaly(image, labels, results, paths["anomaly"])
    render_distribution(results, paths["distribution"])
    return paths
