import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

mpl_dir = Path(tempfile.gettempdir()) / "grain_analysis_test_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src import io_utils, visualization


class TestVisualizationHelpers(unittest.TestCase):
    def test_as_rgb_converts_gray_and_flips_bgr(self):
        gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        rgb_gray = visualization._as_rgb(gray)
        self.assertEqual(rgb_gray.shape, (2, 2, 3))
        self.assertTrue(np.array_equal(rgb_gray[..., 0], gray))

        bgr = np.zeros((1, 1, 3), dtype=np.uint8)
        bgr[0, 0] = [1, 2, 3]
        self.assertEqual(visualization._as_rgb(bgr)[0, 0].tolist(), [3, 2, 1])

    def test_render_distribution_handles_empty_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "distribution.png"
            visualization.render_distribution(
                {
                    "grain_statistics": {
                        "diameters_um": [],
                        "count": 0,
                        "mean_diameter_um": 0,
                        "median_diameter_um": 0,
                        "std_diameter_um": 0,
                        "mean_aspect_ratio": 0,
                        "mean_circularity": 0,
                    }
                },
                str(out_path),
            )
            self.assertTrue(out_path.exists())


class TestRenderAllFromResults(unittest.TestCase):
    def test_render_all_from_results_rebuilds_all_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = np.zeros((40, 40, 3), dtype=np.uint8)
            image[10:30, 10:30, 1] = 255
            image_path = root / "input.png"
            cv2.imwrite(str(image_path), image)

            labels = np.zeros((40, 40), dtype=np.int32)
            labels[2:18, 2:18] = 1
            labels[2:18, 22:38] = 2
            labels[22:38, 2:18] = 3
            labels[22:38, 22:38] = 4
            labels_path = root / "demo_labels.npy"
            np.save(labels_path, labels)

            results = {
                "image_name": "demo",
                "image_path": "input.png",
                "artifacts": {
                    "labels_path": "demo_labels.npy",
                    "grain_props_path": "demo_grain_props.npy",
                },
                "segmentation": {"backend": "optical"},
                "grain_statistics": {
                    "count": 4,
                    "diameters_um": [10.0, 11.0, 12.0, 13.0],
                    "mean_diameter_um": 11.5,
                    "median_diameter_um": 11.5,
                    "std_diameter_um": 1.118,
                    "mean_aspect_ratio": 1.1,
                    "mean_circularity": 0.82,
                },
                "area_method": {
                    "n_inside": 0,
                    "n_edge": 0,
                    "n_corner": 4,
                    "n_equivalent": 1.0,
                    "n_a_per_mm2": 625.0,
                    "mean_diameter_um": 5.0,
                    "astm_g_value": 6.0,
                    "inside_grain_ids": [],
                    "edge_grain_ids": [],
                    "corner_grain_ids": [1, 2, 3, 4],
                },
                "intercept_method": {
                    "total_intersections": 8.0,
                    "total_line_length_um": 200.0,
                    "n_l_per_mm": 40.0,
                    "mean_intercept_length_um": 25.0,
                    "astm_g_value": 8.0,
                    "pattern_elements": [("line", 5, 5, 35, 35), ("circle", 20, 20, 10)],
                    "intersection_points": [(5, 5), (20, 20)],
                    "half_intersection_points": [(35, 35)],
                    "intersected_grain_ids": [1, 2, 3, 4],
                },
                "anomaly_detection": {
                    "has_anomaly": True,
                    "rule_a": {"triggered": True},
                    "rule_b": {"triggered": False},
                    "rule_c": {"triggered": False, "threshold_um": 30.0},
                    "total_anomalous_grains": 1,
                    "anomalous_grain_ids": [4],
                },
            }
            results_path = root / "demo_results.json"
            io_utils.save_json(str(results_path), results)

            paths = visualization.render_all_from_results(str(results_path))

            for key in ["original", "segmented", "area", "intercept", "anomaly", "distribution"]:
                self.assertTrue(Path(paths[key]).exists(), key)

    def test_render_all_from_results_rejects_legacy_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = np.zeros((10, 10, 3), dtype=np.uint8)
            image_path = root / "input.png"
            cv2.imwrite(str(image_path), image)
            labels_path = root / "demo_labels.npy"
            np.save(labels_path, np.ones((10, 10), dtype=np.int32))

            legacy_results = {
                "image_name": "demo",
                "image_path": str(image_path),
                "artifacts": {"labels_path": str(labels_path)},
                "segmentation": {"backend": "optical"},
                "grain_statistics": {"diameters": [], "mean_diameter_px": 0},
                "area_method": {},
                "intercept_method": {"total_line_length_px": 10},
                "anomaly_detection": {
                    "has_anomaly": False,
                    "rule_a": {"triggered": False},
                    "rule_b": {"triggered": False},
                    "rule_c": {"triggered": False},
                    "total_anomalous_grains": 0,
                    "anomalous_grain_ids": [],
                },
            }
            results_path = root / "demo_results.json"
            io_utils.save_json(str(results_path), legacy_results)

            with self.assertRaisesRegex(ValueError, "Legacy results.json"):
                visualization.render_all_from_results(str(results_path))

    def test_render_all_from_results_raises_for_missing_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = np.zeros((10, 10, 3), dtype=np.uint8)
            image_path = root / "input.png"
            cv2.imwrite(str(image_path), image)
            results = {
                "image_name": "demo",
                "image_path": str(image_path),
                "artifacts": {"labels_path": "missing.npy", "grain_props_path": "props.npy"},
                "segmentation": {"backend": "optical"},
                "grain_statistics": {
                    "count": 0,
                    "diameters_um": [],
                    "mean_diameter_um": 0.0,
                    "median_diameter_um": 0.0,
                    "std_diameter_um": 0.0,
                    "mean_aspect_ratio": 0.0,
                    "mean_circularity": 0.0,
                },
                "area_method": {},
                "intercept_method": {"total_line_length_um": 10.0},
                "anomaly_detection": {
                    "has_anomaly": False,
                    "rule_a": {"triggered": False},
                    "rule_b": {"triggered": False},
                    "rule_c": {"triggered": False, "threshold_um": 0.0},
                    "total_anomalous_grains": 0,
                    "anomalous_grain_ids": [],
                },
            }
            results_path = root / "demo_results.json"
            io_utils.save_json(str(results_path), results)

            with self.assertRaises(FileNotFoundError):
                visualization.render_all_from_results(str(results_path))


if __name__ == "__main__":
    unittest.main()
