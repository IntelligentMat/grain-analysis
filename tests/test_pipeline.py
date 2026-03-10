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

import numpy as np
from src import pipeline


class TestPipelineOptical(unittest.TestCase):
    """Exercise the full optical pipeline on a deterministic fixture image."""

    def test_pipeline_runs_optical_flow_on_grid_fixture(self):
        fixture = Path(__file__).resolve().parent / "fixtures" / "grid_fixture.png"
        self.assertTrue(fixture.exists(), f"Missing fixture: {fixture}")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipeline.run(
                image_path=str(fixture),
                output_dir=tmpdir,
                smooth_mode="gaussian",
                gaussian_sigma=0.8,
                median_kernel=1,
                clahe_clip_limit=1.0,
                segmentation_backend="optical",
                min_distance=8,
                closing_disk_size=1,
                opening_disk_size=1,
                min_grain_area=50,
                remove_border=False,
                pixels_per_micron=1.0,
                min_intercept_px=3,
            )

            self.assertEqual(result["segmentation_backend"], "optical")
            self.assertEqual(result["segmentation_method"], "watershed")
            self.assertEqual(result["total_grains"], 9)
            self.assertTrue(Path(result["output_dir"]).as_posix().endswith("/grid_fixture/optical"))

            self.assertAlmostEqual(result["area_result"].n_inside, 1)
            self.assertAlmostEqual(result["area_result"].n_edge, 4)
            self.assertAlmostEqual(result["area_result"].n_corner, 4)
            self.assertAlmostEqual(result["area_result"].n_equivalent, 4.0)
            self.assertEqual(result["area_result"].inside_grain_ids, [5])
            self.assertEqual(result["area_result"].edge_grain_ids, [2, 4, 6, 7])
            self.assertEqual(result["area_result"].corner_grain_ids, [1, 3, 8, 9])

            self.assertAlmostEqual(result["intercept_result"].total_intersections, 21.0)
            self.assertAlmostEqual(result["intercept_result"].n_l_per_px, 0.02489650347979036)
            self.assertAlmostEqual(
                result["intercept_result"].mean_intercept_length_um, 40.16628282006532
            )
            self.assertEqual(
                result["intercept_result"].intersected_grain_ids, [1, 2, 3, 4, 5, 6, 7, 8, 9]
            )

            labels_path = Path(result["paths"]["labels"])
            json_path = Path(result["paths"]["json"])
            self.assertTrue(labels_path.exists())
            self.assertTrue(json_path.exists())
            for key in ["original", "segmented", "area", "intercept", "anomaly"]:
                self.assertTrue(Path(result["paths"][key]).exists(), key)

            labels = np.load(labels_path)
            self.assertEqual(labels.shape, (90, 90))
            self.assertEqual(int(labels.max()), 9)
            centers = [
                (15, 15),
                (15, 45),
                (15, 75),
                (45, 15),
                (45, 45),
                (45, 75),
                (75, 15),
                (75, 45),
                (75, 75),
            ]
            center_labels = [int(labels[r, c]) for r, c in centers]
            self.assertEqual(set(center_labels), set(range(1, 10)))

            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertEqual(payload["segmentation"]["backend"], "optical")
            self.assertEqual(payload["segmentation"]["method"], "watershed")
            self.assertEqual(payload["intercept_method"]["counting_basis"], "grain_segments_n")
            self.assertEqual(payload["area_method"]["n_edge"], 4)
            self.assertEqual(payload["area_method"]["n_corner"], 4)
            self.assertEqual(payload["area_method"]["inside_grain_ids"], [5])
            self.assertEqual(payload["area_method"]["corner_grain_ids"], [1, 3, 8, 9])
            self.assertEqual(
                payload["intercept_method"]["intersected_grain_ids"], [1, 2, 3, 4, 5, 6, 7, 8, 9]
            )
            self.assertAlmostEqual(payload["intercept_method"]["total_intersections"], 21.0)
