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

from src import io_utils
from src.analysis import AreaMethodResult, GrainStatistics, InterceptMethodResult
from src.anomaly import AnomalyResult, RuleAResult, RuleBResult, RuleCResult


class TestIoUtils(unittest.TestCase):
    def test_collect_images_filters_and_sorts_supported_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ["b.png", "a.jpg", "ignore.txt", "c.bmp"]:
                (root / name).write_bytes(b"x")

            paths = io_utils.collect_images(tmpdir)
            self.assertEqual([Path(path).name for path in paths], ["a.jpg", "b.png", "c.bmp"])

    def test_load_image_round_trip_and_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            image = np.zeros((6, 7, 3), dtype=np.uint8)
            image[..., 2] = 255
            cv2.imwrite(str(image_path), image)

            loaded = io_utils.load_image(str(image_path))
            self.assertEqual(loaded.shape, image.shape)
            self.assertEqual(loaded.dtype, np.uint8)

            with self.assertRaises(FileNotFoundError):
                io_utils.load_image(str(Path(tmpdir) / "missing.png"))

    def test_save_json_converts_numpy_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.json"
            payload = {
                "flag": np.bool_(True),
                "count": np.int32(3),
                "score": np.float32(1.5),
                "values": np.array([1, 2, 3], dtype=np.int16),
            }

            io_utils.save_json(str(path), payload)
            saved = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(saved, {"flag": True, "count": 3, "score": 1.5, "values": [1, 2, 3]})

    def test_save_results_json_writes_expected_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "results.json"
            io_utils.save_results_json(
                output_path=str(out_path),
                labels_path="/tmp/labels.npy",
                grain_props_path="/tmp/grain_props.npy",
                image_name="demo",
                image_path="/tmp/demo.png",
                image_shape=(20, 30, 3),
                segmentation_backend="optical",
                segmentation_method="watershed",
                segmentation_params={"min_distance": 10},
                total_grains=3,
                pixels_per_micron=2.25,
                stats=GrainStatistics(
                    count=3,
                    mean_diameter_um=8.5,
                    std_diameter_um=1.2,
                    min_diameter_um=7.0,
                    max_diameter_um=10.0,
                    median_diameter_um=8.0,
                    p10_diameter_um=7.2,
                    p90_diameter_um=9.8,
                    mean_area_um2=50.0,
                    total_area_um2=150.0,
                    mean_aspect_ratio=1.15,
                    mean_circularity=0.81,
                    diameters_um=[7.0, 8.5, 10.0],
                    areas_um2=[40.0, 50.0, 60.0],
                ),
                area_result=AreaMethodResult(
                    n_inside=1,
                    n_intersect=2,
                    n_edge=1,
                    n_corner=1,
                    n_equivalent=1.75,
                    n_a_per_mm2=1234.5,
                    astm_g_value=5.67,
                    mean_grain_area_mm2=0.001,
                    mean_diameter_um=12.3,
                    inside_grain_ids=[1],
                    edge_grain_ids=[2],
                    corner_grain_ids=[3],
                    intersect_grain_ids=[2, 3],
                ),
                intercept_result=InterceptMethodResult(
                    n_lines=4,
                    n_circles=3,
                    total_intersections=9.5,
                    total_line_length_um=88.9,
                    n_l_per_mm=106.862,
                    mean_intercept_length_um=10.526,
                    astm_g_value=8.9,
                    pattern_elements=[("line", 0, 0, 1, 1)],
                    intersection_points=[(1, 1)],
                    half_intersection_points=[(0, 0)],
                    intersected_grain_ids=[1, 2],
                ),
                anomaly_result=AnomalyResult(
                    has_anomaly=True,
                    rule_a=RuleAResult(True, 3.0, 2.5, [3]),
                    rule_b=RuleBResult(True, 5.0, 0.3, 0.4),
                    rule_c=RuleCResult(False, 30.0, []),
                    total_anomalous_grains=1,
                    anomalous_grain_ids=[3],
                ),
                extra_artifacts={"sam3_prompt_json_path": "/tmp/prompts.json"},
                segmentation_details={"mask_conversion": {"kept_masks": 2}},
                config_info={
                    "source_path": "/tmp/config.yml",
                    "effective": {"analysis": {"pixels-per-micron": 2.25}},
                    "cli_overrides": {"analysis": {"pixels-per-micron": 2.25}},
                },
            )

            payload = io_utils.load_results_json(str(out_path))
            self.assertEqual(payload["image_name"], "demo")
            self.assertEqual(payload["pixels_per_micron"], 2.25)
            self.assertEqual(payload["measurement_mode"], "physical_um")
            self.assertEqual(payload["segmentation"]["backend"], "optical")
            self.assertEqual(payload["segmentation"]["details"]["mask_conversion"]["kept_masks"], 2)
            self.assertEqual(payload["artifacts"]["sam3_prompt_json_path"], "/tmp/prompts.json")
            self.assertEqual(payload["artifacts"]["grain_props_path"], "/tmp/grain_props.npy")
            self.assertEqual(payload["grain_statistics"]["mean_aspect_ratio"], 1.15)
            self.assertEqual(payload["grain_statistics"]["mean_circularity"], 0.81)
            self.assertEqual(payload["area_method"]["corner_grain_ids"], [3])
            self.assertEqual(payload["intercept_method"]["counting_basis"], "grain_segments_n")
            self.assertEqual(payload["config"]["source_path"], "/tmp/config.yml")


if __name__ == "__main__":
    unittest.main()
