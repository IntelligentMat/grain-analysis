import os
import sys
import tempfile
import unittest
from pathlib import Path

mpl_dir = Path(tempfile.gettempdir()) / 'grain_analysis_test_mpl'
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(mpl_dir))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.analysis import area_method, intercept_method


class TestAreaMethod(unittest.TestCase):
    def test_area_method_applies_inside_edge_corner_weights(self):
        labels = np.array([
            [3, 2, 2, 2, 0],
            [4, 1, 1, 2, 0],
            [4, 1, 1, 0, 0],
            [4, 4, 0, 5, 5],
            [0, 0, 0, 5, 5],
        ], dtype=np.int32)

        result = area_method(labels, pixels_per_micron=2.0)

        self.assertEqual(result.n_inside, 1)
        self.assertEqual(result.n_intersect, 4)
        self.assertEqual(result.n_edge, 2)
        self.assertEqual(result.n_corner, 2)
        self.assertEqual(result.inside_grain_ids, [1])
        self.assertEqual(result.edge_grain_ids, [2, 4])
        self.assertEqual(result.corner_grain_ids, [3, 5])
        self.assertEqual(result.intersect_grain_ids, [2, 3, 4, 5])
        self.assertAlmostEqual(result.n_equivalent, 2.5)
        self.assertAlmostEqual(result.n_a_per_mm2, 400000.0)
        self.assertAlmostEqual(result.mean_grain_area_mm2, 2.5e-6)
        self.assertAlmostEqual(result.mean_diameter_um, 1.7841241161527712)
        self.assertAlmostEqual(result.astm_g_value, 15.65604329119149)

    def test_area_method_rejects_non_positive_pixels_per_micron(self):
        labels = np.ones((3, 3), dtype=np.int32)
        with self.assertRaisesRegex(ValueError, 'pixels_per_micron'):
            area_method(labels, pixels_per_micron=0)


class TestInterceptMethod(unittest.TestCase):
    def test_intercept_method_counts_grain_segments_n(self):
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[:, :7] = 1
        labels[:, 7:14] = 2
        labels[:, 14:] = 3

        result = intercept_method(
            labels,
            pixels_per_micron=2.0,
            min_intercept_px=1,
            margin_ratio=0.05,
        )

        self.assertEqual(result.n_lines, 4)
        self.assertEqual(result.n_circles, 3)
        self.assertEqual(result.intersected_grain_ids, [1, 2, 3])
        self.assertEqual(len(result.pattern_elements), 7)
        self.assertEqual(len(result.intersection_points), 19)
        self.assertAlmostEqual(result.total_intersections, 16.0)
        self.assertAlmostEqual(result.n_l_per_px, 0.08761639411862952)
        self.assertAlmostEqual(result.mean_intercept_length_px, 11.413389127222413)
        self.assertAlmostEqual(result.mean_intercept_length_um, 5.706694563611206)
        self.assertAlmostEqual(result.astm_g_value, 11.612394673330234)

    def test_intercept_method_rejects_non_positive_pixels_per_micron(self):
        labels = np.ones((5, 5), dtype=np.int32)
        with self.assertRaisesRegex(ValueError, 'pixels_per_micron'):
            intercept_method(labels, pixels_per_micron=-1)


if __name__ == '__main__':
    unittest.main()
