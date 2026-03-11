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

from src import analysis


class TestAnalysisHelpers(unittest.TestCase):
    def test_extract_grain_props_and_statistics(self):
        labels = np.array(
            [
                [1, 1, 0, 2, 2],
                [1, 1, 0, 2, 2],
                [0, 0, 0, 0, 0],
                [3, 3, 3, 0, 0],
                [3, 3, 3, 0, 0],
            ],
            dtype=np.int32,
        )

        props = analysis.extract_grain_props(labels, pixels_per_micron=2.0)
        stats = analysis.compute_grain_statistics(props)

        self.assertEqual([prop.grain_id for prop in props], [1, 2, 3])
        self.assertEqual(stats.count, 3)
        self.assertAlmostEqual(stats.total_area_um2, 3.5)
        self.assertEqual(len(stats.diameters_um), 3)
        self.assertGreater(stats.mean_aspect_ratio, 0)
        self.assertGreaterEqual(stats.mean_circularity, 0)

    def test_compute_grain_statistics_handles_empty_props(self):
        stats = analysis.compute_grain_statistics([])
        self.assertEqual(stats.count, 0)
        self.assertEqual(stats.diameters_um, [])

    def test_segment_helpers_count_open_and_closed_intercepts(self):
        line_labels = np.array([1, 1, 1, 0, 2, 2, 2, 2, 0, 3, 3])
        segments = analysis._get_grain_segments(line_labels)
        self.assertEqual(segments, [(0, 3, 1), (4, 8, 2), (9, 11, 3)])

        groups = analysis._valid_grain_segment_groups(line_labels, min_intercept_px=3)
        self.assertEqual(groups, [[(0, 3, 1)], [(4, 8, 2)]])
        self.assertEqual(analysis._count_intercepts(line_labels, min_intercept_px=3), 1.5)
        self.assertEqual(analysis._intercepted_grain_ids_on_path(line_labels, min_intercept_px=3), {1, 2})

        closed_labels = np.array([4, 4, 0, 5, 5, 5, 0, 4, 4])
        closed_groups = analysis._valid_grain_segment_groups(
            closed_labels, min_intercept_px=2, is_closed=True
        )
        self.assertEqual(closed_groups, [[(7, 9, 4), (0, 2, 4)], [(3, 6, 5)]])
        self.assertEqual(analysis._count_intercepts(closed_labels, min_intercept_px=2, is_closed=True), 2.0)

    def test_intercept_position_helpers_mark_half_segments(self):
        labels = np.array([1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3])
        rr = np.arange(len(labels))
        cc = np.zeros_like(rr)

        points = analysis._intercept_positions_on_path(labels, rr, cc, min_intercept_px=3)
        half_points = analysis._half_intercept_positions_on_path(labels, rr, cc, min_intercept_px=3)

        self.assertEqual(points, [(0, 0), (5, 0), (10, 0)])
        self.assertEqual(half_points, [(0, 0), (10, 0)])

    def test_sort_circle_pixels_orders_by_angle(self):
        rr = np.array([1, 0, -1, 0])
        cc = np.array([0, 1, 0, -1])
        sorted_rr, sorted_cc = analysis._sort_circle_pixels(rr, cc, 0, 0)

        ordered = list(zip(sorted_rr.tolist(), sorted_cc.tolist()))
        self.assertEqual(ordered, [(-1, 0), (0, 1), (1, 0), (0, -1)])


if __name__ == "__main__":
    unittest.main()
