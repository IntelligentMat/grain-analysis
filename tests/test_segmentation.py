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

from src import segmentation


class TestSegmentationHelpers(unittest.TestCase):
    def test_auto_min_distance_and_area_have_safe_lower_bounds(self):
        self.assertEqual(segmentation._auto_min_distance((100, 120), None), 10)
        self.assertEqual(segmentation._auto_min_distance((100, 120), 0), 1)
        self.assertEqual(segmentation._auto_min_grain_area(1), 20)
        self.assertGreater(segmentation._auto_min_grain_area(30), 20)

    def test_fill_unseeded_regions_adds_marker_to_each_component(self):
        grain_mask = np.zeros((7, 7), dtype=bool)
        grain_mask[1:3, 1:3] = True
        grain_mask[4:6, 4:6] = True
        seed_mask = np.zeros_like(grain_mask)
        distance = np.zeros_like(grain_mask, dtype=np.float32)
        distance[1:3, 1:3] = [[1, 2], [2, 3]]
        distance[4:6, 4:6] = [[1, 4], [3, 2]]

        filled = segmentation._fill_unseeded_regions(seed_mask, grain_mask, distance)
        self.assertEqual(int(filled.sum()), 2)
        self.assertTrue(filled[2, 2])
        self.assertTrue(filled[4, 5])

    def test_remove_small_and_border_grains(self):
        labels = np.array(
            [
                [1, 1, 0, 2],
                [1, 1, 0, 2],
                [0, 0, 3, 3],
                [4, 0, 3, 3],
            ],
            dtype=np.int32,
        )

        filtered = segmentation._remove_small_grains(labels, min_area=3)
        self.assertEqual(set(np.unique(filtered)), {0, 1, 3})

        border_removed = segmentation._remove_border_grains(filtered)
        self.assertEqual(set(np.unique(border_removed)), {0})


class TestSegmentationPipeline(unittest.TestCase):
    def test_segment_rejects_invalid_boundary_mode(self):
        enhanced = np.zeros((10, 10), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "boundary_mode"):
            segmentation.segment(enhanced, boundary_mode="unknown")

    def test_segment_splits_bright_grid_into_multiple_grains(self):
        enhanced = np.zeros((90, 90), dtype=np.uint8)
        enhanced[::30, :] = 255
        enhanced[:, ::30] = 255

        labels = segmentation.segment(
            enhanced,
            min_distance=8,
            closing_disk_size=1,
            opening_disk_size=1,
            min_grain_area=50,
            boundary_mode="bright",
            remove_border=False,
        )

        self.assertEqual(labels.dtype, np.int32)
        self.assertEqual(int(labels.max()), 9)
        center_labels = [int(labels[r, c]) for r, c in [(15, 15), (45, 45), (75, 75)]]
        self.assertEqual(len(set(center_labels)), 3)


if __name__ == "__main__":
    unittest.main()
