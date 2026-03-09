import math
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

from src.sam3_backend import masks_to_labels, select_top_grain_prompts


class TestSam3BackendHelpers(unittest.TestCase):
    def test_select_top_grain_prompts_uses_area_descending_ratio(self):
        labels = np.array([
            [1, 1, 1, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 3, 3, 0],
            [3, 0, 0, 4, 0],
        ], dtype=np.int32)

        prompts, masks = select_top_grain_prompts(labels, top_ratio=0.5, mode='both')
        self.assertEqual(len(prompts), math.ceil(4 * 0.5))
        self.assertEqual([prompt.grain_id for prompt in prompts], [1, 3])
        self.assertEqual(masks.shape, (2, 5, 5))

    def test_masks_to_labels_applies_open_close_postprocess(self):
        masks = np.zeros((2, 7, 7), dtype=bool)
        masks[0, 1:6, 1:6] = True
        masks[0, 3, 3] = False
        masks[1, 0, 0] = True
        scores = np.array([0.9, 0.3], dtype=np.float32)

        labels, stats = masks_to_labels(masks, scores=scores, opening_disk_size=1, closing_disk_size=1)
        self.assertEqual(stats['original_masks'], 2)
        self.assertEqual(stats['morphology_processed_masks'], 2)
        self.assertEqual(stats['kept_masks'], 1)
        self.assertEqual(stats['labeled_grains'], 1)
        self.assertEqual(stats['postprocess'], 'opening_then_closing')
        self.assertEqual(stats['opening_disk_size'], 1)
        self.assertEqual(stats['closing_disk_size'], 1)
        self.assertEqual(int(labels.max()), 1)
        self.assertEqual(int(labels[3, 3]), 1)
        self.assertEqual(int(labels[0, 0]), 0)
