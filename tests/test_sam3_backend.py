import math
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
from PIL import Image

from src.sam3_backend import (
    GrainPrompt,
    PromptBox,
    export_prompt_package,
    masks_to_labels,
    run_prompted_sam3,
    select_top_grain_prompts,
)


class TestSam3PromptTypes(unittest.TestCase):
    def test_prompt_box_normalizes_coordinate_order(self):
        box = PromptBox(x1=9, y1=8, x2=3, y2=4)
        self.assertEqual(box.as_xyxy, [3.0, 4.0, 9.0, 8.0])

    def test_grain_prompt_serialization_and_prompt_box(self):
        prompt = GrainPrompt(
            grain_id=7,
            bbox_xyxy=[5, 6, 10, 12],
            centroid_rc=[8.1234, 9.8765],
            area_px=42,
            mask_index=1,
        )

        self.assertEqual(
            prompt.as_dict(),
            {
                "grain_id": 7,
                "bbox_xyxy": [5, 6, 10, 12],
                "centroid_rc": [8.123, 9.877],
                "area_px": 42,
                "mask_index": 1,
            },
        )
        self.assertEqual(prompt.as_prompt_box().as_xyxy, [5.0, 6.0, 10.0, 12.0])


class TestSam3BackendHelpers(unittest.TestCase):
    def test_select_top_grain_prompts_uses_area_descending_ratio(self):
        labels = np.array(
            [
                [1, 1, 1, 2, 2],
                [1, 1, 0, 2, 2],
                [0, 0, 0, 0, 0],
                [3, 3, 3, 3, 0],
                [3, 0, 0, 4, 0],
            ],
            dtype=np.int32,
        )

        prompts, masks = select_top_grain_prompts(labels, top_ratio=0.5, mode="both")
        self.assertIsNotNone(masks)
        assert masks is not None
        self.assertEqual(len(prompts), math.ceil(4 * 0.5))
        self.assertEqual([prompt.grain_id for prompt in prompts], [1, 3])
        self.assertEqual(masks.shape, (2, 5, 5))

    def test_masks_to_labels_applies_open_close_postprocess(self):
        masks = np.zeros((2, 7, 7), dtype=bool)
        masks[0, 1:6, 1:6] = True
        masks[0, 3, 3] = False
        masks[1, 0, 0] = True
        scores = np.array([0.9, 0.3], dtype=np.float32)

        labels, stats = masks_to_labels(
            masks, scores=scores, opening_disk_size=1, closing_disk_size=1
        )
        self.assertEqual(stats["original_masks"], 2)
        self.assertEqual(stats["morphology_processed_masks"], 2)
        self.assertEqual(stats["kept_masks"], 1)
        self.assertEqual(stats["labeled_grains"], 1)
        self.assertEqual(stats["postprocess"], "opening_then_closing")
        self.assertEqual(stats["opening_disk_size"], 1)
        self.assertEqual(stats["closing_disk_size"], 1)
        self.assertEqual(int(labels.max()), 1)
        self.assertEqual(int(labels[3, 3]), 1)
        self.assertEqual(int(labels[0, 0]), 0)

    def test_masks_to_labels_rejects_non_stack_input(self):
        with self.assertRaisesRegex(ValueError, r"shape \(N, H, W\)"):
            masks_to_labels(np.zeros((5, 5), dtype=bool))

    def test_export_prompt_package_writes_json_and_masks(self):
        labels = np.array(
            [
                [1, 1, 0, 2],
                [1, 1, 0, 2],
                [0, 0, 0, 2],
                [3, 3, 3, 0],
            ],
            dtype=np.int32,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            labels_path = Path(tmpdir) / "labels.npy"
            np.save(labels_path, labels)
            output_prefix = Path(tmpdir) / "prompts" / "demo_prompts"

            paths, prompts, masks = export_prompt_package(
                labels_path=labels_path,
                output_prefix=output_prefix,
                top_ratio=0.5,
                mode="both",
            )

            self.assertIsNotNone(masks)
            assert masks is not None
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["masks"].exists())
            self.assertEqual(len(prompts), 2)
            self.assertEqual(masks.shape, (2, 4, 4))

    def test_run_prompted_sam3_returns_empty_labels_when_no_prompts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "image.png"
            Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(image_path)

            labels_path = root / "labels.npy"
            np.save(labels_path, np.zeros((6, 8), dtype=np.int32))

            result = run_prompted_sam3(
                image_path=image_path,
                optical_labels_path=labels_path,
                output_prefix=root / "out" / "demo_prompts",
                model_id="unused",
                device="cpu",
                score_threshold=0.5,
                mask_threshold=0.5,
                prompt_top_ratio=0.5,
            )

            self.assertEqual(result["prompt_selected_grain_ids"], [])
            self.assertEqual(result["sam3_device"], "cpu")
            self.assertIsNone(result["raw_masks_path"])
            self.assertEqual(result["labels"].shape, (6, 8))
            self.assertEqual(int(result["labels"].max()), 0)
