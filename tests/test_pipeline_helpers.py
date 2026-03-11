import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

mpl_dir = Path(tempfile.gettempdir()) / "grain_analysis_test_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src import pipeline


class TestPipelineHelpers(unittest.TestCase):
    def test_normalize_backend_accepts_alias_and_rejects_invalid_values(self):
        self.assertEqual(pipeline._normalize_backend("watershed"), "optical")
        self.assertEqual(pipeline._normalize_backend("optical"), "optical")
        self.assertEqual(pipeline._normalize_backend("sam3"), "sam3")
        with self.assertRaisesRegex(ValueError, "segmentation_backend"):
            pipeline._normalize_backend("bad")

    def test_run_uses_sam3_backend_with_existing_optical_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "sample.png"
            image = np.zeros((12, 12, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), image)

            optical_dir = root / "out" / "sample" / "optical"
            optical_dir.mkdir(parents=True)
            optical_labels_path = optical_dir / "sample_labels.npy"
            np.save(optical_labels_path, np.ones((12, 12), dtype=np.int32))

            fake_sam3_result = {
                "labels": np.full((12, 12), 2, dtype=np.int32),
                "prompt_paths": {
                    "json": root / "prompts.json",
                    "masks": root / "prompts_masks.npz",
                },
                "prompt_selected_grain_ids": [1, 2],
                "mask_conversion": {"kept_masks": 2},
                "sam3_device": "cpu",
                "raw_masks_path": root / "raw_masks.npy",
                "raw_json_path": root / "raw.json",
            }

            with patch(
                "src.pipeline.run_prompted_sam3", return_value=fake_sam3_result
            ) as sam3_mock:
                with patch(
                    "src.pipeline.run_analysis_from_labels", return_value={"ok": True}
                ) as analysis_mock:
                    result = pipeline.run(
                        image_path=str(image_path),
                        output_dir=str(root / "out"),
                        segmentation_backend="sam3",
                    )

            self.assertEqual(result, {"ok": True})
            sam3_mock.assert_called_once()
            analysis_mock.assert_called_once()
            self.assertEqual(analysis_mock.call_args.kwargs["segmentation_backend"], "sam3")
            self.assertEqual(
                analysis_mock.call_args.kwargs["segmentation_details"]["prompt_selected_grain_ids"],
                [1, 2],
            )


if __name__ == "__main__":
    unittest.main()
