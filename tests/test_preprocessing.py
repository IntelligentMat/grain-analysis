import os
import sys
import tempfile
import types
from typing import Any, cast
import unittest
from pathlib import Path
from unittest.mock import patch

mpl_dir = Path(tempfile.gettempdir()) / "grain_analysis_test_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import preprocessing


class TestPreprocessing(unittest.TestCase):
    def test_resolve_gaussian_sigma_returns_explicit_value(self):
        gray = np.zeros((8, 8), dtype=np.uint8)
        sigma = preprocessing._resolve_gaussian_sigma(
            gray,
            gaussian_sigma=1.75,
            noise_to_sigma_k=18.0,
            sigma_bounds=(0.8, 4.0),
        )
        self.assertEqual(sigma, 1.75)

    def test_resolve_gaussian_sigma_uses_noise_estimate_and_clips_bounds(self):
        gray = np.zeros((8, 8), dtype=np.uint8)
        with patch("src.preprocessing._estimate_noise_std", return_value=0.01):
            sigma = preprocessing._resolve_gaussian_sigma(
                gray,
                gaussian_sigma="auto",
                noise_to_sigma_k=50.0,
                sigma_bounds=(0.8, 0.9),
            )
        self.assertEqual(sigma, 0.8)

    def test_preprocess_gaussian_on_color_image_returns_enhanced_gray(self):
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        image[..., 1] = 80
        image[3:9, 3:9, 2] = 255

        enhanced = preprocessing.preprocess(
            image,
            smooth_mode="gaussian",
            gaussian_sigma=1.0,
            median_kernel=1,
            clahe_clip_limit=1.0,
        )

        self.assertEqual(enhanced.shape, image.shape[:2])
        self.assertEqual(enhanced.dtype, np.uint8)
        self.assertGreater(int(enhanced.max()), int(enhanced.min()))

    def test_preprocess_bilateral_on_gray_image(self):
        image = np.zeros((20, 20), dtype=np.uint8)
        image[:, 10:] = 200

        enhanced = preprocessing.preprocess(
            image,
            smooth_mode="bilateral",
            median_kernel=1,
            bilateral_d=5,
            bilateral_sigma_color=30,
            bilateral_sigma_space=30,
        )

        self.assertEqual(enhanced.shape, image.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_preprocess_anisotropic_uses_medpy_backend(self):
        image = np.full((10, 10), 120, dtype=np.uint8)

        def fake_diffusion(gray, niter, kappa, gamma):
            self.assertEqual(niter, 5)
            self.assertEqual(kappa, 30)
            self.assertEqual(gamma, 0.2)
            return gray.astype(np.float32) + 5

        medpy_module = types.ModuleType("medpy")
        filter_module = types.ModuleType("medpy.filter")
        smoothing_module = cast(Any, types.ModuleType("medpy.filter.smoothing"))
        smoothing_module.anisotropic_diffusion = fake_diffusion

        with patch.dict(
            sys.modules,
            {
                "medpy": medpy_module,
                "medpy.filter": filter_module,
                "medpy.filter.smoothing": smoothing_module,
            },
        ):
            enhanced = preprocessing.preprocess(
                image,
                smooth_mode="anisotropic",
                anisotropic_niter=5,
                anisotropic_kappa=30,
                anisotropic_gamma=0.2,
                median_kernel=1,
                clahe_clip_limit=1.0,
            )

        self.assertEqual(enhanced.shape, image.shape)
        self.assertEqual(enhanced.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
