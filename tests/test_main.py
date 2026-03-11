import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

mpl_dir = Path(tempfile.gettempdir()) / "grain_analysis_test_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as cli_module

HAS_YAML = importlib.util.find_spec("yaml") is not None


@unittest.skipUnless(HAS_YAML, "PyYAML not installed")
class TestMainCli(unittest.TestCase):
    def test_main_rejects_config_with_render_mode(self):
        runner = CliRunner()
        result = runner.invoke(
            cli_module.main,
            ["--render-from-results", "demo.json", "--config", "config.yml"],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--config 仅分析模式可用", result.output)

    def test_main_applies_cli_override_over_config(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yml"
            config_path.write_text(
                """
run:
  output: ./from-config
analysis:
  pixels-per-micron: 2.25
""".strip(),
                encoding="utf-8",
            )
            image_path = Path(tmpdir) / "demo.png"
            image_path.write_bytes(b"fake")

            with mock.patch.object(cli_module.io_utils, "collect_images", return_value=[str(image_path)]), mock.patch.object(
                cli_module.pipeline,
                "run",
                return_value={
                    "image_name": "demo",
                    "segmentation_backend": "optical",
                    "total_grains": 3,
                    "astm_g_area": 5.0,
                    "astm_g_intercept": 5.1,
                    "has_anomaly": False,
                },
            ) as run_mock:
                result = runner.invoke(
                    cli_module.main,
                    [
                        "--input",
                        str(tmpdir),
                        "--config",
                        str(config_path),
                        "--pixels-per-micron",
                        "1.5",
                    ],
                )

            self.assertEqual(result.exit_code, 0, result.output)
            kwargs = run_mock.call_args.kwargs
            self.assertEqual(kwargs["output_dir"], "./from-config")
            self.assertEqual(kwargs["pixels_per_micron"], 1.5)
            self.assertEqual(kwargs["config_info"]["source_path"], str(config_path.resolve()))
            self.assertEqual(
                kwargs["config_info"]["effective"]["analysis"]["pixels-per-micron"],
                1.5,
            )
            self.assertEqual(
                kwargs["config_info"]["cli_overrides"],
                {"analysis": {"pixels-per-micron": 1.5}},
            )


if __name__ == "__main__":
    unittest.main()
