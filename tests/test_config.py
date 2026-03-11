import importlib.util
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

from src import config as app_config

HAS_YAML = importlib.util.find_spec("yaml") is not None


@unittest.skipUnless(HAS_YAML, "PyYAML not installed")
class TestConfig(unittest.TestCase):
    def test_load_config_file_accepts_grouped_cli_aligned_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yml"
            config_path.write_text(
                """
run:
  output: ./custom-output
  segmentation-backend: optical
analysis:
  pixels-per-micron: 2.25
  min-intercept-px: 5
sam3:
  sam3-device: cpu
""".strip(),
                encoding="utf-8",
            )

            resolved_path, values = app_config.load_config_file(str(config_path))
            self.assertEqual(resolved_path, str(config_path.resolve()))
            self.assertEqual(values["output_dir"], "./custom-output")
            self.assertEqual(values["segmentation_backend"], "optical")
            self.assertEqual(values["pixels_per_micron"], 2.25)
            self.assertEqual(values["min_intercept_px"], 5)
            self.assertEqual(values["sam3_device"], "cpu")

    def test_load_config_file_rejects_unknown_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yml"
            config_path.write_text(
                """
analysis:
  unknown-param: 1
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unknown config key"):
                app_config.load_config_file(str(config_path))

    def test_load_config_file_rejects_misplaced_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yml"
            config_path.write_text(
                """
run:
  pixels-per-micron: 2.25
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected group 'analysis'"):
                app_config.load_config_file(str(config_path))

    def test_load_config_file_rejects_cli_only_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yml"
            config_path.write_text(
                """
input:
  path: ./image.png
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unknown config group"):
                app_config.load_config_file(str(config_path))

    def test_build_resolved_config_merges_defaults_config_and_cli(self):
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

            cli_values = {
                "output_dir": None,
                "pixels_per_micron": 1.5,
            }
            resolved = app_config.build_resolved_config(
                config_path=str(config_path),
                cli_values=cli_values,
                explicit_param_names={"pixels_per_micron"},
            )

            self.assertEqual(resolved.runtime_values["output_dir"], "./from-config")
            self.assertEqual(resolved.runtime_values["pixels_per_micron"], 1.5)
            self.assertEqual(resolved.effective["analysis"]["pixels-per-micron"], 1.5)
            self.assertEqual(resolved.effective["run"]["output"], "./from-config")
            self.assertEqual(
                app_config.prune_empty_override_groups(resolved.cli_overrides),
                {"analysis": {"pixels-per-micron": 1.5}},
            )


if __name__ == "__main__":
    unittest.main()
