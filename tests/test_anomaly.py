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

from src.analysis import GrainProps, GrainStatistics, compute_grain_statistics
from src.anomaly import detect_anomalies


def make_grain(grain_id: int, area: float, diameter: float) -> GrainProps:
    return GrainProps(
        grain_id=grain_id,
        area_um2=area,
        perimeter_um=max(1.0, diameter * 3.14),
        equivalent_diameter_um=diameter,
        aspect_ratio=1.0,
        circularity=0.9,
        centroid_rc_px=(float(grain_id), float(grain_id)),
        bbox_rc_px=(0, 0, 1, 1),
    )


class TestAnomalyDetection(unittest.TestCase):
    def test_detect_anomalies_returns_empty_result_for_no_grains(self):
        result = detect_anomalies([], GrainStatistics())

        self.assertFalse(result.has_anomaly)
        self.assertEqual(result.total_anomalous_grains, 0)
        self.assertEqual(result.anomalous_grain_ids, [])
        self.assertEqual(result.rule_c.threshold_um, 0.0)

    def test_detect_anomalies_triggers_rules_a_and_b(self):
        grains = [
            make_grain(1, 100, 10),
            make_grain(2, 100, 10),
            make_grain(3, 100, 10),
            make_grain(4, 1600, 40),
        ]
        stats = compute_grain_statistics(grains)

        result = detect_anomalies(
            grains,
            stats,
            rule_a_threshold=2.0,
            rule_b_top_pct=25.0,
            rule_b_area_frac_threshold=0.5,
        )

        self.assertTrue(result.has_anomaly)
        self.assertTrue(result.rule_a.triggered)
        self.assertTrue(result.rule_b.triggered)
        self.assertFalse(result.rule_c.triggered)
        self.assertEqual(result.anomalous_grain_ids, [4])

    def test_detect_anomalies_triggers_rule_c_for_extreme_outlier(self):
        grains = [make_grain(index, 25, 5) for index in range(1, 21)]
        grains.append(make_grain(21, 2500, 200))
        stats = compute_grain_statistics(grains)

        result = detect_anomalies(grains, stats, rule_a_threshold=100.0)

        self.assertTrue(result.rule_c.triggered)
        self.assertGreater(result.rule_c.threshold_um, 0)
        self.assertIn(21, result.rule_c.anomalous_grain_ids)
        self.assertIn(21, result.anomalous_grain_ids)


if __name__ == "__main__":
    unittest.main()
