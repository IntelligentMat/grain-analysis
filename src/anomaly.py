"""
anomaly.py — 异常晶粒判定模块

实现三条规则：
  规则 A：尺寸比法（d_max / d_avg > threshold）
  规则 B：长尾分布法（前 X% 大晶粒占面积比）
  规则 C：统计偏离法（3σ 准则）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.analysis import GrainProps, GrainStatistics


@dataclass
class RuleAResult:
    triggered: bool
    d_max_over_d_avg: float
    threshold: float
    anomalous_grain_ids: List[int] = field(default_factory=list)


@dataclass
class RuleBResult:
    triggered: bool
    top_pct: float
    area_fraction_threshold: float
    top_pct_area_fraction: float


@dataclass
class RuleCResult:
    triggered: bool
    threshold_um: float
    anomalous_grain_ids: List[int] = field(default_factory=list)


@dataclass
class AnomalyResult:
    has_anomaly: bool
    rule_a: RuleAResult
    rule_b: RuleBResult
    rule_c: RuleCResult
    total_anomalous_grains: int
    anomalous_grain_ids: List[int] = field(default_factory=list)


def detect_anomalies(
    grain_props: List[GrainProps],
    stats: GrainStatistics,
    rule_a_threshold: float = 3.0,
    rule_b_top_pct: float = 5.0,
    rule_b_area_frac_threshold: float = 0.30,
) -> AnomalyResult:
    """对晶粒列表执行三规则异常判定。"""
    if not grain_props:
        empty_a = RuleAResult(False, 0.0, rule_a_threshold)
        empty_b = RuleBResult(False, rule_b_top_pct, rule_b_area_frac_threshold, 0.0)
        empty_c = RuleCResult(False, 0.0)
        return AnomalyResult(False, empty_a, empty_b, empty_c, 0)

    diameters = np.array([g.equivalent_diameter_um for g in grain_props], dtype=np.float64)
    areas = np.array([g.area_um2 for g in grain_props], dtype=np.float64)
    ids = np.array([g.grain_id for g in grain_props], dtype=np.int32)

    d_avg = stats.mean_diameter_um
    d_max = stats.max_diameter_um

    ratio = d_max / d_avg if d_avg > 0 else 0.0
    rule_a_triggered = ratio > rule_a_threshold
    rule_a_anomalous = (
        ids[diameters > d_avg * rule_a_threshold].tolist() if rule_a_triggered else []
    )
    rule_a = RuleAResult(
        triggered=rule_a_triggered,
        d_max_over_d_avg=round(ratio, 3),
        threshold=rule_a_threshold,
        anomalous_grain_ids=rule_a_anomalous,
    )

    n_top = max(1, int(len(grain_props) * rule_b_top_pct / 100.0))
    sorted_idx = np.argsort(areas)[::-1]
    top_area_sum = areas[sorted_idx[:n_top]].sum()
    total_area = areas.sum()
    top_frac = top_area_sum / total_area if total_area > 0 else 0.0
    rule_b_triggered = top_frac > rule_b_area_frac_threshold
    rule_b = RuleBResult(
        triggered=rule_b_triggered,
        top_pct=rule_b_top_pct,
        area_fraction_threshold=rule_b_area_frac_threshold,
        top_pct_area_fraction=round(float(top_frac), 4),
    )

    threshold_c = stats.mean_diameter_um + 3 * stats.std_diameter_um
    rule_c_anomalous_mask = diameters > threshold_c
    rule_c_triggered = bool(rule_c_anomalous_mask.any())
    rule_c_anomalous = ids[rule_c_anomalous_mask].tolist()
    rule_c = RuleCResult(
        triggered=rule_c_triggered,
        threshold_um=round(float(threshold_c), 2),
        anomalous_grain_ids=rule_c_anomalous,
    )

    anomalous_set = set(rule_a_anomalous) | set(rule_c_anomalous)
    has_anomaly = rule_a_triggered or rule_b_triggered or rule_c_triggered

    return AnomalyResult(
        has_anomaly=has_anomaly,
        rule_a=rule_a,
        rule_b=rule_b,
        rule_c=rule_c,
        total_anomalous_grains=len(anomalous_set),
        anomalous_grain_ids=sorted(anomalous_set),
    )
