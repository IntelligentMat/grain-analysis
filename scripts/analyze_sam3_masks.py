#!/usr/bin/env python3
"""Analyze SAM3 instance masks with the shared grain-analysis pipeline helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import io_utils, pipeline
from src.sam3_backend import masks_to_labels


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON sidecar not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SAM3 masks with the existing grain analysis pipeline.")
    parser.add_argument("--masks", required=True, help="Path to *_masks.npy exported by the SAM3 GUI")
    parser.add_argument("--json", default=None, help="Optional sidecar JSON with scores/boxes/image_path")
    parser.add_argument("--image", default=None, help="Source image path; overrides the JSON field")
    parser.add_argument("--output", default="./data", help="Output root directory")
    parser.add_argument("--image-name", default=None, help="Override output image name")
    parser.add_argument("--pixels-per-micron", type=float, default=1.0, help="Scale factor for ASTM analysis")
    parser.add_argument("--min-intercept-px", type=int, default=3, help="Minimum intercept length")
    parser.add_argument("--sam3-opening-disk", type=int, default=1, help="Opening disk radius for each SAM3 mask")
    parser.add_argument("--sam3-closing-disk", type=int, default=2, help="Closing disk radius for each SAM3 mask")
    parser.add_argument("--rule-a-threshold", type=float, default=3.0)
    parser.add_argument("--rule-b-top-pct", type=float, default=5.0)
    parser.add_argument("--rule-b-area-frac", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    masks_path = Path(args.masks)
    if not masks_path.exists():
        raise FileNotFoundError(f"Masks file not found: {masks_path}")

    sidecar_path = Path(args.json) if args.json else masks_path.with_name(masks_path.stem.replace("_masks", "") + ".json")
    sidecar = _load_json(sidecar_path if sidecar_path.exists() else None)

    image_path = args.image or sidecar.get("image_path")
    if not image_path:
        raise ValueError("Please provide --image or a JSON sidecar containing image_path")

    image_name = args.image_name or Path(image_path).stem
    masks = np.load(masks_path)
    scores = np.asarray(sidecar.get("scores", []), dtype=np.float32) if sidecar.get("scores") is not None else None
    labels, mask_stats = masks_to_labels(
        masks,
        scores=scores,
        opening_disk_size=args.sam3_opening_disk,
        closing_disk_size=args.sam3_closing_disk,
    )
    image = io_utils.load_image(image_path)
    out_dir = Path(args.output) / image_name / "sam3"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = pipeline.run_analysis_from_labels(
        image_path=image_path,
        image=image,
        labels=labels,
        output_dir=args.output,
        image_name=image_name,
        segmentation_backend="sam3",
        segmentation_method="sam3_masks",
        segmentation_params={
            "source": "sam3_gui_masks",
            "opening_disk_size": int(args.sam3_opening_disk),
            "closing_disk_size": int(args.sam3_closing_disk),
        },
        pixels_per_micron=args.pixels_per_micron,
        min_intercept_px=args.min_intercept_px,
        rule_a_threshold=args.rule_a_threshold,
        rule_b_top_pct=args.rule_b_top_pct,
        rule_b_area_frac=args.rule_b_area_frac,
        extra_artifacts={
            "sam3_masks_path": str(masks_path.resolve()),
            **({"sam3_json_path": str(sidecar_path.resolve())} if sidecar_path.exists() else {}),
        },
        segmentation_details={"postprocess_mode": "opening_then_closing_per_mask", "mask_conversion": mask_stats},
    )

    print(f"Saved labels: {result['paths']['labels']}")
    print(f"Saved results: {result['paths']['json']}")
    print(f"Saved segmented preview: {result['paths']['segmented']}")


if __name__ == "__main__":
    main()
