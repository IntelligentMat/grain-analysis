#!/usr/bin/env python3
"""Convert optical `*_labels.npy` artifacts into SAM3 prompt packages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sam3_backend import export_prompt_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert optical labels.npy into SAM3 prompt packages.")
    parser.add_argument("--labels", required=True, help="Path to *_labels.npy")
    parser.add_argument("--output-prefix", default=None, help="Output prefix for prompt package files")
    parser.add_argument("--top-ratio", type=float, default=0.3, help="Select top-ratio grains by area")
    parser.add_argument("--mode", choices=["boxes", "masks", "both"], default="both", help="Which prompt artifacts to export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels)
    if args.output_prefix is None:
        stem = labels_path.stem[:-7] if labels_path.stem.endswith("_labels") else labels_path.stem
        output_prefix = labels_path.parent / f"{stem}_sam3_prompts"
    else:
        output_prefix = Path(args.output_prefix)

    paths, prompts, _ = export_prompt_package(
        labels_path=labels_path,
        output_prefix=output_prefix,
        top_ratio=args.top_ratio,
        mode=args.mode,
    )
    print(f"Saved prompt package JSON: {paths['json']}")
    if "masks" in paths:
        print(f"Saved mask prompt NPZ: {paths['masks']}")
    print(f"Selected grains: {[prompt.grain_id for prompt in prompts]}")


if __name__ == "__main__":
    main()
