from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.measure import regionprops
from skimage.morphology import binary_closing, binary_opening, disk

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@dataclass
class PromptBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: int = 1

    @property
    def as_xyxy(self) -> list[float]:
        left = min(self.x1, self.x2)
        top = min(self.y1, self.y2)
        right = max(self.x1, self.x2)
        bottom = max(self.y1, self.y2)
        return [float(left), float(top), float(right), float(bottom)]


@dataclass
class GrainPrompt:
    grain_id: int
    bbox_xyxy: list[int]
    centroid_rc: list[float]
    area_px: int
    mask_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "grain_id": int(self.grain_id),
            "bbox_xyxy": [int(v) for v in self.bbox_xyxy],
            "centroid_rc": [round(float(v), 3) for v in self.centroid_rc],
            "area_px": int(self.area_px),
        }
        if self.mask_index is not None:
            payload["mask_index"] = int(self.mask_index)
        return payload

    def as_prompt_box(self) -> PromptBox:
        x1, y1, x2, y2 = self.bbox_xyxy
        return PromptBox(x1=x1, y1=y1, x2=x2, y2=y2, label=1)


class Sam3InferenceError(RuntimeError):
    pass


class TransformersSam3Backend:
    def __init__(self, model_id: str, device: str = "auto") -> None:
        self.model_id = model_id
        self.device_preference = device
        self.device: str | None = None
        self._model = None
        self._processor = None
        self._torch = None

    def _lazy_load(self) -> None:
        if self._model is not None and self._processor is not None and self._torch is not None:
            return

        try:
            import torch
            from transformers import Sam3Model, Sam3Processor
        except Exception as exc:
            raise Sam3InferenceError(
                "Failed to import SAM3 dependencies. Install `transformers`, `torch`, and request access to the gated model first."
            ) from exc

        if self.device_preference == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = self.device_preference

        torch_dtype = torch.float32
        if resolved_device == "cuda":
            torch_dtype = torch.bfloat16

        try:
            model = Sam3Model.from_pretrained(self.model_id, torch_dtype=torch_dtype)
            processor = Sam3Processor.from_pretrained(self.model_id)
        except Exception as exc:
            raise Sam3InferenceError(
                "Could not load the SAM3 checkpoint. Make sure you have accepted access for the gated model, authenticated with Hugging Face, and if needed try `--sam3-device cpu`."
            ) from exc

        self._torch = torch
        self._model = model.to(resolved_device)
        self._model.eval()
        self._processor = processor
        self.device = resolved_device

    def predict(
        self,
        image: Image.Image,
        prompt_boxes: list[PromptBox],
        score_threshold: float,
        mask_threshold: float,
    ) -> dict[str, Any]:
        self._lazy_load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        assert self.device is not None

        if not prompt_boxes:
            raise Sam3InferenceError("SAM3 prompt inference requires at least one prompt box.")

        input_boxes = [[box.as_xyxy for box in prompt_boxes]]
        input_box_labels = [[int(box.label) for box in prompt_boxes]]

        inputs = self._processor(
            images=image,
            input_boxes=input_boxes,
            input_boxes_labels=input_box_labels,
            return_tensors="pt",
        )

        target_sizes = inputs.get("original_sizes")
        prepared_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            prepared_inputs[key] = value.to(self.device) if hasattr(value, "to") else value

        with self._torch.no_grad():
            outputs = self._model(**prepared_inputs)

        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            mask_threshold=mask_threshold,
            target_sizes=target_sizes.tolist() if hasattr(target_sizes, "tolist") else target_sizes,
        )[0]

        masks = (
            results["masks"].detach().cpu().numpy()
            if len(results["masks"])
            else np.empty((0, image.height, image.width), dtype=bool)
        )
        boxes = (
            results["boxes"].detach().cpu().numpy()
            if len(results["boxes"])
            else np.empty((0, 4), dtype=np.float32)
        )
        scores = (
            results["scores"].detach().cpu().numpy()
            if len(results["scores"])
            else np.empty((0,), dtype=np.float32)
        )

        return {
            "masks": masks.astype(bool),
            "boxes": boxes,
            "scores": scores,
            "device": self.device,
        }


def _postprocess_mask(
    mask: np.ndarray, opening_disk_size: int = 1, closing_disk_size: int = 2
) -> np.ndarray:
    result = mask.astype(bool)
    if opening_disk_size > 0:
        result = binary_opening(result, disk(int(opening_disk_size)))
    if closing_disk_size > 0:
        result = binary_closing(result, disk(int(closing_disk_size)))
    return result.astype(bool)


def masks_to_labels(
    masks: np.ndarray,
    scores: np.ndarray | None = None,
    opening_disk_size: int = 1,
    closing_disk_size: int = 2,
) -> tuple[np.ndarray, dict[str, int]]:
    if masks.ndim != 3:
        raise ValueError("masks must have shape (N, H, W)")

    masks = masks.astype(bool)
    num_masks = masks.shape[0]
    if scores is None or len(scores) != num_masks:
        scores = np.ones(num_masks, dtype=np.float32)
    else:
        scores = np.asarray(scores, dtype=np.float32)

    order = np.argsort(-scores) if num_masks else np.empty((0,), dtype=np.int32)
    labels = np.zeros(masks.shape[1:], dtype=np.int32)
    next_id = 1
    kept_masks = 0

    for idx in order:
        processed_mask = _postprocess_mask(
            masks[idx],
            opening_disk_size=opening_disk_size,
            closing_disk_size=closing_disk_size,
        )
        region = processed_mask & (labels == 0)
        if not region.any():
            continue
        labels[region] = next_id
        next_id += 1
        kept_masks += 1

    return labels, {
        "original_masks": int(num_masks),
        "morphology_processed_masks": int(num_masks),
        "kept_masks": int(kept_masks),
        "labeled_grains": int(next_id - 1),
        "postprocess": "opening_then_closing",
        "opening_disk_size": int(opening_disk_size),
        "closing_disk_size": int(closing_disk_size),
    }


def select_top_grain_prompts(
    labels: np.ndarray,
    top_ratio: float = 0.3,
    mode: str = "both",
) -> tuple[list[GrainPrompt], np.ndarray | None]:
    if labels.ndim != 2:
        raise ValueError("labels array must be 2D")
    if not (0 < float(top_ratio) <= 1.0):
        raise ValueError("top_ratio must be in (0, 1]")

    props = list(regionprops(labels))
    if not props:
        empty_masks = (
            np.empty((0, *labels.shape), dtype=bool) if mode in {"masks", "both"} else None
        )
        return [], empty_masks

    selected_count = max(1, int(math.ceil(len(props) * float(top_ratio))))
    selected_props = sorted(props, key=lambda prop: (-int(prop.area), int(prop.label)))[
        :selected_count
    ]

    prompts: list[GrainPrompt] = []
    masks: list[np.ndarray] = []
    for prop in selected_props:
        min_row, min_col, max_row, max_col = prop.bbox
        mask_index = None
        if mode in {"masks", "both"}:
            mask_index = len(masks)
            masks.append(labels == int(prop.label))
        prompts.append(
            GrainPrompt(
                grain_id=int(prop.label),
                bbox_xyxy=[int(min_col), int(min_row), int(max_col - 1), int(max_row - 1)],
                centroid_rc=[float(prop.centroid[0]), float(prop.centroid[1])],
                area_px=int(prop.area),
                mask_index=mask_index,
            )
        )

    masks_array = None
    if mode in {"masks", "both"}:
        masks_array = (
            np.stack(masks, axis=0).astype(bool)
            if masks
            else np.empty((0, *labels.shape), dtype=bool)
        )
    return prompts, masks_array


def export_prompt_package(
    labels_path: str | Path,
    output_prefix: str | Path,
    top_ratio: float = 0.3,
    mode: str = "both",
) -> tuple[dict[str, Path], list[GrainPrompt], np.ndarray | None]:
    labels_path = Path(labels_path)
    labels = np.load(labels_path)
    prompts, masks_array = select_top_grain_prompts(labels, top_ratio=top_ratio, mode=mode)

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_json = output_prefix.with_suffix(".json")

    payload: dict[str, Any] = {
        "source_labels_path": str(labels_path.resolve()),
        "image_shape": list(labels.shape),
        "mode": mode,
        "prompt_top_ratio": float(top_ratio),
        "num_prompts": len(prompts),
        "prompt_selected_grain_ids": [prompt.grain_id for prompt in prompts],
        "prompts": [prompt.as_dict() for prompt in prompts],
    }

    paths: dict[str, Path] = {"json": output_json}
    if mode in {"masks", "both"} and masks_array is not None:
        masks_path = output_prefix.with_name(f"{output_prefix.name}_masks.npz")
        np.savez_compressed(
            masks_path,
            masks=masks_array.astype(np.uint8),
            grain_ids=np.asarray([prompt.grain_id for prompt in prompts], dtype=np.int32),
        )
        payload["mask_prompts_path"] = str(masks_path.resolve())
        paths["masks"] = masks_path

    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths, prompts, masks_array


def run_prompted_sam3(
    image_path: str | Path,
    optical_labels_path: str | Path,
    output_prefix: str | Path,
    model_id: str,
    device: str,
    score_threshold: float,
    mask_threshold: float,
    prompt_top_ratio: float,
    opening_disk_size: int = 1,
    closing_disk_size: int = 2,
) -> dict[str, Any]:
    prompt_paths, prompts, _ = export_prompt_package(
        labels_path=optical_labels_path,
        output_prefix=output_prefix,
        top_ratio=prompt_top_ratio,
        mode="both",
    )
    prompt_boxes = [prompt.as_prompt_box() for prompt in prompts]

    if not prompt_boxes:
        empty_labels = np.zeros(
            np.asarray(Image.open(image_path).convert("RGB")).shape[:2], dtype=np.int32
        )
        return {
            "labels": empty_labels,
            "prompt_paths": prompt_paths,
            "prompt_selected_grain_ids": [],
            "mask_conversion": {
                "original_masks": 0,
                "morphology_processed_masks": 0,
                "kept_masks": 0,
                "labeled_grains": 0,
                "postprocess": "opening_then_closing",
                "opening_disk_size": int(opening_disk_size),
                "closing_disk_size": int(closing_disk_size),
            },
            "sam3_device": device,
            "raw_masks_path": None,
            "raw_json_path": None,
        }

    image = Image.open(image_path).convert("RGB")
    backend = TransformersSam3Backend(model_id=model_id, device=device)
    result = backend.predict(
        image=image,
        prompt_boxes=prompt_boxes,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
    )

    raw_masks_path = Path(f"{output_prefix}_raw_masks.npy")
    np.save(raw_masks_path, result["masks"].astype(np.uint8))

    raw_json_path = Path(f"{output_prefix}_raw.json")
    raw_payload = {
        "image_path": str(Path(image_path).resolve()),
        "model_id": model_id,
        "device": result["device"],
        "score_threshold": float(score_threshold),
        "mask_threshold": float(mask_threshold),
        "scores": np.asarray(result["scores"], dtype=np.float32).tolist(),
        "boxes": np.asarray(result["boxes"], dtype=np.float32).tolist(),
        "prompt_selected_grain_ids": [prompt.grain_id for prompt in prompts],
    }
    raw_json_path.write_text(
        json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    labels, mask_stats = masks_to_labels(
        np.asarray(result["masks"], dtype=bool),
        scores=np.asarray(result["scores"], dtype=np.float32),
        opening_disk_size=opening_disk_size,
        closing_disk_size=closing_disk_size,
    )

    return {
        "labels": labels,
        "prompt_paths": prompt_paths,
        "prompt_selected_grain_ids": [prompt.grain_id for prompt in prompts],
        "mask_conversion": mask_stats,
        "sam3_device": result["device"],
        "raw_masks_path": raw_masks_path,
        "raw_json_path": raw_json_path,
    }
