#!/usr/bin/env python3
"""Interactive SAM3 GUI for image segmentation.

This tool provides a lightweight desktop GUI for SAM3 image prompting with:
- text prompts
- positive / negative exemplar boxes
- overlay visualization
- export of composited preview and raw masks

It uses the Hugging Face `transformers` SAM3 integration by default because the
current project environment already includes `transformers`, while the official
`sam3` package is not installed locally.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from labels_to_sam3_prompts import build_prompt_package

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from PIL import Image


@dataclass
class PromptBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: int

    @property
    def as_xyxy(self) -> list[float]:
        left = min(self.x1, self.x2)
        top = min(self.y1, self.y2)
        right = max(self.x1, self.x2)
        bottom = max(self.y1, self.y2)
        return [float(left), float(top), float(right), float(bottom)]

    def describe(self) -> str:
        kind = "positive" if self.label == 1 else "negative"
        x1, y1, x2, y2 = self.as_xyxy
        return f"{kind:8s} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"


class Sam3InferenceError(RuntimeError):
    pass


class TransformersSam3Backend:
    def __init__(self, model_id: str, device: str = "auto") -> None:
        self.model_id = model_id
        self.device_preference = device
        self.device = None
        self._model = None
        self._processor = None
        self._torch = None
        self._mask_generator = None

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
        elif resolved_device == "mps":
            torch_dtype = torch.float32

        try:
            model = Sam3Model.from_pretrained(self.model_id, torch_dtype=torch_dtype)
            processor = Sam3Processor.from_pretrained(self.model_id)
        except Exception as exc:
            raise Sam3InferenceError(
                "Could not load the SAM3 checkpoint. Make sure you have accepted access for the gated model, authenticated with Hugging Face, and if MPS still fails try `--device cpu`."
            ) from exc

        self._torch = torch
        self._model = model.to(resolved_device)
        self._model.eval()
        self._processor = processor
        self.device = resolved_device

    def _lazy_load_mask_generator(self) -> None:
        if self._mask_generator is not None:
            return

        self._lazy_load()
        assert self._torch is not None

        try:
            from transformers import pipeline
        except Exception as exc:
            raise Sam3InferenceError("Failed to import transformers pipeline for automatic mask generation.") from exc

        torch_dtype = self._torch.float32
        if self.device == "cuda":
            torch_dtype = self._torch.bfloat16
        elif self.device == "mps":
            torch_dtype = self._torch.float32

        device_arg = self.device
        if device_arg is None:
            device_arg = "cpu"

        try:
            self._mask_generator = pipeline(
                "mask-generation",
                model=self.model_id,
                device=device_arg,
                torch_dtype=torch_dtype,
            )
        except Exception as exc:
            raise Sam3InferenceError(
                "Could not initialize SAM3 automatic mask generation pipeline. If MPS fails, try `--device cpu`."
            ) from exc

    def generate_automatic_masks(
        self,
        image: Image.Image,
        points_per_batch: int,
    ) -> dict[str, Any]:
        self._lazy_load_mask_generator()
        assert self._mask_generator is not None
        assert self.device is not None

        outputs = self._mask_generator(image, points_per_batch=points_per_batch)
        raw_masks = outputs.get("masks", [])
        raw_scores = outputs.get("scores", [])
        raw_boxes = outputs.get("bounding_boxes", outputs.get("boxes", []))

        masks = []
        for mask in raw_masks:
            if hasattr(mask, "detach"):
                mask_arr = mask.detach().cpu().numpy()
            elif hasattr(mask, "numpy") and not isinstance(mask, np.ndarray):
                mask_arr = mask.numpy()
            else:
                mask_arr = np.asarray(mask)
            if mask_arr.ndim > 2:
                mask_arr = np.squeeze(mask_arr)
            masks.append(mask_arr.astype(bool))

        boxes = []
        for box in raw_boxes:
            if isinstance(box, dict):
                candidate = box.get("box") or box.get("bbox") or box.get("boxes")
                if candidate is None:
                    continue
                box = candidate
            if hasattr(box, "detach"):
                box = box.detach().cpu().numpy()
            box_arr = np.asarray(box, dtype=np.float32).reshape(-1)
            if box_arr.size == 4:
                boxes.append(box_arr)

        scores = []
        for score in raw_scores:
            if hasattr(score, "detach"):
                score = score.detach().cpu().item()
            elif hasattr(score, "item"):
                score = score.item()
            scores.append(float(score))

        height, width = image.height, image.width
        masks_array = np.stack(masks, axis=0) if masks else np.empty((0, height, width), dtype=bool)
        boxes_array = np.stack(boxes, axis=0) if boxes else np.empty((0, 4), dtype=np.float32)
        scores_array = np.asarray(scores, dtype=np.float32) if scores else np.empty((0,), dtype=np.float32)

        return {
            "masks": masks_array,
            "boxes": boxes_array,
            "scores": scores_array,
            "device": self.device,
        }

    def predict(
        self,
        image: Image.Image,
        text_prompt: str | None,
        prompt_boxes: list[PromptBox],
        score_threshold: float,
        mask_threshold: float,
    ) -> dict[str, Any]:
        self._lazy_load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        assert self.device is not None

        if not text_prompt and not prompt_boxes:
            raise Sam3InferenceError("Please provide a text prompt or at least one box prompt.")

        input_boxes = None
        input_box_labels = None
        if prompt_boxes:
            input_boxes = [[box.as_xyxy for box in prompt_boxes]]
            input_box_labels = [[int(box.label) for box in prompt_boxes]]

        inputs = self._processor(
            images=image,
            text=text_prompt or None,
            input_boxes=input_boxes,
            input_boxes_labels=input_box_labels,
            return_tensors="pt",
        )

        target_sizes = inputs.get("original_sizes")
        prepared_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                prepared_inputs[key] = value.to(self.device)
            else:
                prepared_inputs[key] = value

        with self._torch.no_grad():
            outputs = self._model(**prepared_inputs)

        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            mask_threshold=mask_threshold,
            target_sizes=target_sizes.tolist() if hasattr(target_sizes, "tolist") else target_sizes,
        )[0]

        masks = results["masks"].detach().cpu().numpy() if len(results["masks"]) else np.empty((0, image.height, image.width), dtype=bool)
        boxes = results["boxes"].detach().cpu().numpy() if len(results["boxes"]) else np.empty((0, 4), dtype=np.float32)
        scores = results["scores"].detach().cpu().numpy() if len(results["scores"]) else np.empty((0,), dtype=np.float32)

        return {
            "masks": masks.astype(bool),
            "boxes": boxes,
            "scores": scores,
            "device": self.device,
        }


class Sam3InteractiveApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.root.title("SAM3 Interactive GUI")
        self.root.geometry("1480x920")

        self.args = args
        self.backend = TransformersSam3Backend(model_id=args.model_id, device=args.device)
        self.current_image: Image.Image | None = None
        self.current_image_path: Path | None = None
        self.prompt_boxes: list[PromptBox] = []
        self.result_masks: np.ndarray | None = None
        self.result_boxes: np.ndarray | None = None
        self.result_scores: np.ndarray | None = None
        self.imported_mask_prompts: np.ndarray | None = None
        self.imported_prompt_source: Path | None = None
        self.selector: RectangleSelector | None = None
        self.draw_mode = tk.StringVar(value="positive")
        self.status_var = tk.StringVar(value="Ready")
        self.prompt_var = tk.StringVar(value=args.prompt or "")
        self.score_var = tk.DoubleVar(value=args.score_threshold)
        self.mask_var = tk.DoubleVar(value=args.mask_threshold)
        self.points_per_batch_var = tk.IntVar(value=args.points_per_batch)
        self.import_min_area_var = tk.IntVar(value=args.import_min_area)
        self.import_top_boxes_var = tk.IntVar(value=args.import_top_boxes)

        self._build_layout()
        if args.image:
            self.load_image(Path(args.image))

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=4)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        viewer_frame = ttk.Frame(self.root, padding=6)
        viewer_frame.grid(row=0, column=0, sticky="nsew")
        viewer_frame.rowconfigure(0, weight=1)
        viewer_frame.columnconfigure(0, weight=1)

        controls_frame = ttk.Frame(self.root, padding=10)
        controls_frame.grid(row=0, column=1, sticky="nsew")
        controls_frame.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.figure, master=viewer_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, viewer_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        self.selector = RectangleSelector(
            self.ax,
            self._on_rectangle_selected,
            useblit=True,
            button=[1],
            minspanx=4,
            minspany=4,
            spancoords="pixels",
            interactive=False,
            drag_from_anywhere=False,
        )

        row = 0
        ttk.Button(controls_frame, text="Open Image", command=self.on_open_image).grid(row=row, column=0, sticky="ew")
        row += 1
        ttk.Button(controls_frame, text="Import Traditional Prompts", command=self.on_import_traditional_prompts).grid(row=row, column=0, sticky="ew", pady=(6, 0))
        row += 1

        ttk.Label(controls_frame, text="Import min area").grid(row=row, column=0, sticky="w", pady=(12, 4))
        row += 1
        ttk.Entry(controls_frame, textvariable=self.import_min_area_var).grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Label(controls_frame, text="Import top boxes").grid(row=row, column=0, sticky="w", pady=(12, 4))
        row += 1
        ttk.Entry(controls_frame, textvariable=self.import_top_boxes_var).grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Label(controls_frame, text="Text prompt").grid(row=row, column=0, sticky="w", pady=(12, 4))
        row += 1
        prompt_entry = ttk.Entry(controls_frame, textvariable=self.prompt_var)
        prompt_entry.grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Label(controls_frame, text="Box mode").grid(row=row, column=0, sticky="w", pady=(12, 4))
        row += 1
        mode_frame = ttk.Frame(controls_frame)
        mode_frame.grid(row=row, column=0, sticky="ew")
        ttk.Radiobutton(mode_frame, text="Positive", value="positive", variable=self.draw_mode).pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="Negative", value="negative", variable=self.draw_mode).pack(anchor="w")
        row += 1

        ttk.Label(controls_frame, text="Prompt boxes").grid(row=row, column=0, sticky="w", pady=(12, 4))
        row += 1
        self.box_list = tk.Listbox(controls_frame, height=8)
        self.box_list.grid(row=row, column=0, sticky="nsew")
        controls_frame.rowconfigure(row, weight=1)
        row += 1

        box_buttons = ttk.Frame(controls_frame)
        box_buttons.grid(row=row, column=0, sticky="ew", pady=(6, 0))
        box_buttons.columnconfigure(0, weight=1)
        box_buttons.columnconfigure(1, weight=1)
        ttk.Button(box_buttons, text="Remove Selected", command=self.on_remove_selected_box).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(box_buttons, text="Clear Boxes", command=self.on_clear_boxes).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        row += 1

        ttk.Label(controls_frame, text="Score threshold").grid(row=row, column=0, sticky="w", pady=(12, 4))
        row += 1
        ttk.Scale(controls_frame, from_=0.05, to=0.95, variable=self.score_var, orient="horizontal").grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Label(controls_frame, text="Mask threshold").grid(row=row, column=0, sticky="w", pady=(12, 4))
        row += 1
        ttk.Scale(controls_frame, from_=0.05, to=0.95, variable=self.mask_var, orient="horizontal").grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Label(controls_frame, text="Auto points_per_batch").grid(row=row, column=0, sticky="w", pady=(12, 4))
        row += 1
        ttk.Entry(controls_frame, textvariable=self.points_per_batch_var).grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Button(controls_frame, text="Run SAM3", command=self.on_run_inference).grid(row=row, column=0, sticky="ew", pady=(16, 0))
        row += 1
        ttk.Button(controls_frame, text="Auto Segment All", command=self.on_auto_segment_all).grid(row=row, column=0, sticky="ew", pady=(6, 0))
        row += 1
        ttk.Button(controls_frame, text="Save Result", command=self.on_save_result).grid(row=row, column=0, sticky="ew", pady=(6, 0))
        row += 1
        ttk.Button(controls_frame, text="Reset Overlay", command=self.on_reset_overlay).grid(row=row, column=0, sticky="ew", pady=(6, 0))
        row += 1

        ttk.Label(controls_frame, text="Status").grid(row=row, column=0, sticky="w", pady=(16, 4))
        row += 1
        self.status_label = ttk.Label(controls_frame, textvariable=self.status_var, wraplength=280, justify="left")
        self.status_label.grid(row=row, column=0, sticky="ew")

    def set_status(self, message: str) -> None:
        self.status_var.set(message)
        self.root.update_idletasks()

    def on_open_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if file_path:
            self.load_image(Path(file_path))

    def load_image(self, image_path: Path) -> None:
        self.current_image = Image.open(image_path).convert("RGB")
        self.current_image_path = image_path
        self.prompt_boxes.clear()
        self.result_masks = None
        self.result_boxes = None
        self.result_scores = None
        self.imported_mask_prompts = None
        self.imported_prompt_source = None
        self._refresh_box_list()
        self._redraw()
        self.set_status(f"Loaded image: {image_path}")

    def _resolve_optional_path(self, base_file: Path, raw_path: str | None) -> Path | None:
        if not raw_path:
            return None
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        return (base_file.parent / candidate).resolve()

    def _box_area(self, box: PromptBox) -> float:
        x1, y1, x2, y2 = box.as_xyxy
        return max(0.0, x2 - x1 + 1.0) * max(0.0, y2 - y1 + 1.0)

    def _boxes_overlap(self, box_a: PromptBox, box_b: PromptBox) -> bool:
        ax1, ay1, ax2, ay2 = box_a.as_xyxy
        bx1, by1, bx2, by2 = box_b.as_xyxy
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    def _merge_box_group(self, boxes: list[PromptBox]) -> PromptBox:
        xs1, ys1, xs2, ys2 = [], [], [], []
        label = 1
        for box in boxes:
            x1, y1, x2, y2 = box.as_xyxy
            xs1.append(x1)
            ys1.append(y1)
            xs2.append(x2)
            ys2.append(y2)
            label = max(label, box.label)
        return PromptBox(x1=min(xs1), y1=min(ys1), x2=max(xs2), y2=max(ys2), label=label)

    def _filter_and_merge_imported_prompts(
        self,
        prompt_boxes: list[PromptBox],
        imported_masks: np.ndarray | None,
        areas_px: list[int] | None = None,
    ) -> tuple[list[PromptBox], np.ndarray | None, dict[str, int]]:
        min_area = max(1, int(self.import_min_area_var.get()))
        top_boxes = max(1, int(self.import_top_boxes_var.get()))

        if areas_px is None:
            areas_px = [int(self._box_area(box)) for box in prompt_boxes]

        entries = []
        for index, (box, area_px) in enumerate(zip(prompt_boxes, areas_px)):
            if int(area_px) < min_area:
                continue
            entries.append((index, box, int(area_px)))

        filtered_count = len(prompt_boxes) - len(entries)
        entries.sort(key=lambda item: item[2], reverse=True)
        entries = entries[:top_boxes]

        boxes_only = [item[1] for item in entries]
        kept_indices = [item[0] for item in entries]
        merged_count = 0
        merged_boxes: list[PromptBox] = []
        merged_masks: list[np.ndarray] = []
        used = [False] * len(boxes_only)

        for i, box in enumerate(boxes_only):
            if used[i]:
                continue

            partner_index = None
            for j in range(i + 1, len(boxes_only)):
                if used[j]:
                    continue
                if self._boxes_overlap(box, boxes_only[j]):
                    partner_index = j
                    break

            group = [box]
            group_indices = [kept_indices[i]]
            used[i] = True
            if partner_index is not None:
                used[partner_index] = True
                group.append(boxes_only[partner_index])
                group_indices.append(kept_indices[partner_index])
                merged_count += 1

            merged_boxes.append(self._merge_box_group(group))
            if imported_masks is not None and len(imported_masks) > 0:
                merged_mask = np.zeros_like(imported_masks[0], dtype=bool)
                for original_index in group_indices:
                    if original_index < len(imported_masks):
                        merged_mask |= imported_masks[original_index].astype(bool)
                merged_masks.append(merged_mask)

        merged_masks_array = None
        if imported_masks is not None and merged_masks:
            merged_masks_array = np.stack(merged_masks, axis=0)
        elif imported_masks is not None:
            merged_masks_array = np.empty((0, *imported_masks.shape[1:]), dtype=bool)

        stats = {
            "original": len(prompt_boxes),
            "filtered": filtered_count,
            "kept": len(entries),
            "merged_away": merged_count,
            "final": len(merged_boxes),
        }
        return merged_boxes, merged_masks_array, stats

    def _apply_imported_prompts(self, prompt_boxes: list[PromptBox], imported_masks: np.ndarray | None, source_path: Path, stats: dict[str, int] | None = None) -> None:
        self.prompt_boxes = prompt_boxes
        self.imported_mask_prompts = imported_masks
        self.imported_prompt_source = source_path
        self._refresh_box_list()
        self._redraw()
        mask_info = f", {len(imported_masks)} masks" if imported_masks is not None else ""
        stats_info = ""
        if stats is not None:
            stats_info = f" (orig={stats['original']}, filtered={stats['filtered']}, top={stats['kept']}, merged={stats['merged_away']})"
        self.set_status(f"Imported {len(prompt_boxes)} box prompts{mask_info} from {source_path.name}{stats_info}.")

    def on_import_traditional_prompts(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Import traditional segmentation prompts",
            filetypes=[
                ("Prompt packages", "*.json *.npy"),
                ("JSON", "*.json"),
                ("NumPy labels", "*.npy"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        source_path = Path(file_path)
        try:
            if source_path.suffix.lower() == ".npy":
                labels = np.load(source_path)
                prompts, imported_masks = build_prompt_package(labels, mode="both", min_area=1, max_prompts=None)
                prompt_boxes = [
                    PromptBox(
                        x1=float(prompt.bbox_xyxy[0]),
                        y1=float(prompt.bbox_xyxy[1]),
                        x2=float(prompt.bbox_xyxy[2]),
                        y2=float(prompt.bbox_xyxy[3]),
                        label=1,
                    )
                    for prompt in prompts
                ]
                areas_px = [int(prompt.area_px) for prompt in prompts]
            else:
                payload = json.loads(source_path.read_text(encoding="utf-8"))
                prompts = payload.get("prompts", [])
                prompt_boxes = [
                    PromptBox(
                        x1=float(item["bbox_xyxy"][0]),
                        y1=float(item["bbox_xyxy"][1]),
                        x2=float(item["bbox_xyxy"][2]),
                        y2=float(item["bbox_xyxy"][3]),
                        label=1,
                    )
                    for item in prompts
                    if "bbox_xyxy" in item and len(item["bbox_xyxy"]) == 4
                ]
                areas_px = [
                    int(item.get("area_px", max(1, (item["bbox_xyxy"][2] - item["bbox_xyxy"][0] + 1) * (item["bbox_xyxy"][3] - item["bbox_xyxy"][1] + 1))))
                    for item in prompts
                    if "bbox_xyxy" in item and len(item["bbox_xyxy"]) == 4
                ]
                imported_masks = None
                mask_prompts_path = self._resolve_optional_path(source_path, payload.get("mask_prompts_path"))
                if mask_prompts_path is not None and mask_prompts_path.exists():
                    mask_payload = np.load(mask_prompts_path)
                    imported_masks = np.asarray(mask_payload["masks"]).astype(bool)

            prompt_boxes, imported_masks, stats = self._filter_and_merge_imported_prompts(prompt_boxes, imported_masks, areas_px=areas_px)
            self._apply_imported_prompts(prompt_boxes, imported_masks, source_path, stats=stats)
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Import failed", str(exc))
            self.set_status(f"Import failed: {exc}")

    def on_remove_selected_box(self) -> None:
        selected = list(self.box_list.curselection())
        if not selected:
            return
        for index in sorted(selected, reverse=True):
            self.prompt_boxes.pop(index)
        self._refresh_box_list()
        self._redraw()

    def on_clear_boxes(self) -> None:
        self.prompt_boxes.clear()
        self._refresh_box_list()
        self._redraw()

    def on_reset_overlay(self) -> None:
        self.result_masks = None
        self.result_boxes = None
        self.result_scores = None
        self.imported_mask_prompts = None
        self.imported_prompt_source = None
        self._redraw()
        self.set_status("Cleared model outputs; prompt boxes are kept.")

    def _run_prediction(self, text_prompt: str | None, prompt_boxes: list[PromptBox], status_message: str) -> None:
        if self.current_image is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return

        try:
            self.set_status(status_message)
            result = self.backend.predict(
                image=self.current_image,
                text_prompt=text_prompt,
                prompt_boxes=prompt_boxes,
                score_threshold=float(self.score_var.get()),
                mask_threshold=float(self.mask_var.get()),
            )
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("SAM3 inference failed", str(exc))
            self.set_status(f"Inference failed: {exc}")
            return

        self.result_masks = result["masks"]
        self.result_boxes = result["boxes"]
        self.result_scores = result["scores"]
        self._redraw()
        self.set_status(
            f"Done on {result['device']}. Found {len(self.result_scores)} objects with threshold={self.score_var.get():.2f}."
        )

    def on_run_inference(self) -> None:
        text_prompt = self.prompt_var.get().strip()
        if not text_prompt:
            text_prompt = None
        self._run_prediction(
            text_prompt=text_prompt,
            prompt_boxes=self.prompt_boxes,
            status_message="Loading SAM3 and running inference...",
        )

    def on_auto_segment_all(self) -> None:
        if self.current_image is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return

        try:
            points_per_batch = max(1, int(self.points_per_batch_var.get()))
        except Exception:
            messagebox.showwarning("Invalid value", "points_per_batch must be a positive integer.")
            return

        try:
            self.set_status(f"Running SAM3 automatic mask generation (points_per_batch={points_per_batch})...")
            result = self.backend.generate_automatic_masks(
                image=self.current_image,
                points_per_batch=points_per_batch,
            )
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("SAM3 auto mask generation failed", str(exc))
            self.set_status(f"Auto segmentation failed: {exc}")
            return

        self.result_masks = result["masks"]
        self.result_boxes = result["boxes"]
        self.result_scores = result["scores"]
        self._redraw()
        self.set_status(
            f"Auto segmentation done on {result['device']}. Generated {len(self.result_masks)} masks (points_per_batch={points_per_batch})."
        )

    def on_save_result(self) -> None:
        if self.current_image is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return
        if self.result_masks is None:
            messagebox.showwarning("No result", "Run SAM3 first.")
            return

        default_dir = self.current_image_path.parent if self.current_image_path else Path.cwd()
        output_path = filedialog.asksaveasfilename(
            title="Save composited preview",
            initialdir=str(default_dir),
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not output_path:
            return

        output_png = Path(output_path)
        output_json = output_png.with_suffix(".json")
        output_masks = output_png.with_name(f"{output_png.stem}_masks.npy")
        self.figure.savefig(output_png, dpi=160, bbox_inches="tight")
        np.save(output_masks, self.result_masks.astype(np.uint8))

        payload = {
            "image_path": str(self.current_image_path) if self.current_image_path else None,
            "prompt": self.prompt_var.get().strip() or None,
            "points_per_batch": int(self.points_per_batch_var.get()),
            "score_threshold": float(self.score_var.get()),
            "mask_threshold": float(self.mask_var.get()),
            "prompt_boxes": [box.as_xyxy + [box.label] for box in self.prompt_boxes],
            "imported_prompt_source": str(self.imported_prompt_source) if self.imported_prompt_source else None,
            "scores": self.result_scores.tolist() if self.result_scores is not None else [],
            "boxes": self.result_boxes.tolist() if self.result_boxes is not None else [],
            "masks_path": str(output_masks),
        }
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.set_status(f"Saved preview to {output_png} and masks to {output_masks}.")

    def _on_rectangle_selected(self, eclick, erelease) -> None:
        if self.current_image is None or eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return
        label = 1 if self.draw_mode.get() == "positive" else 0
        new_box = PromptBox(
            x1=float(eclick.xdata),
            y1=float(eclick.ydata),
            x2=float(erelease.xdata),
            y2=float(erelease.ydata),
            label=label,
        )
        left, top, right, bottom = new_box.as_xyxy
        if abs(right - left) < 3 or abs(bottom - top) < 3:
            return
        self.prompt_boxes.append(new_box)
        self._refresh_box_list()
        self._redraw()

    def _refresh_box_list(self) -> None:
        self.box_list.delete(0, tk.END)
        for prompt_box in self.prompt_boxes:
            self.box_list.insert(tk.END, prompt_box.describe())

    def _build_overlay(self, base_rgb: np.ndarray) -> np.ndarray:
        overlay = base_rgb.astype(np.float32).copy()

        if self.imported_mask_prompts is not None and len(self.imported_mask_prompts) > 0:
            ref_color = np.array([255.0, 215.0, 0.0], dtype=np.float32)
            for mask in self.imported_mask_prompts:
                overlay[mask] = 0.88 * overlay[mask] + 0.12 * ref_color

        if self.result_masks is None or len(self.result_masks) == 0:
            return np.clip(overlay / 255.0, 0.0, 1.0)

        rng = np.random.default_rng(12345)
        for mask in self.result_masks:
            color = rng.uniform(0.15, 0.95, size=3).astype(np.float32)
            alpha = 0.42
            overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color * 255.0
        return np.clip(overlay / 255.0, 0.0, 1.0)

    def _redraw(self) -> None:
        self.ax.clear()
        self.ax.set_axis_off()

        if self.current_image is None:
            self.ax.text(0.5, 0.5, "Open an image to begin", ha="center", va="center", fontsize=16)
            self.canvas.draw_idle()
            return

        base_rgb = np.asarray(self.current_image)
        display_image = self._build_overlay(base_rgb)
        self.ax.imshow(display_image)

        for prompt_box in self.prompt_boxes:
            x1, y1, x2, y2 = prompt_box.as_xyxy
            color = "lime" if prompt_box.label == 1 else "red"
            self.ax.add_patch(
                Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2.0, linestyle="--")
            )

        if self.result_boxes is not None and self.result_scores is not None:
            for index, (box, score) in enumerate(zip(self.result_boxes, self.result_scores)):
                x1, y1, x2, y2 = [float(value) for value in box]
                self.ax.add_patch(
                    Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="cyan", linewidth=1.8)
                )
                self.ax.text(
                    x1,
                    max(0.0, y1 - 4.0),
                    f"#{index + 1} {score:.2f}",
                    fontsize=9,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
                )

        title = self.current_image_path.name if self.current_image_path else "image"
        if self.imported_prompt_source is not None:
            title += f" — prompts:{self.imported_prompt_source.stem}"
        if self.result_scores is not None:
            title += f" — {len(self.result_scores)} masks"
        self.ax.set_title(title)
        self.canvas.draw_idle()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive SAM3 GUI for image prompting.")
    parser.add_argument("--image", type=str, default=None, help="Optional image to open on startup.")
    parser.add_argument("--prompt", type=str, default=None, help="Optional initial text prompt.")
    parser.add_argument("--model-id", type=str, default="facebook/sam3", help="Hugging Face model id.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device selection.")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Instance score threshold.")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Binary mask threshold.")
    parser.add_argument("--points-per-batch", type=int, default=64, help="points_per_batch for automatic mask generation.")
    parser.add_argument("--import-min-area", type=int, default=200, help="Minimum area in pixels for imported traditional boxes.")
    parser.add_argument("--import-top-boxes", type=int, default=10, help="Keep only the top-N largest imported boxes before pairwise merging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = Sam3InteractiveApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
