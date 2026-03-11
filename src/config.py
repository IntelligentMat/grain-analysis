"""config.py — 严格对齐 main.py CLI 的 YAML 配置解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_GROUPS = ("run", "preprocessing", "segmentation", "analysis", "sam3")
FORBIDDEN_CONFIG_KEYS = {"input", "render-from-results"}


@dataclass(frozen=True)
class OptionSpec:
    group: str
    cli_name: str
    param_name: str
    pipeline_kwarg: str | None
    default: Any
    kind: str
    allow_none: bool = False
    choices: tuple[Any, ...] | None = None
    flags: tuple[str, ...] = ()


OPTION_SPECS = (
    OptionSpec(
        group="run",
        cli_name="output",
        param_name="output_dir",
        pipeline_kwarg="output_dir",
        default="./data",
        kind="str",
        flags=("--output", "-o"),
    ),
    OptionSpec(
        group="run",
        cli_name="segmentation-backend",
        param_name="segmentation_backend",
        pipeline_kwarg="segmentation_backend",
        default="optical",
        kind="str",
        choices=("optical", "sam3"),
        flags=("--segmentation-backend",),
    ),
    OptionSpec(
        group="preprocessing",
        cli_name="smooth-mode",
        param_name="smooth_mode",
        pipeline_kwarg="smooth_mode",
        default="gaussian",
        kind="str",
        choices=("gaussian", "bilateral", "anisotropic"),
        flags=("--smooth-mode",),
    ),
    OptionSpec(
        group="preprocessing",
        cli_name="gaussian-sigma",
        param_name="gaussian_sigma",
        pipeline_kwarg="gaussian_sigma",
        default=None,
        kind="float",
        allow_none=True,
        flags=("--gaussian-sigma",),
    ),
    OptionSpec(
        group="preprocessing",
        cli_name="median-kernel",
        param_name="median_kernel",
        pipeline_kwarg="median_kernel",
        default=3,
        kind="int",
        flags=("--median-kernel",),
    ),
    OptionSpec(
        group="preprocessing",
        cli_name="clahe-clip",
        param_name="clahe_clip",
        pipeline_kwarg="clahe_clip_limit",
        default=2.0,
        kind="float",
        flags=("--clahe-clip",),
    ),
    OptionSpec(
        group="segmentation",
        cli_name="min-distance",
        param_name="min_distance",
        pipeline_kwarg="min_distance",
        default=None,
        kind="int",
        allow_none=True,
        flags=("--min-distance",),
    ),
    OptionSpec(
        group="segmentation",
        cli_name="closing-disk",
        param_name="closing_disk",
        pipeline_kwarg="closing_disk_size",
        default=2,
        kind="int",
        flags=("--closing-disk",),
    ),
    OptionSpec(
        group="segmentation",
        cli_name="opening-disk",
        param_name="opening_disk",
        pipeline_kwarg="opening_disk_size",
        default=1,
        kind="int",
        flags=("--opening-disk",),
    ),
    OptionSpec(
        group="segmentation",
        cli_name="min-grain-area",
        param_name="min_grain_area",
        pipeline_kwarg="min_grain_area",
        default=None,
        kind="int",
        allow_none=True,
        flags=("--min-grain-area",),
    ),
    OptionSpec(
        group="segmentation",
        cli_name="remove-border",
        param_name="remove_border",
        pipeline_kwarg="remove_border",
        default=False,
        kind="bool",
        flags=("--remove-border", "--keep-border"),
    ),
    OptionSpec(
        group="analysis",
        cli_name="pixels-per-micron",
        param_name="pixels_per_micron",
        pipeline_kwarg="pixels_per_micron",
        default=1.0,
        kind="float",
        flags=("--pixels-per-micron",),
    ),
    OptionSpec(
        group="analysis",
        cli_name="min-intercept-px",
        param_name="min_intercept_px",
        pipeline_kwarg="min_intercept_px",
        default=3,
        kind="int",
        flags=("--min-intercept-px",),
    ),
    OptionSpec(
        group="analysis",
        cli_name="rule-a-threshold",
        param_name="rule_a_threshold",
        pipeline_kwarg="rule_a_threshold",
        default=3.0,
        kind="float",
        flags=("--rule-a-threshold",),
    ),
    OptionSpec(
        group="analysis",
        cli_name="rule-b-top-pct",
        param_name="rule_b_top_pct",
        pipeline_kwarg="rule_b_top_pct",
        default=5.0,
        kind="float",
        flags=("--rule-b-top-pct",),
    ),
    OptionSpec(
        group="analysis",
        cli_name="rule-b-area-frac",
        param_name="rule_b_area_frac",
        pipeline_kwarg="rule_b_area_frac",
        default=0.30,
        kind="float",
        flags=("--rule-b-area-frac",),
    ),
    OptionSpec(
        group="sam3",
        cli_name="sam3-model-id",
        param_name="sam3_model_id",
        pipeline_kwarg="sam3_model_id",
        default="facebook/sam3",
        kind="str",
        flags=("--sam3-model-id",),
    ),
    OptionSpec(
        group="sam3",
        cli_name="sam3-device",
        param_name="sam3_device",
        pipeline_kwarg="sam3_device",
        default="auto",
        kind="str",
        choices=("auto", "cpu", "cuda", "mps"),
        flags=("--sam3-device",),
    ),
    OptionSpec(
        group="sam3",
        cli_name="sam3-score-threshold",
        param_name="sam3_score_threshold",
        pipeline_kwarg="sam3_score_threshold",
        default=0.5,
        kind="float",
        flags=("--sam3-score-threshold",),
    ),
    OptionSpec(
        group="sam3",
        cli_name="sam3-mask-threshold",
        param_name="sam3_mask_threshold",
        pipeline_kwarg="sam3_mask_threshold",
        default=0.5,
        kind="float",
        flags=("--sam3-mask-threshold",),
    ),
    OptionSpec(
        group="sam3",
        cli_name="sam3-opening-disk",
        param_name="sam3_opening_disk",
        pipeline_kwarg="sam3_opening_disk_size",
        default=1,
        kind="int",
        flags=("--sam3-opening-disk",),
    ),
    OptionSpec(
        group="sam3",
        cli_name="sam3-closing-disk",
        param_name="sam3_closing_disk",
        pipeline_kwarg="sam3_closing_disk_size",
        default=2,
        kind="int",
        flags=("--sam3-closing-disk",),
    ),
    OptionSpec(
        group="sam3",
        cli_name="sam3-prompt-top-ratio",
        param_name="sam3_prompt_top_ratio",
        pipeline_kwarg="sam3_prompt_top_ratio",
        default=0.05,
        kind="float",
        flags=("--sam3-prompt-top-ratio",),
    ),
)


SPECS_BY_PARAM = {spec.param_name: spec for spec in OPTION_SPECS}
SPECS_BY_CLI_NAME = {spec.cli_name: spec for spec in OPTION_SPECS}


@dataclass
class ResolvedConfig:
    source_path: str | None
    effective: dict[str, dict[str, Any]]
    cli_overrides: dict[str, dict[str, Any]]
    runtime_values: dict[str, Any]

    @property
    def pipeline_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for spec in OPTION_SPECS:
            if spec.pipeline_kwarg is not None:
                kwargs[spec.pipeline_kwarg] = self.runtime_values[spec.param_name]
        return kwargs


def _load_yaml_module():
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required for --config support. Install the environment from environment.yml."
        ) from exc
    return yaml


def _validate_value(spec: OptionSpec, value: Any) -> Any:
    if value is None:
        if spec.allow_none:
            return None
        raise ValueError(f"Config key '{spec.cli_name}' does not allow null values.")

    if spec.kind == "bool":
        if isinstance(value, bool):
            normalized = value
        else:
            raise ValueError(f"Config key '{spec.cli_name}' must be a boolean.")
    elif spec.kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Config key '{spec.cli_name}' must be an integer.")
        normalized = int(value)
    elif spec.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Config key '{spec.cli_name}' must be a number.")
        normalized = float(value)
    elif spec.kind == "str":
        if not isinstance(value, str):
            raise ValueError(f"Config key '{spec.cli_name}' must be a string.")
        normalized = value
    else:
        raise ValueError(f"Unsupported config kind '{spec.kind}' for '{spec.cli_name}'.")

    if spec.choices is not None and normalized not in spec.choices:
        raise ValueError(
            f"Config key '{spec.cli_name}' must be one of {list(spec.choices)}; got {normalized!r}."
        )
    return normalized


def _normalize_grouped(values_by_param: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped = {group: {} for group in CONFIG_GROUPS}
    for spec in OPTION_SPECS:
        grouped[spec.group][spec.cli_name] = values_by_param[spec.param_name]
    return grouped


def _flatten_yaml_config(data: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for group, group_values in data.items():
        if group not in CONFIG_GROUPS:
            raise ValueError(
                f"Unknown config group '{group}'. Allowed groups: {', '.join(CONFIG_GROUPS)}."
            )
        if not isinstance(group_values, dict):
            raise ValueError(f"Config group '{group}' must contain a mapping.")

        for key, value in group_values.items():
            if key in FORBIDDEN_CONFIG_KEYS:
                raise ValueError(f"Config key '{key}' is CLI-only and cannot appear in config.")
            spec = SPECS_BY_CLI_NAME.get(key)
            if spec is None:
                raise ValueError(f"Unknown config key '{key}' in group '{group}'.")
            if spec.group != group:
                raise ValueError(
                    f"Config key '{key}' is not allowed in group '{group}'; expected group '{spec.group}'."
                )
            flat[spec.param_name] = _validate_value(spec, value)
    return flat


def load_config_file(config_path: str) -> tuple[str, dict[str, Any]]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    yaml = _load_yaml_module()
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a top-level mapping.")

    flat = _flatten_yaml_config(payload)
    return str(path), flat


def default_runtime_values() -> dict[str, Any]:
    return {spec.param_name: spec.default for spec in OPTION_SPECS}


def build_resolved_config(
    config_path: str | None,
    cli_values: dict[str, Any],
    explicit_param_names: set[str],
) -> ResolvedConfig:
    runtime_values = default_runtime_values()
    source_path: str | None = None
    if config_path:
        source_path, config_values = load_config_file(config_path)
        runtime_values.update(config_values)

    cli_overrides_by_param: dict[str, Any] = {}
    for param_name in explicit_param_names:
        spec = SPECS_BY_PARAM.get(param_name)
        if spec is None:
            continue
        normalized = _validate_value(spec, cli_values[param_name])
        runtime_values[param_name] = normalized
        cli_overrides_by_param[param_name] = normalized

    return ResolvedConfig(
        source_path=source_path,
        effective=_normalize_grouped(runtime_values),
        cli_overrides=_normalize_grouped(
            {spec.param_name: cli_overrides_by_param.get(spec.param_name) for spec in OPTION_SPECS}
        ),
        runtime_values=runtime_values,
    )


def prune_empty_override_groups(
    cli_overrides: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    cleaned: dict[str, dict[str, Any]] = {}
    for group, values in cli_overrides.items():
        group_values = {key: value for key, value in values.items() if value is not None}
        if group_values:
            cleaned[group] = group_values
    return cleaned


def was_option_explicit(arg: str, option_flags: tuple[str, ...]) -> bool:
    return any(
        arg == flag or (flag.startswith("--") and arg.startswith(f"{flag}="))
        for flag in option_flags
    )
