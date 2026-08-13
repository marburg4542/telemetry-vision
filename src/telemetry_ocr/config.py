from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import ConfigError, DependencyError

Channel = Literal[
    "color", "gray", "b", "g", "r", "lab_l", "lab_a", "lab_b", "hsv_s", "hsv_v"
]
Morphology = Literal["none", "tophat", "blackhat"]
Threshold = Literal["none", "otsu", "adaptive"]
Profile = Literal["fast", "stable"]
DataType = Literal["string", "integer", "float", "dms"]
Normalizer = Literal["none", "numeric", "dms", "optics"]


@dataclass(slots=True)
class RecipeConfig:
    name: str = "lab_l_clahe"
    channel: Channel = "lab_l"
    scale: float = 2.0
    clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    unsharp: bool = True
    unsharp_amount: float = 0.8
    morphology: Morphology = "none"
    morphology_kernel: int = 3
    threshold: Threshold = "none"
    adaptive_block_size: int = 21
    adaptive_c: float = 5.0
    invert: bool = False


@dataclass(slots=True)
class FieldConfig:
    field_id: str
    box: tuple[float, float, float, float]
    allowed_chars: str
    regex: str
    data_type: DataType = "string"
    normalizer: Normalizer = "none"
    decimals: int | None = None
    dms_degrees_digits: int | None = None
    dms_decimal_places: int = 2
    hemisphere: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    profile: Profile = "stable"
    max_delta_per_second: float | None = None
    derive_from_source_time: bool = False
    ocr_interval: int = 1
    ocr_phase: int = 0
    padding: float | None = None
    primary: RecipeConfig = field(default_factory=RecipeConfig)
    fallback: RecipeConfig = field(
        default_factory=lambda: RecipeConfig(
            name="gray_adaptive",
            channel="gray",
            clahe=True,
            unsharp=True,
            threshold="adaptive",
        )
    )


@dataclass(slots=True)
class RuntimeConfig:
    model_name: str = "PP-OCRv6_tiny_rec"
    model_source: str = "BOS"
    model_cache_dir: str = "models/paddlex"
    device: str = "cpu"
    enable_hpi: bool = True
    enable_mkldnn: bool = True
    cpu_threads: int = 4
    recognizer_workers: int = 1
    batch_size: int = 10
    warmup_iterations: int = 2
    ocr_hz: float = 12.0
    queue_size: int = 2
    realtime: bool = True
    confidence_threshold: float = 0.80
    enable_fallback: bool = True
    padding: float = 0.08
    aspect_bucket_base: float = 4.0
    max_rois: int = 10


@dataclass(slots=True)
class TemporalProfileConfig:
    window: int
    required: int


@dataclass(slots=True)
class TemporalConfig:
    fast: TemporalProfileConfig = field(
        default_factory=lambda: TemporalProfileConfig(window=3, required=2)
    )
    stable: TemporalProfileConfig = field(
        default_factory=lambda: TemporalProfileConfig(window=5, required=3)
    )


@dataclass(slots=True)
class AppConfig:
    version: int = 1
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    fields: list[FieldConfig] = field(default_factory=list)


_CHANNELS = {"color", "gray", "b", "g", "r", "lab_l", "lab_a", "lab_b", "hsv_s", "hsv_v"}
_MORPHOLOGY = {"none", "tophat", "blackhat"}
_THRESHOLDS = {"none", "otsu", "adaptive"}
_PROFILES = {"fast", "stable"}
_DATA_TYPES = {"string", "integer", "float", "dms"}
_NORMALIZERS = {"none", "numeric", "dms", "optics"}


def _recipe(data: dict[str, Any] | None, default: RecipeConfig | None = None) -> RecipeConfig:
    base = asdict(default or RecipeConfig())
    if data:
        base.update(data)
    try:
        return RecipeConfig(**base)
    except TypeError as exc:
        raise ConfigError(f"Invalid preprocessing recipe: {exc}") from exc


def _field(data: dict[str, Any]) -> FieldConfig:
    item = dict(data)
    primary_data = item.pop("primary", None)
    fallback_data = item.pop("fallback", None)
    box = item.get("box")
    if isinstance(box, list):
        item["box"] = tuple(box)
    try:
        result = FieldConfig(**item)
    except TypeError as exc:
        raise ConfigError(f"Invalid field entry: {exc}") from exc
    result.primary = _recipe(primary_data, result.primary)
    result.fallback = _recipe(fallback_data, result.fallback)
    return result


def parse_config(data: dict[str, Any]) -> AppConfig:
    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a mapping")
    try:
        runtime = RuntimeConfig(**(data.get("runtime") or {}))
        temporal_data = data.get("temporal") or {}
        temporal = TemporalConfig(
            fast=TemporalProfileConfig(
                **(temporal_data.get("fast") or {"window": 3, "required": 2})
            ),
            stable=TemporalProfileConfig(
                **(temporal_data.get("stable") or {"window": 5, "required": 3})
            ),
        )
        config = AppConfig(
            version=int(data.get("version", 1)),
            runtime=runtime,
            temporal=temporal,
            fields=[_field(item) for item in (data.get("fields") or [])],
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc
    try:
        validate_config(config)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid configuration value type: {exc}") from exc
    return config


def load_config(path: str | Path) -> AppConfig:
    try:
        import yaml
    except ImportError as exc:
        raise DependencyError("PyYAML is required to read roi_config.yaml") from exc
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"Cannot parse {config_path}: {exc}") from exc
    return parse_config(data)


def dump_config(config: AppConfig, path: str | Path) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise DependencyError("PyYAML is required to write roi_config.yaml") from exc
    validate_config(config)
    data = asdict(config)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def clone_config(config: AppConfig) -> AppConfig:
    return copy.deepcopy(config)


def validate_config(config: AppConfig) -> None:
    if config.version != 1:
        raise ConfigError(f"Unsupported config version: {config.version}")
    runtime = config.runtime
    if not config.fields:
        raise ConfigError("At least one telemetry field is required")
    if len(config.fields) > runtime.max_rois:
        raise ConfigError(f"Configured {len(config.fields)} fields; max_rois is {runtime.max_rois}")
    if runtime.ocr_hz <= 0 or runtime.ocr_hz > 60:
        raise ConfigError("runtime.ocr_hz must be in (0, 60]")
    if runtime.queue_size not in (1, 2):
        raise ConfigError("runtime.queue_size must be 1 or 2")
    if runtime.cpu_threads <= 0 or runtime.batch_size <= 0:
        raise ConfigError("cpu_threads and batch_size must be positive")
    if runtime.recognizer_workers not in (1, 2):
        raise ConfigError("recognizer_workers must be 1 or 2")
    if runtime.warmup_iterations < 0 or runtime.warmup_iterations > 10:
        raise ConfigError("warmup_iterations must be between 0 and 10")
    if not runtime.model_name or not runtime.model_source or not runtime.model_cache_dir:
        raise ConfigError("model_name, model_source and model_cache_dir cannot be empty")
    if not 0 <= runtime.confidence_threshold <= 1:
        raise ConfigError("confidence_threshold must be between 0 and 1")
    if not 0 <= runtime.padding <= 0.5:
        raise ConfigError("runtime.padding must be between 0 and 0.5")
    if runtime.aspect_bucket_base <= 1:
        raise ConfigError("aspect_bucket_base must be greater than 1")

    for name, profile in (("fast", config.temporal.fast), ("stable", config.temporal.stable)):
        if profile.window <= 0 or profile.required <= 0 or profile.required > profile.window:
            raise ConfigError(f"temporal.{name} must satisfy 0 < required <= window")

    seen: set[str] = set()
    for item in config.fields:
        if not item.field_id or item.field_id in seen:
            raise ConfigError(f"Duplicate or empty field_id: {item.field_id!r}")
        seen.add(item.field_id)
        if len(item.box) != 4:
            raise ConfigError(f"{item.field_id}: box must have four normalized values")
        x, y, width, height = item.box
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise ConfigError(f"{item.field_id}: box must remain inside normalized frame bounds")
        if not item.allowed_chars:
            raise ConfigError(f"{item.field_id}: allowed_chars cannot be empty")
        try:
            re.compile(item.regex)
        except re.error as exc:
            raise ConfigError(f"{item.field_id}: invalid regex: {exc}") from exc
        if item.profile not in _PROFILES:
            raise ConfigError(f"{item.field_id}: profile must be fast or stable")
        if item.data_type not in _DATA_TYPES:
            raise ConfigError(f"{item.field_id}: unsupported data_type {item.data_type}")
        if item.normalizer not in _NORMALIZERS:
            raise ConfigError(f"{item.field_id}: unsupported normalizer {item.normalizer}")
        if item.decimals is not None and (item.decimals < 0 or item.data_type != "float"):
            raise ConfigError(f"{item.field_id}: decimals is only valid for float fields")
        if item.data_type == "dms":
            if item.normalizer != "dms":
                raise ConfigError(f"{item.field_id}: dms data_type requires normalizer: dms")
            if item.dms_degrees_digits not in (2, 3):
                raise ConfigError(f"{item.field_id}: dms_degrees_digits must be 2 or 3")
            if item.dms_decimal_places < 0 or item.dms_decimal_places > 4:
                raise ConfigError(f"{item.field_id}: dms_decimal_places must be between 0 and 4")
            if item.hemisphere not in {"N", "S", "E", "W"}:
                raise ConfigError(f"{item.field_id}: dms hemisphere must be N, S, E or W")
        elif item.normalizer == "dms":
            raise ConfigError(f"{item.field_id}: normalizer dms requires data_type: dms")
        if (
            item.min_value is not None
            and item.max_value is not None
            and item.min_value > item.max_value
        ):
            raise ConfigError(f"{item.field_id}: min_value cannot exceed max_value")
        if item.max_delta_per_second is not None and item.max_delta_per_second <= 0:
            raise ConfigError(f"{item.field_id}: max_delta_per_second must be positive")
        if item.derive_from_source_time:
            if item.data_type != "string" or item.normalizer != "none":
                raise ConfigError(
                    f"{item.field_id}: derive_from_source_time requires a plain string field"
                )
            if re.fullmatch(item.regex, "00:00:00") is None:
                raise ConfigError(
                    f"{item.field_id}: derive_from_source_time requires an HH:MM:SS regex"
                )
        if item.ocr_interval <= 0:
            raise ConfigError(f"{item.field_id}: ocr_interval must be positive")
        if item.ocr_phase < 0 or item.ocr_phase >= item.ocr_interval:
            raise ConfigError(
                f"{item.field_id}: ocr_phase must satisfy 0 <= phase < ocr_interval"
            )
        if item.padding is not None and not 0 <= item.padding <= 0.5:
            raise ConfigError(f"{item.field_id}: padding must be between 0 and 0.5")
        _validate_recipe(item.field_id, item.primary)
        _validate_recipe(item.field_id, item.fallback)


def _validate_recipe(field_id: str, recipe: RecipeConfig) -> None:
    prefix = f"{field_id}/{recipe.name}"
    if not recipe.name:
        raise ConfigError(f"{field_id}: preprocessing recipe needs a name")
    if recipe.channel not in _CHANNELS:
        raise ConfigError(f"{prefix}: unsupported channel {recipe.channel}")
    if recipe.morphology not in _MORPHOLOGY:
        raise ConfigError(f"{prefix}: unsupported morphology {recipe.morphology}")
    if recipe.threshold not in _THRESHOLDS:
        raise ConfigError(f"{prefix}: unsupported threshold {recipe.threshold}")
    if recipe.scale <= 0 or recipe.clahe_clip <= 0 or recipe.clahe_grid <= 0:
        raise ConfigError(f"{prefix}: scale and CLAHE values must be positive")
    if recipe.unsharp_amount < 0:
        raise ConfigError(f"{prefix}: unsharp_amount cannot be negative")
    if recipe.morphology_kernel <= 0 or recipe.morphology_kernel % 2 == 0:
        raise ConfigError(f"{prefix}: morphology_kernel must be a positive odd number")
    if recipe.adaptive_block_size <= 1 or recipe.adaptive_block_size % 2 == 0:
        raise ConfigError(f"{prefix}: adaptive_block_size must be an odd number greater than 1")
