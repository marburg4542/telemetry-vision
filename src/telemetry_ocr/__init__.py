"""Fixed-ROI telemetry OCR pipeline."""

from .config import AppConfig, FieldConfig, load_config

__all__ = ["AppConfig", "FieldConfig", "load_config"]
# Keep in step with `version` in pyproject.toml; both CLIs report this one.
__version__ = "0.5.0"
