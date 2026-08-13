"""Detection-side tooling: dataset preparation, YOLO evaluation and reporting.

The realtime telemetry reader lives in `telemetry_ocr`. This package covers
everything that produces or measures the checkpoints it consumes.
"""

from telemetry_ocr import __version__

__all__ = ["__version__"]
