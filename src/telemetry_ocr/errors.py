class TelemetryOCRError(Exception):
    """Base exception for expected application errors."""


class ConfigError(TelemetryOCRError):
    """Raised when the ROI configuration is invalid."""


class DependencyError(TelemetryOCRError):
    """Raised when an optional runtime dependency is unavailable."""


class VideoError(TelemetryOCRError):
    """Raised when a video cannot be decoded."""


class RecognitionError(TelemetryOCRError):
    """Raised when the OCR backend returns malformed output."""

