"""Import Ultralytics the same way the realtime OCR worker does.

Both halves of this project load the same checkpoints, so they have to agree on
three things: where Ultralytics keeps its settings, how a CUDA request that
cannot be honoured is reported, and how `torch.load` is relaxed for the older
checkpoints in `models/`. The rules live in `telemetry_ocr.detection`; this
module reuses them for the offline commands.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from telemetry_ocr.detection import configure_ultralytics_home, resolve_torch_device
from telemetry_ocr.errors import DependencyError


def import_ultralytics() -> tuple[Any, Any]:
    """Return `(torch, YOLO)` with the Ultralytics settings directory pinned."""
    configure_ultralytics_home()
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise DependencyError(
            "Ultralytics YOLO is not installed in this environment; run "
            '`python -m pip install -e ".[ai]"`.'
        ) from exc
    return torch, YOLO


@contextmanager
def relaxed_checkpoint_loading(torch: Any) -> Iterator[None]:
    """Allow PyTorch 2.6+ to unpickle the model objects in these checkpoints.

    `weights_only=True` became the default and refuses them. The override is
    scoped to the `with` block rather than left installed for the process,
    because it disables the protection for every later load as well — and these
    checkpoints already execute code on load, so the narrower the window the
    better.
    """
    original_load = torch.load

    def compatible_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compatible_load
    try:
        yield
    finally:
        torch.load = original_load


def describe_environment() -> dict[str, Any]:
    """Report what PyTorch and Ultralytics see, for `vision-lab check`."""
    report: dict[str, Any] = {"torch": None, "cuda_available": False, "devices": []}
    try:
        import torch
    except ImportError:
        report["error"] = (
            "PyTorch is not installed. Install the CUDA build with:\n"
            "  python -m pip install torch==2.13.0+cu126 torchvision==0.28.0+cu126 "
            "--index-url https://download.pytorch.org/whl/cu126"
        )
        return report

    report["torch"] = str(torch.__version__)
    report["cuda_build"] = str(torch.version.cuda or "cpu-only build")
    report["cuda_available"] = bool(torch.cuda.is_available())
    if report["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            report["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_gb": round(properties.total_memory / (1024**3), 2),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    report["selected_device"] = resolve_torch_device(torch, None)

    try:
        import ultralytics

        report["ultralytics"] = str(ultralytics.__version__)
    except ImportError:
        report["ultralytics"] = None
    return report
