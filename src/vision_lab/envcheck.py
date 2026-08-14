"""Report whether this environment can run OCR, detection, or both.

The two halves of the project have incompatible CUDA requirements in a single
process, so "is my install working" is genuinely two questions. This checks
each stack independently, in the order that tells you the most, and reports
which assets the path manifest can actually reach.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from .paths import ProjectPaths
from .torch_env import describe_environment


def check_paddle() -> dict[str, Any]:
    """Probe PaddlePaddle in a child process.

    Importing paddle here would pull cuDNN into this interpreter, and if torch
    is imported later in the same run the second one fails to load its own copy
    (`WinError 127`). A child process keeps the diagnostic from breaking the
    thing it is diagnosing.
    """
    probe = (
        "import json;"
        "import paddle;"
        "print(json.dumps({"
        "'version': str(paddle.__version__),"
        "'compiled_with_cuda': bool(paddle.device.is_compiled_with_cuda()),"
        "'gpu_count': int(paddle.device.cuda.device_count())"
        "if paddle.device.is_compiled_with_cuda() else 0}))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"could not probe paddle: {exc}"}
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        return {"error": detail[-1] if detail else "paddle import failed"}
    try:
        return dict(json.loads(completed.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        return {"error": "paddle probe returned unexpected output"}


def check_assets(paths: ProjectPaths) -> dict[str, Any]:
    models = {}
    for name, path in sorted(paths.models.items()):
        models[name] = {
            "path": str(path),
            "present": path.is_file(),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 1) if path.is_file() else None,
        }
    directories = {
        name: {"path": str(path), "present": path.is_dir()}
        for name, path in sorted(paths.asset_dirs.items())
    }
    return {
        "repo_root": str(paths.root),
        "asset_root": str(paths.asset_root),
        "asset_root_present": paths.asset_root.is_dir(),
        "default_model": paths.default_model,
        "models": models,
        "asset_dirs": directories,
    }


def collect(paths: ProjectPaths | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": describe_environment(),
        "paddle": check_paddle(),
    }
    if paths is not None:
        report["assets"] = check_assets(paths)
    return report


def render(report: dict[str, Any]) -> str:
    lines = [f"python {report['python']}", f"  {report['executable']}", ""]

    torch_report = report["torch"]
    if torch_report.get("torch") is None:
        lines += ["detection stack (torch + ultralytics): NOT INSTALLED"]
        if torch_report.get("error"):
            lines += [f"  {line}" for line in str(torch_report["error"]).splitlines()]
    else:
        cuda = "yes" if torch_report["cuda_available"] else "no"
        lines += [
            "detection stack (torch + ultralytics)",
            f"  torch {torch_report['torch']} (cuda build: {torch_report['cuda_build']})",
            f"  ultralytics {torch_report.get('ultralytics') or 'NOT INSTALLED'}",
            f"  cuda available: {cuda}  -> selected device: "
            f"{torch_report.get('selected_device') or 'UNRESOLVED'}",
        ]
        if torch_report.get("device_error"):
            lines += [f"  {line}" for line in str(torch_report["device_error"]).splitlines()]
        for device in torch_report["devices"]:
            lines.append(
                f"  gpu {device['index']}: {device['name']} "
                f"({device['total_memory_gb']} GB, compute {device['compute_capability']})"
            )
    lines.append("")

    paddle_report = report["paddle"]
    if "error" in paddle_report:
        lines += ["ocr stack (paddlepaddle): NOT USABLE", f"  {paddle_report['error']}"]
    else:
        cuda = "yes" if paddle_report["compiled_with_cuda"] else "no (cpu build)"
        lines += [
            "ocr stack (paddlepaddle)",
            f"  paddle {paddle_report['version']}",
            f"  compiled with cuda: {cuda}, visible gpus: {paddle_report['gpu_count']}",
        ]

    assets = report.get("assets")
    if assets:
        lines += ["", f"repository: {assets['repo_root']}"]
        marker = "ok" if assets["asset_root_present"] else "MISSING"
        lines.append(f"asset root: {assets['asset_root']}  [{marker}]")
        for name, entry in assets["asset_dirs"].items():
            marker = "ok" if entry["present"] else "MISSING"
            lines.append(f"  {name:<16} {entry['path']}  [{marker}]")
        lines.append("checkpoints:")
        for name, entry in assets["models"].items():
            default = " (default)" if name == assets["default_model"] else ""
            marker = f"{entry['size_mb']} MB" if entry["present"] else "MISSING"
            lines.append(f"  {name:<16} {marker}{default}")

    if torch_report.get("cuda_available") and not report["paddle"].get("error"):
        lines += [
            "",
            "note: both stacks work, but never in one process on CUDA. The OCR "
            "server hides torch from itself and runs YOLO in a spawn child.",
        ]
    return "\n".join(lines)
