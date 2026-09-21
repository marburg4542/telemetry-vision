"""Train the vehicle detector on this machine's GPU, from two starting points.

The recipe is the one that trained 14th_opt on the GTX 1650 (4 GB): 640 px,
batch 2, AMP off (Ultralytics turns it off on GTX 16xx anyway), AdamW.

    14th       fine-tune 14th_opt, which already knows this camera's classes;
               fewer epochs and a lower learning rate
    visdrone   start again from yolov8m-visdrone, as 14th_opt itself did

The dataset is a Roboflow version (after the draft labels from
`telemetry-ocr training-frames` were corrected there), downloaded with the
ROBOFLOW_API_KEY from the environment or the project's .env; or any YOLOv8
data.yaml given with --data.

    .venv-cuda\\Scripts\\python.exe scripts\\train_local.py --version 14 --start both
    .venv-cuda\\Scripts\\python.exe scripts\\train_local.py --data path\\to\\data.yaml --start 14th
    .venv-cuda\\Scripts\\python.exe scripts\\train_local.py --version 14 --start both --resume

Runs are written to output/training/runs/<name>/, best weights in weights/best.pt.
Stop the web server's render first: both need the same 4 GB.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "output" / "training" / "runs"
DATASETS = ROOT / "output" / "training" / "datasets"

ROBOFLOW_WORKSPACE = "objects-detection-1orcw"
ROBOFLOW_PROJECT = "car-detection-yntbe"
VISDRONE_REPO = "mshamrai/yolov8m-visdrone"
FOURTEENTH = ROOT / "models" / "weights" / "14th_opt.pt"

# What differs between the two starts. The rest is shared (COMMON).
STARTS: dict[str, dict[str, Any]] = {
    "14th": {"epochs": 60, "lr0": 0.0005, "warmup_epochs": 1, "patience": 30},
    "visdrone": {"epochs": 150, "lr0": 0.001, "warmup_epochs": 3, "patience": 50},
}
COMMON: dict[str, Any] = {
    "imgsz": 640,
    # batch 4 grew to 3.9 GB after the first validation and spilled into RAM,
    # three times slower. Ultralytics accumulates gradients to 64 either way.
    "batch": 2,
    "workers": 4,
    "device": 0,
    "amp": False,
    "optimizer": "AdamW",
    "lrf": 0.01,
    "cos_lr": True,
    "weight_decay": 0.01,
    "warmup_bias_lr": 0.0,
    "scale": 0.6,
    "mixup": 0.1,
    "degrees": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "close_mosaic": 10,
    "pretrained": True,
    "exist_ok": True,
    "verbose": True,
}


def api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    env = ROOT / ".env"
    if not key and env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ROBOFLOW_API_KEY":
                key = value.strip().strip("\"'")
    if not key:
        sys.exit(f"ROBOFLOW_API_KEY not set: add ROBOFLOW_API_KEY=<key> to {env}")
    return key


def download(version: int) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("pip install roboflow into .venv-cuda, or download the dataset and pass --data")
    folder = DATASETS / f"car-detection-{version}"
    # Roboflow skips a folder that exists even when an earlier download broke off.
    project = Roboflow(api_key=api_key()).workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(version).download(
        "yolov8", location=str(folder), overwrite=not (folder / "data.yaml").is_file()
    )
    return Path(dataset.location) / "data.yaml"


def start_weights(start: str) -> str:
    if start == "14th":
        if not FOURTEENTH.is_file():
            sys.exit(f"missing {FOURTEENTH}")
        return str(FOURTEENTH)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=VISDRONE_REPO, filename="best.pt")


def train(start: str, data: Path, name: str, epochs: int | None, resume: bool) -> Path:
    from ultralytics import YOLO

    last = RUNS / name / "weights" / "last.pt"
    if resume and last.is_file():
        print(f"[{start}] resuming {last}")
        YOLO(str(last)).train(resume=True)
    else:
        settings = {**COMMON, **STARTS[start]}
        if epochs:
            settings["epochs"] = epochs
        print(f"[{start}] {name}: {settings['epochs']} epochs from {start_weights(start)}")
        YOLO(start_weights(start)).train(
            data=str(data), project=str(RUNS), name=name, **settings
        )
    return RUNS / name / "weights" / "best.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--version", type=int, help="Roboflow dataset version to download")
    source.add_argument("--data", type=Path, help="A YOLOv8 data.yaml already on disk")
    parser.add_argument("--start", choices=["14th", "visdrone", "both"], default="both")
    parser.add_argument("--epochs", type=int, help="Override the start's epoch count")
    parser.add_argument("--name", help="Run name prefix (default: v<version> or the data folder)")
    parser.add_argument("--resume", action="store_true", help="Carry on from last.pt")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        sys.exit("CUDA not available: run with .venv-cuda")
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU {gpu.name} ({gpu.total_memory / 1024**3:.1f} GB)")

    data = args.data.resolve() if args.data else download(args.version)
    if not data.is_file():
        sys.exit(f"missing {data}")
    prefix = args.name or (f"v{args.version}" if args.version else data.parent.name)
    starts = ["14th", "visdrone"] if args.start == "both" else [args.start]
    best = {
        start: train(start, data, f"{prefix}-from-{start}", args.epochs, args.resume)
        for start in starts
    }

    print("\nDone. Compare against 14th_opt on the held-out flights:")
    models = " ".join(f'"{path}"' for path in [FOURTEENTH, *best.values()])
    print(
        f"  .venv-cuda\\Scripts\\python.exe scripts\\compare_models.py --models {models} "
        f'--data "{data}" --video videos\\<held-out flight>.mp4 --start 60 --length 40'
    )
    return 0


# Windows DataLoader workers re-import this file; without the guard they train too.
if __name__ == "__main__":
    raise SystemExit(main())
