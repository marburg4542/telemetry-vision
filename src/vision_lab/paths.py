"""Locate the repository, the checkpoints and the media kept outside of it.

The training-side media is roughly eight gigabytes, so it lives next to the
repository rather than in it. Every command that needs a video, a frame dump or
a dataset resolves it through here, which means a machine with the media on
another disk only edits `configs/paths.yaml` or sets one environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from telemetry_ocr.errors import ConfigError

MANIFEST_RELATIVE_PATH = Path("configs") / "paths.yaml"
# Written by the web app when models are added, removed or made the default.
# Untracked: it describes this machine's checkpoints, not the project's.
OVERLAY_RELATIVE_PATH = Path("configs") / "models.local.yaml"
SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv"})


def repo_root() -> Path:
    """Return the project root, the directory that holds `configs/paths.yaml`."""
    override = os.environ.get("TELEMETRY_VISION_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / MANIFEST_RELATIVE_PATH).is_file():
            raise ConfigError(
                f"TELEMETRY_VISION_ROOT points at {root}, which has no "
                f"{MANIFEST_RELATIVE_PATH.as_posix()}"
            )
        return root

    # src/vision_lab/paths.py -> src -> root. True for the editable install the
    # project is developed and run from.
    packaged = Path(__file__).resolve().parents[2]
    if (packaged / MANIFEST_RELATIVE_PATH).is_file():
        return packaged

    # A wheel install puts the package under site-packages instead, so fall
    # back to searching upwards from the working directory.
    cwd = Path.cwd().resolve()
    for directory in (cwd, *cwd.parents):
        if (directory / MANIFEST_RELATIVE_PATH).is_file():
            return directory

    raise ConfigError(
        "Could not locate the project root. Run the command from inside the "
        "repository, or set TELEMETRY_VISION_ROOT to its path."
    )


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    root: Path
    asset_root: Path
    asset_dirs: dict[str, Path]
    # Where the web app looks for videos to play, which is not where the CLI
    # keeps the training footage.
    video_dir: Path
    models: dict[str, Path]
    default_model: str

    def asset_dir(self, name: str) -> Path:
        try:
            return self.asset_dirs[name]
        except KeyError:
            known = ", ".join(sorted(self.asset_dirs)) or "none"
            raise ConfigError(f"Unknown asset directory {name!r}; known: {known}") from None

    def resolve_model(self, reference: str | Path | None) -> Path:
        """Resolve a registry name such as `best50`, or a path, to a checkpoint."""
        key = self.default_model if reference is None else str(reference)
        candidate = self.models.get(key)
        if candidate is None:
            candidate = self.root / key if not Path(key).is_absolute() else Path(key)
        candidate = candidate.resolve()
        if not candidate.is_file():
            known = ", ".join(sorted(self.models)) or "none"
            raise ConfigError(
                f"Checkpoint not found: {candidate}\nRegistered names: {known}"
            )
        return candidate

    def resolve_video(self, reference: str | Path) -> Path:
        """Resolve a video by absolute path, path relative to the root, or bare name.

        Bare names are looked up in the external video directory, which is what
        makes `vision-lab evaluate 15-31-25-0.mp4` work from anywhere.
        """
        given = Path(reference).expanduser()
        candidates = [given] if given.is_absolute() else [
            Path.cwd() / given,
            self.root / given,
            self.video_dir / given,
            self.asset_dirs.get("videos", self.asset_root) / given,
        ]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file():
                if resolved.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
                    raise ConfigError(
                        f"{resolved} is not one of "
                        f"{sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
                    )
                return resolved
        searched = "\n  ".join(str(candidate) for candidate in candidates)
        raise ConfigError(f"Video not found: {reference}\nLooked in:\n  {searched}")

    def output_dir(self, *parts: str) -> Path:
        destination = self.root.joinpath("output", *parts)
        destination.mkdir(parents=True, exist_ok=True)
        return destination


def load_paths(
    manifest: Path | None = None,
    *,
    root: Path | None = None,
    overlay: bool = True,
) -> ProjectPaths:
    """Read configs/paths.yaml, with the app's model overlay merged in.

    `overlay=False` gives the tracked base registry alone, which is what the
    registry needs to tell a model the base declares from one the app added.
    """
    import yaml

    root = root.resolve() if root is not None else repo_root()
    manifest_path = manifest or root / MANIFEST_RELATIVE_PATH
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Cannot read {manifest_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{manifest_path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{manifest_path} must contain a mapping")

    asset_root = _resolve_asset_root(root, raw)
    asset_dirs = {
        str(name): _anchor(asset_root, value)
        for name, value in _mapping(raw, "asset_dirs", manifest_path).items()
    }
    models = {
        str(name): _anchor(root, value)
        for name, value in _mapping(raw, "models", manifest_path).items()
    }
    video_dir = _resolve_video_dir(root, raw)

    default_model = str(raw.get("default_model", "")).strip()
    if default_model and default_model not in models:
        raise ConfigError(
            f"default_model {default_model!r} in {manifest_path} is not in the models table"
        )
    if overlay:
        models, default_model = _apply_overlay(root, models, default_model)

    return ProjectPaths(
        root=root,
        asset_root=asset_root,
        asset_dirs=asset_dirs,
        video_dir=video_dir,
        models=models,
        default_model=default_model,
    )


def _resolve_video_dir(root: Path, raw: dict[str, Any]) -> Path:
    """The app's video folder: the environment, then the manifest, then videos/.

    The environment variable is what lets a second server run against a folder
    of test clips without touching the manifest the real one reads.
    """
    override = os.environ.get("TELEMETRY_OCR_VIDEOS", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _anchor(root, raw.get("video_dir") or "videos")


def read_overlay(root: Path) -> dict[str, Any]:
    """The app's model changes, or an empty overlay when there are none."""
    import yaml

    path = root / OVERLAY_RELATIVE_PATH
    empty: dict[str, Any] = {"added": {}, "removed": [], "default": ""}
    if not path.is_file():
        return empty
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping")
    added = raw.get("added") or {}
    removed = raw.get("removed") or []
    if not isinstance(added, dict) or not isinstance(removed, list):
        raise ConfigError(f"{path}: `added` must be a mapping and `removed` a list")
    return {
        "added": {str(name): str(value) for name, value in added.items()},
        "removed": [str(name) for name in removed],
        "default": str(raw.get("default") or "").strip(),
    }


def _apply_overlay(
    root: Path, models: dict[str, Path], default_model: str
) -> tuple[dict[str, Path], str]:
    changes = read_overlay(root)
    merged = {name: path for name, path in models.items() if name not in changes["removed"]}
    # After the removals, so a base name that was removed and then re-added under
    # the same name points at the file it was re-added with.
    for name, value in changes["added"].items():
        merged[name] = _anchor(root, value)
    chosen = changes["default"] if changes["default"] in merged else default_model
    if chosen not in merged:
        # The default itself was removed: fall back to the first model left
        # rather than to a name nothing resolves.
        chosen = next(iter(merged), "")
    return merged, chosen


def _resolve_asset_root(root: Path, raw: dict[str, Any]) -> Path:
    # The environment variable wins so that a second machine, or a CI job with
    # no media at all, does not have to edit a tracked file.
    override = os.environ.get("TELEMETRY_VISION_ASSETS")
    configured = override or raw.get("asset_root") or "."
    return _anchor(root, configured)


def _anchor(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _mapping(raw: dict[str, Any], key: str, manifest_path: Path) -> dict[Any, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{key} in {manifest_path} must be a mapping")
    return value
