"""Add, remove and choose checkpoints from the running app.

configs/paths.yaml stays the tracked, commented base registry and is never
rewritten: yaml.safe_dump would strip every comment in it, and those comments
are the only record of why best50 is the default. What the app changes goes into
configs/models.local.yaml instead, a small overlay `load_paths` merges on top,
so the CLI and the web page always agree on what is registered.

Nothing here accepts a path from a caller. A checkpoint is added by the id of an
entry `candidates()` itself discovered under this repository, because a .pt is a
pickle that runs code as it loads, and what gets unpickled should be decided by
what is on this machine rather than by what a request names.

Removing a model forgets it and never deletes the file. A trained checkpoint is
hours of GPU time; a button that could unlink one is not worth the convenience.

Imports YAML and nothing else heavy: the web server that uses this holds
PaddleOCR, and torch must never load in that process.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import OVERLAY_RELATIVE_PATH, load_paths, read_overlay, repo_root

NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}")

# Where trained checkpoints are looked for, relative to the repository root.
_SEARCH = (("models", "**/*.pt"), ("runs", "*/weights/*.pt"))

_OVERLAY_HEADER = (
    "# Written by the Telemetry OCR web app when models are added, removed or made\n"
    "# the default. It overlays configs/paths.yaml, which is never rewritten.\n"
    "# Safe to delete: the registry falls back to configs/paths.yaml alone.\n"
)


@dataclass(frozen=True, slots=True)
class ModelEntry:
    name: str
    path: Path
    # Repository-relative where possible. The absolute path stays on the server.
    label: str
    available: bool
    size_bytes: int
    default: bool


@dataclass(frozen=True, slots=True)
class Candidate:
    """A checkpoint on disk that is not registered yet."""

    id: str
    path: Path
    label: str
    size_bytes: int
    modified: float
    suggested_name: str


class ModelRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or repo_root()).resolve()
        self._lock = threading.Lock()

    # -- reading ----------------------------------------------------------------

    def entries(self) -> list[ModelEntry]:
        paths = load_paths(root=self.root)
        listed = []
        for name, path in paths.models.items():
            present = path.is_file()
            size = 0
            if present:
                with suppress(OSError):
                    size = path.stat().st_size
            listed.append(
                ModelEntry(
                    name=name,
                    path=path,
                    label=self._label(path),
                    available=present,
                    size_bytes=size,
                    default=name == paths.default_model,
                )
            )
        return listed

    def get(self, name: str) -> ModelEntry | None:
        return next((entry for entry in self.entries() if entry.name == name), None)

    def default(self) -> str | None:
        return load_paths(root=self.root).default_model or None

    def candidates(self) -> list[Candidate]:
        registered = {path.resolve() for path in load_paths(root=self.root).models.values()}
        found: set[Path] = set()
        for directory, pattern in _SEARCH:
            base = self.root / directory
            if not base.is_dir():
                continue
            for path in base.glob(pattern):
                resolved = path.resolve()
                # A link out of the repository is not something this machine's
                # training produced, so it is not offered.
                if resolved.is_file() and resolved.is_relative_to(self.root):
                    found.add(resolved)

        taken = {entry.name for entry in self.entries()}
        offered = []
        for path in sorted(found - registered):
            name = _unique(self._suggest(path), taken)
            taken.add(name)
            try:
                stat = path.stat()
            except OSError:
                continue
            label = self._label(path)
            offered.append(
                Candidate(
                    id=_candidate_id(label),
                    path=path,
                    label=label,
                    size_bytes=stat.st_size,
                    modified=stat.st_mtime,
                    suggested_name=name,
                )
            )
        return offered

    # -- changing ---------------------------------------------------------------

    def add(self, candidate_id: str, name: str) -> ModelEntry:
        name = name.strip()
        _check_name(name)
        with self._lock:
            candidate = next(
                (item for item in self.candidates() if item.id == candidate_id), None
            )
            if candidate is None:
                raise ValueError(
                    "That checkpoint is not among the files this machine offers to add; "
                    "refresh the list and choose again"
                )
            if any(entry.name == name for entry in self.entries()):
                raise ValueError(f"A model named {name!r} is already registered")
            overlay = read_overlay(self.root)
            overlay["added"][name] = candidate.label
            # Re-adding a name that was removed from the base registry.
            overlay["removed"] = [item for item in overlay["removed"] if item != name]
            self._write(overlay)
        entry = self.get(name)
        assert entry is not None
        return entry

    def remove(self, name: str) -> None:
        with self._lock:
            paths = load_paths(root=self.root)
            if name not in paths.models:
                raise ValueError(f"Unknown model {name!r}")
            overlay = read_overlay(self.root)
            overlay["added"].pop(name, None)
            base = load_paths(root=self.root, overlay=False).models
            if name in base and name not in overlay["removed"]:
                overlay["removed"].append(name)
            if paths.default_model == name:
                remaining = [item for item in paths.models if item != name]
                overlay["default"] = remaining[0] if remaining else ""
            self._write(overlay)

    def set_default(self, name: str) -> None:
        with self._lock:
            if name not in load_paths(root=self.root).models:
                raise ValueError(f"Unknown model {name!r}")
            overlay = read_overlay(self.root)
            overlay["default"] = name
            self._write(overlay)

    # -- helpers ----------------------------------------------------------------

    def _label(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    @staticmethod
    def _suggest(path: Path) -> str:
        if path.parent.name == "weights":
            run = path.parent.parent.name
            base = run if path.stem == "best" else f"{run}-{path.stem}"
        else:
            base = path.stem
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", base).strip("-._")[:40]
        return cleaned or "model"

    def _write(self, overlay: dict[str, Any]) -> None:
        import yaml

        path = self.root / OVERLAY_RELATIVE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        body = _OVERLAY_HEADER + yaml.safe_dump(
            {
                "added": overlay["added"],
                "removed": overlay["removed"],
                "default": overlay["default"],
            },
            allow_unicode=True,
            sort_keys=False,
        )
        # Written beside the target and swapped in, so a crash mid-write leaves
        # the previous registry rather than half of one.
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".models.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(body)
            os.replace(temporary, path)
        except BaseException:
            with suppress(OSError):
                os.unlink(temporary)
            raise


def _check_name(name: str) -> None:
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "A model name is 1-40 letters, digits, dots, dashes or underscores, "
            "starting with a letter or digit"
        )


def _unique(name: str, taken: set[str]) -> str:
    if name not in taken:
        return name
    counter = 2
    while f"{name}-{counter}" in taken:
        counter += 1
    return f"{name}-{counter}"


def _candidate_id(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()[:16]
