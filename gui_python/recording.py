"""Recording a scan's frames, to replay them through the fusion offline.

Opt-in: with ALL3DSCAN_RECORD_DIR=<dir>, every frame of a scan is kept, with
when it arrived, whether it was handed to the fusion and how it sat there,
and the whole scan is written out once it completes:

    <dir>/<YYYYmmdd-HHMMSS>-<scan name>/
        manifest.json     settings, and one entry per frame
        frame_0000.npz    points, colors, normals (N, 3) and depth_mm (H, W)
        ...

Nothing is written while scanning: the frames are held in memory (about a
megabyte each at the POP 4's coverage) so that the disk cannot change the
frame rate being recorded. tools/replay_fusion.py reads it back.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

ENVIRONMENT = "ALL3DSCAN_RECORD_DIR"


class ScanRecorder:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._frames: list = []
        self._entries: list[dict] = []
        self._awaiting_fit: list[int] = []  # offered, oldest first
        self._settings: dict = {}
        self._started = 0.0

    @classmethod
    def from_environment(cls) -> ScanRecorder | None:
        root = os.environ.get(ENVIRONMENT)
        return cls(Path(root)) if root else None

    def start(self, settings: dict) -> None:
        """Begins a scan, forgetting the last one."""
        self._frames.clear()
        self._entries.clear()
        self._awaiting_fit.clear()
        self._settings = dict(settings)
        self._started = time.perf_counter()

    def frame(self, frame) -> None:
        """Keeps a frame that arrived while scanning."""
        self._frames.append(frame)
        self._entries.append({
            "file": f"frame_{len(self._entries):04d}.npz",
            "index": int(frame.index),
            "arrived_s": round(time.perf_counter() - self._started, 4),
            "points": int(len(frame.points)),
            "offered": False,
        })

    def offered(self, step: int) -> None:
        """The newest frame was handed to the fusion, every `step`th point."""
        if self._entries:
            self._entries[-1]["offered"] = True
            self._entries[-1]["step"] = int(step)
            self._awaiting_fit.append(len(self._entries) - 1)

    def fit(self, fit: dict) -> None:
        """How the oldest frame still waiting sat on the model."""
        if self._awaiting_fit:
            entry = self._entries[self._awaiting_fit.pop(0)]
            entry["fit"] = {key: (list(value) if key == "pose" else value)
                            for key, value in fit.items()}

    def finish(self, name: str, summary: dict) -> Path | None:
        """Writes the scan out; returns where, or None with nothing kept."""
        if not self._frames:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "scan"
        directory = self.root / f"{time.strftime('%Y%m%d-%H%M%S')}-{safe}"
        directory.mkdir(parents=True, exist_ok=True)
        for frame, entry in zip(self._frames, self._entries):
            np.savez(directory / entry["file"],
                     points=np.asarray(frame.points, dtype=np.float32),
                     colors=np.asarray(frame.colors, dtype=np.uint8),
                     normals=np.asarray(frame.normals, dtype=np.float32),
                     depth_mm=np.asarray(frame.depth_mm, dtype=np.uint16))
        manifest = {
            "name": name,
            "commit": _commit(),
            "settings": self._settings,
            "summary": summary,
            "frames": self._entries,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                                 encoding="utf-8")
        self._frames.clear()
        return directory


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"], cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=5, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
