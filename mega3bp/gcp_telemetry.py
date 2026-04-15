"""Standard telemetry for GCP-bound experiments.

Every GCP experiment writes `status.json` in its experiment dir. Query via:
    ssh ... "cat ~/3bp/experiments/orbit_verification/NN_X/status.json"

Or aggregate all active experiments via:
    ssh ... "find ~/3bp/experiments/ -name status.json -exec sh -c 'echo === {}; cat {}' \\;"

Schema:
    stage (str)              current stage name
    progress (int)           how many units processed
    total (int)              total units
    fraction (float)         progress / total
    elapsed_s (float)        wall time since Telemetry() was constructed
    eta_s (float)            estimated remaining seconds
    rate_per_s (float)       items per second
    last_update_utc (str)    ISO timestamp
    extra (dict)             arbitrary experiment-specific fields

Usage:
    from mega3bp.gcp_telemetry import Telemetry

    t = Telemetry(Path("status.json"), total=24582, stage="init")
    for i, entry in enumerate(entries):
        process(entry)
        t.update(progress=i + 1, stage="sweeping")
    t.done(stage="complete")
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


class Telemetry:
    """Atomic, rate-limited status writer for long-running GCP jobs."""

    def __init__(self, path, total: int, stage: str = "init",
                 min_interval_s: float = 0.5):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.total = int(total)
        self.start_ts = time.time()
        self.stage = stage
        self.progress = 0
        self._last_write_ts = 0.0
        self._min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self.extra: dict = {}
        self._force_write()

    def update(self, progress=None, stage=None, force=False, **extra):
        """Update telemetry. Writes at most every min_interval_s unless force=True
        or progress == total."""
        with self._lock:
            if progress is not None:
                self.progress = int(progress)
            if stage is not None:
                self.stage = str(stage)
            self.extra.update(extra)
            now = time.time()
            if (force or self.progress == self.total
                    or now - self._last_write_ts >= self._min_interval_s):
                self._write_locked()
                self._last_write_ts = now

    def done(self, stage: str = "done", **extra):
        with self._lock:
            self.stage = stage
            self.progress = self.total
            self.extra.update(extra)
            self._write_locked()

    def error(self, message: str, **extra):
        with self._lock:
            self.stage = "error"
            self.extra["error"] = message
            self.extra.update(extra)
            self._write_locked()

    def _force_write(self):
        with self._lock:
            self._write_locked()

    def _write_locked(self):
        elapsed = time.time() - self.start_ts
        rate = self.progress / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total - self.progress)
        eta = remaining / rate if rate > 1e-12 else (0.0 if remaining == 0
                                                     else float("inf"))
        doc = {
            "stage": self.stage,
            "progress": self.progress,
            "total": self.total,
            "fraction": (self.progress / self.total) if self.total > 0 else 1.0,
            "elapsed_s": elapsed,
            "eta_s": eta if eta != float("inf") else None,
            "rate_per_s": rate,
            "last_update_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime()),
            "extra": self.extra,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2, default=str))
        os.replace(str(tmp), str(self.path))
