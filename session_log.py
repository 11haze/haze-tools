"""Timestamped CSV log of every bot action: logs/session_<date>_<time>.csv"""
import csv
import threading
import time
from datetime import datetime
from pathlib import Path

FIELDS = ["timestamp", "elapsed_s", "state", "action", "detail"]


class SessionLog:
    def __init__(self, log_dir="logs"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.path = Path(log_dir) / f"session_{stamp}.csv"
        self._start = time.monotonic()
        self._lock = threading.Lock()
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(FIELDS)
        self._file.flush()
        self.state = "INIT"

    def log(self, action, detail="", state=None):
        row = [
            datetime.now().isoformat(timespec="milliseconds"),
            f"{time.monotonic() - self._start:.3f}",
            state or self.state,
            action,
            detail,
        ]
        with self._lock:
            if self._file.closed:
                return
            self._writer.writerow(row)
            self._file.flush()
        print(" | ".join(row[1:]))

    def close(self):
        with self._lock:
            if not self._file.closed:
                self._file.close()
