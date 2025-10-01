# cbm_power/logger.py
from __future__ import annotations
import contextlib
import datetime as _dt
import io
import logging
import os
import sys
from typing import Optional

class _Tee(io.TextIOBase):
    """Write to multiple text streams (e.g., console + file)."""
    def __init__(self, *streams: io.TextIOBase):
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()

def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def make_base_path(save_path: Optional[str]) -> str:
    """Return base path without extension. If empty/None, make cbm_<timestamp>."""
    if not save_path:
        return f"cbm_{_timestamp()}"
    base, _ext = os.path.splitext(save_path)
    return base

class RunLogger(contextlib.AbstractContextManager):
    """
    Context manager that mirrors everything printed to console (stdout & stderr)
    into <base>.log, and also attaches a logging.FileHandler so all `logging`
    output (e.g., Ax [INFO] lines) is written to the same log file.
    """
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.log_path = f"{base_path}.log"
        self._log_file = None
        self._old_stdout = None
        self._old_stderr = None
        self._root_logger = None
        self._file_handler = None

    def __enter__(self):
        # open log file line-buffered
        self._log_file = open(self.log_path, "w", buffering=1, encoding="utf-8")

        # tee stdout & stderr
        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        tee = _Tee(self._old_stdout, self._log_file)
        sys.stdout = tee
        sys.stderr = tee

        # attach logging.FileHandler to root so all logging goes to file too
        self._root_logger = logging.getLogger()
        self._root_logger.setLevel(logging.INFO)  # keep INFO so Ax [INFO] shows up
        self._file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        self._file_handler.setFormatter(logging.Formatter(
            "[%(levelname)s %(asctime)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        self._root_logger.addHandler(self._file_handler)

        # ensure library loggers propagate to root (so file handler catches them)
        for name in ("ax", "botorch", "linear_operator"):
            logging.getLogger(name).propagate = True

        return self

    def __exit__(self, exc_type, exc, tb):
        # remove logging handler
        if self._root_logger and self._file_handler:
            self._root_logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

        # restore std streams
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
        if self._old_stderr is not None:
            sys.stderr = self._old_stderr

        # close file
        if self._log_file:
            self._log_file.close()
            self._log_file = None

        # don't suppress exceptions
        return False