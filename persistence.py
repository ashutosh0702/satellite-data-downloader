"""Atomic artifacts and an advisory lock for a single output directory (POSIX)."""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkstemp
import fcntl


@contextmanager
def atomic_path(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = mkstemp(prefix=f".{destination.stem}-", suffix=destination.suffix, dir=destination.parent)
    os.close(fd)
    path = Path(name)
    try:
        yield path
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(path, destination)
    finally:
        path.unlink(missing_ok=True)
        Path(str(path) + ".msk").unlink(missing_ok=True)
        Path(str(path) + ".aux.xml").unlink(missing_ok=True)


def write_json(destination, value):
    with atomic_path(destination) as path:
        path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


@contextmanager
def directory_lock(directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".processing.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This output directory is already being processed.") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
