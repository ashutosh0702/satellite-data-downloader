"""Process-wide task, I/O and estimated working-memory admission limits."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager


def positive_env(name, default):
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


class CancelledError(Exception):
    pass


def check_cancel(event):
    if event is not None and event.is_set():
        raise CancelledError("Job cancelled; committed dates can be resumed.")


class ResourceBudget:
    def __init__(self, memory_mb=512, tasks=4):
        self.memory_mb = memory_mb
        self.tasks = tasks
        self.active = self.used_mb = 0
        self.condition = threading.Condition()

    @contextmanager
    def reserve(self, memory_mb, cancel_event=None):
        memory_mb = max(1, memory_mb)
        if memory_mb > self.memory_mb:
            raise ValueError(
                f"Task needs an estimated {memory_mb} MiB; budget is {self.memory_mb} MiB. "
                "Reduce window size or increase SATELLITE_MEMORY_MB."
            )
        with self.condition:
            while self.active >= self.tasks or self.used_mb + memory_mb > self.memory_mb:
                check_cancel(cancel_event)
                self.condition.wait(timeout=0.1)
            check_cancel(cancel_event)
            self.active += 1
            self.used_mb += memory_mb
        try:
            yield
        finally:
            with self.condition:
                self.active -= 1
                self.used_mb -= memory_mb
                self.condition.notify_all()


BUDGET = ResourceBudget(positive_env("SATELLITE_MEMORY_MB", 512), positive_env("SATELLITE_GLOBAL_TASKS", 4))
IO_SLOTS = threading.BoundedSemaphore(positive_env("SATELLITE_IO_WORKERS", 8))


@contextmanager
def io_slot(cancel_event=None):
    while not IO_SLOTS.acquire(timeout=0.1):
        check_cancel(cancel_event)
    try:
        check_cancel(cancel_event)
        yield
    finally:
        IO_SLOTS.release()


def bounded_results(function, values, workers, cancel_event=None):
    """Only worker-count futures are queued; workers must check cancellation."""
    iterator = iter(values)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}

        def submit_one():
            if cancel_event is not None and cancel_event.is_set():
                return
            try:
                value = next(iterator)
            except StopIteration:
                return
            pending[executor.submit(function, value)] = value

        for _ in range(workers):
            submit_one()
        while pending:
            done, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                value = pending.pop(future)
                yield value, future
                submit_one()
