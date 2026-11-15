"""Thread-safe structured progress; fractions describe handled image work, not bytes."""

import threading
import time


class ProcessingProgress:
    def __init__(self, total_dates, clock=time.monotonic):
        self.total = total_dates
        self.clock = clock
        self.started = clock()
        self.lock = threading.Lock()
        self.active = {}
        self.finished = {}

    def event(self, event):
        with self.lock:
            date = event["date"]
            if event["stage"] == "date_finished":
                self.active.pop(date, None)
                self.finished[date] = {
                    "saved": bool(event.get("saved")),
                    "resumed": bool(event.get("resumed")),
                }
            elif date not in self.finished:
                self.active[date] = dict(event)

    def snapshot(self):
        with self.lock:
            elapsed = max(0, self.clock() - self.started)
            saved = sum(row["saved"] for row in self.finished.values())
            skipped = len(self.finished) - saved
            work = len(self.finished)
            for row in self.active.values():
                total = row.get("total_windows", 0)
                if total:
                    work += min(1, row.get("completed_windows", 0) / total)
            fraction = min(1, work / self.total) if self.total else None
            stages = {row["stage"] for row in self.active.values()}
            stage = (
                "processing"
                if "processing" in stages
                else "finalizing"
                if "finalizing" in stages
                else "validating"
                if "validating" in stages
                else "waiting_for_resources"
                if stages == {"waiting_for_resources"}
                else "preparing"
            )
            eta = None
            if (
                saved >= 2
                and not skipped
                and elapsed >= 5
                and fraction
                and fraction < 1
                and not any(row["resumed"] for row in self.finished.values())
            ):
                eta = round(elapsed * (1 - fraction) / fraction)
            return {
                "stage": stage,
                "completed": len(self.finished),
                "total": self.total,
                "saved": saved,
                "skipped": skipped,
                "fraction": fraction,
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": eta,
                "active_dates": [self.active[date] for date in sorted(self.active)],
            }
