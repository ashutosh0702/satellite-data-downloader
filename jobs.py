"""Persistent local jobs; workers never call Streamlit APIs.

Designed for one trusted Streamlit server process. UUID receipts are recovery
capabilities, not a substitute for authentication on a public deployment.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import time
from uuid import uuid4

import pystac
from shapely.geometry import shape

from persistence import write_json
from processing_utils import process_items, extract_pixel_timeseries
from resource_budget import CancelledError, positive_env
from stac_utils import search_stac

_EXECUTOR = ThreadPoolExecutor(
    max_workers=positive_env("SATELLITE_JOB_WORKERS", 2), thread_name_prefix="satellite-job"
)
_LOCK = threading.RLock()
_FUTURES = {}
_EVENTS = {}
_UUID = re.compile(r"^[a-f0-9]{32}$")


def _now():
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, root="output_data"):
        self.root = Path(root).resolve()

    def directory(self, owner, job_id):
        if not _UUID.fullmatch(owner) or not _UUID.fullmatch(job_id):
            raise ValueError("Invalid job receipt.")
        directory = self.root / owner / job_id
        if directory.resolve() != directory:
            raise ValueError("Job directories must not be symlinks.")
        return directory

    def _update(self, directory, **changes):
        with _LOCK:
            path = directory / "job.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            state.update(changes, updated_at=_now())
            write_json(path, state)
            return state

    def create(self, owner, request):
        job_id = uuid4().hex
        directory = self.directory(owner, job_id)
        directory.mkdir(parents=True, exist_ok=False)
        if request.get("kind") not in ("download", "timeseries"):
            raise ValueError("Unsupported job kind.")
        write_json(directory / "request.json", request)
        self._update(
            directory,
            id=job_id,
            owner=owner,
            kind=request["kind"],
            status="created",
            progress={"completed": 0, "total": 0},
            created_at=_now(),
            error=None,
            parameters={
                key: request[key]
                for key in (
                    "collection",
                    "bands",
                    "indices",
                    "resolution",
                    "index",
                    "geometry",
                    "search",
                    "radiometry_policy",
                )
                if key in request
            },
        )
        try:
            self.resume(owner, job_id)
        except ValueError as exc:
            # Keep the immutable request recoverable when the queue is full.
            self._update(directory, status="created", error=str(exc))
        return job_id

    def get(self, owner, job_id):
        directory = self.directory(owner, job_id)
        with _LOCK:
            state = json.loads((directory / "job.json").read_text())
            key = str(directory)
            if state["status"] in ("queued", "running", "cancelling") and key not in _FUTURES:
                pid = state.get("pid")
                alive = False
                if pid is not None:
                    try:
                        os.kill(pid, 0)
                        alive = True
                    except (OSError, ValueError):
                        pass
                if not alive or pid == os.getpid():
                    state = self._update(
                        directory,
                        status="interrupted",
                        detail="Worker stopped; resume to recover completed dates.",
                    )
            return state

    def cancel(self, owner, job_id):
        directory = self.directory(owner, job_id)
        key = str(directory)
        with _LOCK:
            state = self.get(owner, job_id)
            if state["status"] not in ("queued", "running", "cancelling"):
                return state
            event = _EVENTS.get(key)
            if event is not None:
                event.set()
            future = _FUTURES.get(key)
            if future is not None and future.cancel():
                return self._update(directory, status="cancelled", detail="Cancelled before starting.")
            return self._update(
                directory, status="cancelling", detail="Finishing the current read before cancellation."
            )

    def resume(self, owner, job_id):
        directory = self.directory(owner, job_id)
        key = str(directory)
        with _LOCK:
            future = _FUTURES.get(key)
            if future is not None and not future.done():
                raise ValueError("Job is already active.")
            # Bound registry/queue growth across browser sessions.
            for old in [k for k, value in _FUTURES.items() if value.done()]:
                _FUTURES.pop(old, None)
                _EVENTS.pop(old, None)
            if len(_FUTURES) >= positive_env("SATELLITE_MAX_QUEUED_JOBS", 16):
                raise ValueError("The local job queue is full; retry after another job finishes.")
            if not (directory / "request.json").is_file():
                raise ValueError("Saved job request is missing.")
            event = threading.Event()
            _EVENTS[key] = event
            self._update(
                directory,
                status="queued",
                error=None,
                result=None,
                pid=os.getpid(),
                detail="Waiting for a job worker.",
            )
            _FUTURES[key] = _EXECUTOR.submit(self._run, directory, event)

    def _run(self, directory, event):
        last_detail = 0.0

        def progress(completed, total):
            self._update(directory, progress={"completed": completed, "total": total})

        def detail(message):
            nonlocal last_detail
            if time.monotonic() - last_detail >= 0.5:
                self._update(directory, detail=message)
                last_detail = time.monotonic()

        try:
            request = json.loads((directory / "request.json").read_text())
            self._update(directory, status="running", detail="Preparing inputs.")
            if request["kind"] == "download":
                items = [pystac.Item.from_dict(item) for item in request["items"]]
                _, elapsed, report = process_items(
                    items,
                    shape(request["geometry"]),
                    collection=request["collection"],
                    bands=request["bands"],
                    indices=request.get("indices", []),
                    resolution=request.get("resolution", 10),
                    target_crs=request.get("target_crs"),
                    apply_scl_filter=request.get("apply_scl_filter", False),
                    requester_pays=request.get("requester_pays", False),
                    out_dir=directory,
                    return_report=True,
                    cancel_event=event,
                    progress_callback=progress,
                    detail_callback=detail,
                    radiometry_policy=request.get("radiometry_policy", "strict"),
                )
                state = (
                    "partial"
                    if report["skipped_dates"] or not request.get("search", {}).get("complete", True)
                    else "completed"
                )
                if report["total_dates"] and not report["saved_dates"] and not report["cancelled"]:
                    state = "failed"
                result = {
                    "elapsed_seconds": elapsed,
                    "manifest": "manifest.json",
                    "saved_dates": report["saved_dates"],
                    "skipped_dates": report["skipped_dates"],
                    "date_details": report["date_details"],
                    "search": request.get("search", {}),
                }
            else:
                detail("Searching the catalog.")
                search = search_stac(**request["search"], return_report=True, cancel_event=event)
                observations, elapsed, report = extract_pixel_timeseries(
                    search.items,
                    shape(request["geometry"]),
                    index=request["index"],
                    apply_scl_filter=request.get("apply_scl_filter", True),
                    return_report=True,
                    cancel_event=event,
                    progress_callback=progress,
                    radiometry_policy=request.get("radiometry_policy", "strict"),
                )
                result = {
                    "observations": observations,
                    "report": report,
                    "search": search.summary(),
                    "elapsed_seconds": elapsed,
                }
                write_json(directory / "series.json", result)
                state = "partial" if search.truncated or report["read_failures"] else "completed"
            if event.is_set():
                state = "cancelled"
            self._update(
                directory, status=state, result=result, detail="Processing finished.", finished_at=_now()
            )
        except CancelledError:
            self._update(directory, status="cancelled", detail="Cancelled; resume to retry unfinished work.")
        except Exception as exc:
            self._update(
                directory,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                detail="Processing failed. Completed date bundles remain resumable.",
            )

    def receipt(self, owner, job_id):
        self.directory(owner, job_id)
        return {"version": 1, "owner": owner, "job_id": job_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume a saved local satellite job.")
    parser.add_argument("job_directory", type=Path)
    args = parser.parse_args()
    path = args.job_directory.resolve()
    manager = JobManager(path.parent.parent)
    manager.resume(path.parent.name, path.name)
    # CLI waits for this job while Streamlit normally polls without blocking.
    _FUTURES[str(path)].result()
    state = manager.get(path.parent.name, path.name)
    print(json.dumps(state, indent=2))
    raise SystemExit(0 if state["status"] == "completed" else 1)
