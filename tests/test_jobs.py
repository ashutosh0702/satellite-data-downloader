import threading
from unittest.mock import patch
from uuid import uuid4

from shapely.geometry import mapping

import jobs
from jobs import JobManager
from tests.fixtures import RasterCase


class JobTests(RasterCase):
    def request(self):
        return {
            "kind": "download",
            "geometry": mapping(self.aoi()),
            "collection": "sentinel-2-l2a",
            "bands": ["nir", "red"],
            "indices": ["NDVI"],
            "target_crs": 3857,
            "items": [self.scene({"nir": 3000, "red": 2000}).to_dict()],
        }

    def test_background_download_persists_request_state_and_outputs(self):
        manager = JobManager(self.root / "jobs")
        owner = uuid4().hex
        job_id = manager.create(owner, self.request())
        directory = manager.directory(owner, job_id)
        jobs._FUTURES[str(directory)].result(timeout=10)
        state = manager.get(owner, job_id)
        self.assertEqual(state["status"], "completed", state)
        self.assertTrue((directory / "request.json").exists())
        self.assertTrue((directory / "manifest.json").exists())
        self.assertEqual(state["result"]["saved_dates"], 1)
        manager.resume(owner, job_id)
        jobs._FUTURES[str(directory)].result(timeout=10)
        self.assertTrue(manager.get(owner, job_id)["result"]["date_details"][0]["resumed"])
        self.assertEqual(manager.receipt(owner, job_id)["job_id"], job_id)

    def test_failure_is_actionable_and_resumable(self):
        manager = JobManager(self.root / "jobs")
        owner = uuid4().hex
        with patch("jobs.process_items", side_effect=OSError("disk full")):
            job_id = manager.create(owner, self.request())
            directory = manager.directory(owner, job_id)
            jobs._FUTURES[str(directory)].result(timeout=10)
        state = manager.get(owner, job_id)
        self.assertEqual(state["status"], "failed")
        self.assertIn("disk full", state["error"])
        manager.resume(owner, job_id)
        jobs._FUTURES[str(directory)].result(timeout=10)
        self.assertEqual(manager.get(owner, job_id)["status"], "completed")

    def test_cancellation_is_cooperative_and_resume_works(self):
        manager = JobManager(self.root / "jobs")
        owner = uuid4().hex
        started = threading.Event()
        from resource_budget import CancelledError

        def wait_for_cancel(*args, **kwargs):
            started.set()
            self.assertTrue(kwargs["cancel_event"].wait(2))
            raise CancelledError("cancelled")

        with patch("jobs.process_items", side_effect=wait_for_cancel):
            job_id = manager.create(owner, self.request())
            directory = manager.directory(owner, job_id)
            self.assertTrue(started.wait(2))
            manager.cancel(owner, job_id)
            jobs._FUTURES[str(directory)].result(timeout=5)
        self.assertEqual(manager.get(owner, job_id)["status"], "cancelled")
        manager.resume(owner, job_id)
        jobs._FUTURES[str(directory)].result(timeout=10)
        self.assertEqual(manager.get(owner, job_id)["status"], "completed")

    def test_restart_marks_running_job_interrupted(self):
        manager = JobManager(self.root / "jobs")
        owner = uuid4().hex
        job_id = uuid4().hex
        directory = manager.directory(owner, job_id)
        manager._update(directory, status="running", pid=999999999, id=job_id, owner=owner)
        self.assertEqual(manager.get(owner, job_id)["status"], "interrupted")

    def test_paths_cannot_escape_session_root(self):
        manager = JobManager(self.root / "jobs")
        for owner, job_id in [("../outside", uuid4().hex), (uuid4().hex, "../../etc")]:
            with self.assertRaises(ValueError):
                manager.directory(owner, job_id)

    def test_sessions_get_distinct_directories(self):
        manager = JobManager(self.root / "jobs")
        job_id = uuid4().hex
        self.assertNotEqual(manager.directory(uuid4().hex, job_id), manager.directory(uuid4().hex, job_id))

    def test_timeseries_job_persists_observations_and_completeness(self):
        from stac_utils import SearchResult

        manager = JobManager(self.root / "jobs")
        owner = uuid4().hex
        item = self.scene({"nir": 3000, "red": 2000, "scl": 4})
        search = SearchResult([item], True, 1, "2025-04-01T00:00:00Z")
        request = {
            "kind": "timeseries",
            "geometry": mapping(self.point()),
            "index": "NDVI",
            "apply_scl_filter": True,
            "search": {
                "bbox": None,
                "start_date": "2025-04-01",
                "end_date": "2025-04-02",
                "collection": "sentinel-2-l2a",
            },
        }
        with patch("jobs.search_stac", return_value=search):
            job_id = manager.create(owner, request)
            directory = manager.directory(owner, job_id)
            jobs._FUTURES[str(directory)].result(timeout=10)
        state = manager.get(owner, job_id)
        self.assertEqual(state["status"], "partial")
        self.assertFalse(state["result"]["search"]["complete"])
        self.assertAlmostEqual(state["result"]["observations"][0]["Index"], 0.2)
        self.assertTrue((directory / "series.json").exists())
        self.assertEqual(state["parameters"]["index"], "NDVI")
