import threading
import unittest

from job_progress import ProcessingProgress


class ProgressTests(unittest.TestCase):
    def test_fraction_tracks_windows_and_finalization_separately(self):
        progress = ProcessingProgress(2)
        progress.event(
            {"stage": "processing", "date": "2025-04-01", "completed_windows": 2, "total_windows": 4}
        )
        self.assertEqual(progress.snapshot()["fraction"], 0.25)
        progress.event(
            {"stage": "finalizing", "date": "2025-04-01", "completed_windows": 4, "total_windows": 4}
        )
        snapshot = progress.snapshot()
        self.assertEqual(snapshot["fraction"], 0.5)
        self.assertEqual(snapshot["completed"], 0)
        self.assertEqual(snapshot["stage"], "finalizing")
        progress.event({"stage": "date_finished", "date": "2025-04-01", "saved": True})
        self.assertEqual(progress.snapshot()["saved"], 1)

    def test_skipped_dates_are_handled_but_not_counted_as_saved(self):
        progress = ProcessingProgress(2)
        progress.event({"stage": "date_finished", "date": "first", "saved": False})
        progress.event({"stage": "date_finished", "date": "second", "saved": True})
        snapshot = progress.snapshot()
        self.assertEqual(snapshot["fraction"], 1)
        self.assertEqual(snapshot["saved"], 1)
        self.assertEqual(snapshot["skipped"], 1)

    def test_eta_uses_observed_work_and_is_suppressed_for_resumed_dates(self):
        now = [0]
        progress = ProcessingProgress(4, clock=lambda: now[0])
        for date in ["first", "second"]:
            progress.event({"stage": "date_finished", "date": date, "saved": True})
        self.assertIsNone(progress.snapshot()["eta_seconds"])
        now[0] = 10
        self.assertEqual(progress.snapshot()["eta_seconds"], 10)
        progress.event({"stage": "date_finished", "date": "third", "saved": True, "resumed": True})
        self.assertIsNone(progress.snapshot()["eta_seconds"])

    def test_out_of_order_events_do_not_resurrect_finished_dates(self):
        progress = ProcessingProgress(1)
        progress.event({"stage": "date_finished", "date": "first", "saved": True})
        progress.event({"stage": "processing", "date": "first", "completed_windows": 1, "total_windows": 4})
        self.assertEqual(progress.snapshot()["active_dates"], [])
        self.assertEqual(progress.snapshot()["fraction"], 1)

    def test_concurrent_updates_have_consistent_counts(self):
        progress = ProcessingProgress(20)
        threads = [
            threading.Thread(
                target=progress.event, args=({"stage": "date_finished", "date": str(i), "saved": True},)
            )
            for i in range(20)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(progress.snapshot()["completed"], 20)
        self.assertEqual(progress.snapshot()["saved"], 20)
