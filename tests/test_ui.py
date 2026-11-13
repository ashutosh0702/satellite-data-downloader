"""Streamlit event/state tests; browser map rendering requires a live browser."""

from unittest.mock import patch
from pathlib import Path

from shapely.geometry import mapping
from streamlit.testing.v1 import AppTest

from stac_utils import SearchResult
from tests.fixtures import RasterCase

ROOT = Path(__file__).resolve().parents[1]


class UITests(RasterCase):
    def test_pages_start_without_errors(self):
        with patch("streamlit_folium.st_folium", return_value={}):
            for path in ("app.py", "pages/1_Downloader.py", "pages/2_Timeseries_Analysis.py"):
                with self.subTest(path=path):
                    app = AppTest.from_file(str(ROOT / path)).run(timeout=10)
                    self.assertEqual(len(app.exception), 0, list(app.exception))

    def test_downloader_invalidates_results_when_collection_changes(self):
        item = self.scene({"blue": 1000, "green": 2000, "red": 3000})
        result = SearchResult([item], False, 1000, "2025-04-01T00:00:00Z")
        with (
            patch("streamlit_folium.st_folium", return_value={}),
            patch("stac_utils.search_stac", return_value=result),
        ):
            app = AppTest.from_file(str(ROOT / "pages/1_Downloader.py")).run(timeout=10)
            app.radio[0].set_value("Draw polygon")
            app.session_state["drawn_geometry"] = mapping(self.aoi())
            app.run()
            next(button for button in app.button if button.label == "Search catalog").click().run()
            process = next(button for button in app.button if button.label == "Process & Save Outputs")
            self.assertFalse(process.disabled)
            app.selectbox[0].set_value("sentinel-1-grd").run()
            process = next(button for button in app.button if button.label == "Process & Save Outputs")
            self.assertTrue(process.disabled)
            self.assertTrue(any("changed" in warning.value for warning in app.warning))
            self.assertEqual(len(app.exception), 0)

    def test_finished_job_renders_report_and_recovery_receipt(self):
        from uuid import uuid4
        import jobs

        manager = jobs.JobManager(self.root / "jobs")
        owner = uuid4().hex
        item = self.scene({"nir": 3000, "red": 2000})
        request = {
            "kind": "download",
            "geometry": mapping(self.aoi()),
            "target_crs": 3857,
            "collection": "sentinel-2-c1-l2a",
            "bands": ["nir", "red"],
            "indices": ["NDVI"],
            "items": [item.to_dict()],
        }
        job_id = manager.create(owner, request)
        jobs._FUTURES[str(manager.directory(owner, job_id))].result(timeout=10)
        with (
            patch("ui_components.job_manager", return_value=manager),
            patch("streamlit_folium.st_folium", return_value={}),
        ):
            app = AppTest.from_file(str(ROOT / "pages/1_Downloader.py"))
            app.session_state["job_download"] = (owner, job_id)
            app.run(timeout=10)
            self.assertEqual(len(app.exception), 0, list(app.exception))
            self.assertTrue(any("completed" in element.value for element in app.markdown))
            self.assertEqual(len(app.dataframe), 1)
