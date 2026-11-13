from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import threading
import unittest
from unittest.mock import patch
import zipfile

from shapely.geometry import Point, Polygon, mapping

from input_utils import read_vector_upload, coordinate_point
from persistence import atomic_path, directory_lock
from processing_utils import preview_processing, preflight_access
from resource_budget import ResourceBudget, CancelledError, bounded_results
from tests.fixtures import RasterCase


class InputTests(RasterCase):
    def test_geojson_upload_and_exact_single_point(self):
        data = json.dumps(
            {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": mapping(self.point())}],
            }
        ).encode()
        frame = read_vector_upload(data, "point.geojson", point=True)
        self.assertEqual(len(frame), 1)
        with self.assertRaises(ValueError):
            read_vector_upload(data, "point.geojson")

    def test_missing_shapefile_crs_rejected_and_zip_temp_removed(self):
        import geopandas as gpd

        frame = gpd.GeoDataFrame(geometry=[self.aoi()], crs=4326)
        shapefile = self.root / "aoi.shp"
        frame.to_file(shapefile)
        (self.root / "aoi.prj").unlink()
        data = BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            for path in self.root.glob("aoi.*"):
                archive.write(path, path.name)
        from tempfile import TemporaryDirectory

        created = []

        def temporary(*args, **kwargs):
            result = TemporaryDirectory(dir=self.root, **kwargs)
            created.append(result.name)
            return result

        with patch("input_utils.TemporaryDirectory", side_effect=temporary):
            with self.assertRaisesRegex(ValueError, "no CRS"):
                read_vector_upload(data.getvalue(), "aoi.zip")
        from pathlib import Path

        self.assertTrue(all(not Path(path).exists() for path in created))

    def test_oversize_upload_rejected_before_parsing(self):
        with patch("input_utils.MAX_UPLOAD_BYTES", 2), self.assertRaisesRegex(ValueError, "20 MiB"):
            read_vector_upload(b"abc", "aoi.geojson")

    def test_empty_invalid_and_out_of_range_geometry(self):
        from raster_engine import validate_geometry

        for geometry in [Polygon(), Polygon([(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)]), Point(181, 0)]:
            with self.subTest(geometry=geometry), self.assertRaises(ValueError):
                validate_geometry(geometry, point=geometry.geom_type == "Point")
        with self.assertRaises(ValueError):
            coordinate_point(0, 91)

    def test_pixel_limit_and_destination_units(self):
        with patch.dict("os.environ", {"SATELLITE_MAX_PIXELS": "1"}):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                preview_processing(self.aoi(), target_crs=3857)
        with self.assertRaisesRegex(ValueError, "metres"):
            preview_processing(self.aoi(), target_crs=4326)

    def test_requester_pays_is_explicit(self):
        item = self.scene({"vv": "s3://sentinel-s1-l1c/example.tiff"})
        with self.assertRaisesRegex(ValueError, "requester-pays"):
            preflight_access([item], ["vv"])


class PersistenceTests(RasterCase):
    def test_atomic_failure_preserves_previous_file(self):
        path = self.root / "artifact.json"
        path.write_text("original")
        with self.assertRaises(OSError):
            with atomic_path(path) as temporary:
                temporary.write_text("partial")
                raise OSError("write failed")
        self.assertEqual(path.read_text(), "original")
        self.assertEqual(list(self.root.glob(".artifact-*")), [])

    def test_output_lock_prevents_concurrent_writer(self):
        with directory_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "already being processed"):
                with directory_lock(self.root):
                    pass
        with directory_lock(self.root):
            pass


class BudgetTests(unittest.TestCase):
    def test_global_memory_and_task_admission(self):
        budget = ResourceBudget(memory_mb=10, tasks=2)
        second_started = threading.Event()

        def blocked():
            with budget.reserve(6):
                second_started.set()

        with ThreadPoolExecutor(max_workers=1) as pool:
            with budget.reserve(6):
                future = pool.submit(blocked)
                self.assertFalse(second_started.wait(0.05))
                self.assertEqual(budget.active, 1)
                self.assertEqual(budget.used_mb, 6)
            future.result(timeout=2)
        self.assertTrue(second_started.is_set())
        self.assertEqual(budget.used_mb, 0)

    def test_waiting_task_cancels_and_resources_release_on_failure(self):
        budget = ResourceBudget(memory_mb=10, tasks=1)
        event = threading.Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            with budget.reserve(10):
                future = pool.submit(lambda: budget.reserve(1, event).__enter__())
                event.set()
                with self.assertRaises(CancelledError):
                    future.result(timeout=2)
        with self.assertRaises(OSError):
            with budget.reserve(5):
                raise OSError("failed")
        self.assertEqual(budget.active, 0)
        with self.assertRaises(ValueError):
            with budget.reserve(11):
                pass

    def test_bounded_submission_respects_cancellation(self):
        event = threading.Event()
        started = []

        def work(value):
            started.append(value)
            event.set()
            return value

        rows = list(bounded_results(work, range(100), 1, event))
        self.assertEqual(len(rows), 1)
        self.assertEqual(started, [0])


class CredentialPreflightTests(RasterCase):
    def test_missing_credentials_rejected_and_valid_provider_checked(self):
        from unittest.mock import MagicMock
        import boto3

        item = self.scene({"vv": "s3://sentinel-s1-l1c/example.tiff"})
        with patch.object(boto3, "Session") as session:
            session.return_value.get_credentials.return_value = None
            with self.assertRaisesRegex(ValueError, "No AWS credentials"):
                preflight_access([item], ["vv"], requester_pays=True)
            credentials = MagicMock()
            session.return_value.get_credentials.return_value = credentials
            preflight_access([item], ["vv"], requester_pays=True)
            credentials.get_frozen_credentials.assert_called_once()
