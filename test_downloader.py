"""Offline regression tests; live catalog benchmarks are explicit scripts."""

from pathlib import Path
import threading
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.control import GroundControlPoint

from tests.fixtures import RasterCase
from processing_utils import process_items, preview_processing
from satellite_config import select_bands, S2_BANDS, COMMON_NODATA


class DownloaderTests(RasterCase):
    def test_partial_coverage_and_negative_nodata(self):
        item = self.scene({"nir": 3000, "red": 2000})
        files, _, report = process_items(
            [item],
            self.aoi((-20, -20, 80, 80)),
            bands=["nir", "red"],
            out_dir=self.root / "output",
            target_crs=3857,
            return_report=True,
        )
        self.assertEqual(report["saved_dates"], 1)
        with rasterio.open(files[0]) as src:
            values = src.read(1)
            self.assertEqual(src.dtypes[0], "float32")
            self.assertTrue(np.any(values == COMMON_NODATA))
            np.testing.assert_allclose(values[values != COMMON_NODATA], 0.3, rtol=1e-6)

    def test_offsets_applied_before_mosaic_and_integrated_indices(self):
        item = self.scene({"nir": 3000, "red": 2000}, scales=0.0001, offsets=-0.1)
        files, _, report = self.process([item], indices=["NDVI", "SAVI"])
        with rasterio.open(report["index_files"][0]) as src:
            np.testing.assert_allclose(src.read(1, masked=True).compressed(), 1 / 3, rtol=1e-6)
        with rasterio.open(files[0]) as src:
            self.assertEqual(src.tags()["radiometry"], "reflectance")
            self.assertEqual(src.tags(1)["units"], "reflectance")
            self.assertEqual(src.scales, (1, 1, 1))
        self.assertEqual(report["date_details"][0]["radiometry"][0]["origin"], "stac")

    def test_mixed_crs_inputs(self):
        to_utm = __import__("pyproj").Transformer.from_crs(3857, 32631, always_xy=True)
        west, north = to_utm.transform(0, 100)
        shifted = self.raster(4000, crs=32631, transform=from_origin(west, north, 10, 10))
        half = np.full((10, 10), 3000, dtype="uint16")
        half[:, :5] = 0
        first = self.scene({"nir": half, "red": np.where(half > 0, 2000, 0)}, scene_id="first")
        second = self.scene({"nir": shifted, "red": shifted}, scene_id="second")
        _, _, report = self.process([first, second])
        self.assertEqual(report["saved_dates"], 1, report)
        self.assertEqual(report["date_details"][0]["source_opens"], 3)

    def test_polygon_hole_is_nodata(self):
        from shapely.geometry import Polygon

        outer = self.aoi()
        inner = self.aoi((30, 30, 70, 70))
        geometry = Polygon(outer.exterior.coords, [inner.exterior.coords])
        files, _ = process_items(
            [self.scene({"nir": 3000, "red": 2000})],
            geometry,
            bands=["nir", "red"],
            out_dir=self.root / "output",
            target_crs=3857,
        )
        with rasterio.open(files[0]) as src:
            row, col = src.index(55, 55)
            self.assertEqual(src.read_masks(1)[row, col], 0)
            self.assertEqual(src.read(1)[row, col], COMMON_NODATA)

    def test_default_bands_are_optical_only(self):
        item = self.scene({b: 1000 for b in S2_BANDS})
        _, _, report = self.process([item], bands=None)
        self.assertEqual(report["saved_dates"], 1)
        self.assertEqual(select_bands("sentinel-2-l2a"), ["blue", "green", "red"])

    def test_index_requirements_added_and_invalid_bands_rejected(self):
        self.assertIn("nir", select_bands("sentinel-2-l2a", ["red"], ["NDVI"]))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            select_bands("sentinel-2-l2a", ["vv"])
        with self.assertRaises(ValueError):
            select_bands("sentinel-1-grd", ["vv"], ["NDVI"])

    def test_native_resolution_avoids_upsampling(self):
        preview = preview_processing(
            self.aoi(), bands=["red", "swir16"], resolution="native", target_crs=3857
        )
        self.assertEqual(preview["grid"]["resolution"], 20)
        self.assertEqual(preview["pixels_per_date"], 25)

    def test_missing_assets_distinguished_from_read_failures(self):
        _, _, missing = self.process([self.scene({"red": 1000})])
        self.assertEqual(missing["date_details"][0]["reason"], "missing_bands")
        self.assertEqual(missing["missing_band_counts"], {"nir": 1})
        bad = self.scene({"nir": self.root / "absent.tif", "red": 1000})
        _, _, failed = self.process([bad], out_dir=self.root / "bad")
        row = failed["date_details"][0]
        self.assertEqual(row["reason"], "processing_failed")
        self.assertIn("RasterioIOError", row["error"])
        self.assertEqual(row["missing_bands"], [])

    def test_resume_uses_commit_record(self):
        item = self.scene({"nir": 3000, "red": 2000})
        files, _, _ = self.process([item], indices=["NDVI"])
        timestamp = Path(files[0]).stat().st_mtime_ns
        _, _, report = self.process([item], indices=["NDVI"])
        self.assertTrue(report["date_details"][0]["resumed"])
        self.assertEqual(Path(files[0]).stat().st_mtime_ns, timestamp)

    def test_corrupt_output_recomputed_on_resume(self):
        item = self.scene({"nir": 3000, "red": 2000})
        files, _, _ = self.process([item])
        Path(files[0]).write_bytes(b"broken")
        _, _, report = self.process([item])
        self.assertFalse(report["date_details"][0]["resumed"])
        self.assertEqual(report["saved_dates"], 1)

    def test_different_request_cannot_mix_outputs(self):
        item = self.scene({"nir": 3000, "red": 2000})
        self.process([item])
        with self.assertRaisesRegex(ValueError, "different request"):
            self.process([item], indices=["NDVI"])

    def test_cancellation_then_resume(self):
        event = threading.Event()
        item = self.scene({"nir": 3000, "red": 2000})
        event.set()
        files, _, report = self.process([item], cancel_event=event)
        self.assertEqual(files, [])
        self.assertTrue(report["cancelled"])
        event.clear()
        files, _, report = self.process([item], cancel_event=event)
        self.assertEqual(len(files), 1)

    def test_failed_write_has_no_published_tiffs(self):
        item = self.scene({"nir": 3000, "red": 2000})
        with patch("processing_utils._index_arrays", side_effect=OSError("simulated disk failure")):
            _, _, report = self.process([item])
        self.assertEqual(report["saved_dates"], 0)
        self.assertEqual(list((self.root / "output").glob("*.tif")), [])
        self.assertEqual(list((self.root / "output").glob(".raster-stage-*")), [])
        self.assertFalse((self.root / "output/.dates/2025-04-01.json").exists())

    def test_window_reads_are_bounded_and_handles_reused(self):
        item = self.scene({"nir": np.full((40, 40), 3000), "red": np.full((40, 40), 2000)})
        with patch(
            "processing_utils.windows", wraps=lambda grid: __import__("raster_engine").windows(grid, size=4)
        ):
            _, _, report = self.process([item])
        row = report["date_details"][0]
        self.assertGreater(row["windows_processed"], 1)
        self.assertEqual(row["source_opens"], 2)
        self.assertEqual(row["window_reads"], row["windows_processed"] * 2)

    def test_zero_reflectance_remains_valid_after_normalizing(self):
        item = self.scene({"nir": 1000, "red": 2000}, scales=0.0001, offsets=-0.1)
        _, _, report = self.process([item], indices=["NDVI"])
        with rasterio.open(report["index_files"][0]) as src:
            np.testing.assert_allclose(src.read(1, masked=True).compressed(), -1, atol=1e-6)

    def test_same_pixel_bands_come_from_same_scene(self):
        red = np.full((10, 10), 2000, dtype="uint16")
        red[:, :5] = 0
        first = self.scene({"nir": 3000, "red": red}, scene_id="first", cloud=0)
        second = self.scene({"nir": 4000, "red": 1000}, scene_id="second", cloud=10)
        _, _, report = self.process([first, second], indices=["NDVI"])
        with rasterio.open(report["index_files"][0]) as src:
            values = src.read(1)
            np.testing.assert_allclose(values[:, :5], 0.6, atol=1e-6)
            np.testing.assert_allclose(values[:, 5:], 0.2, atol=1e-6)

    def test_per_scene_calibration_before_mosaicking(self):
        left = np.full((10, 10), 3000, dtype="uint16")
        left[:, 5:] = 0
        first = self.scene(
            {"nir": left, "red": np.where(left > 0, 2000, 0)},
            scene_id="first",
            cloud=0,
            scales=0.0001,
            offsets=-0.1,
        )
        second = self.scene({"nir": 2000, "red": 1000}, scene_id="second", cloud=10, scales=0.0001, offsets=0)
        _, _, report = self.process([first, second], indices=["NDVI"])
        with rasterio.open(report["index_files"][0]) as src:
            np.testing.assert_allclose(src.read(1), 1 / 3, atol=1e-6)

    def test_gcp_only_raster_warped(self):
        from rasterio.crs import CRS

        corners = [(0, 0, 0, 100), (0, 10, 100, 100), (10, 0, 0, 0), (10, 10, 100, 0)]
        gcps = ([GroundControlPoint(row=r, col=c, x=x, y=y) for r, c, x, y in corners], CRS.from_epsg(3857))
        path = self.raster(3000, crs=None, gcps=gcps)
        item = self.scene({"nir": path, "red": path})
        _, _, report = self.process([item])
        self.assertEqual(report["saved_dates"], 1, report)

    def test_scl_filter_masks_clouds_and_reports_missing_scl(self):
        scl = np.full((10, 10), 4, dtype="uint16")
        scl[:, :5] = 9
        item = self.scene({"nir": 3000, "red": 2000, "scl": scl})
        files, _, report = self.process([item], apply_scl_filter=True)
        with rasterio.open(files[0]) as src:
            self.assertTrue((src.read_masks(1)[:, :5] == 0).all())
            self.assertTrue((src.read_masks(1)[:, 5:] == 255).all())
        missing = self.scene({"nir": 3000, "red": 2000})
        _, _, report = self.process([missing], apply_scl_filter=True, out_dir=self.root / "missing-scl")
        self.assertEqual(report["date_details"][0]["reason"], "missing_scl")

    def test_no_overlap_has_precise_reason(self):
        item = self.scene({"nir": 0, "red": 0})
        _, _, report = self.process([item])
        self.assertEqual(report["date_details"][0]["reason"], "no_valid_overlap")
        self.assertEqual(list((self.root / "output").glob("*.tif")), [])

    def test_invalid_date_marker_recomputed(self):
        item = self.scene({"nir": 3000, "red": 2000})
        self.process([item])
        (self.root / "output/.dates/2025-04-01.json").write_text("broken")
        _, _, report = self.process([item])
        self.assertEqual(report["saved_dates"], 1)
        self.assertFalse(report["date_details"][0]["resumed"])

    def test_large_grid_reads_only_bounded_windows(self):
        from shapely.geometry import mapping
        from raster_engine import SourceCache, WINDOW_SIZE

        size = 1024
        geometry = self.aoi((0, 0, size * 10, size * 10))
        nir = self.raster(np.full((size, size), 3000), transform=from_origin(0, size * 10, 10, 10))
        red = self.raster(np.full((size, size), 2000), transform=from_origin(0, size * 10, 10, 10))
        item = self.scene({"nir": nir, "red": red}, geometry=mapping(geometry))
        reads = []
        original = SourceCache.read

        def record(cache, asset, band, window, *args, **kwargs):
            reads.append((window.width, window.height))
            return original(cache, asset, band, window, *args, **kwargs)

        with patch.object(SourceCache, "read", record):
            files, _, report = process_items(
                [item],
                geometry,
                bands=["nir", "red"],
                target_crs=3857,
                out_dir=self.root / "large",
                return_report=True,
            )
        self.assertTrue(all(w <= WINDOW_SIZE and h <= WINDOW_SIZE for w, h in reads))
        self.assertGreater(len(reads), 2)
        self.assertEqual(report["date_details"][0]["valid_pixels"], size**2)
        with rasterio.open(files[0]) as src:
            self.assertTrue(src.overviews(1))

    def test_sar_amplitude_units_are_preserved(self):
        item = self.scene({"vv": 3000, "vh": 2000})
        files, _, report = self.process([item], collection="sentinel-1-grd", bands=["vv", "vh"])
        self.assertEqual(report["saved_dates"], 1)
        with rasterio.open(files[0]) as src:
            self.assertEqual(src.tags()["radiometry"], "amplitude")
            self.assertEqual(src.tags(1)["units"], "amplitude")
            np.testing.assert_allclose(src.read(1), 3000)

    def test_modified_local_source_invalidates_resume(self):
        item = self.scene({"nir": 3000, "red": 2000})
        self.process([item])
        with rasterio.open(item.assets["nir"].href, "r+") as source:
            source.write(np.full((10, 10), 4000, dtype="uint16"), 1)
        with self.assertRaisesRegex(ValueError, "different request"):
            self.process([item])

    def test_interrupted_staging_is_cleaned_on_resume(self):
        item = self.scene({"nir": 3000, "red": 2000})
        self.process([item])
        stale = self.root / "output/.raster-stage-interrupted"
        stale.mkdir()
        (stale / "partial.tif").write_bytes(b"partial")
        self.process([item])
        self.assertFalse(stale.exists())

    def test_distant_footprints_do_not_open_irrelevant_assets(self):
        from shapely.geometry import mapping

        distant = self.scene(
            {"nir": self.root / "missing-nir.tif", "red": self.root / "missing-red.tif"},
            scene_id="distant",
            cloud=0,
            geometry=mapping(self.aoi((10000, 10000, 10100, 10100))),
        )
        covered = self.scene({"nir": 3000, "red": 2000}, scene_id="covered", cloud=10)
        _, _, report = self.process([distant, covered])
        self.assertEqual(report["saved_dates"], 1)
        self.assertEqual(report["date_details"][0]["source_opens"], 2)
