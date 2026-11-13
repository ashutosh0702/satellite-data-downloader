"""Offline point-series correctness and quality-selection tests."""

from unittest.mock import patch
import numpy as np
from rasterio.transform import from_origin

from tests.fixtures import RasterCase
from processing_utils import process_single_item, extract_pixel_timeseries


class TimeseriesTests(RasterCase):
    def sample(self, item, **kwargs):
        point = self.point()
        return process_single_item(item, point.x, point.y, "NDVI", kwargs.get("scl", False))

    def test_offset_applied(self):
        item = self.scene({"nir": 3000, "red": 2000}, scales=0.0001, offsets=-0.1)
        self.assertAlmostEqual(self.sample(item)["Index"], 1 / 3)

    def test_either_nodata_band_rejected(self):
        for nir, red in [(1000, 0), (0, 1000), (0, 0)]:
            with self.subTest(nir=nir, red=red):
                self.assertIsNone(self.sample(self.scene({"nir": nir, "red": red})))

    def test_nonfinite_sample_rejected(self):
        bad = self.raster(np.nan, dtype="float32", nodata=None)
        self.assertIsNone(self.sample(self.scene({"nir": bad, "red": 1000})))

    def test_missing_scl_fails_closed_and_is_reported(self):
        item = self.scene({"nir": 3000, "red": 2000})
        observations, _, report = extract_pixel_timeseries(
            [item], self.point(), apply_scl_filter=True, return_report=True
        )
        self.assertEqual(observations, [])
        self.assertEqual(report["date_details"][0]["attempts"][0]["reason"], "missing_scl")

    def test_cloudy_scene_falls_back_to_clear_scene_in_any_order(self):
        cloudy = self.scene({"nir": 3000, "red": 2000, "scl": 9}, scene_id="a-cloudy", cloud=0)
        clear = self.scene({"nir": 3000, "red": 2000, "scl": 4}, scene_id="b-clear", cloud=10)
        for items in ([cloudy, clear], [clear, cloudy]):
            with self.subTest(order=[item.id for item in items]):
                results, _, report = extract_pixel_timeseries(
                    items, self.point(), apply_scl_filter=True, return_report=True
                )
                self.assertEqual(len(results), 1)
                self.assertEqual(report["date_details"][0]["selected_scene"], clear.id)
                self.assertAlmostEqual(results[0]["Index"], 0.2)

    def test_each_band_uses_its_own_crs_and_bounds(self):
        from pyproj import Transformer

        x, y = Transformer.from_crs(3857, 32631, always_xy=True).transform(0, 100)
        red = self.raster(2000, crs=32631, transform=from_origin(x, y, 10, 10))
        self.assertAlmostEqual(self.sample(self.scene({"nir": 3000, "red": red}))["Index"], 0.2)
        outside = self.raster(2000, transform=from_origin(1000, 100, 10, 10))
        self.assertIsNone(self.sample(self.scene({"nir": 3000, "red": outside})))

    def test_internal_mask_rejected(self):
        masked = self.raster(2000, mask=np.zeros((10, 10)))
        self.assertIsNone(self.sample(self.scene({"nir": 3000, "red": masked})))

    def test_dataset_opened_once_per_asset(self):
        item = self.scene({"nir": 3000, "red": 2000, "scl": 4})
        import rasterio

        with patch("processing_utils.rasterio.open", wraps=rasterio.open) as opened:
            self.assertIsNotNone(self.sample(item, scl=True))
        self.assertEqual(opened.call_count, 3)

    def test_invalid_point_and_index_rejected(self):
        from shapely.geometry import Point

        with self.assertRaises(ValueError):
            extract_pixel_timeseries([], Point(200, 0))
        with self.assertRaises(ValueError):
            extract_pixel_timeseries([], self.point(), index="invalid")

    def test_results_sorted_by_date_and_selection_is_deterministic(self):
        later = self.scene({"nir": 3000, "red": 2000}, day=2)
        first = self.scene({"nir": 3000, "red": 2000}, scene_id="a", day=1, cloud=5)
        second = self.scene({"nir": 4000, "red": 2000}, scene_id="b", day=1, cloud=10)
        results, _, report = extract_pixel_timeseries(
            [later, second, first], self.point(), return_report=True
        )
        self.assertEqual([r["Date"] for r in results], ["2025-04-01", "2025-04-02"])
        self.assertEqual(report["date_details"][0]["selected_scene"], "a")

    def test_acquisition_dates_are_normalized_to_utc(self):
        from datetime import datetime, timezone, timedelta

        item = self.scene({"nir": 3000, "red": 2000})
        item.datetime = datetime(2025, 4, 2, 1, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        result = self.sample(item)
        self.assertEqual(result["Date"], "2025-04-01")

    def test_stac_nodata_respected_when_tiff_has_no_nodata_tag(self):
        red = self.raster(0, nodata=None)
        item = self.scene({"nir": 3000, "red": red})
        item.assets["red"].extra_fields["raster:bands"] = [{"nodata": 0, "scale": 0.0001}]
        self.assertIsNone(self.sample(item))

    def test_external_mask_sidecar_is_respected(self):
        import rasterio

        path = self.raster(2000)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False), rasterio.open(path, "r+") as src:
            src.write_mask(np.zeros((10, 10), dtype="uint8"))
        self.assertTrue(__import__("pathlib").Path(str(path) + ".msk").exists())
        item = self.scene({"nir": 3000, "red": path})
        self.assertIsNone(self.sample(item))
