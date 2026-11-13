from types import SimpleNamespace
from unittest.mock import patch

import rasterio

from raster_engine import calibration
from processing_utils import process_single_item, extract_pixel_timeseries
from tests.fixtures import RasterCase


class RadiometryPolicyTests(RasterCase):
    def test_ambiguous_legacy_offsets_rejected_by_default(self):
        path = self.raster(3000)
        asset = SimpleNamespace(
            href="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/test.tif",
            extra_fields={"raster:bands": [{"scale": 0.0001, "offset": -0.1}]},
        )
        with rasterio.open(path) as src:
            with self.assertRaisesRegex(ValueError, "Ambiguous legacy"):
                calibration(src, asset, "nir")
            self.assertEqual(calibration(src, asset, "nir", radiometry_policy="metadata")[:2], (0.0001, -0.1))
            self.assertEqual(calibration(src, asset, "nir", radiometry_policy="harmonized")[:2], (0.0001, 0))

    def test_point_and_series_apply_explicit_legacy_policy(self):
        item = self.scene({"nir": 3000, "red": 2000}, scales=0.0001, offsets=-0.1)
        paths = {}
        for band, asset in item.assets.items():
            path = asset.href
            asset.href = f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/{band}.tif"
            paths[asset.href] = path
        point = self.point()
        original = rasterio.open
        with patch(
            "processing_utils.rasterio.open",
            side_effect=lambda href, *args, **kw: original(paths[href], *args, **kw),
        ):
            with self.assertLogs("processing_utils", level="WARNING"):
                self.assertIsNone(process_single_item(item, point.x, point.y, "NDVI", False))
            harmonized = process_single_item(
                item, point.x, point.y, "NDVI", False, radiometry_policy="harmonized"
            )
            declared = process_single_item(
                item, point.x, point.y, "NDVI", False, radiometry_policy="metadata"
            )
            series, _ = extract_pixel_timeseries([item], point, radiometry_policy="harmonized")
        self.assertAlmostEqual(harmonized["Index"], 0.2)
        self.assertAlmostEqual(declared["Index"], 1 / 3)
        self.assertAlmostEqual(series[0]["Index"], 0.2)

    def test_collection_one_supports_indices_and_cloud_filtering(self):
        item = self.scene({"nir": 3000, "red": 2000, "scl": 4}, scales=0.0001, offsets=-0.1)
        item.collection_id = "sentinel-2-c1-l2a"
        _, _, report = self.process(
            [item], collection="sentinel-2-c1-l2a", indices=["NDVI"], apply_scl_filter=True
        )
        self.assertEqual(report["saved_dates"], 1)
        with rasterio.open(report["index_files"][0]) as src:
            self.assertAlmostEqual(float(src.read(1)[0, 0]), 1 / 3, places=6)
