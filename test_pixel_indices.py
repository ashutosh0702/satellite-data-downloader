import unittest
import warnings
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

from processing_utils import process_single_item


class PixelIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def extract_index(self, index_name, band_values):
        assets = {}
        for band_name, value in band_values.items():
            path = Path(self.temp_dir.name) / f"{band_name}.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=1,
                height=1,
                count=1,
                dtype="uint16",
                crs="EPSG:3857",
                transform=from_origin(-5, 5, 10, 10),
                nodata=0,
            ) as dest:
                dest.write(np.array([[value]], dtype=np.uint16), 1)
            assets[band_name] = SimpleNamespace(href=str(path))

        item = SimpleNamespace(
            datetime=datetime(2025, 4, 3, tzinfo=timezone.utc),
            properties={"eo:cloud_cover": 12.5},
            assets=assets,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            result = process_single_item(item, 0, 0, index_name, False)

        runtime_warnings = [
            str(warning.message) for warning in caught if issubclass(warning.category, RuntimeWarning)
        ]
        self.assertEqual(runtime_warnings, [])
        return result

    def test_ndvi_negative_difference(self):
        result = self.extract_index("NDVI", {"nir": 1000, "red": 2000})
        self.assertIsNotNone(result)
        self.assertEqual(set(result), {"Date", "Index", "CloudCover"})
        self.assertEqual(result["Date"], "2025-04-03")
        self.assertEqual(result["CloudCover"], 12.5)
        self.assertAlmostEqual(result["Index"], -1 / 3)

    def test_ndwi_negative_difference(self):
        result = self.extract_index("NDWI", {"green": 1000, "nir": 2000})
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["Index"], -1 / 3)

    def test_sum_exceeding_uint16_range(self):
        for index_name, first_band, second_band in [
            ("NDVI", "nir", "red"),
            ("NDWI", "green", "nir"),
        ]:
            with self.subTest(index=index_name):
                result = self.extract_index(index_name, {first_band: 40000, second_band: 30000})
                self.assertIsNotNone(result)
                self.assertAlmostEqual(result["Index"], 1 / 7)

    def test_zero_pixels_return_no_observation(self):
        for index_name in ("NDVI", "NDWI"):
            with self.subTest(index=index_name):
                result = self.extract_index(index_name, {"nir": 0, "red": 0, "green": 0})
                self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
