"""Known-value and alignment regressions using local GeoTIFF fixtures."""

import numpy as np
import rasterio
from rasterio.transform import from_origin

from processing_utils import calculate_indices
from satellite_config import COMMON_NODATA, INDEX_FUNCS
from tests.fixtures import RasterCase


class IndexTests(RasterCase):
    def composite(
        self, values, descriptions, *, transform=None, units="reflectance", dtype="float32", **kwargs
    ):
        original = self.raster(
            np.stack([np.full((10, 10), v) for v in values]),
            descriptions=descriptions,
            units=units,
            dtype=dtype,
            transform=transform,
            nodata=COMMON_NODATA if dtype == "float32" else 0,
            **kwargs,
        )
        path = original.with_name(f"2025-04-01_{original.stem}.tif")
        original.rename(path)
        return path

    def test_all_indices_against_known_values(self):
        values = {"nir": 0.6, "red": 0.2, "green": 0.3, "blue": 0.1, "swir16": 0.15, "rededge1": 0.4}
        path = self.composite(list(values.values()), list(values))
        outputs = calculate_indices([path], list(INDEX_FUNCS), self.root / "indices")
        expected = {
            "NDVI": 0.5,
            "NDWI": -1 / 3,
            "NDMI": 0.6,
            "NDSI": 1 / 3,
            "EVI": 2.5 * 0.4 / (0.6 + 6 * 0.2 - 7.5 * 0.1 + 1),
            "SAVI": 1.5 * 0.4 / (0.6 + 0.2 + 0.5),
            "NDRE": 0.2,
        }
        self.assertEqual(len(outputs), len(expected))
        for path in outputs:
            with rasterio.open(path) as src:
                name = src.descriptions[0]
                np.testing.assert_allclose(src.read(1, masked=True).compressed(), expected[name], atol=1e-6)
                self.assertEqual(src.nodata, COMMON_NODATA)

    def test_equal_shape_shifted_grids_do_not_combine(self):
        nir = self.composite([0.3], ["nir"])
        red = self.composite([0.1], ["red"], transform=from_origin(1000, 100, 10, 10))
        output = calculate_indices([nir, red], ["NDVI"], self.root / "indices")[0]
        with rasterio.open(output) as src:
            self.assertTrue(src.read(1, masked=True).mask.all())

    def test_lower_resolution_alpha_uses_original_transform(self):
        nir = self.composite([0.3], ["nir"])
        red = self.raster(
            np.stack([np.full((5, 5), 0.1), np.tile([0, 0, 1, 1, 1], (5, 1))]),
            dtype="float32",
            nodata=COMMON_NODATA,
            descriptions=["red", "alpha"],
            units="reflectance",
            transform=from_origin(0, 100, 20, 20),
        )
        path = red.with_name("2025-04-01_red.tif")
        red.rename(path)
        output = calculate_indices([nir, path], ["NDVI"], self.root / "indices")[0]
        with rasterio.open(output) as src:
            mask = src.read_masks(1)
            self.assertTrue((mask[:, :4] == 0).all())
            self.assertTrue((mask[:, 4:] == 255).all())

    def test_geotiff_scale_and_offset(self):
        path = self.composite(
            [3000, 2000],
            ["nir", "red"],
            dtype="uint16",
            units=None,
            scales=[0.0001, 0.0001],
            offsets=[-0.1, -0.1],
        )
        output = calculate_indices([path], ["NDVI"], self.root / "indices")[0]
        with rasterio.open(output) as src:
            np.testing.assert_allclose(src.read(1), 1 / 3, atol=1e-6)

    def test_legacy_float_requires_explicit_units(self):
        path = self.composite([3000, 2000], ["nir", "red"], units=None)
        with self.assertRaisesRegex(ValueError, "unknown units"):
            calculate_indices([path], ["NDVI"], self.root / "indices")
        output = calculate_indices([path], ["NDVI"], self.root / "indices", legacy_scale=0.0001)[0]
        with rasterio.open(output) as src:
            np.testing.assert_allclose(src.read(1), 0.2, atol=1e-6)

    def test_zero_denominator_is_nodata(self):
        path = self.composite([0.1, -0.1], ["nir", "red"])
        output = calculate_indices([path], ["NDVI"], self.root / "indices")[0]
        with rasterio.open(output) as src:
            self.assertTrue(src.read(1, masked=True).mask.all())

    def test_missing_band_and_duplicate_band_are_actionable(self):
        path = self.composite([0.3], ["nir"])
        with self.assertRaisesRegex(ValueError, "Missing bands"):
            calculate_indices([path], ["NDVI"], self.root / "indices")
        with self.assertRaisesRegex(ValueError, "Duplicate band"):
            calculate_indices([path, path], ["NDVI"], self.root / "indices")
