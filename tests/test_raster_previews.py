from io import BytesIO

import numpy as np
from PIL import Image

from result_artifacts import committed_files, preview_png, summary_csv
from tests.fixtures import RasterCase


class PreviewTests(RasterCase):
    def test_thumbnail_is_bounded_and_retains_transparent_nodata(self):
        values = np.full((1024, 2048), 0.2, dtype="float32")
        values[:, :100] = -99999
        path = self.raster(values, dtype="float32", nodata=-99999, descriptions=["NDVI"])
        png, _ = preview_png(path, max_size=128)
        with Image.open(BytesIO(png)) as image:
            self.assertLessEqual(max(image.size), 128)
            self.assertEqual(image.mode, "RGBA")
            rgba = np.asarray(image)
            self.assertTrue((rgba[:, :5, 3] == 0).all())
            self.assertTrue((rgba[:, -5:, 3] == 255).all())

    def test_rgb_uses_only_requested_bands_and_fixed_stretch(self):
        path = self.raster(
            np.stack([np.full((10, 10), 0.3), np.full((10, 10), 0.15), np.zeros((10, 10))]),
            dtype="float32",
            nodata=-99999,
            descriptions=["red", "green", "blue"],
        )
        png, legend = preview_png(path, rgb=True)
        with Image.open(BytesIO(png)) as image:
            np.testing.assert_array_equal(np.asarray(image)[0, 0], [255, 128, 0, 255])
        self.assertIn("fixed", legend)
        with self.assertRaises(ValueError):
            preview_png(self.raster(0.2, dtype="float32"), rgb=True)

    def test_summary_csv_has_band_statistics_and_zero_values(self):
        _, _, report = self.process(
            [self.scene({"nir": 1000, "red": 2000}, scales=0.0001, offsets=-0.1)], indices=["NDVI"]
        )
        files = committed_files(self.root / "output")
        csv = summary_csv(files)
        self.assertIn("NDVI", csv)
        self.assertIn("quantiles_approximate", csv)
        nir = next(row for row in files.values() if row["name"] == "composite")
        self.assertEqual(nir["statistics"]["nir"]["mean"], 0)
        self.assertEqual(nir["units"], "reflectance")
        self.assertEqual(nir["resolution_m"], 10)
