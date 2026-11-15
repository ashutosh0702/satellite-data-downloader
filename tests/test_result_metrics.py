import json

import numpy as np
from shapely.geometry import Polygon

from result_metrics import PixelStatistics
from tests.fixtures import RasterCase


class StatisticsTests(RasterCase):
    def test_exact_statistics_keep_zeros_and_exclude_invalid_values(self):
        stats = PixelStatistics()
        values = np.array([0, 1, 2, np.nan, 100], dtype="float32")
        stats.update(values, np.array([1, 1, 1, 1, 0], dtype=bool))
        result = stats.summary()
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["mean"], 1)
        self.assertEqual(result["minimum"], 0)
        self.assertEqual(result["median"], 1)
        self.assertFalse(result["quantiles_approximate"])

    def test_sampling_is_bounded_and_reproducible(self):
        first, second = PixelStatistics(seed=42, limit=8), PixelStatistics(seed=42, limit=8)
        for offset in range(10):
            values = np.arange(offset * 100, (offset + 1) * 100, dtype="float32")
            for stats in (first, second):
                stats.update(values, np.ones(100, dtype=bool))
        self.assertEqual(first.count, 1000)
        self.assertEqual(first.summary()["mean"], 499.5)
        self.assertEqual(len(first.sample), 8)
        self.assertTrue(first.summary()["quantiles_approximate"])
        np.testing.assert_array_equal(first.sample, second.sample)

    def test_empty_statistics_are_json_safe(self):
        result = PixelStatistics().summary()
        self.assertIsNone(result["median"])
        json.dumps(result, allow_nan=False)

    def test_partial_coverage_uses_aoi_pixels(self):
        nir = np.full((10, 10), 3000, dtype="uint16")
        nir[:, :5] = 0
        _, _, report = self.process([self.scene({"nir": nir, "red": 2000})], indices=["NDVI"])
        row = report["date_details"][0]
        self.assertEqual(row["aoi_pixels"], 100)
        self.assertEqual(row["coverage_percent"], 50)
        self.assertEqual(len(row["products"]), 2)
        self.assertGreater(row["products"][0]["size_bytes"], 0)
        self.assertAlmostEqual(row["products"][1]["statistics"]["NDVI"]["mean"], 0.2, places=6)

    def test_holes_are_excluded_from_coverage_denominator(self):
        from processing_utils import process_items

        outer, inner = self.aoi(), self.aoi((30, 30, 70, 70))
        geometry = Polygon(outer.exterior.coords, [inner.exterior.coords])
        _, _, report = process_items(
            [self.scene({"nir": 3000, "red": 2000})],
            geometry,
            bands=["nir", "red"],
            target_crs=3857,
            out_dir=self.root / "hole",
            return_report=True,
        )
        self.assertEqual(report["date_details"][0]["aoi_pixels"], 84)
        self.assertEqual(report["date_details"][0]["coverage_percent"], 100)

    def test_index_validity_can_differ_from_composite(self):
        item = self.scene({"nir": 1000, "red": 1000}, scales=0.0001, offsets=-0.1)
        _, _, report = self.process([item], indices=["NDVI"])
        composite, index = report["date_details"][0]["products"]
        self.assertEqual(composite["coverage_percent"], 100)
        self.assertEqual(index["coverage_percent"], 0)
        self.assertEqual(index["statistics"]["NDVI"]["count"], 0)
        self.assertEqual(composite["statistics"]["nir"]["mean"], 0)

    def test_old_committed_results_get_summaries_without_raster_rewrite(self):
        item = self.scene({"nir": 3000, "red": 2000})
        paths, _, _ = self.process([item], indices=["NDVI"])
        before = __import__("pathlib").Path(paths[0]).stat().st_mtime_ns
        marker = self.root / "output/.dates/2025-04-01.json"
        saved = json.loads(marker.read_text())
        for key in ["products", "summary_version", "aoi_pixels", "coverage_percent"]:
            saved.pop(key, None)
        marker.write_text(json.dumps(saved))
        _, _, report = self.process([item], indices=["NDVI"])
        self.assertTrue(report["date_details"][0]["resumed"])
        self.assertEqual(len(report["date_details"][0]["products"]), 2)
        self.assertEqual(__import__("pathlib").Path(paths[0]).stat().st_mtime_ns, before)

    def test_processing_emits_structured_stages_and_committed_completion(self):
        events = []
        self.process([self.scene({"nir": 3000, "red": 2000})], event_callback=events.append)
        stages = [event["stage"] for event in events]
        self.assertIn("processing", stages)
        self.assertIn("finalizing", stages)
        self.assertEqual(events[-1]["stage"], "date_finished")
        self.assertTrue(events[-1]["saved"])
        for event in events:
            if event["stage"] == "processing":
                self.assertLessEqual(event["completed_windows"], event["total_windows"])
