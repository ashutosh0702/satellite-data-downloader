"""Explicit public-data smoke test. Uses a London fixture, never a user's AOI."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import rasterio
    from shapely.geometry import box, Point
    from stac_utils import search_stac
    from processing_utils import process_items, process_single_item, get_transformer, gdal_options

    collection = "sentinel-2-c1-l2a"
    bbox = [-0.13, 51.50, -0.129, 51.501]
    search = search_stac(
        bbox, "2025-04-01", "2025-04-27", collection, max_items=1, return_report=True, use_cache=False
    )
    if not search.items:
        raise RuntimeError("No scenes returned for the public smoke fixture.")
    item = search.items[0]
    point = Point(-0.1295, 51.5005)
    observation = process_single_item(item, point.x, point.y, "NDVI", False)
    if observation is None:
        raise RuntimeError("Point extraction failed.")
    raw_values, physical = {}, {}
    with rasterio.Env(**gdal_options()):
        for band in ("nir", "red"):
            asset = item.assets[band]
            with rasterio.open(asset.href) as src:
                xy = get_transformer(src.crs.to_string()).transform(point.x, point.y)
                sample = next(src.sample([xy], masked=True))[0]
                if np.ma.is_masked(sample):
                    raise RuntimeError("Smoke fixture has a nodata sample.")
                raw_values[band] = float(sample)
            metadata = asset.extra_fields["raster:bands"][0]
            physical[band] = raw_values[band] * metadata.get("scale", 1) + metadata.get("offset", 0)
    expected = (physical["nir"] - physical["red"]) / (physical["nir"] + physical["red"])
    if not np.isclose(observation["Index"], expected, atol=1e-8):
        raise RuntimeError("Point result disagrees with independently evaluated calibration.")
    with TemporaryDirectory(prefix="satellite-live-smoke-") as temporary:
        files, seconds, report = process_items(
            [item],
            box(*bbox),
            collection=collection,
            bands=["red", "nir"],
            indices=["NDVI"],
            out_dir=temporary,
            return_report=True,
        )
        if report["saved_dates"] != 1:
            raise RuntimeError(report["date_details"])
        with rasterio.open(files[0]) as src:
            if src.tags()["radiometry"] != "reflectance" or src.read(1, masked=True).count() == 0:
                raise RuntimeError("Composite did not contain normalized valid reflectance.")
        with rasterio.open(report["index_files"][0]) as src:
            if not np.isfinite(src.read(1, masked=True).compressed()).all():
                raise RuntimeError("Nonfinite output index.")
        result = {
            "collection": collection,
            "search": search.summary(),
            "scene": item.id,
            "point_index": observation,
            "independent_expected_ndvi": expected,
            "raw_samples": raw_values,
            "reflectance_samples": physical,
            "seconds": seconds,
            "saved_dates": report["saved_dates"],
            "resources": report["resources"],
            "calibration": report["date_details"][0]["radiometry"],
        }
    output = ROOT / "docs/live_smoke_results.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
