"""Reproducible local pipeline benchmark, with optional original-source comparison.

Run: python scripts/benchmark_pipeline.py --baseline-source /tmp/original.py
The parent prepares fixtures outside measured child processes. No network is used.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BANDS = ["blue", "green", "red", "rededge1", "nir", "nir08", "swir16", "swir22"]
INDICES = ["NDVI", "EVI", "SAVI"]


def peak_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if sys.platform == "darwin" else 1024)


def worker(args):
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import box

    if args.mode == "baseline":
        spec = importlib.util.spec_from_file_location("baseline", args.baseline_source)
        pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline)
    else:
        import processing_utils as pipeline
    assets = {
        b: SimpleNamespace(href=str(args.work / f"{b}.tif"), media_type="image/tiff", extra_fields={})
        for b in BANDS
    }
    item = SimpleNamespace(
        id="synthetic",
        datetime=datetime(2025, 4, 1, tzinfo=timezone.utc),
        properties={"proj:epsg": 3857},
        assets=assets,
    )
    inverse = Transformer.from_crs(3857, 4326, always_xy=True)
    aoi = box(*inverse.transform(0, 0), *inverse.transform(args.size * 10, args.size * 10))
    initial_peak = peak_mib()
    start = time.perf_counter()
    output = args.work / args.mode
    if args.mode == "baseline":
        files, _ = pipeline.process_items([item], aoi, bands=BANDS, out_dir=str(output))
        indices = pipeline.calculate_indices(files, INDICES, str(output))
        stats = {}
    else:
        files, _, report = pipeline.process_items(
            [item],
            aoi,
            bands=BANDS,
            indices=INDICES,
            out_dir=output,
            target_crs=3857,
            return_report=True,
            overviews=False,
        )
        indices = report["index_files"]
        stats = {
            k: report["date_details"][0][k] for k in ("windows_processed", "source_opens", "window_reads")
        }
    elapsed = time.perf_counter() - start
    if len(files) != 1 or len(indices) != len(INDICES):
        raise RuntimeError("Benchmark did not produce the expected outputs.")
    metrics = {
        "mode": args.mode,
        "size": args.size,
        "pixels": args.size**2,
        "seconds": round(elapsed, 3),
        "initial_peak_mib": round(initial_peak, 2),
        "peak_rss_mib": round(peak_mib(), 2),
        "output_bytes": sum(p.stat().st_size for p in output.glob("*.tif")),
        **stats,
    }
    for path in indices:
        with rasterio.open(path) as src:
            sample = src.read(1, window=rasterio.windows.Window(100, 100, 1, 1), masked=True)
            metrics.setdefault("index_samples", {})[src.descriptions[0]] = float(sample[0, 0])
    print("METRICS=" + json.dumps(metrics))


def main(args):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    results = []
    env = {
        **os.environ,
        "GDAL_CACHEMAX": "64",
        "SATELLITE_GDAL_CACHE_MB": "64",
        "SATELLITE_DATE_WORKERS": "1",
        "SATELLITE_BAND_WORKERS": "1",
        "SATELLITE_GLOBAL_TASKS": "1",
        "SATELLITE_WINDOW_SIZE": "512",
    }
    with TemporaryDirectory(prefix="satellite-benchmark-") as temporary:
        for size in map(int, args.sizes.split(",")):
            work = Path(temporary) / str(size)
            work.mkdir()
            rng = np.random.default_rng(42)
            for band in BANDS:
                # Padding keeps the original pipeline's unsigned-fill bug out of
                # this benchmark; correctness is checked separately in unit tests.
                with rasterio.open(
                    work / f"{band}.tif",
                    "w",
                    driver="GTiff",
                    width=size + 2,
                    height=size + 2,
                    count=1,
                    dtype="uint16",
                    crs=3857,
                    transform=from_origin(-10, (size + 1) * 10, 10, 10),
                    nodata=0,
                    tiled=True,
                    compress="lzw",
                ) as dst:
                    for _, window in dst.block_windows(1):
                        dst.write(
                            rng.integers(
                                1000, 9000, size=(int(window.height), int(window.width)), dtype=np.uint16
                            ),
                            1,
                            window=window,
                        )
            for mode in ["baseline", "windowed"] if args.baseline_source else ["windowed"]:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--mode",
                    mode,
                    "--work",
                    str(work),
                    "--size",
                    str(size),
                ]
                if args.baseline_source:
                    command.extend(["--baseline-source", str(args.baseline_source.resolve())])
                completed = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
                row = json.loads(
                    next(
                        line.removeprefix("METRICS=")
                        for line in completed.stdout.splitlines()
                        if line.startswith("METRICS=")
                    )
                )
                results.append(row)
                print(json.dumps(row), flush=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "rasterio": rasterio.__version__,
        "gdal": rasterio.__gdal_version__,
        "bands": BANDS,
        "indices": INDICES,
        "settings": {k: env[k] for k in env if k.startswith("SATELLITE_")},
        "notes": "Local synthetic COG inputs, warm OS cache, one date/worker; overview generation disabled. Timings exclude fixture generation, imports and catalog/network access. RSS includes imports and GDAL caches. Not a cloud latency benchmark.",
        "results": results,
        "pipeline_source_hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "processing_utils.py",
                "raster_engine.py",
                "resource_budget.py",
                "satellite_config.py",
            )
        },
    }
    if args.baseline_source:
        report["baseline_sha256"] = hashlib.sha256(args.baseline_source.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="1024,2048")
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/benchmark_results.json")
    parser.add_argument("--mode", choices=["baseline", "windowed"])
    parser.add_argument("--work", type=Path)
    parser.add_argument("--size", type=int)
    args = parser.parse_args()
    if args.mode:
        worker(args)
    else:
        main(args)
