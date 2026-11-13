"""Windowed satellite composites and deterministic, quality-aware point series.

Existing tuple return shapes are retained. Optional reports, integrated indices,
resolution, cancellation and resumable manifests extend the original API.
"""

from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import Point

from persistence import directory_lock, write_json
from raster_engine import (
    Grid,
    SourceCache,
    calibration,
    ensure_georeferenced,
    estimated_task_mb,
    make_grid,
    polygon_mask,
    raster_bundle,
    valid_raster,
    validate_geometry,
    windows,
    SceneIndex,
    WINDOW_SIZE,
)
from resource_budget import BUDGET, CancelledError, bounded_results, check_cancel, io_slot, positive_env
from satellite_config import COMMON_NODATA, INDEX_FUNCS, SCL_BAD, select_bands, S2_COLLECTIONS
from satellite_config import S2_SCALE_FACTOR as S2_SCALE_FACTOR

logger = logging.getLogger(__name__)


class NoValidOverlapError(ValueError):
    pass


DATE_WORKER_LIMIT = positive_env("SATELLITE_DATE_WORKERS", 4)
# Kept for compatibility with old benchmark clients. Bands now share a window,
# so nested band executors are deliberately removed.
BAND_WORKER_LIMIT = 1
TIMESERIES_WORKER_LIMIT = positive_env("SATELLITE_TIMESERIES_WORKERS", 8)
DEFAULT_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "tif,tiff",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_TCP_KEEPALIVE": "YES",
    "GDAL_HTTP_CONNECTTIMEOUT": "15",
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_TIFF_INTERNAL_MASK": True,
}


def gdal_options(requester_pays=False):
    options = {k: os.environ.get(k, v) for k, v in DEFAULT_GDAL_ENV.items()}
    options["GDAL_CACHEMAX"] = positive_env("SATELLITE_GDAL_CACHE_MB", 64) * 1024**2
    if requester_pays:
        options["AWS_REQUEST_PAYER"] = "requester"
    return options


@lru_cache(maxsize=32)
def get_transformer(crs_string):
    return Transformer.from_crs("EPSG:4326", crs_string, always_xy=True)


def item_datetime(item):
    value = item.datetime or item.properties.get("datetime")
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value is None:
        raise ValueError("Every scene must have an acquisition datetime.")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def scene_key(item):
    cloud = item.properties.get("eo:cloud_cover", 100)
    cloud = float(cloud) if cloud is not None else 100.0
    if not math.isfinite(cloud):
        cloud = 100.0
    return (
        cloud,
        item_datetime(item).isoformat(),
        getattr(item, "id", ""),
        tuple(sorted(asset.href for asset in item.assets.values())),
    )


def _asset_signature(asset):
    signature = {"href": asset.href, "metadata": getattr(asset, "extra_fields", {})}
    if "://" not in asset.href:
        path = Path(asset.href)
        if path.is_file():
            stat = path.stat()
            signature["local_file"] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return signature


def _item_signature(item):
    return {
        "id": getattr(item, "id", ""),
        "datetime": item_datetime(item).isoformat(),
        "properties": item.properties,
        "assets": {key: _asset_signature(asset) for key, asset in sorted(item.assets.items())},
    }


def preflight_access(items, bands, requester_pays=False):
    s3_assets = [
        item.assets[b].href
        for item in items
        for b in bands
        if b in item.assets and item.assets[b].href.startswith("s3://")
    ]
    if not s3_assets:
        return
    if any(href.startswith("s3://sentinel-s1-l1c/") for href in s3_assets) and not requester_pays:
        raise ValueError(
            "Sentinel-1 assets require requester-pays access. Enable it and configure AWS credentials."
        )
    try:
        import boto3
        import botocore.session
        from botocore.exceptions import BotoCoreError

        session = botocore.session.Session()
        session.set_config_variable("metadata_service_timeout", 1)
        session.set_config_variable("metadata_service_num_attempts", 1)
        credentials = boto3.Session(botocore_session=session).get_credentials()
        if credentials is None:
            raise ValueError("No AWS credentials found. Configure an AWS profile or environment credentials.")
        credentials.get_frozen_credentials()  # Check expired/unrefreshable providers before starting writers.
    except ImportError as exc:
        raise ValueError("Install boto3 to access S3 assets.") from exc
    except BotoCoreError as exc:
        raise ValueError(
            f"AWS credential preflight failed ({type(exc).__name__}); check your AWS profile."
        ) from exc


def preview_processing(
    aoi_geom, collection="sentinel-2-l2a", bands=None, indices=None, resolution=10, target_crs=None, dates=1
):
    selected = select_bands(collection, bands, indices)
    grid = make_grid(aoi_geom, selected, collection, resolution, target_crs)
    task_mb = estimated_task_mb(len(selected))
    if task_mb > BUDGET.memory_mb:
        raise ValueError("Window working set exceeds SATELLITE_MEMORY_MB; reduce SATELLITE_WINDOW_SIZE.")
    workers = max(1, min(DATE_WORKER_LIMIT, BUDGET.tasks, BUDGET.memory_mb // task_mb))
    return {
        "bands": selected,
        "grid": grid.to_dict(),
        "pixels_per_date": grid.width * grid.height,
        "estimated_task_mb": task_mb,
        "date_workers": workers,
        "estimated_output_bytes": grid.width
        * grid.height
        * 4
        * (len(selected) + 1 + len(indices or []))
        * dates,
        "window_size": WINDOW_SIZE,
        "memory_budget_mb": BUDGET.memory_mb,
    }


def _index_arrays(band_arrays, names, common_valid):
    results = {}
    for name in names:
        spec = INDEX_FUNCS[name]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            values = spec["func"](*(band_arrays[b] for b in spec["bands"]))
        valid = common_valid & np.isfinite(values)
        results[name] = (np.where(valid, values, COMMON_NODATA).astype(np.float32), valid)
    return results


def _date_report(date_str, count):
    return {
        "date": date_str,
        "item_count": count,
        "saved": False,
        "output_count": 0,
        "reason": None,
        "reason_label": None,
        "missing_bands": [],
        "error": None,
        "files": [],
        "index_files": [],
        "resumed": False,
    }


def _process_date(
    date_str,
    date_items,
    grid,
    bands,
    indices,
    collection,
    out_dir,
    fingerprint,
    cancel_event,
    resume,
    apply_scl_filter,
    requester_pays,
    radiometry_policy,
    detail_callback,
    overviews,
):
    detail = _date_report(date_str, len(date_items))
    marker = out_dir / ".dates" / f"{date_str}.json"
    with BUDGET.reserve(estimated_task_mb(len(bands)), cancel_event):
        check_cancel(cancel_event)
        with rasterio.Env(**gdal_options(requester_pays)):
            if resume and marker.exists():
                try:
                    saved = json.loads(marker.read_text())
                except (OSError, ValueError):
                    saved = {}
                paths = saved.get("files", []) + saved.get("index_files", [])
                if (
                    saved.get("fingerprint") == fingerprint
                    and paths
                    and all(
                        Path(p).name == p and valid_raster(out_dir / p, fingerprint, cancel_event)
                        for p in paths
                    )
                ):
                    saved.update(resumed=True)
                    return saved
            missing = [b for b in bands if not any(b in item.assets for item in date_items)]
            if missing:
                detail.update(
                    reason="missing_bands", reason_label="Missing selected bands", missing_bands=missing
                )
                return detail
            candidates = [i for i in sorted(date_items, key=scene_key) if all(b in i.assets for b in bands)]
            if apply_scl_filter:
                candidates = [i for i in candidates if "scl" in i.assets]
            if not candidates:
                detail.update(
                    reason="missing_scl" if apply_scl_filter else "incomplete_scenes",
                    reason_label="No scene has all required assets",
                )
                return detail
            required_disk = math.ceil(grid.width * grid.height * 4 * (len(bands) + 1 + len(indices)) * 1.4)
            if required_disk > shutil.disk_usage(out_dir).free * 0.8:
                raise ValueError("Insufficient free disk space for this date's staged outputs.")
            paths = {"composite": out_dir / f"{date_str}_{grid.resolution:g}m_all_bands.tif"}
            paths.update({name: out_dir / f"{date_str}_{name}.tif" for name in indices})
            descriptions = {"composite": bands + ["alpha"], **{name: [name] for name in indices}}
            marker.unlink(missing_ok=True)
            valid_pixels = window_count = 0
            scene_index = SceneIndex(candidates, grid)
            with SourceCache(grid, cancel_event, requester_pays, radiometry_policy) as cache:
                with raster_bundle(paths, grid, descriptions, fingerprint, overviews) as writers:
                    if collection not in S2_COLLECTIONS:
                        writers["composite"].update_tags(radiometry="amplitude", units="amplitude")
                        for bidx in range(1, len(bands) + 1):
                            writers["composite"].update_tags(bidx, units="amplitude")
                    for window in windows(grid):
                        check_cancel(cancel_event)
                        inside = polygon_mask(grid, window)
                        shape = inside.shape
                        arrays = {b: np.full(shape, COMMON_NODATA, dtype=np.float32) for b in bands}
                        filled = np.zeros(shape, dtype=bool)
                        for item in scene_index.query(window):
                            if not np.any(inside & ~filled):
                                break
                            scene_arrays = {}
                            valid = inside & ~filled
                            if apply_scl_filter:
                                scl, scl_valid = cache.read(item.assets["scl"], "scl", window, collection)
                                valid &= scl_valid & ~np.isin(scl, list(SCL_BAD))
                            if not valid.any():
                                continue
                            for band in bands:
                                data, coverage = cache.read(item.assets[band], band, window, collection)
                                scene_arrays[band] = data
                                valid &= coverage
                            # All bands at a pixel come from the same acquisition.
                            for band in bands:
                                arrays[band][valid] = scene_arrays[band][valid]
                            filled |= valid
                        for bidx, band in enumerate(bands, 1):
                            writers["composite"].write(arrays[band], bidx, window=window)
                        writers["composite"].write(filled.astype(np.float32), len(bands) + 1, window=window)
                        writers["composite"].write_mask(filled.astype(np.uint8) * 255, window=window)
                        for name, (data, valid) in _index_arrays(arrays, indices, filled).items():
                            writers[name].write(data, 1, window=window)
                            writers[name].write_mask(valid.astype(np.uint8) * 255, window=window)
                        valid_pixels += int(filled.sum())
                        window_count += 1
                        if detail_callback:
                            detail_callback(f"{date_str}: window {window_count}")
                    if not valid_pixels:
                        raise NoValidOverlapError("No valid data overlap with the AOI after masking.")
                detail.update(
                    saved=True,
                    output_count=1,
                    files=[paths["composite"].name],
                    index_files=[paths[n].name for n in indices],
                    fingerprint=fingerprint,
                    valid_pixels=valid_pixels,
                    windows_processed=window_count,
                    source_opens=cache.opens,
                    window_reads=cache.reads,
                    radiometry=list(cache.provenance.values()),
                    source_ids=[getattr(i, "id", "") for i in candidates],
                )
            # The marker is the commit record for the entire composite/index bundle.
            write_json(marker, detail)
    return detail


def process_items(
    items,
    aoi_geom,
    collection="sentinel-2-l2a",
    bands=None,
    out_dir="downloads",
    progress_callback=None,
    return_report=False,
    *,
    indices=None,
    resolution=10,
    target_crs=None,
    cancel_event=None,
    resume=True,
    apply_scl_filter=False,
    requester_pays=False,
    detail_callback=None,
    overviews=True,
    radiometry_policy="strict",
):
    started = time.perf_counter()
    items, indices = list(items), list(dict.fromkeys(indices or []))
    bands = select_bands(collection, bands, indices)
    if radiometry_policy not in ("strict", "metadata", "harmonized"):
        raise ValueError("Unknown radiometry policy.")
    if apply_scl_filter and collection not in S2_COLLECTIONS:
        raise ValueError("SCL filtering is only available for Sentinel-2.")
    grid = make_grid(aoi_geom, bands, collection, resolution, target_crs)
    preflight_access(items, bands, requester_pays)
    by_date = defaultdict(list)
    for item in items:
        by_date[item_datetime(item).strftime("%Y-%m-%d")].append(item)
    config = {
        "pipeline_version": 2,
        "collection": collection,
        "bands": bands,
        "indices": indices,
        "grid": grid.to_dict(),
        "aoi": aoi_geom.normalize().wkb_hex,
        "apply_scl_filter": apply_scl_filter,
        "radiometry_policy": radiometry_policy,
        "items": sorted((_item_signature(i) for i in items), key=lambda i: json.dumps(i, sort_keys=True)),
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    summary = preview_processing(
        aoi_geom, collection, bands, indices, grid.resolution, grid.crs, len(by_date)
    )
    if summary["estimated_task_mb"] > BUDGET.memory_mb:
        raise ValueError(
            "Window working set exceeds the process memory budget; reduce SATELLITE_WINDOW_SIZE."
        )
    report = {
        "total_dates": len(by_date),
        "saved_dates": 0,
        "skipped_dates": 0,
        "dates_with_missing_bands": 0,
        "missing_band_counts": {},
        "saved_date_labels": [],
        "skipped_date_labels": [],
        "date_details": [],
        "index_files": [],
        "cancelled": False,
        "fingerprint": fingerprint,
        "config": {k: v for k, v in config.items() if k not in ("items", "aoi")},
        "config_file": "processing-config.json",
        "resources": summary,
        "manifest_version": 2,
    }

    def save_manifest():
        report["date_details"].sort(key=lambda row: row["date"])
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_json(manifest_path, report)

    with directory_lock(out_dir):
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text())
            if prior.get("fingerprint") != fingerprint:
                raise ValueError("Output directory belongs to a different request; choose a new directory.")
        config_path = out_dir / "processing-config.json"
        if config_path.exists():
            try:
                saved_config = json.loads(config_path.read_text())
            except (OSError, ValueError):
                saved_config = None
            if (
                saved_config is not None
                and hashlib.sha256(json.dumps(saved_config, sort_keys=True).encode()).hexdigest()
                != fingerprint
            ):
                raise ValueError("Saved processing configuration differs; choose a new output directory.")
        write_json(config_path, config)
        # A process crash may leave staging directories. The directory lock
        # proves no cooperating writer is still using them.
        for stale in out_dir.glob(".raster-stage-*"):
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale)
        # Only reserve disk for dates that do not have a completed marker.
        remaining = sum(not (resume and (out_dir / ".dates" / f"{d}.json").exists()) for d in by_date)
        per_date = summary["estimated_output_bytes"] // max(1, len(by_date))
        if per_date * remaining * 1.4 > shutil.disk_usage(out_dir).free * 0.8:
            raise ValueError("Insufficient free disk space for estimated uncompressed outputs.")
        save_manifest()

        def run_date(value):
            d, scenes = value
            try:
                return _process_date(
                    d,
                    scenes,
                    grid,
                    bands,
                    indices,
                    collection,
                    out_dir,
                    fingerprint,
                    cancel_event,
                    resume,
                    apply_scl_filter,
                    requester_pays,
                    radiometry_policy,
                    detail_callback,
                    overviews,
                )
            except CancelledError as exc:
                row = _date_report(d, len(scenes))
                row.update(reason="cancelled", reason_label="Cancelled", error=str(exc))
                return row
            except Exception as exc:
                logger.exception("Date %s failed", d)
                row = _date_report(d, len(scenes))
                row.update(
                    reason="no_valid_overlap"
                    if isinstance(exc, NoValidOverlapError)
                    else "processing_failed",
                    reason_label="No valid overlap with the AOI"
                    if isinstance(exc, NoValidOverlapError)
                    else "Processing failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                return row

        processed = set()
        for (d, _), future in bounded_results(
            run_date, sorted(by_date.items()), summary["date_workers"], cancel_event
        ):
            row = future.result()
            processed.add(d)
            report["date_details"].append(row)
            if row["saved"]:
                report["saved_dates"] += 1
                report["saved_date_labels"].append(d)
            else:
                report["skipped_dates"] += 1
                report["skipped_date_labels"].append(d)
                if row["missing_bands"]:
                    report["dates_with_missing_bands"] += 1
                    for b in row["missing_bands"]:
                        report["missing_band_counts"][b] = report["missing_band_counts"].get(b, 0) + 1
            save_manifest()
            if progress_callback:
                progress_callback(len(processed), len(by_date))
        for d in sorted(by_date.keys() - processed):
            row = _date_report(d, len(by_date[d]))
            row.update(reason="cancelled", reason_label="Cancelled before processing")
            report["date_details"].append(row)
            report["skipped_dates"] += 1
            report["skipped_date_labels"].append(d)
        report["cancelled"] = cancel_event is not None and cancel_event.is_set()
        results = [str(out_dir / p) for row in report["date_details"] if row["saved"] for p in row["files"]]
        report["index_files"] = [
            str(out_dir / p) for row in report["date_details"] if row["saved"] for p in row["index_files"]
        ]
        report["saved_date_labels"].sort()
        report["skipped_date_labels"].sort()
        save_manifest()
    elapsed = round(time.perf_counter() - started, 3)
    return (sorted(results), elapsed, report) if return_report else (sorted(results), elapsed)


def _sample(src, lon, lat, asset=None):
    ensure_georeferenced(src)
    # GDAL handles GCP-only SAR inputs in the raster pipeline. Optical point
    # samples need a regular grid; use an on-demand VRT if only GCPs are present.
    if src.crs is None:
        from rasterio.vrt import WarpedVRT

        with WarpedVRT(src, crs="EPSG:4326", dtype="float32", nodata=COMMON_NODATA) as vrt:
            return _sample(vrt, lon, lat, asset)
    x, y = get_transformer(src.crs.to_string()).transform(lon, lat)
    row, col = src.index(x, y)
    if not (0 <= row < src.height and 0 <= col < src.width):
        return None
    sample = next(src.sample([(x, y)], masked=True))[0]
    if np.ma.is_masked(sample) or not np.isfinite(sample):
        return None
    records = getattr(asset, "extra_fields", {}).get("raster:bands", []) if asset else []
    if records and records[0].get("nodata") is not None and sample == records[0]["nodata"]:
        return None
    return float(sample)


def _single_observation(
    item, lon, lat, index_name, apply_scl_filter, cancel_event=None, radiometry_policy="strict"
):
    if index_name not in ("NDVI", "NDWI"):
        raise ValueError("Point extraction supports NDVI and NDWI.")
    bands = INDEX_FUNCS[index_name]["bands"]
    if any(b not in item.assets for b in bands):
        return None, "missing_bands"
    if apply_scl_filter and "scl" not in item.assets:
        return None, "missing_scl"
    check_cancel(cancel_event)
    try:
        with rasterio.Env(**gdal_options()), ExitStack() as stack:
            opened = {}

            def source(band):
                href = item.assets[band].href
                if href not in opened:
                    local_options = {"GDAL_DISABLE_READDIR_ON_OPEN": "FALSE"} if "://" not in href else {}
                    with io_slot(cancel_event), rasterio.Env(**local_options):
                        opened[href] = stack.enter_context(rasterio.open(href))
                return opened[href]

            if apply_scl_filter:
                scl_source = source("scl")
                with (
                    io_slot(cancel_event),
                    rasterio.Env(
                        GDAL_DISABLE_READDIR_ON_OPEN="FALSE" if "://" not in scl_source.name else "EMPTY_DIR"
                    ),
                ):
                    scl = _sample(scl_source, lon, lat, item.assets["scl"])
                if scl is None or scl in SCL_BAD:
                    return None, "cloud_or_invalid_scl"
            values = []
            for band in bands:
                check_cancel(cancel_event)
                src = source(band)
                with (
                    io_slot(cancel_event),
                    rasterio.Env(
                        GDAL_DISABLE_READDIR_ON_OPEN="FALSE" if "://" not in src.name else "EMPTY_DIR"
                    ),
                ):
                    value = _sample(src, lon, lat, item.assets[band])
                if value is None:
                    return None, "nodata_or_outside"
                scale, offset, _ = calibration(
                    src,
                    item.assets[band],
                    band,
                    collection=getattr(item, "collection_id", None) or "sentinel-2-l2a",
                    radiometry_policy=radiometry_policy,
                )
                values.append(value * scale + offset)
            denominator = values[0] + values[1]
            if denominator == 0:
                return None, "zero_denominator"
            result = (values[0] - values[1]) / denominator
            if not math.isfinite(result):
                return None, "nonfinite"
            return {
                "Date": item_datetime(item).strftime("%Y-%m-%d"),
                "Index": result,
                "CloudCover": item.properties.get("eo:cloud_cover", 0),
            }, None
    except CancelledError:
        raise
    except Exception as exc:
        logger.warning("Scene %s failed: %s", getattr(item, "id", "unknown"), exc)
        return None, f"read_failed: {type(exc).__name__}: {exc}"


def process_single_item(item, lon, lat, index_name, apply_scl_filter, *, radiometry_policy="strict"):
    """Compatibility wrapper returning an observation or None."""
    validate_geometry(Point(lon, lat), point=True)
    with BUDGET.reserve(16):
        return _single_observation(
            item, lon, lat, index_name, apply_scl_filter, radiometry_policy=radiometry_policy
        )[0]


def extract_pixel_timeseries(
    items,
    point_geom,
    index="NDVI",
    apply_scl_filter=False,
    progress_callback=None,
    *,
    return_report=False,
    cancel_event=None,
    radiometry_policy="strict",
):
    validate_geometry(point_geom, point=True)
    if radiometry_policy not in ("strict", "metadata", "harmonized"):
        raise ValueError("Unknown radiometry policy.")
    if index not in ("NDVI", "NDWI"):
        raise ValueError("Point extraction supports NDVI and NDWI.")
    started = time.perf_counter()
    by_date = defaultdict(list)
    for item in items:
        # Exact footprint test avoids irrelevant bbox hits before opening COGs.
        geometry = getattr(item, "geometry", None)
        if geometry:
            from shapely.geometry import shape

            if not shape(geometry).covers(point_geom):
                continue
        by_date[item_datetime(item).strftime("%Y-%m-%d")].append(item)
    results, details = [], []

    def run_date(value):
        d, candidates = value
        row = {"date": d, "attempts": [], "selected_scene": None}
        with BUDGET.reserve(16, cancel_event):
            for item in sorted(candidates, key=scene_key):
                observation, reason = _single_observation(
                    item, point_geom.x, point_geom.y, index, apply_scl_filter, cancel_event, radiometry_policy
                )
                row["attempts"].append({"id": getattr(item, "id", ""), "reason": reason})
                if observation is not None:
                    row["selected_scene"] = getattr(item, "id", "")
                    return observation, row
        return None, row

    for _, future in bounded_results(
        run_date, sorted(by_date.items()), min(TIMESERIES_WORKER_LIMIT, BUDGET.tasks), cancel_event
    ):
        try:
            observation, row = future.result()
        except CancelledError:
            continue
        details.append(row)
        if observation is not None:
            results.append(observation)
        if progress_callback:
            progress_callback(len(details), len(by_date))
    results.sort(key=lambda row: row["Date"])
    elapsed = round(time.perf_counter() - started, 3)
    report = {
        "total_dates": len(by_date),
        "valid_observations": len(results),
        "cancelled": cancel_event is not None and cancel_event.is_set(),
        "date_details": sorted(details, key=lambda row: row["date"]),
        "read_failures": sum(
            str(attempt.get("reason", "")).startswith("read_failed")
            for row in details
            for attempt in row["attempts"]
        ),
        "selection": "lowest scene cloud cover, then acquisition time and scene id; first valid pixel",
    }
    return (results, elapsed, report) if return_report else (results, elapsed)


def calculate_indices(file_paths, indices, out_dir, *, legacy_scale=None, cancel_event=None):
    """Windowed index calculation for explicit composites, including shifted grids.

    Unknown float units must be supplied explicitly via legacy_scale; old DN
    float composites use 0.0001, already-normalized reflectance uses 1.
    """
    indices = list(dict.fromkeys(indices))
    if any(name not in INDEX_FUNCS for name in indices):
        raise ValueError("Unknown spectral index.")
    if not indices or not file_paths:
        return []
    out_dir = Path(out_dir).resolve()
    by_date = defaultdict(list)
    for path in file_paths:
        by_date[Path(path).name.split("_")[0]].append(Path(path).resolve())
    generated = []
    with (
        directory_lock(out_dir),
        BUDGET.reserve(estimated_task_mb(4), cancel_event),
        rasterio.Env(**gdal_options()),
    ):
        for date_str, paths in sorted(by_date.items()):
            check_cancel(cancel_event)
            with ExitStack() as stack:
                sources = [stack.enter_context(rasterio.open(path)) for path in paths]
                pool = {}
                masks = {}
                for src in sources:
                    ensure_georeferenced(src)
                    for bidx, name in enumerate(src.descriptions, 1):
                        if name and name != "alpha":
                            if name in pool:
                                raise ValueError(
                                    f"Duplicate band {name} for {date_str}; pass one explicit composite set."
                                )
                            pool[name] = (src, bidx)
                        elif name == "alpha":
                            masks[src.name] = bidx
                needed = {b for name in indices for b in INDEX_FUNCS[name]["bands"]}
                missing = needed - pool.keys()
                if missing:
                    raise ValueError(f"Missing bands for {date_str}: {', '.join(sorted(missing))}")
                reference = min((pool[b][0] for b in needed), key=lambda src: abs(src.transform.determinant))
                grid = Grid(
                    reference.crs,
                    reference.transform,
                    reference.width,
                    reference.height,
                    None,
                    float(reference.res[0]),
                )
                # Windowed VRT reads always compare full geography, not only shape.
                fingerprint = hashlib.sha256(
                    json.dumps(
                        {
                            "paths": [str(p) for p in paths],
                            "mtimes": [p.stat().st_mtime_ns for p in paths],
                            "indices": indices,
                            "legacy_scale": legacy_scale,
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                output_paths = {name: out_dir / f"{date_str}_{name}.tif" for name in indices}
                with (
                    SourceCache(grid, cancel_event) as cache,
                    raster_bundle(
                        output_paths, grid, {name: [name] for name in indices}, fingerprint
                    ) as writers,
                ):
                    for window in windows(grid):
                        check_cancel(cancel_event)
                        arrays, coverage = {}, {}
                        for band in needed:
                            src, bidx = pool[band]
                            asset = SimpleNamespace(href=src.name, extra_fields={})
                            data, valid = cache.read(
                                asset, band, window, bidx=bidx, legacy_scale=legacy_scale, require_units=True
                            )
                            if src.name in masks:
                                alpha, alpha_valid = cache.read(asset, "alpha", window, bidx=masks[src.name])
                                valid &= alpha_valid & (alpha > 0)
                            arrays[band], coverage[band] = data, valid
                        for name in indices:
                            valid = np.logical_and.reduce([coverage[b] for b in INDEX_FUNCS[name]["bands"]])
                            data, valid = _index_arrays(arrays, [name], valid)[name]
                            writers[name].write(data, 1, window=window)
                            writers[name].write_mask(valid.astype(np.uint8) * 255, window=window)
                generated.extend(str(path) for path in output_paths.values())
    return generated
