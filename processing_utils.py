import os
import logging
import time
import concurrent.futures
from collections import defaultdict
from functools import lru_cache

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from pyproj import Transformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GDAL / Rasterio environment tuning
# ---------------------------------------------------------------------------
def _get_int_env(name, default):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %d", name, raw_value, default)
        return default

    return max(1, value)


DEFAULT_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    # Let GDAL size the cache from usable RAM instead of pinning 20% of total RAM.
    "GDAL_CACHEMAX": "5%",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "tif,tiff",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_TCP_KEEPALIVE": "YES",
    "GDAL_HTTP_CONNECTTIMEOUT": "30",
}

for env_name, env_value in DEFAULT_GDAL_ENV.items():
    os.environ.setdefault(env_name, env_value)

DATE_WORKER_LIMIT = _get_int_env("SATELLITE_DATE_WORKERS", 8)
BAND_WORKER_LIMIT = _get_int_env("SATELLITE_BAND_WORKERS", 3)
TIMESERIES_WORKER_LIMIT = _get_int_env("SATELLITE_TIMESERIES_WORKERS", 30)

# Sentinel-2 L2A reflectance scale factor (DN → [0, 1])
S2_SCALE_FACTOR = 10000.0
# Common float nodata sentinel for GeoTIFF outputs. QGIS reads this cleanly.
COMMON_NODATA = -99999.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def get_transformer(crs_string):
    """Caches Transformer objects per CRS string to avoid PyProj overhead."""
    return Transformer.from_crs("EPSG:4326", crs_string, always_xy=True)


def _detect_crs(date_items, collection, bands):
    """Fast CRS detection from STAC metadata, falling back to remote file header."""
    for item in date_items:
        if "proj:epsg" in item.properties and item.properties["proj:epsg"] is not None:
            return f"EPSG:{item.properties['proj:epsg']}"

        asset_key = "red" if collection == "sentinel-2-l2a" else (bands[0] if bands else None)
        if asset_key and asset_key in item.assets:
            try:
                with rasterio.open(item.assets[asset_key].href) as src:
                    return src.crs
            except Exception as e:
                logger.warning("CRS detection failed for %s: %s", asset_key, e)
        else:
            for v in item.assets.values():
                if v.media_type and "tiff" in v.media_type:
                    try:
                        with rasterio.open(v.href) as src:
                            return src.crs
                    except Exception as e:
                        logger.warning("CRS detection fallback failed: %s", e)
            break
    return "EPSG:4326"


# ---------------------------------------------------------------------------
# process_items  –  download, merge, clip
# ---------------------------------------------------------------------------

def process_items(items, aoi_geom, collection="sentinel-2-l2a", bands=None,
                  out_dir="downloads", progress_callback=None, return_report=False):
    """
    Downloads, merges, and clips selected bands from STAC items grouped by
    date. Saves one composite per date containing all selected bands.
    Returns (list_of_file_paths, elapsed_seconds) by default.
    When return_report=True, also returns a per-date processing summary.
    """
    os.makedirs(out_dir, exist_ok=True)
    results = []
    start_time = time.time()

        # Group items by date
    items_by_date: dict[str, list] = {}
    for item in items:
        date_str = (item.datetime.strftime("%Y-%m-%d")
                    if item.datetime
                    else item.properties.get("datetime", "unknown_date")[:10])
        items_by_date.setdefault(date_str, []).append(item)

    report = {
        "total_dates": len(items_by_date),
        "saved_dates": 0,
        "skipped_dates": 0,
        "dates_with_missing_bands": 0,
        "missing_band_counts": defaultdict(int),
        "saved_date_labels": [],
        "skipped_date_labels": [],
        "date_details": [],
    }

    print(f"\n--- Starting processing for {len(items_by_date)} unique dates ---")

    def reason_label(reason_code):
        labels = {
            "missing_bands": "Missing selected bands",
            "no_valid_overlap": "No valid overlap with the AOI",
            "aoi_out_of_bounds": "AOI did not intersect the raster grid",
            "no_bands_selected": "No bands selected",
            "processing_failed": "Processing failed",
            "unknown_failure": "Composite was not generated",
        }
        return labels.get(reason_code, reason_code.replace("_", " ").title())

    # ------------------------------------------------------------------
    def process_date(date_str, date_items):
        date_results = []
        date_report = {
            "date": date_str,
            "item_count": len(date_items),
            "saved": False,
            "output_count": 0,
            "reason": None,
            "reason_label": None,
            "missing_bands": [],
        }
        print(f"  [>] Processing date: {date_str} ({len(date_items)} items)")
        logger.info("Processing date: %s with %d items", date_str, len(date_items))

        crs = _detect_crs(date_items, collection, bands)

        # Project AOI to target CRS
        geom = gpd.GeoSeries([aoi_geom], crs="EPSG:4326").to_crs(crs)
        aoi_crs_geom = geom.iloc[0]
        aoi_bounds = aoi_crs_geom.bounds  # (minx, miny, maxx, maxy)

        # ---- inner: fetch a single band -----------------------------------
        def fetch_band(band, target_resolution):
            urls = [item.assets[band].href
                    for item in date_items if band in item.assets]
            if not urls:
                logger.warning("Band %s missing for %s", band, date_str)
                return None, None, None, None

            src_files = []
            try:
                for url in urls:
                    src_files.append(rasterio.open(url))

                if not src_files:
                    return None, None, None, None

                if target_resolution is None:
                    target_res = tuple(map(float, src_files[0].res))
                else:
                    target_res = tuple(map(float, target_resolution))

                # Use the same merge/grid path for single-tile and multi-tile dates.
                # This preserves partial tile overlap instead of dropping one-tile dates.
                mosaic, trans = merge(
                    src_files,
                    bounds=aoi_bounds,
                    res=target_res,
                    nodata=0,
                    masked=True,
                    resampling=Resampling.bilinear,
                )
                if mosaic.size == 0:
                    return None, None, None, None

                band_data = np.asarray(mosaic[0].filled(COMMON_NODATA), dtype=np.float32)
                coverage_mask = ~np.ma.getmaskarray(mosaic[0])
                meta = src_files[0].meta.copy()
                return band_data, trans, meta, coverage_mask
            except Exception as e:
                logger.error("Failed to fetch band %s for %s: %s",
                             band, date_str, e)
                return None, None, None, None
            finally:
                for f in src_files:
                    f.close()

        # ---- inner: create multiband composite ----------------------------
        def create_composite(bands_list, resolution, name_suffix):
            # Fetch bands in parallel with a configurable cap to avoid throttling.
            band_results = {}
            max_band_workers = min(BAND_WORKER_LIMIT, len(bands_list))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_band_workers) as pool:
                future_map = {
                    pool.submit(fetch_band, b, resolution): b
                    for b in bands_list
                }
                for future in concurrent.futures.as_completed(future_map):
                    bname = future_map[future]
                    try:
                        arr, trans, meta, coverage_mask = future.result()
                    except Exception as e:
                        logger.error("Band future %s raised: %s", bname, e)
                        continue
                    if arr is not None:
                        band_results[bname] = (arr, trans, meta, coverage_mask)

            if len(band_results) != len(bands_list):
                missing_bands = [b for b in bands_list if b not in band_results]
                print(f"Missing bands for {date_str} {name_suffix}: {missing_bands}")
                logger.warning("Missing bands for %s %s – %s",
                               date_str, name_suffix, missing_bands)
                return None, {
                    "reason": "missing_bands",
                    "missing_bands": missing_bands,
                }

            first_arr, first_trans, first_meta, _ = band_results[bands_list[0]]
            
            # Apply polygon mask directly on numpy
            gmask = geometry_mask(
                [aoi_crs_geom],
                out_shape=(first_arr.shape[0], first_arr.shape[1]),
                transform=first_trans,
                invert=True,
            )
            
            # Crop to tight bounding box of the polygon mask
            rows = np.any(gmask, axis=1)
            cols = np.any(gmask, axis=0)
            if not rows.any() or not cols.any():
                logger.warning("AOI does not intersect raster grid for %s %s",
                               date_str, name_suffix)
                return None, {
                    "reason": "aoi_out_of_bounds",
                    "missing_bands": [],
                }
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Crop the gmask itself
            cropped_mask = gmask[rmin:rmax + 1, cmin:cmax + 1]
            combined_valid_mask = cropped_mask.copy()

            for b in bands_list:
                coverage = band_results[b][3][rmin:rmax + 1, cmin:cmax + 1]
                combined_valid_mask &= coverage

            if not combined_valid_mask.any():
                logger.warning("No valid data overlap for %s %s", date_str, name_suffix)
                return None, {
                    "reason": "no_valid_overlap",
                    "missing_bands": [],
                }
            
            # Gather and crop all bands
            cropped_bands = []
            for b in bands_list:
                arr = band_results[b][0][rmin:rmax + 1, cmin:cmax + 1].copy()
                arr[~combined_valid_mask] = COMMON_NODATA
                cropped_bands.append(arr)
            
            # Add a simple mask band for downstream AOI extraction logic.
            alpha = combined_valid_mask.astype(np.float32)
            cropped_bands.append(alpha)
            
            stacked = np.stack(cropped_bands)

            crop_transform = rasterio.transform.from_bounds(
                first_trans.c + cmin * first_trans.a,
                first_trans.f + (rmax + 1) * first_trans.e,
                first_trans.c + (cmax + 1) * first_trans.a,
                first_trans.f + rmin * first_trans.e,
                cmax - cmin + 1,
                rmax - rmin + 1,
            )

            final_meta = first_meta.copy()
            final_meta.update(
                driver="GTiff",
                height=stacked.shape[1],
                width=stacked.shape[2],
                transform=crop_transform,
                count=len(bands_list) + 1, # +1 for Alpha
                dtype="float32",
                nodata=COMMON_NODATA,
                interleave='pixel', # Better for some viewers
                tiled=True,
                compress='lzw'
            )

            final_path = os.path.join(out_dir, f"{date_str}_{name_suffix}.tif")
            with rasterio.open(final_path, "w", **final_meta) as dest:
                dest.write(stacked)
                for i, b in enumerate(bands_list, 1):
                    dest.set_band_description(i, b)
                dest.set_band_description(len(bands_list) + 1, "alpha")
            return final_path, {
                "reason": None,
                "missing_bands": [],
            }
        # ---- end inner helpers --------------------------------------------

        if collection == "sentinel-2-l2a":
            all_bands = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "nir09", "swir16", "swir22", "vh", "vv"]
            selected_bands = list(bands) if bands else all_bands
            path = None
            composite_info = {
                "reason": "no_bands_selected",
                "missing_bands": [],
            }
            if selected_bands: 
                path, composite_info = create_composite(selected_bands, (10, 10), "10m_all_bands")
            if path:
                print(f"    [+] Created 10m all-bands composite: {os.path.basename(path)}")
                date_results.append(path)  # Single composite only
                date_report["saved"] = True
                date_report["output_count"] = 1
            else:
                date_report["reason"] = composite_info["reason"]
                date_report["reason_label"] = reason_label(composite_info["reason"])
                date_report["missing_bands"] = composite_info["missing_bands"]
        else:
            selected_bands = list(bands) if bands else []
            if not selected_bands:
                date_report["reason"] = "no_bands_selected"
                date_report["reason_label"] = reason_label("no_bands_selected")
                return date_results, date_report

            path, composite_info = create_composite(selected_bands, None, "all_bands")
            if path:
                print(f"    [+] Created all-bands composite: {os.path.basename(path)}")
                date_results.append(path)
                date_report["saved"] = True
                date_report["output_count"] = 1
            else:
                date_report["reason"] = composite_info["reason"]
                date_report["reason_label"] = reason_label(composite_info["reason"])
                date_report["missing_bands"] = composite_info["missing_bands"]

        if not date_report["saved"] and date_report["reason"] is None:
            date_report["reason"] = "unknown_failure"
            date_report["reason_label"] = reason_label("unknown_failure")

        return date_results, date_report
    # ------------------------------------------------------------------

    workers = min(DATE_WORKER_LIMIT, len(items_by_date)) if items_by_date else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_date = {
            executor.submit(process_date, d, itms): d
            for d, itms in items_by_date.items()
        }
        for i, future in enumerate(concurrent.futures.as_completed(future_to_date)):
            d_str = future_to_date[future]
            try:
                res, date_report = future.result()
            except Exception as e:
                logger.error("Date %s failed: %s", d_str, e)
                res = []
                date_report = {
                    "date": d_str,
                    "item_count": len(items_by_date[d_str]),
                    "saved": False,
                    "output_count": 0,
                    "reason": "processing_failed",
                    "reason_label": reason_label("processing_failed"),
                    "missing_bands": [],
                }

            report["date_details"].append(date_report)
            if date_report["saved"]:
                report["saved_dates"] += 1
                report["saved_date_labels"].append(date_report["date"])
            else:
                report["skipped_dates"] += 1
                report["skipped_date_labels"].append(date_report["date"])
                if date_report["missing_bands"]:
                    report["dates_with_missing_bands"] += 1
                    for band_name in sorted(set(date_report["missing_bands"])):
                        report["missing_band_counts"][band_name] += 1

            if res:
                results.extend(res)
            if progress_callback:
                progress_callback(i + 1, len(future_to_date))

    t_sec = round(time.time() - start_time, 2)
    print(f"--- Processing complete! Found {len(results)} files in {t_sec}s ---\n")
    logger.info("Composites generated in %.2fs", t_sec)
    report["date_details"] = sorted(report["date_details"], key=lambda row: row["date"])
    report["missing_band_counts"] = dict(sorted(report["missing_band_counts"].items()))
    logger.info(
        "Processing summary: %d dates total, %d saved, %d skipped, %d dates with missing bands",
        report["total_dates"],
        report["saved_dates"],
        report["skipped_dates"],
        report["dates_with_missing_bands"],
    )

    if return_report:
        return results, t_sec, report
    return results, t_sec


# ---------------------------------------------------------------------------
# Timeseries extraction
# ---------------------------------------------------------------------------

# SCL values that indicate bad pixels
_SCL_BAD = frozenset([0, 1, 3, 8, 9, 10])


def process_single_item(item, lon, lat, index_name, apply_scl_filter):
    """Extract a single index value for one STAC item at (lon, lat)."""
    date_str = (item.datetime.strftime("%Y-%m-%d")
                if item.datetime
                else item.properties.get("datetime", "unknown")[:10])
    assets = item.assets
    cloud_cover = item.properties.get("eo:cloud_cover", 0)

    if index_name == "NDVI":
        b1_name, b2_name = "nir", "red"
    elif index_name == "NDWI":
        b1_name, b2_name = "green", "nir"
    else:
        return None

    if b1_name not in assets or b2_name not in assets:
        return None

    url_b1 = assets[b1_name].href
    url_b2 = assets[b2_name].href

    try:
        with rasterio.Env():
            # Transform coordinates ONCE (reused for SCL + bands)
            # Use the first available band to determine CRS
            crs_source = url_b1
            if apply_scl_filter and "scl" in assets:
                crs_source = assets["scl"].href

            with rasterio.open(crs_source) as ref_src:
                transformer = get_transformer(ref_src.crs.to_string())
                x, y = transformer.transform(lon, lat)

            # --- SCL filter (optional) ---
            if apply_scl_filter and "scl" in assets:
                with rasterio.open(assets["scl"].href) as src_scl:
                    if not (src_scl.bounds.left <= x <= src_scl.bounds.right and
                            src_scl.bounds.bottom <= y <= src_scl.bounds.top):
                        return None
                    scl_val = next(src_scl.sample([(x, y)]))[0]
                    if scl_val in _SCL_BAD:
                        return None

            # --- Band extraction (reuse x, y from above) ---
            with rasterio.open(url_b1) as src1, rasterio.open(url_b2) as src2:
                if not (src1.bounds.left <= x <= src1.bounds.right and
                        src1.bounds.bottom <= y <= src1.bounds.top):
                    return None

                val1 = next(src1.sample([(x, y)]))[0]
                val2 = next(src2.sample([(x, y)]))[0]

                if val1 == 0 and val2 == 0:
                    return None

                denom = float(val1 + val2)
                if denom != 0:
                    index_val = (val1 - val2) / denom
                    return {
                        "Date": date_str,
                        "Index": index_val,
                        "CloudCover": cloud_cover,
                    }
    except Exception as e:
        logger.warning("Item %s pixel extraction failed: %s", date_str, e)
    return None


def extract_pixel_timeseries(items, point_geom, index="NDVI",
                             apply_scl_filter=False, progress_callback=None):
    """Extract a timeseries of index values at a point from STAC items."""
    results = []
    lat, lon = point_geom.y, point_geom.x

    items_by_date = {}
    for item in items:
        date_str = (item.datetime.strftime("%Y-%m-%d")
                    if item.datetime
                    else item.properties.get("datetime", "unknown")[:10])
        if date_str not in items_by_date:
            items_by_date[date_str] = item

    unique_items = list(items_by_date.values())
    start_time = time.time()

    timeseries_workers = min(TIMESERIES_WORKER_LIMIT, len(unique_items)) if unique_items else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=timeseries_workers) as executor:
        futures = [
            executor.submit(process_single_item, item, lon, lat,
                            index, apply_scl_filter)
            for item in unique_items
        ]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                res = future.result()
            except Exception as e:
                logger.warning("Timeseries future failed: %s", e)
                res = None
            if res:
                results.append(res)
            if progress_callback:
                progress_callback(i + 1, len(futures))

    results = sorted(results, key=lambda x: x["Date"])
    t_sec = round(time.time() - start_time, 2)
    logger.info("Timeseries: %d observations in %.2fs from %d items",
                len(results), t_sec, len(unique_items))
    return results, t_sec


# ---------------------------------------------------------------------------
# Index calculation on downloaded composites
# ---------------------------------------------------------------------------

# Index definitions: name → {bands, func}
INDEX_FUNCS = {
    "NDVI":  {"bands": ["nir", "red"],
              "func": lambda n, r: (n - r) / (n + r + 1e-8)},
    "NDWI":  {"bands": ["green", "nir"],
              "func": lambda g, n: (g - n) / (g + n + 1e-8)},
    "NDMI":  {"bands": ["nir", "swir16"],
              "func": lambda n, s: (n - s) / (n + s + 1e-8)},
    "NDSI":  {"bands": ["green", "swir16"],
              "func": lambda g, s: (g - s) / (g + s + 1e-8)},
    "EVI":   {"bands": ["nir", "red", "blue"],
              "func": lambda n, r, b: 2.5 * (n - r) / (n + 6*r - 7.5*b + 1 + 1e-8)},
    "SAVI":  {"bands": ["nir", "red"],
              "func": lambda n, r: 1.5 * (n - r) / (n + r + 0.5 + 1e-8)},
    "NDRE":  {"bands": ["nir", "rededge1"],
              "func": lambda n, re: (n - re) / (n + re + 1e-8)},
}

# Indices that need reflectance-scaled inputs (DN → [0-1])
_REFLECTANCE_INDICES = {"EVI", "SAVI"}


def calculate_indices(file_paths, indices, out_dir):
    """
    Calculate spectral indices from multiband composite GeoTIFFs.
    Handles cross-resolution bands by resampling 20 m data to 10 m using
    bilinear interpolation via rasterio.warp.
    Only reads bands actually required by the requested indices.
    Returns a list of generated index file paths.
    """
    if not indices or not file_paths:
        return []

    index_files = []

    # Determine which bands we actually need across all requested indices
    needed_bands = set()
    for idx_name in indices:
        if idx_name in INDEX_FUNCS:
            needed_bands.update(INDEX_FUNCS[idx_name]["bands"])

    # Group files by date prefix
    date_files = defaultdict(list)
    for fpath in file_paths:
        basename = os.path.basename(fpath)
        date_prefix = basename.split("_")[0]
        date_files[date_prefix].append(fpath)

    for date_prefix, fpaths in date_files.items():
        # Build band pool – only read bands we actually need
        band_pool = {}          # band_name → {"array", "meta", "res"}
        ref_meta = None         # highest-resolution file's metadata
        ref_transform = None
        aoi_mask = None         # To be derived from alpha channel

        for fpath in fpaths:
            try:
                with rasterio.open(fpath) as src:
                    band_desc = src.descriptions
                    res = src.res
                    meta = src.meta.copy()

                    if ref_meta is None or res[0] < ref_meta["transform"][0]:
                        ref_meta = meta
                        ref_transform = src.transform

                    for i, desc in enumerate(band_desc):
                        if desc == "alpha":
                            # Use alpha channel as mask (0 = transparent)
                            alpha = src.read(i + 1)
                            if aoi_mask is None or alpha.shape == (ref_meta["height"], ref_meta["width"]):
                                aoi_mask = (alpha > 0)
                        elif desc and desc in needed_bands:
                            band_pool[desc] = {
                                "array": src.read(i + 1).astype(np.float32),
                                "meta": meta,
                                "res": res,
                                "nodata": src.nodata,
                            }
            except Exception as e:
                logger.error("Error reading %s: %s", fpath, e)

        if not band_pool or ref_meta is None:
            continue

        ref_meta.update(
            count=1, 
            dtype="float32", 
            nodata=COMMON_NODATA,
            compress='lzw',
            tiled=True
        )
        ref_shape = (ref_meta["height"], ref_meta["width"])

        for idx_name in indices:
            if idx_name not in INDEX_FUNCS:
                continue

            req_bands = INDEX_FUNCS[idx_name]["bands"]
            if not all(b in band_pool for b in req_bands):
                logger.warning("Missing bands %s for %s on %s – skipping",
                               req_bands, idx_name, date_prefix)
                continue

            # Gather arrays; resample lower-res bands with bilinear interpolation
            arrays = []
            valid_mask = np.ones(ref_shape, dtype=bool)
            for b in req_bands:
                arr = band_pool[b]["array"]
                band_nodata = band_pool[b].get("nodata")
                if arr.shape != ref_shape:
                    src_meta = band_pool[b]["meta"]
                    resampled = np.full(ref_shape, COMMON_NODATA, dtype=np.float32)
                    reproject(
                        arr, resampled,
                        src_transform=src_meta["transform"],
                        src_crs=src_meta["crs"],
                        dst_transform=ref_transform,
                        dst_crs=ref_meta["crs"],
                        resampling=Resampling.bilinear,
                        src_nodata=band_nodata if band_nodata is not None else COMMON_NODATA,
                        dst_nodata=COMMON_NODATA,
                        init_dest_nodata=True,
                    )
                    arr = resampled

                invalid = ~np.isfinite(arr) | np.isclose(arr, COMMON_NODATA)
                if band_nodata is not None:
                    invalid |= np.isclose(arr, band_nodata)
                valid_mask &= ~invalid
                arrays.append(np.where(invalid, np.nan, arr))

            # Normalize to reflectance [0-1] for indices that need it
            if idx_name in _REFLECTANCE_INDICES:
                scaled_arrays = []
                for a in arrays:
                    finite_values = a[np.isfinite(a)]
                    if finite_values.size and np.nanmax(finite_values) > 2:
                        scaled_arrays.append(a / S2_SCALE_FACTOR)
                    else:
                        scaled_arrays.append(a)
                arrays = scaled_arrays

            with np.errstate(divide="ignore", invalid="ignore"):
                idx_array = INDEX_FUNCS[idx_name]["func"](*arrays)

            # Keep a single explicit nodata sentinel in the written GeoTIFF.
            idx_array = idx_array.astype(np.float32)
            idx_array[~np.isfinite(idx_array)] = COMMON_NODATA
            
            # Apply AOI mask derived from alpha channel if available
            if aoi_mask is not None:
                if idx_array.shape == aoi_mask.shape:
                    valid_mask &= aoi_mask
                else:
                    # Resample mask if it doesn't match (unlikely but possible)
                    resampled_mask = np.zeros(idx_array.shape, dtype=np.uint8)
                    reproject(
                        aoi_mask.astype(np.uint8), resampled_mask,
                        src_transform=ref_transform,
                        src_crs=ref_meta["crs"],
                        dst_transform=ref_transform,
                        dst_crs=ref_meta["crs"],
                        resampling=Resampling.nearest,
                        src_nodata=0,
                        dst_nodata=0,
                        init_dest_nodata=True,
                    )
                    valid_mask &= (resampled_mask > 0)

            idx_array[~valid_mask] = COMMON_NODATA

            out_path = os.path.join(out_dir, f"{date_prefix}_{idx_name}.tif")
            with rasterio.open(out_path, "w", **ref_meta) as dest:
                dest.write(idx_array, 1)
                dest.set_band_description(1, idx_name)

            index_files.append(out_path)

    return index_files
