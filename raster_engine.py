"""Bounded raster windows, radiometry, reprojection and staged GeoTIFF writes."""

from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, transform as window_transform
from pyproj import Transformer
from shapely.ops import transform as transform_geometry

from resource_budget import check_cancel, io_slot, positive_env
from satellite_config import COMMON_NODATA, COLLECTION_BANDS, S2_SCALE_FACTOR, S2_COLLECTIONS

WINDOW_SIZE = min(2048, positive_env("SATELLITE_WINDOW_SIZE", 512))
SOURCE_CACHE_SIZE = min(64, positive_env("SATELLITE_SOURCE_CACHE", 16))
WARP_MEMORY_MB = positive_env("SATELLITE_WARP_MEMORY_MB", 8)


def validate_geometry(geometry, point=False):
    expected = {"Point"} if point else {"Polygon", "MultiPolygon"}
    if geometry is None or geometry.is_empty or geometry.geom_type not in expected:
        raise ValueError(f"Provide a nonempty {'point' if point else 'polygon or multipolygon'}.")
    if not geometry.is_valid:
        raise ValueError("Geometry is invalid; repair it before processing.")
    bounds = geometry.bounds
    if not all(math.isfinite(v) for v in bounds):
        raise ValueError("Coordinates must be finite.")
    if bounds[0] < -180 or bounds[2] > 180 or bounds[1] < -90 or bounds[3] > 90:
        raise ValueError("Coordinates must be WGS84 longitude/latitude.")
    if bounds[2] - bounds[0] > 180:
        raise ValueError("Split AOIs crossing the antimeridian into separate jobs.")
    if not point and (bounds[1] <= -90 or bounds[3] >= 90):
        raise ValueError("AOIs must exclude the exact poles.")
    # Bound rasterization cost independently of the upload byte limit.
    from shapely import get_num_coordinates

    if get_num_coordinates(geometry) > 200000:
        raise ValueError("Simplify this geometry to at most 200,000 coordinates.")
    return geometry


@dataclass(frozen=True)
class Grid:
    crs: CRS
    transform: object
    width: int
    height: int
    geometry: object
    resolution: float

    @property
    def shape(self):
        return self.height, self.width

    def to_dict(self):
        return {
            "crs": self.crs.to_string(),
            "transform": list(self.transform)[:6],
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
        }


def make_grid(aoi, bands, collection, resolution=10, target_crs=None):
    validate_geometry(aoi)
    if resolution == "native":
        resolution = max(COLLECTION_BANDS[collection][b] for b in bands)
    resolution = float(resolution)
    if not math.isfinite(resolution) or not 0 < resolution <= 10000:
        raise ValueError("Resolution must be in (0, 10000] metres, or 'native'.")
    if target_crs is None:
        lon, lat = aoi.centroid.x, aoi.centroid.y
        if lat > 84:
            target_crs = "EPSG:3413"
        elif lat < -80:
            target_crs = "EPSG:3031"
        else:
            zone = max(1, min(60, int((lon + 180) // 6) + 1))
            target_crs = f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"
    crs = CRS.from_user_input(target_crs)
    if not crs.is_projected or crs.linear_units != "metre":
        raise ValueError("Destination CRS must be projected in metres.")
    transformer = Transformer.from_crs(4326, crs, always_xy=True)
    projected = transform_geometry(transformer.transform, aoi)
    if projected.is_empty or not all(math.isfinite(x) for x in projected.bounds):
        raise ValueError("AOI cannot be projected to the destination CRS.")
    west, south, east, north = projected.bounds
    west, south = math.floor(west / resolution) * resolution, math.floor(south / resolution) * resolution
    east, north = math.ceil(east / resolution) * resolution, math.ceil(north / resolution) * resolution
    width, height = max(1, round((east - west) / resolution)), max(1, round((north - south) / resolution))
    if width * height > positive_env("SATELLITE_MAX_PIXELS", 250000000):
        raise ValueError("Output exceeds SATELLITE_MAX_PIXELS; use a smaller AOI or coarser resolution.")
    return Grid(crs, from_origin(west, north, resolution, resolution), width, height, projected, resolution)


def windows(grid, size=WINDOW_SIZE):
    for row in range(0, grid.height, size):
        for col in range(0, grid.width, size):
            yield Window(col, row, min(size, grid.width - col), min(size, grid.height - row))


def polygon_mask(grid, window):
    return geometry_mask(
        [grid.geometry],
        out_shape=(int(window.height), int(window.width)),
        transform=window_transform(window, grid.transform),
        invert=True,
    )


def raster_profile(grid, count):
    return dict(
        driver="GTiff",
        crs=grid.crs,
        transform=grid.transform,
        width=grid.width,
        height=grid.height,
        count=count,
        dtype="float32",
        nodata=COMMON_NODATA,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="lzw",
        predictor=3,
        interleave="band",
        BIGTIFF="IF_SAFER",
    )


def calibration(
    src,
    asset,
    band,
    collection="sentinel-2-l2a",
    bidx=1,
    legacy_scale=None,
    require_units=False,
    radiometry_policy="strict",
):
    """Metadata first. Never infer radiometry from observed pixel magnitudes."""
    if radiometry_policy not in ("strict", "metadata", "harmonized"):
        raise ValueError("Unknown radiometry policy.")
    tags = src.tags(bidx)
    if tags.get("units") == "reflectance" or src.tags().get("radiometry") == "reflectance":
        return 1.0, 0.0, "normalized_reflectance"
    extra = getattr(asset, "extra_fields", {}) if asset is not None else {}
    records = extra.get("raster:bands", extra.get("bands", []))
    record = records[min(bidx - 1, len(records) - 1)] if records else {}
    if "scale" in record or "offset" in record:
        scale, offset, origin = record.get("scale", 1), record.get("offset", 0), "stac"
    elif src.scales[bidx - 1] != 1 or src.offsets[bidx - 1] != 0:
        scale, offset, origin = src.scales[bidx - 1], src.offsets[bidx - 1], "geotiff"
    elif legacy_scale is not None:
        scale, offset, origin = legacy_scale, 0, "explicit_legacy_scale"
    elif (
        collection in S2_COLLECTIONS
        and band != "scl"
        and np.issubdtype(np.dtype(src.dtypes[bidx - 1]), np.integer)
    ):
        scale, offset, origin = 1 / S2_SCALE_FACTOR, 0, "sentinel2_integer_default"
    elif require_units and collection in S2_COLLECTIONS and band != "scl":
        raise ValueError(
            "Untagged floating-point composite has unknown units. "
            "Pass legacy_scale=0.0001 for old DN composites, or 1 for reflectance."
        )
    else:
        scale, offset, origin = 1, 0, "identity"
    if not math.isfinite(float(scale)) or not math.isfinite(float(offset)) or float(scale) <= 0:
        raise ValueError(f"Invalid radiometry metadata for {band}.")
    if collection in S2_COLLECTIONS and band != "scl":
        if radiometry_policy == "harmonized":
            offset, origin = 0.0, origin + "_harmonized_override"
        elif (
            radiometry_policy == "strict"
            and float(offset) != 0
            and asset is not None
            and "sentinel-s2-l2a-cogs/" in asset.href
        ):
            raise ValueError(
                "Ambiguous legacy Earth Search COG offset (provider issue #71). "
                "Use sentinel-2-c1-l2a, or explicitly select metadata/harmonized radiometry."
            )
    return float(scale), float(offset), origin


def ensure_georeferenced(src):
    if src.crs is None and not (src.gcps[0] and src.gcps[1]):
        raise ValueError("Raster has neither a CRS nor georeferenced GCPs.")


class SourceCache:
    """Task-local LRU: dataset handles are never shared between worker threads."""

    def __init__(self, grid, cancel_event=None, requester_pays=False, radiometry_policy="strict"):
        self.grid, self.cancel_event = grid, cancel_event
        self.sources = OrderedDict()
        self.provenance = {}
        self.calibrations = {}
        self.requester_pays = requester_pays
        self.radiometry_policy = radiometry_policy
        self.opens = self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for src, vrt in self.sources.values():
            vrt.close()
            src.close()
        self.sources.clear()

    def source(self, asset, categorical=False):
        href = asset.href
        key = (href, categorical)
        if key in self.sources:
            self.sources.move_to_end(key)
            return self.sources[key]
        check_cancel(self.cancel_event)
        local_options = {"GDAL_DISABLE_READDIR_ON_OPEN": "FALSE"} if "://" not in href else {}
        with io_slot(self.cancel_event), rasterio.Env(**local_options):
            src = rasterio.open(href)
            try:
                ensure_georeferenced(src)
                extra = getattr(asset, "extra_fields", {})
                records = extra.get("raster:bands", [])
                nodata = src.nodata
                if nodata is None and records:
                    nodata = records[0].get("nodata")
                vrt = WarpedVRT(
                    src,
                    crs=self.grid.crs,
                    transform=self.grid.transform,
                    width=self.grid.width,
                    height=self.grid.height,
                    src_nodata=nodata,
                    nodata=COMMON_NODATA,
                    dtype="float32",
                    resampling=Resampling.nearest if categorical else Resampling.bilinear,
                    warp_mem_limit=WARP_MEMORY_MB,
                )
            except BaseException:
                src.close()
                raise
        self.opens += 1
        self.sources[key] = (src, vrt)
        if len(self.sources) > SOURCE_CACHE_SIZE:
            old_src, old_vrt = self.sources.popitem(last=False)[1]
            old_vrt.close()
            old_src.close()
        return src, vrt

    def read(
        self, asset, band, window, collection="sentinel-2-l2a", bidx=1, legacy_scale=None, require_units=False
    ):
        src, vrt = self.source(asset, categorical=band in ("scl", "alpha"))
        local_options = {"GDAL_DISABLE_READDIR_ON_OPEN": "FALSE"} if "://" not in asset.href else {}
        with io_slot(self.cancel_event), rasterio.Env(**local_options):
            raw = vrt.read(bidx, window=window, masked=True, out_dtype="float32")
        self.reads += 1
        data = np.asarray(raw.data, dtype=np.float32)
        valid = ~np.ma.getmaskarray(raw) & np.isfinite(data) & (data != COMMON_NODATA)
        if band in ("scl", "alpha"):
            scale, offset, origin = 1, 0, "categorical"
        else:
            key = (asset.href, bidx, band, collection, legacy_scale, require_units)
            if key not in self.calibrations:
                self.calibrations[key] = calibration(
                    src, asset, band, collection, bidx, legacy_scale, require_units, self.radiometry_policy
                )
            scale, offset, origin = self.calibrations[key]
        self.provenance[f"{asset.href}#{bidx}:{band}"] = {
            "href": asset.href,
            "band_index": bidx,
            "band": band,
            "scale": scale,
            "offset": offset,
            "origin": origin,
        }
        # A new float32 array is at most one window, regardless of AOI dimensions.
        data *= np.float32(scale)
        data += np.float32(offset)
        valid &= np.isfinite(data)
        return data, valid


def estimated_task_mb(band_count):
    # Window arrays + masks/formula temporaries + bounded VRT cache and writer buffers.
    arrays = WINDOW_SIZE**2 * (band_count * 12 + 64) / 1024**2
    return math.ceil(arrays + SOURCE_CACHE_SIZE * WARP_MEMORY_MB + 32)


def valid_raster(path, fingerprint=None, cancel_event=None):
    try:
        with rasterio.open(path) as src:
            if src.width <= 0 or src.height <= 0 or src.count <= 0:
                return False
            if fingerprint and src.tags().get("request_fingerprint") != fingerprint:
                return False
            # Decode every block, including masks, with bounded memory: resume must not
            # accept a valid TIFF header whose compressed tail was truncated.
            for _, window in src.block_windows(1):
                check_cancel(cancel_event)
                src.read(window=window, masked=True)
            return True
    except (OSError, ValueError, rasterio.errors.RasterioError):
        return False


@contextmanager
def raster_bundle(paths, grid, descriptions, fingerprint, overviews=True):
    """Publish only after all writers close successfully; manifests commit the bundle."""
    parent = Path(next(iter(paths.values()))).parent
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".raster-stage-", dir=parent) as temp:
        stages = {name: Path(temp) / Path(path).name for name, path in paths.items()}
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with ExitStack() as stack:
                writers = {}
                for name, path in stages.items():
                    desc = descriptions[name]
                    writer = stack.enter_context(rasterio.open(path, "w", **raster_profile(grid, len(desc))))
                    for bidx, band in enumerate(desc, 1):
                        writer.set_band_description(bidx, band)
                        writer.update_tags(
                            bidx,
                            units="reflectance"
                            if name == "composite" and band != "alpha"
                            else "dimensionless",
                        )
                    writer.update_tags(
                        request_fingerprint=fingerprint,
                        radiometry="reflectance" if name == "composite" else "index",
                        pipeline_version="2",
                    )
                    writers[name] = writer
                yield writers
                if overviews:
                    for writer in writers.values():
                        factors = [x for x in (2, 4, 8, 16) if min(writer.width, writer.height) // x >= 64]
                        if factors:
                            writer.build_overviews(factors, Resampling.average)
                            writer.update_tags(ns="rio_overview", resampling="average")
        for name, path in stages.items():
            with rasterio.open(path) as src:
                if src.count != len(descriptions[name]) or src.shape != grid.shape:
                    raise RuntimeError("Output validation failed.")
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        for name, path in stages.items():
            os.replace(path, paths[name])


class SceneIndex:
    """Coarse projected-footprint index; avoid opening distant tiles per window."""

    def __init__(self, items, grid):
        from shapely.geometry import shape, box
        from shapely.strtree import STRtree

        self.items = items
        self.grid = grid
        self.unknown = []
        footprints, self.indices = [], []
        transformer = Transformer.from_crs(4326, grid.crs, always_xy=True)
        for index, item in enumerate(items):
            geometry = getattr(item, "geometry", None)
            try:
                projected = transform_geometry(transformer.transform, shape(geometry)) if geometry else None
                if (
                    projected is None
                    or projected.is_empty
                    or not all(math.isfinite(v) for v in projected.bounds)
                ):
                    self.unknown.append(index)
                else:
                    footprints.append(box(*projected.bounds).buffer(grid.resolution * 2))
                    self.indices.append(index)
            except (ValueError, TypeError):
                self.unknown.append(index)
        self.tree = STRtree(footprints)

    def query(self, window):
        from rasterio.windows import bounds
        from shapely.geometry import box

        matched = self.tree.query(box(*bounds(window, self.grid.transform)))
        indices = sorted(set(self.unknown + [self.indices[int(i)] for i in matched]))
        return [self.items[index] for index in indices]
