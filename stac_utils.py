"""STAC search with bounded requests, pagination, explicit completeness and TTL caching."""

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
import threading
import time
import warnings

import geopandas as gpd
import pystac
import pystac_client
from pystac_client.stac_api_io import StacApiIO
from urllib3.util.retry import Retry

from resource_budget import check_cancel
from satellite_config import COLLECTION_BANDS, S2_COLLECTIONS

ELEMENT84_URL = "https://earth-search.aws.element84.com/v1"
MAX_STAC_ITEMS = None  # Legacy name; default searches now paginate to exhaustion.
_CACHE = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CLIENTS = threading.local()
CACHE_TTL_SECONDS = 300
CACHE_MAX_ENTRIES = 32
CACHE_MAX_BYTES = 32 * 1024**2


class CatalogSearchError(RuntimeError):
    pass


@dataclass
class SearchResult:
    items: list
    truncated: bool
    limit: int | None
    fetched_at: str
    cached: bool = False

    @property
    def complete(self):
        return not self.truncated

    def summary(self):
        return {
            "returned": len(self.items),
            "complete": self.complete,
            "truncated": self.truncated,
            "limit": self.limit,
            "fetched_at": self.fetched_at,
            "cached": self.cached,
        }


def get_stac_client():
    if not hasattr(_CLIENTS, "client"):
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST"]),
            respect_retry_after_header=True,
            backoff_max=8,
            retry_after_max=15,
        )
        io = StacApiIO(timeout=(10, 45), max_retries=retry)
        _CLIENTS.client = pystac_client.Client.open(ELEMENT84_URL, stac_io=io)
    return _CLIENTS.client


def clear_search_cache():
    with _CACHE_LOCK:
        _CACHE.clear()


def search_fingerprint(
    bbox, start_date, end_date, collection, max_cloud_cover=100, intersects=None, max_items=None
):
    return hashlib.sha256(
        json.dumps(
            {
                "bbox": list(bbox) if bbox is not None else None,
                "start": str(start_date),
                "end": str(end_date),
                "collection": collection,
                "cloud": max_cloud_cover if collection in S2_COLLECTIONS else None,
                "intersects": intersects,
                "max_items": max_items,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def search_stac(
    bbox,
    start_date,
    end_date,
    collection,
    max_cloud_cover=100,
    *,
    max_items=None,
    return_report=False,
    intersects=None,
    use_cache=True,
    cancel_event=None,
):
    if collection not in COLLECTION_BANDS:
        raise ValueError("Unsupported satellite collection.")
    start, end = date.fromisoformat(str(start_date)), date.fromisoformat(str(end_date))
    if start > end:
        raise ValueError("Start date must not be after end date.")
    if bbox is not None:
        if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
            raise ValueError("Provide four finite bounding-box coordinates.")
        west, south, east, north = bbox
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("Bounding box must be ordered WGS84 coordinates.")
    elif intersects is None:
        raise ValueError("A bounding box or intersects geometry is required.")
    if not 0 <= max_cloud_cover <= 100:
        raise ValueError("Cloud cover must be between 0 and 100.")
    if max_items is not None and (
        isinstance(max_items, bool) or not isinstance(max_items, int) or max_items <= 0
    ):
        raise ValueError("max_items must be a positive integer or None.")
    if intersects is not None:
        from shapely.geometry import shape
        from raster_engine import validate_geometry

        geometry = shape(intersects)
        validate_geometry(geometry, point=geometry.geom_type == "Point")
    key = search_fingerprint(bbox, start, end, collection, max_cloud_cover, intersects, max_items)
    check_cancel(cancel_event)
    if use_cache:
        with _CACHE_LOCK:
            entry = _CACHE.get(key)
            if entry is not None and time.monotonic() - entry[0] < CACHE_TTL_SECONDS:
                _CACHE.move_to_end(key)
                result = SearchResult(
                    [pystac.Item.from_dict(d) for d in json.loads(entry[1])],
                    entry[2],
                    max_items,
                    entry[3],
                    cached=True,
                )
                if result.truncated and not return_report:
                    warnings.warn(
                        "Cached catalog results are incomplete; increase max_items.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                return result if return_report else result.items
    query = {}
    if collection in S2_COLLECTIONS and max_cloud_cover < 100:
        query = {"eo:cloud_cover": {"lte": max_cloud_cover}}
    kwargs = {
        "collections": [collection],
        "datetime": f"{start}/{end}",
        "query": query,
        "max_items": None if max_items is None else max_items + 1,
        "limit": 100,
    }
    kwargs["intersects" if intersects is not None else "bbox"] = (
        intersects if intersects is not None else bbox
    )
    try:
        search = get_stac_client().search(**kwargs)
        items, seen = [], set()
        for item in search.items():
            check_cancel(cancel_event)
            identity = (getattr(item, "collection_id", None), item.id)
            if identity not in seen:
                seen.add(identity)
                items.append(item)
        truncated = max_items is not None and len(items) > max_items
        if truncated:
            items = items[:max_items]
    except Exception as exc:
        from resource_budget import CancelledError

        if isinstance(exc, CancelledError):
            raise
        raise CatalogSearchError(
            f"Catalog search failed ({type(exc).__name__}). Retry or narrow the search."
        ) from exc
    fetched = datetime.now(timezone.utc).isoformat()
    result = SearchResult(items, truncated, max_items, fetched)
    if use_cache:
        payload = json.dumps([i.to_dict() for i in items], separators=(",", ":"))
        with _CACHE_LOCK:
            if len(payload.encode("utf-8")) <= CACHE_MAX_BYTES:
                _CACHE[key] = (time.monotonic(), payload, truncated, fetched)
                _CACHE.move_to_end(key)
            while (
                len(_CACHE) > CACHE_MAX_ENTRIES
                or sum(len(row[1].encode("utf-8")) for row in _CACHE.values()) > CACHE_MAX_BYTES
            ):
                _CACHE.popitem(last=False)
    if truncated and not return_report:
        warnings.warn(
            "Catalog results are incomplete; increase max_items or narrow the query.",
            RuntimeWarning,
            stacklevel=2,
        )
    return result if return_report else result.items


def items_to_gdf(items):
    if not items:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame.from_features([item.to_dict() for item in items], crs="EPSG:4326")
    gdf["datetime"] = [item.datetime.isoformat() if item.datetime else None for item in items]
    gdf["stac_id"] = [item.id for item in items]
    if "eo:cloud_cover" in gdf.columns:
        gdf["cloud_cover"] = gdf["eo:cloud_cover"]
    return gdf
