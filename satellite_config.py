"""Collection capabilities and shared, floating-point index definitions."""

import numpy as np

S2_BANDS = {
    "blue": 10,
    "green": 10,
    "red": 10,
    "rededge1": 20,
    "rededge2": 20,
    "rededge3": 20,
    "nir": 10,
    "nir08": 20,
    "nir09": 60,
    "swir16": 20,
    "swir22": 20,
}
S2_COLLECTIONS = ("sentinel-2-c1-l2a", "sentinel-2-l2a")
COLLECTION_BANDS = {name: S2_BANDS for name in S2_COLLECTIONS}
COLLECTION_BANDS["sentinel-1-grd"] = {"vv": 10, "vh": 10, "hh": 10, "hv": 10}
DEFAULT_BANDS = {name: ["blue", "green", "red"] for name in S2_COLLECTIONS}
DEFAULT_BANDS["sentinel-1-grd"] = ["vv", "vh"]
COMMON_NODATA = -99999.0
S2_SCALE_FACTOR = 10000.0
SCL_BAD = frozenset([0, 1, 3, 8, 9, 10])


def ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float32), np.asarray(denominator, dtype=np.float32)
    )
    out = np.full(numerator.shape, np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(numerator, denominator, out=out, where=np.isfinite(denominator) & (denominator != 0))
    return out


INDEX_FUNCS = {
    "NDVI": {"bands": ["nir", "red"], "func": lambda n, r: ratio(n - r, n + r)},
    "NDWI": {"bands": ["green", "nir"], "func": lambda g, n: ratio(g - n, g + n)},
    "NDMI": {"bands": ["nir", "swir16"], "func": lambda n, s: ratio(n - s, n + s)},
    "NDSI": {"bands": ["green", "swir16"], "func": lambda g, s: ratio(g - s, g + s)},
    "EVI": {
        "bands": ["nir", "red", "blue"],
        "func": lambda n, r, b: ratio(2.5 * (n - r), n + 6 * r - 7.5 * b + 1),
    },
    "SAVI": {"bands": ["nir", "red"], "func": lambda n, r: ratio(1.5 * (n - r), n + r + 0.5)},
    "NDRE": {"bands": ["nir", "rededge1"], "func": lambda n, r: ratio(n - r, n + r)},
}


def select_bands(collection, bands=None, indices=None):
    if collection not in COLLECTION_BANDS:
        raise ValueError(f"Unsupported collection: {collection}")
    selected = list(DEFAULT_BANDS[collection] if bands is None else dict.fromkeys(bands))
    if not selected:
        raise ValueError("Select at least one band.")
    for name in indices or []:
        if collection not in S2_COLLECTIONS or name not in INDEX_FUNCS:
            raise ValueError(f"Unsupported index {name} for {collection}")
        selected.extend(b for b in INDEX_FUNCS[name]["bands"] if b not in selected)
    invalid = set(selected) - COLLECTION_BANDS[collection].keys()
    if invalid:
        raise ValueError(f"Bands unavailable for {collection}: {', '.join(sorted(invalid))}")
    return selected
