"""Bounded, reproducible valid-pixel summaries collected during raster processing."""

import math

import numpy as np

SAMPLE_LIMIT = 16384
SUMMARY_VERSION = 1


class PixelStatistics:
    """Exact count/range/mean; uniformly sampled quantiles with bounded storage."""

    def __init__(self, seed=0, limit=SAMPLE_LIMIT):
        self.limit = limit
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.sample = np.empty(0, dtype=np.float32)

    def update(self, array, valid):
        values = np.asarray(array, dtype=np.float32)[valid & np.isfinite(array)]
        count = values.size
        if not count:
            return
        self.total += float(values.sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        size = min(self.limit, self.count + count)
        # Hypergeometric allocation merges two uniform reservoirs without
        # generating a random priority for every pixel in the raster window.
        take_new = int(self.rng.hypergeometric(count, self.count, size)) if self.count else size
        take_old = size - take_new
        old = (
            self.sample[self.rng.choice(len(self.sample), take_old, replace=False)]
            if take_old
            else self.sample[:0]
        )
        new = values[self.rng.choice(count, take_new, replace=False)] if take_new else values[:0]
        self.sample = np.concatenate((old, new))
        self.count += count

    def summary(self):
        if not self.count:
            return dict(
                count=0,
                mean=None,
                minimum=None,
                maximum=None,
                p10=None,
                median=None,
                p90=None,
                sample_count=0,
                quantiles_approximate=False,
            )
        p10, median, p90 = np.quantile(self.sample, [0.1, 0.5, 0.9])
        return dict(
            count=self.count,
            mean=self.total / self.count,
            minimum=self.minimum,
            maximum=self.maximum,
            p10=float(p10),
            median=float(median),
            p90=float(p90),
            sample_count=len(self.sample),
            quantiles_approximate=self.count > len(self.sample),
        )


def product_record(path, name, bands, statistics, aoi_pixels, valid_pixels, grid=None, units=None):
    return {
        "name": name,
        "filename": path.name,
        "kind": "Composite" if name == "composite" else "Index",
        "bands": list(bands),
        "size_bytes": path.stat().st_size,
        "aoi_pixels": aoi_pixels,
        "valid_pixels": valid_pixels,
        "coverage_percent": 100 * valid_pixels / aoi_pixels if aoi_pixels else None,
        "statistics": {band: stat.summary() for band, stat in statistics.items()},
        "resolution_m": grid.resolution if grid is not None else None,
        "crs": grid.crs.to_string() if grid is not None else None,
        "units": units,
    }


def summarize_saved(paths, grid, cancel_event=None):
    """Upgrade older committed results without changing their raster files."""
    import rasterio
    from raster_engine import windows, polygon_mask
    from resource_budget import check_cancel

    products = []
    for name, path in paths.items():
        with rasterio.open(path) as source:
            indices = [i for i, desc in enumerate(source.descriptions, 1) if desc != "alpha"]
            bands = [source.descriptions[i - 1] or f"band_{i}" for i in indices]
            stats = {band: PixelStatistics(seed=i) for i, band in enumerate(bands)}
            aoi_pixels = valid_pixels = 0
            for window in windows(grid):
                check_cancel(cancel_event)
                inside = polygon_mask(grid, window)
                data = source.read(indices, window=window, masked=True, out_dtype="float32")
                valid = inside & ~np.ma.getmaskarray(data).any(axis=0) & np.isfinite(data.data).all(axis=0)
                if "alpha" in source.descriptions:
                    valid &= source.read(source.descriptions.index("alpha") + 1, window=window) > 0
                for index, band in enumerate(bands):
                    stats[band].update(data.data[index], valid)
                aoi_pixels += int(inside.sum())
                valid_pixels += int(valid.sum())
            products.append(
                product_record(
                    path,
                    name,
                    bands,
                    stats,
                    aoi_pixels,
                    valid_pixels,
                    grid=grid,
                    units="dimensionless" if name != "composite" else source.tags().get("radiometry"),
                )
            )
    return products
