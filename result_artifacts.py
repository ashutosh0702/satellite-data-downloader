"""Read only committed raster artifacts and create small, bounded previews."""

import csv
from io import StringIO, BytesIO
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image


def committed_files(directory):
    """Do not expose staging files, unrelated TIFFs or a bundle without its marker."""
    directory = Path(directory).resolve()
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        return {}
    try:
        report = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(report, dict) or not isinstance(report.get("fingerprint"), str):
        return {}
    found = {}
    for row in report.get("date_details", []):
        if not isinstance(row, dict) or not row.get("saved"):
            continue
        date = row.get("date", "")
        if not isinstance(date, str) or len(date) != 10 or any(c not in "0123456789-" for c in date):
            continue
        marker_path = directory / ".dates" / f"{date}.json"
        try:
            marker = json.loads(marker_path.read_text())
        except (OSError, ValueError):
            continue
        if (
            not isinstance(marker, dict)
            or not marker.get("saved")
            or marker.get("fingerprint") != report.get("fingerprint")
        ):
            continue
        products = {
            item["filename"]: item
            for item in marker.get("products", [])
            if isinstance(item, dict) and isinstance(item.get("filename"), str)
        }
        for name in marker.get("files", []) + marker.get("index_files", []):
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".tif"):
                continue
            path = directory / name
            if path.is_symlink() or path.resolve().parent != directory or not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            entry = dict(products.get(name, {}))
            entry.update(filename=name, date=date, size_bytes=size)
            if not entry.get("name"):
                entry.update(
                    name="composite" if name in marker.get("files", []) else Path(name).stem.split("_")[-1],
                    kind="Composite" if name in marker.get("files", []) else "Index",
                )
            entry.setdefault("coverage_percent", marker.get("coverage_percent"))
            found[name] = entry
    return found


def summary_csv(files):
    output = StringIO()
    fields = [
        "date",
        "product",
        "filename",
        "size_bytes",
        "coverage_percent",
        "valid_pixels",
        "aoi_pixels",
        "band",
        "mean",
        "median",
        "p10",
        "p90",
        "minimum",
        "maximum",
        "sample_count",
        "quantiles_approximate",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for name, item in sorted(files.items()):
        for band, statistics in (item.get("statistics") or {"": {}}).items():
            writer.writerow(
                {
                    "date": item["date"],
                    "product": item["name"],
                    "filename": name,
                    **{
                        key: item.get(key)
                        for key in ("size_bytes", "coverage_percent", "valid_pixels", "aoi_pixels")
                    },
                    "band": band,
                    **{key: statistics.get(key) for key in fields[8:]},
                }
            )
    return output.getvalue()


def preview_png(path, band=None, rgb=False, max_size=640):
    """Read a thumbnail using raster overviews; never read the full source grid."""
    max_size = min(1024, max(16, int(max_size)))
    with rasterio.open(path) as src:
        ratio = min(1, max_size / max(src.width, src.height))
        height, width = max(1, round(src.height * ratio)), max(1, round(src.width * ratio))
        names = src.descriptions
        if rgb:
            if not all(name in names for name in ("red", "green", "blue")):
                raise ValueError("This composite does not contain red, green and blue bands.")
            indices = [names.index(name) + 1 for name in ("red", "green", "blue")]
        else:
            indices = [names.index(band) + 1 if band in names else 1]
        data = src.read(
            indices, out_shape=(len(indices), height, width), masked=True, resampling=Resampling.nearest
        )
        valid = ~np.ma.getmaskarray(data).any(axis=0) & np.isfinite(data.data).all(axis=0)
        # Older products can use the explicit compatibility alpha band.
        if "alpha" in names:
            alpha = src.read(
                names.index("alpha") + 1, out_shape=(height, width), resampling=Resampling.nearest
            )
            valid &= alpha > 0
        values = data.filled(0).astype(np.float32)
        is_index = src.tags().get("radiometry") == "index"
        if rgb:
            # Fixed reflectance stretch makes different dates visually comparable.
            color = np.moveaxis(np.clip(values / 0.3, 0, 1), 0, -1)
            legend = "RGB reflectance stretch: 0 to 0.30 (fixed across dates)."
        elif is_index:
            t = np.clip((values[0] + 1) / 2, 0, 1)
            low, middle, high = (
                np.array([0.15, 0.28, 0.65]),
                np.array([0.94, 0.94, 0.84]),
                np.array([0.10, 0.50, 0.24]),
            )
            color = np.where(
                (t <= 0.5)[..., None],
                low + (middle - low) * (t * 2)[..., None],
                middle + (high - middle) * ((t - 0.5) * 2)[..., None],
            )
            # Flag out-of-scale valid values; do not silently alter the data.
            color[valid & ((values[0] < -1) | (values[0] > 1))] = [0.85, 0.1, 0.75]
            legend = "Index colour scale: blue −1 · cream 0 · green +1; magenta outside this scale."
        else:
            finite = values[0][valid]
            lower, upper = (float(finite.min()), float(finite.max())) if finite.size else (0, 1)
            normalized = (
                np.clip((values[0] - lower) / (upper - lower), 0, 1)
                if upper > lower
                else np.full((height, width), 0.5)
            )
            color = np.repeat(normalized[..., None], 3, axis=2)
            legend = f"Band preview stretch: {lower:.4g} to {upper:.4g}."
        rgba = np.dstack((np.round(np.clip(color, 0, 1) * 255).astype("uint8"), valid.astype("uint8") * 255))
    output = BytesIO()
    Image.fromarray(rgba).save(output, format="PNG")
    return output.getvalue(), legend
