from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pystac
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
from shapely.geometry import box, mapping, Point


class RasterCase(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.counter = 0
        self.inverse = Transformer.from_crs(3857, 4326, always_xy=True)

    def aoi(self, bounds=(0, 0, 100, 100)):
        return box(*self.inverse.transform(*bounds[:2]), *self.inverse.transform(*bounds[2:]))

    def point(self, x=55, y=55):
        return Point(*self.inverse.transform(x, y))

    def raster(
        self,
        values,
        *,
        dtype="uint16",
        nodata=0,
        crs=3857,
        transform=None,
        descriptions=None,
        scales=None,
        offsets=None,
        units=None,
        mask=None,
        gcps=None,
    ):
        self.counter += 1
        path = self.root / f"source-{self.counter}.tif"
        values = np.asarray(values, dtype=dtype)
        if values.ndim == 0:
            values = np.full((10, 10), values, dtype=dtype)
        if values.ndim == 2:
            values = values[None]
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=values.shape[2],
            height=values.shape[1],
            count=values.shape[0],
            dtype=dtype,
            crs=crs,
            transform=transform or from_origin(0, 100, 10, 10),
            nodata=nodata,
        ) as dst:
            dst.write(values)
            if descriptions:
                for bidx, desc in enumerate(descriptions, 1):
                    dst.set_band_description(bidx, desc)
                    if units:
                        dst.update_tags(bidx, units=units)
            if scales is not None:
                dst.scales = tuple(scales)
            if offsets is not None:
                dst.offsets = tuple(offsets)
            if mask is not None:
                dst.write_mask(np.asarray(mask, dtype="uint8") * 255)
            if gcps:
                dst.gcps = gcps
        return path

    def scene(self, bands, *, scene_id="scene", day=1, cloud=0, scales=None, offsets=None, geometry=None):
        item = pystac.Item(
            scene_id,
            geometry or mapping(self.aoi()),
            list(self.aoi().bounds),
            datetime(2025, 4, day, tzinfo=timezone.utc),
            {"eo:cloud_cover": cloud},
            collection="sentinel-2-l2a",
        )
        for band, value in bands.items():
            path = value if isinstance(value, (str, Path)) else self.raster(value)
            asset = pystac.Asset(str(path), media_type=pystac.MediaType.COG)
            if scales is not None or offsets is not None:
                asset.extra_fields["raster:bands"] = [
                    {"scale": scales if scales is not None else 1, "offset": offsets or 0}
                ]
            item.add_asset(band, asset)
        return item

    def process(self, items, **kwargs):
        from processing_utils import process_items

        options = dict(
            bands=["nir", "red"],
            out_dir=self.root / "output",
            target_crs=3857,
            return_report=True,
            overviews=False,
        )
        options.update(kwargs)
        return process_items(items, self.aoi(), **options)
