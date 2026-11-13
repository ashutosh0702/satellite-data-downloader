"""Validated uploads with temporary-file cleanup and explicit AOI selection."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import geopandas as gpd
from shapely.geometry import Point

from raster_engine import validate_geometry

MAX_UPLOAD_BYTES = 20 * 1024**2
MAX_UNCOMPRESSED_BYTES = 100 * 1024**2


def read_vector_upload(data, filename, point=False):
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload must be at most 20 MiB.")
    suffix = Path(filename).suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(BytesIO(data)) as archive:
            if sum(info.file_size for info in archive.infolist()) > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Uncompressed archive must be at most 100 MiB.")
            if sum(name.lower().endswith(".shp") for name in archive.namelist()) != 1:
                raise ValueError("ZIP must contain exactly one shapefile.")
        # GDAL reads the archive directly; no untrusted paths are extracted.
        with TemporaryDirectory(prefix="satellite-upload-") as temporary:
            path = Path(temporary) / "upload.zip"
            path.write_bytes(data)
            frame = gpd.read_file(f"zip://{path}")
        if frame.crs is None:
            raise ValueError("Shapefile has no CRS; include its .prj file.")
    elif suffix in (".geojson", ".json"):
        frame = gpd.read_file(BytesIO(data))
        if frame.crs is None:
            frame = frame.set_crs(4326)  # GeoJSON coordinates are WGS84.
    else:
        raise ValueError("Upload GeoJSON or one zipped shapefile.")
    if frame.empty or frame.geometry.isna().any():
        raise ValueError("Upload must contain nonempty geometries.")
    frame = frame.to_crs(4326)
    if point:
        if len(frame) != 1:
            raise ValueError("Upload exactly one Point for a single-pixel series.")
        validate_geometry(frame.geometry.iloc[0], point=True)
    else:
        for geometry in frame.geometry:
            validate_geometry(geometry)
        validate_geometry(frame.union_all())
    return frame


def coordinate_point(longitude, latitude):
    return validate_geometry(Point(longitude, latitude), point=True)
