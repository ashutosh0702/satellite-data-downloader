import numpy as np
import rasterio
from rasterio.transform import from_origin
from processing_utils import calculate_indices

# Create dummy 4-band TIFF (red, green, blue, nir, swir16, rededge1)
# Bands needed: 1=red, 2=green, 3=blue, 4=nir, 5=swir16, 6=rededge1
data = np.random.rand(6, 10, 10).astype(np.float32)

transform = from_origin(0, 0, 10, 10)
meta = {
    'driver': 'GTiff',
    'height': 10,
    'width': 10,
    'count': 6,
    'dtype': 'float32',
    'transform': transform,
    'crs': 'EPSG:4326'
}

with rasterio.open("test_output/dummy.tif", "w", **meta) as dest:
    dest.write(data)
    dest.set_band_description(1, "red")
    dest.set_band_description(2, "green")
    dest.set_band_description(3, "blue")
    dest.set_band_description(4, "nir")
    dest.set_band_description(5, "swir16")
    dest.set_band_description(6, "rededge1")

# Test ALL indices
indices_to_test = ["NDVI", "NDWI", "NDMI", "NDSI", "EVI", "SAVI", "NDRE"]
results = calculate_indices(["test_output/dummy.tif"], indices_to_test, "test_output")

print(f"Generated indices: {results}")

for f in results:
    with rasterio.open(f) as src:
        print(f"{f}: bands={src.count}, desc={src.descriptions[0]}, shape={src.shape}")
