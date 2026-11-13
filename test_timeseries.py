import geopandas as gpd
import pandas as pd
from stac_utils import search_stac
from processing_utils import extract_pixel_timeseries

def test_timeseries():
    print("Testing Timeseries Extraction...")
    gdf = gpd.read_file('test_data/test.geojson')
    point_geom = gdf.geometry.iloc[0]
    lon, lat = point_geom.x, point_geom.y
    bbox = [lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001]
    
    items = search_stac(
        bbox=bbox,
        start_date="2025-03-18",
        end_date="2026-03-18",
        collection="sentinel-2-l2a",
        max_cloud_cover=100
    )
    
    print(f"Found {len(items)} items for timeseries")
    
    if items:
        results, t_sec = extract_pixel_timeseries(items, point_geom, "NDVI", True)
        print(f"Extracted {len(results)} valid observations.")
        if results:
            df = pd.DataFrame(results)
            print(df.head())
        print("Timeseries test completed.")

if __name__ == "__main__":
    test_timeseries()
