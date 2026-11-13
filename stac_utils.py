import logging

import pystac_client
import geopandas as gpd
from shapely.geometry import box

logger = logging.getLogger(__name__)

ELEMENT84_URL = "https://earth-search.aws.element84.com/v1"
MAX_STAC_ITEMS = 100

def get_stac_client():
    return pystac_client.Client.open(ELEMENT84_URL)

def search_stac(bbox, start_date, end_date, collection, max_cloud_cover=100):
    client = get_stac_client()
    datetime_range = f"{start_date}/{end_date}"
    
    query = {}
    if collection == "sentinel-2-l2a":
        query = {"eo:cloud_cover": {"lt": max_cloud_cover}}

    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime_range,
        query=query,
        max_items=MAX_STAC_ITEMS,
    )
    
    items = list(search.items())
    
    if len(items) >= MAX_STAC_ITEMS:
        logger.warning(
            "STAC query returned %d items (the limit). Results may be "
            "truncated — consider narrowing the date range or AOI.",
            len(items),
        )
    
    return items

def items_to_gdf(items):
    """Converts a list of pystac Items to a GeoDataFrame for visualization"""
    if not items:
        return gpd.GeoDataFrame()
    
    features = [item.to_dict() for item in items]
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    
    # Store essential metadata in columns
    gdf['datetime'] = [item.datetime.isoformat() if item.datetime else None for item in items]
    gdf['stac_id'] = [item.id for item in items]
    if 'eo:cloud_cover' in gdf.columns:
        gdf['cloud_cover'] = gdf['eo:cloud_cover']
    else:
        gdf['cloud_cover'] = 0  # e.g., for Sentinel-1 which has no cloud cover
        
    return gdf
