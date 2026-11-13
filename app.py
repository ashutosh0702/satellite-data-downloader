import streamlit as st

st.set_page_config(
    page_title="Satellite Data Portal",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Satellite Data Portal")
st.markdown("""
Welcome to the Satellite Data Portal!

This application allows you to search, visualize, and analyze Sentinel satellite data from the Element84 STAC catalog.
It is built utilizing modern cloud-native geospatial techniques (Cloud Optimized GeoTIFFs, STAC).

### Available Tools:

**1. Downloader**
* Search Sentinel-1 and Sentinel-2 data over any region.
* Define your Area of Interest (AOI) via GeoJSON or ZIP Shapefile uploads.
* Download precisely clipped and perfectly merged spatial mosaics of your selected bands for specific dates.

**2. Timeseries Analysis**
* Generate on-the-fly timeseries charts (e.g. NDVI or NDWI) for a single pixel.
* Stream values directly from cloud storage without downloading huge image tiles locally. 
* Optionally filter out cloudy pixels using the Scene Classification Layer (SCL).

👈 **Select a module from the sidebar to get started!**
""")
