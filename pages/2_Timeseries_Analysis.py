import streamlit as st
import leafmap.foliumap as leafmap
import geopandas as gpd
from shapely.geometry import shape, Point
from datetime import date, timedelta
import os
from tempfile import NamedTemporaryFile
import pandas as pd
import plotly.express as px

from stac_utils import search_stac, items_to_gdf
from processing_utils import extract_pixel_timeseries

st.set_page_config(page_title="Timeseries Analysis", layout="wide")

st.title("📈 Single-Pixel Timeseries Analysis")
st.markdown("Extract an on-the-fly index timeseries directly from the cloud for a specific point.")

st.sidebar.header("Analysis Parameters")

index_to_calculate = st.sidebar.selectbox("Vegetation Index", ["NDVI", "NDWI"])

date_range = st.sidebar.date_input(
    "Date Range",
    [date.today() - timedelta(days=90), date.today()]
)

apply_scl_filter = st.sidebar.checkbox("Filter Cloudy Pixels (SCL Layer)", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Point Selection")

point_method = st.sidebar.radio("Selection Method", ["Upload File", "Enter Coordinates"])

point_gdf = None

if point_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload Point (GeoJSON/ZIP)", type=["geojson", "zip"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".zip"):
                with NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp.flush()
                    gdf = gpd.read_file(tmp.name)
            else:
                gdf = gpd.read_file(uploaded_file)
                
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            elif gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
                
            if not all(gdf.geometry.type == 'Point'):
                st.sidebar.error("Uploaded file must contain Point geometries only.")
            else:
                point_gdf = gdf.head(1) # Take just the first point
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            
else:
    lat_input = st.sidebar.number_input("Latitude", value=46.5) # Example default
    lon_input = st.sidebar.number_input("Longitude", value=6.5)
    point_gdf = gpd.GeoDataFrame(geometry=[Point(lon_input, lat_input)], crs="EPSG:4326")

col1, col2 = st.columns([1, 1])

with col1:
    m = leafmap.Map(center=[0, 0], zoom=2)
    m.add_basemap("SATELLITE")
    if point_gdf is not None and not point_gdf.empty:
        m.add_gdf(point_gdf, layer_name="Selected Point", marker_icon="red")
        bounds = point_gdf.total_bounds
        m.set_center(bounds[0], bounds[1], zoom=12)
    m.to_streamlit(height=500)
    
with col2:
    if st.button("Generate Timeseries", type="primary", width="stretch"):
        if point_gdf is None or point_gdf.empty:
            st.error("Please provide a valid point first.")
        elif len(date_range) != 2:
            st.error("Please select a valid date range.")
        else:
            with st.spinner("Finding scenes and querying pixel values (this can take ~30 seconds for a full year)..."):
                # Use a small bbox around the point to query STAC
                lon, lat = point_gdf.geometry.iloc[0].x, point_gdf.geometry.iloc[0].y
                bbox = [lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001]
                
                # We do not use max_cloud_cover in the initial query because SCL filtering handles it per-pixel
                # But we can pass 100 to make sure we get everything
                items = search_stac(
                    bbox=bbox,
                    start_date=date_range[0].strftime("%Y-%m-%d"),
                    end_date=date_range[1].strftime("%Y-%m-%d"),
                    collection="sentinel-2-l2a",
                    max_cloud_cover=100
                )
                
                if not items:
                    st.warning("No scenes found for this date range and location.")
                else:
                    st.info(f"Found {len(items)} scenes intersecting the point. Extracting network pixel streams...")
                    progress_bar = st.progress(0, text="Initializing pixel extraction...")
                    
                    def ui_progress_callback(current, total):
                        pct = current / total
                        progress_bar.progress(pct, text=f"Extracted {current} of {total} points")
                        
                    results, t_sec = extract_pixel_timeseries(
                        items=items,
                        point_geom=point_gdf.geometry.iloc[0],
                        index=index_to_calculate,
                        apply_scl_filter=apply_scl_filter,
                        progress_callback=ui_progress_callback
                    )
                    
                    progress_bar.empty()
                    
                    if not results:
                        st.error("Could not extract any valid pixel data (perhaps all were cloudy?).")
                    else:
                        st.success(f"Extracted {len(results)} valid observations in {t_sec}s.")
                        df = pd.DataFrame(results)
                        df['Date'] = pd.to_datetime(df['Date'])
                        
                        fig = px.line(
                            df, x='Date', y='Index', markers=True, 
                            title=f"{index_to_calculate} Timeseries",
                            labels={'Index': index_to_calculate, 'Date': 'Observation Date'}
                        )
                        fig.update_traces(line_color='#2ca02c') # nice green color
                        fig.update_layout(hovermode="x unified")
                        
                        st.plotly_chart(fig, width="stretch")
                        
                        st.write("### Data Table")
                        st.dataframe(df.sort_values(by='Date', ascending=False), width="stretch")
