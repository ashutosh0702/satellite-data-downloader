import streamlit as st
import leafmap.foliumap as leafmap
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from datetime import date, timedelta
import datetime
import os
from tempfile import NamedTemporaryFile

from stac_utils import search_stac, items_to_gdf
from processing_utils import process_items, calculate_indices

st.set_page_config(page_title="Sentinel Data Downloader", layout="wide")

st.title("🛰️ Sentinel Data Downloader")
st.markdown("Search, visualize, process, and save Sentinel STAC data. Outputs are merged and clipped to your area of interest using the Element84 API.")


def list_output_rasters(output_dir):
    if not output_dir or not os.path.isdir(output_dir):
        return []

    return sorted(
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.lower().endswith(".tif")
    )

# Sidebar Controls
st.sidebar.header("Search Parameters")

collection = st.sidebar.selectbox(
    "Satellite Collection",
    ["sentinel-2-l2a", "sentinel-1-grd"]
)

date_range = st.sidebar.date_input(
    "Date Range",
    [date.today() - timedelta(days=30), date.today()]
)

max_cloud_cover = 100
if collection == "sentinel-2-l2a":
    max_cloud_cover = st.sidebar.slider("Max Cloud Cover (%)", 0, 100, 20)

bands_to_download = st.sidebar.multiselect(
    "Bands to Download",
    ["blue", "green", "red","rededge1","rededge2","rededge3", "nir", "nir08","nir09", "swir16", "swir22", "vh", "vv"],
    default=["blue", "green", "red"] if collection == "sentinel-2-l2a" else ["vh", "vv"]
)

indices_to_calculate = []
if collection == "sentinel-2-l2a":
    indices_to_calculate = st.sidebar.multiselect(
        "Calculate Indices",
        ["NDVI", "NDWI", "NDMI", "NDSI", "EVI", "SAVI", "NDRE"],
        default=[]
    )

st.sidebar.markdown("---")
st.sidebar.header("Area of Interest")
uploaded_file = st.sidebar.file_uploader("Upload GeoJSON or Zipped Shapefile", type=["geojson", "zip"])

# Main layout
col1, col2 = st.columns([2, 1])

# Initialize session state for AOI
if "aoi_gdf" not in st.session_state:
    st.session_state.aoi_gdf = None
if "stac_items" not in st.session_state:
    st.session_state.stac_items = []
if "stac_gdf" not in st.session_state:
    st.session_state.stac_gdf = gpd.GeoDataFrame()
if "download_dir" not in st.session_state:
    st.session_state.download_dir = None
if "indices_to_calculate" not in st.session_state:
    st.session_state.indices_to_calculate = []
if "processed_collection" not in st.session_state:
    st.session_state.processed_collection = None
if "generated_file_count" not in st.session_state:
    st.session_state.generated_file_count = 0
if "last_processing_time" not in st.session_state:
    st.session_state.last_processing_time = None
if "processing_report" not in st.session_state:
    st.session_state.processing_report = None

# Map component
with col1:
    m = leafmap.Map(center=[0, 0], zoom=2, draw_export=True)
    m.add_basemap("SATELLITE")
    
    # Handle uploaded file
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
                
            st.session_state.aoi_gdf = gdf
                
            # Add to map and center
            m.add_gdf(st.session_state.aoi_gdf, layer_name="Uploaded AOI", fill_colors=["blue"])
            bounds = st.session_state.aoi_gdf.total_bounds
            m.set_center((bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2, zoom=10)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            
    # Visualize STAC footprints if they exist
    if not st.session_state.stac_gdf.empty:
        m.add_gdf(
            st.session_state.stac_gdf,
            layer_name="STAC Footprints",
            style={"color": "red", "fillOpacity": 0.1, "weight": 2}
        )
        
        # Adjust center to footprints if no AOI is explicitly centering things
        bounds = st.session_state.stac_gdf.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        
    st_data = m.to_streamlit(height=600, bidirectional=True)
    
    # Capture drawn features
    if st_data and st_data.get("last_active_drawing"):
        try:
            geom = shape(st_data["last_active_drawing"]["geometry"])
            st.session_state.drawn_aoi = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        except Exception:
            st.session_state.drawn_aoi = None
    else:
        st.session_state.drawn_aoi = None

with col2:
    st.subheader("Data Search")
    
    if st.button("Search catalog", width="stretch"):
        # Determine AOI bounding box
        bbox = None
        active_aoi = st.session_state.aoi_gdf
        
        # If no uploaded AOI, try to use drawn AOI
        if (active_aoi is None or active_aoi.empty) and st.session_state.get("drawn_aoi") is not None:
            active_aoi = st.session_state.drawn_aoi
            st.session_state.aoi_gdf = active_aoi # Set it globally so Process uses it
            
        if active_aoi is not None and not active_aoi.empty:
            bbox = list(active_aoi.total_bounds)
        else:
            st.warning("Please upload a file or draw a polygon on the map to establish an AOI bounding box for search.")
        
        if bbox is not None and len(date_range) == 2:
            with st.spinner("Querying Element84 STAC API..."):
                items = search_stac(
                    bbox=bbox,
                    start_date=date_range[0].strftime("%Y-%m-%d"),
                    end_date=date_range[1].strftime("%Y-%m-%d"),
                    collection=collection,
                    max_cloud_cover=max_cloud_cover
                )
                st.session_state.stac_items = items
                st.session_state.stac_gdf = items_to_gdf(items)
                
            if items:
                st.success(f"Found {len(items)} items matching criteria!")
                st.rerun()
            else:
                st.info("No items found for the specificed parameters.")
                
    # Display search results summary
    if not st.session_state.stac_gdf.empty:
        st.write("### Found Data Summary")
        
        mc1, mc2 = st.columns(2)
        mc1.metric("Total Scenes", len(st.session_state.stac_items))
        
        avg_cloud = 0
        if 'cloud_cover' in st.session_state.stac_gdf.columns:
            avg_cloud = st.session_state.stac_gdf['cloud_cover'].mean()
            mc2.metric("Avg Cloud Cover", f"{avg_cloud:.1f}%")
        
        display_cols = ['stac_id', 'datetime']
        if 'cloud_cover' in st.session_state.stac_gdf.columns:
            display_cols.append('cloud_cover')
            
        st.dataframe(st.session_state.stac_gdf[display_cols].head(10), width="stretch")
        #if len(st.session_state.stac_items) > 10:
        #    st.caption(f"...and {len(st.session_state.stac_items) - 10} more.")
            
        st.markdown("---")
        st.write("### Process & Save")
        st.info("This will stream the selected STAC assets, merge adjacent tiles for your AOI, clip the result, and save the outputs directly under `output_data/`.")
        
        if st.button("Process & Save Outputs", type="primary", width="stretch"):
            if not bands_to_download:
                st.error("Select at least one band to process.")
            elif st.session_state.aoi_gdf is None or st.session_state.aoi_gdf.empty:
                st.error("No valid Area of Interest (AOI) shape found to clip against.")
            else:
                try:
                    st.session_state.download_dir = None
                    st.session_state.indices_to_calculate = []
                    st.session_state.processed_collection = None
                    st.session_state.generated_file_count = 0
                    st.session_state.last_processing_time = None
                    st.session_state.processing_report = None

                    # Get single unified geometry for clipping
                    aoi_geom = st.session_state.aoi_gdf.union_all()
                    
                    st.info("Downloading images and clipping to AOI... ")
                    progress_bar = st.progress(0, text="Initializing processing...")
                    
                    def ui_progress_callback(current, total):
                        pct = current / total
                        progress_bar.progress(pct, text=f"Processed {current} of {total} items")
                        
                    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    download_dir = os.path.join("output_data", f"download_{timestamp_str}")
                    os.makedirs(download_dir, exist_ok=True)
                    
                    generated_files, proc_time, processing_report = process_items(
                        st.session_state.stac_items,
                        aoi_geom=aoi_geom,
                        collection=collection,
                        bands=bands_to_download,
                        out_dir=download_dir,
                        progress_callback=ui_progress_callback,
                        return_report=True,
                    )
                    
                    progress_bar.empty()

                    st.session_state.download_dir = download_dir
                    st.session_state.indices_to_calculate = list(indices_to_calculate)
                    st.session_state.processed_collection = collection
                    st.session_state.generated_file_count = len(generated_files)
                    st.session_state.last_processing_time = proc_time
                    st.session_state.processing_report = processing_report

                    if generated_files:
                        st.success(f"Successfully processed and saved {len(generated_files)} raster files in {proc_time} seconds!")
                        st.rerun()
                    else:
                        st.warning("Processing finished but no files were generated. Check the skipped-date summary below for missing bands or overlap issues.")
                except Exception as e:
                    st.error(f"Processing failed: {e}")
       
    if st.session_state.get("download_dir"):
        download_dir = st.session_state.download_dir
        saved_indices = st.session_state.get("indices_to_calculate", [])
        available_rasters = list_output_rasters(download_dir)
        processing_report = st.session_state.get("processing_report")

        st.markdown("---")
        st.write("### Output Folder")

        if not os.path.isdir(download_dir):
            st.warning(f"Output folder is not available: `{download_dir}`")
        else:
            st.success(f"Outputs are saved directly to `{download_dir}`")
            st.write(f"Composite rasters from the last run: `{st.session_state.generated_file_count}`")
            st.write(f"Current GeoTIFF files in folder: `{len(available_rasters)}`")

            if st.session_state.last_processing_time is not None:
                st.write(f"Last processing time: `{st.session_state.last_processing_time}` seconds")

            if processing_report:
                st.write("### Processing Summary")
                metric_cols = st.columns(4)
                metric_cols[0].metric("Dates Found", processing_report["total_dates"])
                metric_cols[1].metric("Dates Saved", processing_report["saved_dates"])
                metric_cols[2].metric("Dates Skipped", processing_report["skipped_dates"])
                metric_cols[3].metric("Dates Missing Bands", processing_report["dates_with_missing_bands"])

                missing_band_counts = processing_report.get("missing_band_counts", {})
                if missing_band_counts:
                    st.write("#### Missing Band Frequency")
                    missing_band_df = pd.DataFrame(
                        [
                            {"band": band_name, "dates_missing": count}
                            for band_name, count in missing_band_counts.items()
                        ]
                    ).set_index("band")
                    st.bar_chart(missing_band_df)

                skipped_rows = [
                    {
                        "Date": row["date"],
                        "Scenes": row["item_count"],
                        "Reason": row["reason_label"],
                        "Missing Bands": ", ".join(row["missing_bands"]) if row["missing_bands"] else "-",
                    }
                    for row in processing_report["date_details"]
                    if not row["saved"]
                ]
                if skipped_rows:
                    with st.expander("Skipped Date Details"):
                        st.dataframe(pd.DataFrame(skipped_rows), width="stretch", hide_index=True)

            if (
                st.session_state.get("processed_collection") == "sentinel-2-l2a"
                and saved_indices
            ):
                st.write("### Post-Processing")
                st.caption("Selected index rasters will be written to the same output folder.")

                if st.button("Calculate Selected Indices", type="secondary", width="stretch"):
                    if not available_rasters:
                        st.warning("No GeoTIFF files were found in the output folder.")
                    else:
                        st.info("Calculating selected indices on saved composites...")
                        idx_progress = st.progress(0, text="Computing indices...")
                        try:
                            index_files = calculate_indices(
                                available_rasters,
                                indices=saved_indices,
                                out_dir=download_dir
                            )
                            if index_files:
                                st.session_state.indices_to_calculate = []
                                st.success(f"Saved {len(index_files)} index rasters to `{download_dir}`")
                                st.rerun()
                            else:
                                st.warning("No index rasters were generated from the saved composites.")
                        except Exception as e:
                            st.warning(f"Index calculation had issues: {e}")
                        idx_progress.empty()
