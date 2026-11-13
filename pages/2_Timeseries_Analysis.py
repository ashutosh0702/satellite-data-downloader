from datetime import date, timedelta

import folium
import streamlit as st
from streamlit_folium import st_folium
from shapely.geometry import mapping

from input_utils import coordinate_point
from satellite_config import S2_COLLECTIONS
from ui_components import radiometry_controls
from ui_components import cached_upload, job_manager, recovery_controls, render_job, session_owner

st.set_page_config(page_title="Timeseries Analysis", layout="wide")
st.title("📈 Single-Pixel Timeseries Analysis")
st.write("Generate a daily index series using the first valid observation, ranked by scene cloud cover.")
collection = st.sidebar.selectbox("Satellite Collection", list(S2_COLLECTIONS))
radiometry_policy = radiometry_controls(collection, "timeseries_radiometry")
index = st.sidebar.selectbox("Spectral index", ["NDVI", "NDWI"])
dates = st.sidebar.date_input("Date Range", [date.today() - timedelta(days=90), date.today()])
scl = st.sidebar.checkbox("Filter Cloudy Pixels (SCL Layer)", value=True)
limit = int(
    st.sidebar.number_input("Maximum search scenes", min_value=1, max_value=50000, value=1000, step=100)
)
method = st.sidebar.radio("Selection Method", ["Upload File", "Enter Coordinates"])
point = None
if method == "Upload File":
    upload = st.sidebar.file_uploader("Upload exactly one Point (GeoJSON/ZIP)", type=["geojson", "zip"])
    if upload is not None:
        try:
            point = cached_upload(upload.getvalue(), upload.name, point=True).geometry.iloc[0]
        except Exception as exc:
            st.sidebar.error(str(exc))
else:
    latitude = st.sidebar.number_input("Latitude", min_value=-89.999999, max_value=89.999999, value=46.5)
    longitude = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=6.5)
    point = coordinate_point(longitude, latitude)
left, right = st.columns([2, 1])
with left:
    view = folium.Map(
        location=[point.y, point.x] if point is not None else [20, 0],
        zoom_start=12 if point is not None else 2,
    )
    if point is not None:
        folium.Marker([point.y, point.x]).add_to(view)
    st_folium(view, height=500, use_container_width=True, key="timeseries_map", returned_objects=[])
with right:
    active = False
    if st.session_state.get("job_timeseries"):
        owner, job_id = st.session_state.job_timeseries
        active = job_manager().get(owner, job_id)["status"] in ("queued", "running", "cancelling")
    if st.button(
        "Generate Timeseries",
        type="primary",
        width="stretch",
        disabled=point is None or len(dates) != 2 or active,
    ):
        owner = session_owner()
        try:
            # Exact point intersection removes the old small-bbox false positives.
            request = {
                "kind": "timeseries",
                "geometry": mapping(point),
                "index": index,
                "apply_scl_filter": scl,
                "radiometry_policy": radiometry_policy,
                "search": {
                    "bbox": None,
                    "start_date": str(dates[0]),
                    "end_date": str(dates[1]),
                    "collection": collection,
                    "max_cloud_cover": 100,
                    "max_items": limit,
                    "intersects": mapping(point),
                },
            }
            job_id = job_manager().create(owner, request)
            st.session_state.job_timeseries = (owner, job_id)
        except Exception as exc:
            st.error(str(exc))
    st.caption(
        "Series and diagnostics remain available when controls change. "
        "Start a new job to apply new parameters. Missing SCL pixels are excluded when filtering is enabled."
    )
    recovery_controls("timeseries")
    render_job("timeseries")
