from datetime import date, timedelta

import folium
from folium.plugins import Draw
import streamlit as st
from streamlit_folium import st_folium
from shapely.geometry import shape, mapping

from processing_utils import preview_processing
from raster_engine import validate_geometry
from satellite_config import COLLECTION_BANDS, DEFAULT_BANDS, INDEX_FUNCS, select_bands, S2_COLLECTIONS
from stac_utils import search_stac, search_fingerprint, items_to_gdf
from ui_components import radiometry_controls
from ui_components import cached_upload, job_manager, recovery_controls, render_job, session_owner

st.set_page_config(page_title="Sentinel Data Downloader", layout="wide")
st.title("🛰️ Sentinel Data Downloader")
st.write(
    "Search satellite scenes, clip mosaics to a polygon, and save bands and indices in a background job."
)

collection = st.sidebar.selectbox("Satellite Collection", list(COLLECTION_BANDS))
radiometry_policy = radiometry_controls(collection, "download_radiometry")
dates = st.sidebar.date_input("Date Range", [date.today() - timedelta(days=30), date.today()])
cloud = st.sidebar.slider("Max Cloud Cover (%)", 0, 100, 20) if collection in S2_COLLECTIONS else 100
bands = st.sidebar.multiselect(
    "Bands to Download",
    list(COLLECTION_BANDS[collection]),
    default=DEFAULT_BANDS[collection],
    key=f"bands_{collection}",
)
indices = (
    st.sidebar.multiselect("Calculate Indices", list(INDEX_FUNCS)) if collection in S2_COLLECTIONS else []
)
resolution = st.sidebar.selectbox(
    "Output resolution",
    [10, 20, 60, "native"],
    format_func=lambda value: (
        "Coarsest selected native resolution" if value == "native" else f"{value} metres"
    ),
)
scl = (
    st.sidebar.checkbox("Filter cloudy pixels using SCL", value=False)
    if collection in S2_COLLECTIONS
    else False
)
requester_pays = False
if collection == "sentinel-1-grd":
    st.sidebar.caption(
        "Sentinel-1 exports are amplitude values, without SAR calibration or terrain correction. "
        "S3 assets require configured AWS credentials and requester-pays access."
    )
    requester_pays = st.sidebar.checkbox("Enable AWS requester-pays access", value=False)
limit = int(
    st.sidebar.number_input("Maximum search scenes", min_value=1, max_value=50000, value=1000, step=100)
)
source = st.sidebar.radio("AOI selection", ["Upload file", "Draw polygon"])
geometry = None
if source == "Upload file":
    upload = st.sidebar.file_uploader("GeoJSON or zipped shapefile", type=["geojson", "zip"])
    if upload is not None:
        try:
            geometry = cached_upload(upload.getvalue(), upload.name).union_all()
        except Exception as exc:
            st.sidebar.error(str(exc))

if source == "Draw polygon" and st.session_state.get("drawn_geometry"):
    try:
        geometry = validate_geometry(shape(st.session_state.drawn_geometry))
    except ValueError as exc:
        st.sidebar.error(str(exc))

left, right = st.columns([2, 1])
with left:
    map_view = folium.Map(location=[20, 0], zoom_start=2)
    if source == "Draw polygon":
        Draw(
            export=True,
            edit_options={"edit": False, "remove": False},
            draw_options={"polyline": False, "circle": False, "circlemarker": False, "marker": False},
        ).add_to(map_view)
    if geometry is not None:
        folium.GeoJson(mapping(geometry), name="AOI").add_to(map_view)
        west, south, east, north = geometry.bounds
        map_view.fit_bounds([[south, west], [north, east]])
    previous = st.session_state.get("catalog_result")
    if previous is not None and previous.items:
        features = [
            {"type": "Feature", "geometry": item.geometry, "properties": {"id": item.id}}
            for item in previous.items
            if item.geometry
        ]
        folium.GeoJson(
            {"type": "FeatureCollection", "features": features},
            name="Scene footprints",
            style_function=lambda _: {"color": "red", "fillOpacity": 0.05},
        ).add_to(map_view)
    drawing = st_folium(
        map_view,
        height=550,
        use_container_width=True,
        key=f"downloader_map_{st.session_state.get('draw_version', 0)}",
        returned_objects=["last_active_drawing", "all_drawings"],
    )
    if source == "Draw polygon":
        if drawing and drawing.get("all_drawings") is not None:
            drawings = drawing["all_drawings"]
            if drawings:
                feature = drawing.get("last_active_drawing") or drawings[-1]
                st.session_state.drawn_geometry = feature["geometry"]
        if st.button("Clear drawn AOI"):
            st.session_state.pop("drawn_geometry", None)
            st.session_state.draw_version = st.session_state.get("draw_version", 0) + 1
            st.rerun()
        if st.session_state.get("drawn_geometry"):
            try:
                geometry = validate_geometry(shape(st.session_state.drawn_geometry))
            except ValueError as exc:
                st.error(str(exc))

with right:
    valid_dates = len(dates) == 2
    fingerprint = None
    if geometry is not None and valid_dates:
        fingerprint = search_fingerprint(
            list(geometry.bounds), dates[0], dates[1], collection, cloud, mapping(geometry), limit
        )
    if st.button("Search catalog", width="stretch", disabled=fingerprint is None):
        try:
            with st.spinner("Searching catalog pages..."):
                result = search_stac(
                    list(geometry.bounds),
                    str(dates[0]),
                    str(dates[1]),
                    collection,
                    cloud,
                    max_items=limit,
                    intersects=mapping(geometry),
                    return_report=True,
                )
            st.session_state.catalog_result = result
            st.session_state.catalog_fingerprint = fingerprint
        except Exception as exc:
            st.session_state.pop("catalog_result", None)
            st.session_state.pop("catalog_fingerprint", None)
            st.error(str(exc))
    if not valid_dates:
        st.info("Select both a start and end date.")
    if geometry is None:
        st.info("Upload a polygon or choose Draw polygon to define an AOI.")
    result = st.session_state.get("catalog_result")
    current = fingerprint is not None and fingerprint == st.session_state.get("catalog_fingerprint")
    if result is not None:
        if not current:
            st.warning("Search parameters or AOI changed. Search again before processing.")
        st.metric("Scenes returned", len(result.items))
        st.caption(f"Catalog snapshot: {result.fetched_at}" + (" (cached)" if result.cached else ""))
        if result.truncated:
            st.warning(
                f"Incomplete results: more than {result.limit} scenes match. "
                "Increase Maximum search scenes or narrow the query."
            )
        if result.items:
            frame = items_to_gdf(result.items)
            columns = [name for name in ("stac_id", "datetime", "cloud_cover") if name in frame]
            st.dataframe(frame[columns], hide_index=True, width="stretch")
        else:
            st.info("No scenes matched this search.")
    ready = False
    if current and result is not None and result.items:
        try:
            selected = select_bands(collection, bands, indices)
            added = [b for b in selected if b not in bands]
            if added:
                st.info("Required index bands added: " + ", ".join(added))
            unique_dates = len({item.datetime.date() for item in result.items})
            preview = preview_processing(
                geometry, collection, selected, indices, resolution, dates=unique_dates
            )
            st.caption(
                f"Output grid: {preview['grid']['width']} × {preview['grid']['height']} pixels · "
                f"{preview['grid']['resolution']:g} m · "
                f"estimated uncompressed output {preview['estimated_output_bytes'] / 1024**2:.1f} MiB"
            )
            ready = True
        except ValueError as exc:
            st.error(str(exc))
    active = False
    if st.session_state.get("job_download"):
        owner, job_id = st.session_state.job_download
        active = job_manager().get(owner, job_id)["status"] in ("queued", "running", "cancelling")
    if st.button("Process & Save Outputs", type="primary", width="stretch", disabled=not ready or active):
        try:
            owner = session_owner()
            request = {
                "kind": "download",
                "geometry": mapping(geometry),
                "collection": collection,
                "bands": selected,
                "indices": indices,
                "resolution": resolution,
                "apply_scl_filter": scl,
                "requester_pays": requester_pays,
                "radiometry_policy": radiometry_policy,
                "items": [item.to_dict() for item in result.items],
                "search": result.summary(),
            }
            job_id = job_manager().create(owner, request)
            st.session_state.job_download = (owner, job_id)
        except Exception as exc:
            st.error(str(exc))
    recovery_controls("download")
    render_job("download")
