"""Full-width result exploration; raster reads are limited to cached thumbnails."""

from datetime import datetime, timezone
import json

import pandas as pd
import streamlit as st

from artifact_delivery import export_url, MAX_BUNDLE_FILES
from result_artifacts import committed_files, preview_png, summary_csv


def format_size(value):
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def format_duration(value):
    seconds = max(0, int(value))
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def elapsed_seconds(state):
    start = state.get("started_at") if state.get("status") != "queued" else state.get("queued_at")
    start = start or state.get("created_at")
    if not start:
        return 0
    end = state.get("finished_at") if state.get("status") not in ("queued", "running", "cancelling") else None
    return max(
        0,
        (
            (datetime.fromisoformat(end) if end else datetime.now(timezone.utc))
            - datetime.fromisoformat(start)
        ).total_seconds(),
    )


def failure_guidance(row):
    reason, error = row.get("reason"), str(row.get("error") or "").lower()
    if reason == "missing_bands":
        return (
            "Missing selected bands",
            "Choose available bands or another collection, then start a new run.",
            False,
        )
    if reason in ("missing_scl", "incomplete_scenes"):
        return (
            "Required scene assets unavailable",
            "Choose scenes with all required bands and SCL, or explicitly change the output/filter settings.",
            False,
        )
    if reason == "no_valid_overlap":
        return (
            "No valid pixels in the AOI",
            "Inspect the area, source coverage and cloud-filter settings.",
            False,
        )
    if reason == "cancelled":
        return "Cancelled before completion", "Resume this saved request to finish its remaining dates.", True
    if "ambiguous legacy" in error:
        return (
            "Ambiguous legacy calibration",
            "Choose Collection 1, or review the advanced legacy calibration policy in a new run.",
            False,
        )
    if "disk" in error or "space" in error:
        return "Storage problem", "Free server disk space before resuming the saved request.", False
    if any(word in error for word in ("credential", "requester-pays", "access denied", "403")):
        return (
            "Asset access problem",
            "Check the AWS credentials, permissions and requester-pays setting before resuming.",
            False,
        )
    if any(
        word in error
        for word in ("timeout", "timed out", "connection", "429", "502", "503", "504", "temporarily")
    ):
        return "Temporary read failure", "Retry unfinished dates; saved outputs will be reused.", True
    return (
        "Processing failed",
        "Expand technical diagnostics and resolve the reported cause before resuming.",
        False,
    )


def date_rows(details):
    return [
        {
            "Date": row["date"],
            "Outcome": "Saved (reused)" if row.get("resumed") else "Saved" if row.get("saved") else "Skipped",
            "Valid AOI (%)": row.get("coverage_percent"),
            "Raster files": len(row.get("files", [])) + len(row.get("index_files", [])),
            "Reason": "" if row.get("saved") else failure_guidance(row)[0],
            "Next step": "" if row.get("saved") else failure_guidance(row)[1],
        }
        for row in details
    ]


@st.cache_data(max_entries=16, ttl=600, show_spinner=False)
def thumbnail(path, modified_ns, size_bytes, band, rgb):
    return preview_png(path, band=band, rgb=rgb)


def _preview(directory, files, key):
    if not files:
        st.info("Previews appear when the first date's files are committed.")
        return
    name = st.selectbox(
        "Raster to preview",
        list(files),
        format_func=lambda name: (
            f"{files[name]['date']} · {files[name]['name']} · {format_size(files[name]['size_bytes'])}"
        ),
        key=f"{key}_preview_file",
    )
    record = files[name]
    bands = [band for band in record.get("bands", []) if band != "alpha"]
    # Older commit records need only the small header to discover band names.
    if not bands:
        import rasterio

        with rasterio.open(directory / name) as source:
            bands = [name for name in source.descriptions if name and name != "alpha"]
    choices = (["RGB"] if all(band in bands for band in ("red", "green", "blue")) else []) + bands
    band = st.selectbox("Preview layer", choices or ["Band 1"], key=f"{key}_preview_band_{name}")
    path = directory / name
    try:
        with st.spinner("Loading a small raster preview..."):
            image, legend = thumbnail(
                str(path), path.stat().st_mtime_ns, path.stat().st_size, band, band == "RGB"
            )
        st.image(image, caption=f"{record['date']} · {band}", width="stretch")
        st.caption(
            legend
            + " Transparent pixels are nodata. Preview is limited to 640 pixels per side; source files retain their full resolution."
        )
        if record.get("coverage_percent") is not None:
            st.caption(f"Valid AOI coverage for this product: {record['coverage_percent']:.1f}%")
    except (OSError, ValueError) as exc:
        st.warning(f"Preview unavailable: {exc}")


def _files(directory, files, key):
    if not files:
        st.info("No committed raster files are currently available.")
        return
    columns = st.columns(2)
    with columns[0]:
        dates = st.multiselect(
            "Acquisition dates",
            sorted({row["date"] for row in files.values()}),
            default=sorted({row["date"] for row in files.values()}),
            key=f"{key}_file_dates",
        )
    with columns[1]:
        products = st.multiselect(
            "Products",
            sorted({row["name"] for row in files.values()}),
            default=sorted({row["name"] for row in files.values()}),
            key=f"{key}_file_products",
        )
    filtered = {name: row for name, row in files.items() if row["date"] in dates and row["name"] in products}
    if not filtered:
        st.info("Choose at least one date and product to browse or download files.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Date": row["date"],
                    "Product": row["name"],
                    "File": name,
                    "Resolution (m)": row.get("resolution_m"),
                    "Size (MiB)": row["size_bytes"] / 1024**2,
                    "Valid AOI (%)": row.get("coverage_percent"),
                }
                for name, row in filtered.items()
            ]
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Size (MiB)": st.column_config.NumberColumn(format="%.2f"),
            "Valid AOI (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        },
    )
    selected = st.selectbox("File to download", list(filtered), key=f"{key}_download_file")
    base_path = st.get_option("server.baseUrlPath")
    buttons = st.columns(2)
    try:
        with buttons[0]:
            st.link_button(
                f"Download TIFF · {format_size(filtered[selected]['size_bytes'])}",
                export_url(directory, [selected], base_path=base_path),
                type="primary",
                width="stretch",
            )
        with buttons[1]:
            if len(filtered) <= MAX_BUNDLE_FILES:
                st.link_button(
                    f"Download filtered files as ZIP ({len(filtered)})",
                    export_url(directory, list(filtered), bundle=True, base_path=base_path),
                    width="stretch",
                )
            else:
                st.info(f"Narrow the filters to at most {MAX_BUNDLE_FILES} files for a ZIP download.")
    except (ValueError, OSError) as exc:
        st.warning(str(exc))
    st.caption(
        "TIFF and ZIP downloads stream to your computer. Links expire after one hour; refresh this view for new links."
    )
    st.button("Refresh download links", key=f"{key}_refresh_links")
    st.download_button(
        "Export filtered summary CSV",
        summary_csv(filtered),
        file_name="raster-summary.csv",
        mime="text/csv",
        key=f"{key}_summary_csv",
    )


def _quality(files, details, key):
    st.dataframe(
        pd.DataFrame(date_rows(details)),
        hide_index=True,
        width="stretch",
        column_config={
            "Valid AOI (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")
        },
    )
    st.caption(
        "Coverage counts valid pixels inside the AOI, including its holes correctly. "
        "A saved date can have partial coverage. Each index has its own validity mask."
    )
    stats = []
    for record in files.values():
        for band, row in record.get("statistics", {}).items():
            stats.append(
                {
                    "Date": record["date"],
                    "Product": record["name"],
                    "Band": band,
                    "Valid pixels": row["count"],
                    "Units": record.get("units") or "Not recorded",
                    "Mean": row["mean"],
                    "Median": row["median"],
                    "P10": row["p10"],
                    "P90": row["p90"],
                    "Minimum": row["minimum"],
                    "Maximum": row["maximum"],
                    "Quantile samples": row["sample_count"],
                }
            )
    if stats:
        st.subheader("Valid-pixel statistics")
        st.dataframe(pd.DataFrame(stats), hide_index=True, width="stretch")
        st.caption(
            "Count, mean and range are exact over valid pixels. Median/P10/P90 use a reproducible uniform sample "
            "of up to 16,384 pixels per band/product; zero reflectance values are retained."
        )
    elif files:
        st.info(
            "This older run has no recorded statistics. Resume it from Run details to add summaries without rewriting valid raster files."
        )
    skipped = [row for row in details if not row.get("saved")]
    if skipped:
        st.subheader("Dates needing attention")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Date": row["date"],
                        "Issue": failure_guidance(row)[0],
                        "Recommended action": failure_guidance(row)[1],
                    }
                    for row in skipped
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        with st.expander("Technical error details"):
            st.json(skipped)


def render_download_results(manager, owner, job_id, state):
    directory = manager.directory(owner, job_id)
    result = state.get("result")
    if not result:
        try:
            result = json.loads((directory / "manifest.json").read_text())
        except (OSError, ValueError):
            result = {}
    details = result.get("date_details", [])
    files = committed_files(directory)
    if not result and not files:
        st.info(
            "Results will appear here as dates finish. You can keep using the map while processing continues."
            if state["status"] in ("queued", "running", "cancelling")
            else "No raster files were committed for this run. Review its error or resume the saved request after resolving the issue."
        )
        return
    st.subheader("Processing results")
    expected = sum(
        len(row.get("files", [])) + len(row.get("index_files", [])) for row in details if row.get("saved")
    )
    if len(files) < expected:
        st.warning(
            "Some saved files are no longer available on the server. Use Run details to restore missing outputs."
        )
    metrics = st.columns(5)
    total = result.get("total_dates", len(details))
    for column, label, value in zip(
        metrics,
        ["Dates with imagery", "Dates saved", "Dates skipped", "Raster files", "Output size"],
        [
            total,
            result.get("saved_dates", 0),
            result.get("skipped_dates", 0),
            len(files),
            format_size(sum(row["size_bytes"] for row in files.values())),
        ],
    ):
        column.metric(label, value)
    search = result.get("search", state.get("parameters", {}).get("search", {}))
    if search.get("complete") is False:
        st.warning(
            "Catalog search is incomplete. Increase the scene limit or narrow the search and create a new run; resuming cannot add scenes to this saved request."
        )
    elif search.get("complete") is True:
        st.caption("Catalog search: complete for the saved query.")
    parameters = state.get("parameters", {})
    scl = parameters.get("apply_scl_filter", result.get("config", {}).get("apply_scl_filter"))
    st.caption(
        "Cloud-pixel screening: "
        + (
            "SCL applied."
            if scl is True
            else "not requested; valid-data coverage does not mean cloud-free coverage."
            if scl is False
            else "not recorded in this older run."
        )
    )
    key = f"results_{job_id}"
    preview, downloads, quality, run = st.tabs(["Preview", "Files", "Quality", "Run details"])
    with preview:
        _preview(directory, files, key)
    with downloads:
        _files(directory, files, key)
    with quality:
        _quality(files, details, key)
    with run:
        st.write("**Saved on server**")
        st.code(str(directory))
        period = parameters.get("search_parameters", parameters.get("search", {}))
        settings = {
            "Area": parameters.get("aoi_name", "Area of interest"),
            "Collection": parameters.get("collection"),
            "Requested period": f"{period['start_date']} to {period['end_date']}"
            if "start_date" in period
            else "Not recorded in this older run",
            "Scene cloud threshold (%)": period.get("max_cloud_cover", "Not recorded"),
            "Bands": parameters.get("bands", []),
            "Indices": parameters.get("indices", []),
            "Resolution": parameters.get("resolution"),
            "Calibration policy": parameters.get("radiometry_policy", "strict"),
            "Elapsed": format_duration(elapsed_seconds(state)),
        }
        st.table(
            pd.DataFrame(
                [
                    {
                        "Setting": name,
                        "Value": ", ".join(map(str, value))
                        if isinstance(value, list)
                        else str(value)
                        if value is not None
                        else "Not recorded",
                    }
                    for name, value in settings.items()
                ]
            )
        )
        if result.get("resources"):
            with st.expander("Output grid and processing resources"):
                st.json(result["resources"])
        manifest = directory / "manifest.json"
        if manifest.exists():
            st.download_button(
                "Download processing manifest",
                manifest.read_bytes(),
                file_name=f"manifest-{job_id}.json",
                mime="application/json",
                key=f"{key}_manifest",
            )
        if (
            len(files) < expected or (files and any(not row.get("statistics") for row in files.values()))
        ) and (state["status"] not in ("queued", "running", "cancelling")):
            if st.button("Restore outputs / update summaries", key=f"{key}_upgrade"):
                manager.resume(owner, job_id)
                st.rerun()
