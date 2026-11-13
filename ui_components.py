"""Streamlit-only presentation; processing workers use plain callbacks."""

import json
from uuid import uuid4

import pandas as pd
import streamlit as st

from input_utils import read_vector_upload
from jobs import JobManager


@st.cache_data(ttl=600, max_entries=16, show_spinner=False)
def cached_upload(data, filename, point=False):
    return read_vector_upload(data, filename, point)


@st.cache_resource
def job_manager():
    return JobManager()


def session_owner():
    if "portal_owner" not in st.session_state:
        st.session_state.portal_owner = uuid4().hex
    return st.session_state.portal_owner


def recovery_controls(kind):
    with st.expander("Recover a saved job"):
        receipt = st.file_uploader("Upload the job receipt JSON", type=["json"], key=f"receipt_{kind}")
        if receipt is not None and st.button("Open saved job", key=f"recover_{kind}"):
            try:
                if receipt.size > 4096:
                    raise ValueError("Receipt must be a small job receipt JSON file.")
                data = json.loads(receipt.getvalue())
                state = job_manager().get(data["owner"], data["job_id"])
                if state["kind"] != kind:
                    raise ValueError("This receipt belongs to the other module.")
                st.session_state[f"job_{kind}"] = (data["owner"], data["job_id"])
            except (ValueError, KeyError, OSError) as exc:
                st.error(f"Cannot recover this job: {exc}")


@st.fragment(run_every=2)
def render_job(kind):
    reference = st.session_state.get(f"job_{kind}")
    if not reference:
        return
    owner, job_id = reference
    manager = job_manager()
    state = manager.get(owner, job_id)
    st.subheader("Job progress")
    st.write(f"Status: **{state['status']}**")
    parameters = state.get("parameters", {})
    if kind == "timeseries" and parameters:
        coordinates = parameters.get("geometry", {}).get("coordinates", [])
        search_parameters = parameters.get("search", {})
        st.caption(
            f"{parameters.get('index', '')} · point {coordinates} · "
            f"{search_parameters.get('start_date', '')} to {search_parameters.get('end_date', '')}"
        )
    elif parameters:
        st.caption(
            f"{parameters.get('collection', '')} · bands: {', '.join(parameters.get('bands', []))} · "
            f"resolution: {parameters.get('resolution', 10)}"
        )
    progress = state.get("progress", {})
    completed, total = progress.get("completed", 0), progress.get("total", 0)
    if total:
        st.progress(min(1.0, completed / total), text=f"{completed} of {total} dates")
    st.caption(state.get("detail", ""))
    if state.get("error"):
        st.error(state["error"])
    active = state["status"] in ("queued", "running", "cancelling")
    if active:
        if st.button("Cancel job", key=f"cancel_{kind}", disabled=state["status"] == "cancelling"):
            manager.cancel(owner, job_id)
            st.rerun(scope="fragment")
    elif state["status"] in ("failed", "cancelled", "interrupted", "created") or (
        state["status"] == "partial" and state.get("result", {}).get("skipped_dates", 0) > 0
    ):
        if st.button("Resume unfinished work", key=f"resume_{kind}"):
            try:
                manager.resume(owner, job_id)
                st.rerun(scope="fragment")
            except ValueError as exc:
                st.error(str(exc))
    st.download_button(
        "Save job receipt",
        json.dumps(manager.receipt(owner, job_id), indent=2),
        file_name=f"satellite-job-{job_id}.json",
        mime="application/json",
        key=f"download_receipt_{kind}",
    )
    result = state.get("result", {})
    if kind == "download" and result:
        if not result.get("search", {}).get("complete", True):
            st.warning(
                "This job used incomplete catalog results. Increase the search limit and start a new job for full coverage."
            )
        st.write(f"Saved dates: **{result['saved_dates']}** · Skipped dates: **{result['skipped_dates']}**")
        st.code(str(manager.directory(owner, job_id)))
        rows = result.get("date_details", [])
        if rows:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            k: row.get(k)
                            for k in ("date", "saved", "resumed", "reason", "missing_bands", "error")
                        }
                        for row in rows
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        manifest = manager.directory(owner, job_id) / "manifest.json"
        if manifest.exists():
            st.download_button(
                "Download processing manifest",
                manifest.read_bytes(),
                file_name=f"manifest-{job_id}.json",
                mime="application/json",
                key=f"manifest_{kind}",
            )
    elif kind == "timeseries" and result:
        search = result["search"]
        if not search["complete"]:
            st.warning(
                f"Incomplete series: the catalog search reached its {search['limit']}-scene limit. "
                "Increase the search limit and start a new job."
            )
        if result["report"].get("read_failures"):
            st.warning(
                "Some scene reads failed. See diagnostics and start a new job to retry the catalog snapshot."
            )
        observations = result["observations"]
        if observations:
            frame = pd.DataFrame(observations)
            st.line_chart(frame.set_index("Date")[["Index"]])
            st.dataframe(frame, hide_index=True, width="stretch")
            st.download_button(
                "Download observations CSV",
                frame.to_csv(index=False),
                file_name="timeseries.csv",
                mime="text/csv",
                key="series_csv",
            )
        else:
            st.warning("No valid observations. Inspect the scene diagnostics below.")
        with st.expander("Scene diagnostics"):
            st.json(result["report"])
        st.download_button(
            "Download series and diagnostics",
            json.dumps(result, indent=2),
            file_name="timeseries.json",
            mime="application/json",
            key="series_json",
        )


def radiometry_controls(collection, key):
    if collection != "sentinel-2-l2a":
        return "strict"
    st.sidebar.warning(
        "The legacy Sentinel-2 catalog has known inconsistent offset metadata. "
        "Prefer sentinel-2-c1-l2a. Strict mode rejects ambiguous legacy COG offsets."
    )
    return st.sidebar.selectbox(
        "Legacy reflectance policy",
        ["strict", "metadata", "harmonized"],
        format_func=lambda value: {
            "strict": "Reject ambiguous offsets (recommended)",
            "metadata": "Use declared catalog scale and offset",
            "harmonized": "Use scale with zero offset",
        }[value],
        key=key,
    )
