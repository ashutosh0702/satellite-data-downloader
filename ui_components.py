"""Streamlit-only presentation; processing workers use plain callbacks."""

import json
import os
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
    return JobManager(os.environ.get("SATELLITE_OUTPUT_ROOT", "output_data"))


def session_owner():
    if "portal_owner" not in st.session_state:
        st.session_state.portal_owner = uuid4().hex
    return st.session_state.portal_owner


def recovery_controls(kind):
    with st.expander("Recover a saved run"):
        receipt = st.file_uploader("Upload the recovery file", type=["json"], key=f"receipt_{kind}")
        if receipt is not None and st.button("Open saved run", key=f"recover_{kind}"):
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


def render_job(kind):
    reference = st.session_state.get(f"job_{kind}")
    if not reference:
        return
    state = job_manager().get(*reference)
    was_active = state["status"] in ("queued", "running", "cancelling")

    @st.fragment(run_every=2 if was_active else None)
    def panel():
        _render_job(kind, reference, was_active)

    panel()


def _render_job(kind, reference, was_active):
    from results_ui import render_download_results, elapsed_seconds, format_duration, failure_guidance

    owner, job_id = reference
    manager = job_manager()
    state = manager.get(owner, job_id)
    active = state["status"] in ("queued", "running", "cancelling")
    if was_active and not active:
        # Re-register the fragment without a timer after the run finishes.
        st.rerun()
    st.subheader("Run progress")
    parameters = state.get("parameters", {})
    if kind == "timeseries" and parameters:
        point = parameters.get("geometry", {}).get("coordinates", [])
        period = parameters.get("search", {})
        st.caption(
            f"{parameters.get('index', '')} · point {point} · "
            f"{period.get('start_date', '')} to {period.get('end_date', '')}"
        )
    elif parameters:
        st.caption(
            f"{parameters.get('aoi_name', 'Area of interest')} · {parameters.get('collection', '')} · "
            f"{parameters.get('resolution', 10)} m output"
        )
    progress = state.get("progress", {})
    completed, total = progress.get("completed", 0), progress.get("total", 0)
    stage_names = {
        "queued": "Waiting for a job worker",
        "preparing": "Preparing imagery",
        "processing": "Processing imagery and indices",
        "finalizing": "Finalizing raster files",
        "validating": "Checking saved outputs",
        "waiting_for_resources": "Waiting for processing capacity",
        "extracting": "Extracting daily observations",
    }
    if active:
        label = (
            "Cancelling after the current operation returns"
            if state["status"] == "cancelling"
            else stage_names.get(progress.get("stage"), state.get("detail", "Preparing inputs"))
        )
        status = st.status(label, state="running", expanded=False)
        activity = progress.get("active_dates", [])
        if activity:
            status.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Date": row["date"],
                            "Activity": stage_names.get(row["stage"], row["stage"]),
                            "Imagery processed (%)": 100
                            * row.get("completed_windows", 0)
                            / row["total_windows"]
                            if row.get("total_windows")
                            else 0,
                        }
                        for row in activity
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        if total:
            fraction = progress.get("fraction")
            st.progress(
                fraction if fraction is not None else min(1, completed / total),
                text=f"{completed} of {total} acquisition dates handled",
            )
            st.caption(
                f"Saved: {progress.get('saved', 0)} · Skipped: {progress.get('skipped', 0)} · "
                f"Elapsed: {format_duration(elapsed_seconds(state))}"
            )
            st.caption(
                "Progress measures image work, not transferred bytes. File finalization may continue after all pixels are processed."
            )
        else:
            st.caption(f"Elapsed: {format_duration(elapsed_seconds(state))}")
        if progress.get("eta_seconds") is not None:
            st.caption(
                f"Approximate time remaining: {format_duration(progress['eta_seconds'])}, based on observed progress."
            )
        if st.button("Cancel run", key=f"cancel_{kind}", disabled=state["status"] == "cancelling"):
            manager.cancel(owner, job_id)
            st.rerun(scope="fragment")
    elif state["status"] == "completed":
        st.success(f"Processing completed · {format_duration(elapsed_seconds(state))}")
    elif state["status"] == "partial":
        st.warning("Finished with issues. Review skipped dates, catalog completeness and coverage below.")
    elif state["status"] == "cancelled":
        st.info("Run cancelled. Committed outputs remain available and can be reused when you resume.")
    elif state["status"] == "interrupted":
        st.warning(
            "The worker stopped before finishing. Resume the saved request to recover its committed outputs."
        )
    else:
        st.error(
            state.get("error") or "The run could not finish. Review the diagnostics and suggested actions."
        )
    if state.get("error") and state["status"] != "failed":
        st.error(state["error"])
    rows = (state.get("result") or {}).get("date_details", [])
    retry = state["status"] in ("failed", "cancelled", "interrupted", "created") or (
        state["status"] == "partial"
        and any(not row.get("saved") and failure_guidance(row)[2] for row in rows)
    )
    if not active and retry:
        if st.button(
            "Resume saved request",
            key=f"resume_{kind}",
            help="Uses the original settings. Saved raster bundles are checked and reused; unfinished dates are retried.",
        ):
            try:
                manager.resume(owner, job_id)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        st.caption(
            "Resuming keeps the original settings. Missing bands or changed cloud/calibration settings require a new run."
        )
    if kind == "download":
        render_download_results(manager, owner, job_id, state)
    elif state.get("result"):
        result = state["result"]
        if not result["search"]["complete"]:
            st.warning(
                "Catalog search is incomplete. Increase the scene limit and start a new run for complete temporal coverage."
            )
        if result["report"].get("read_failures"):
            st.warning("Some scene reads failed. See diagnostics and start a new run to retry them.")
        frame = pd.DataFrame(result["observations"])
        total_dates = result["report"]["total_dates"]
        metrics = st.columns(3)
        metrics[0].metric("Acquisition dates", total_dates)
        metrics[1].metric("Valid observations", len(frame))
        metrics[2].metric("Dates without valid data", total_dates - len(frame))
        if not frame.empty:
            # Keep rejected acquisition dates as gaps; do not invent daily observations.
            dates = [row["date"] for row in result["report"]["date_details"]]
            chart = frame.set_index("Date")[["Index"]].reindex(dates)
            chart.index = pd.to_datetime(chart.index)
            st.line_chart(chart)
            st.dataframe(frame, hide_index=True, width="stretch")
            st.download_button(
                "Download observations CSV",
                frame.to_csv(index=False),
                file_name="timeseries.csv",
                mime="text/csv",
                key=f"{job_id}_series_csv",
            )
        else:
            st.info("No valid observations. The scene diagnostics explain why each date was rejected.")
        with st.expander("Scene diagnostics"):
            st.json(result["report"])
        st.download_button(
            "Download series and diagnostics",
            json.dumps(result, indent=2),
            file_name="timeseries.json",
            mime="application/json",
            key=f"{job_id}_series_json",
        )
    with st.expander("Save recovery file"):
        st.caption("Keep this file to reopen the run after reconnecting or restarting the server.")
        st.download_button(
            "Download recovery file",
            json.dumps(manager.receipt(owner, job_id), indent=2),
            file_name=f"satellite-job-{job_id}.json",
            mime="application/json",
            key=f"download_receipt_{kind}",
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
