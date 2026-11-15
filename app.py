"""ASGI entry point: Streamlit UI and bounded-memory raster delivery."""

from pathlib import Path

import streamlit as st

from artifact_delivery import export_routes

app = st.App(
    Path(__file__).with_name("portal.py"), routes=export_routes(base_path=st.get_option("server.baseUrlPath"))
)
