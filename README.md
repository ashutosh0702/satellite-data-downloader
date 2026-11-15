# Satellite Data Downloader

Search Sentinel scenes, create polygon-clipped mosaics and spectral indices, and generate daily single-pixel NDVI/NDWI series. Processing uses fixed-size raster windows and background jobs with cancellation and recovery.

## Run

Use macOS or Linux with Python 3.12–3.14. Python 3.14 is the project default. The old checked-in-workspace `venv` may reference a removed Homebrew interpreter; the project uses a separate `.venv`.

```bash
uv sync --locked
uv run --frozen streamlit run app.py
```

Alternatively, inside a supported Python virtual environment:

```bash
python -m pip install --require-hashes -r requirements.txt
python -m streamlit run app.py
```

`pyproject.toml` defines supported dependencies; `uv.lock` and `requirements.lock` pin the resolved environment. After intentional dependency changes, regenerate both:

```bash
uv lock
uv export --locked --format requirements-txt --no-dev --no-emit-project --output-file requirements.lock
```

## Use

1. Keep the recommended `sentinel-2-c1-l2a` Collection 1 source selected, then upload GeoJSON/a zipped shapefile, or choose Draw polygon for the AOI. Uploads are limited to 20 MiB; shapefiles need a CRS.
2. Search the catalog. If the scene limit is reached, increase it or narrow the search; incomplete results are explicitly labelled.
3. Choose bands, indices and resolution. Required index bands are added automatically. “Coarsest selected native resolution” avoids expanding coarse bands to 10 m.
4. Start processing. Jobs continue across ordinary UI reruns. Save the job receipt for later recovery, including after a server restart.
5. Review the full-width Preview, Files, Quality and Run details tabs. Download individual TIFFs, a filtered ZIP, summary CSV and the processing manifest. Point observations also have charts and CSV/JSON exports.

Progress distinguishes preparation, raster windows and finalization, with separate saved/skipped counts. Quality summaries show valid pixel coverage inside the AOI, per product, and actionable reasons for skipped dates. Counts, minimum, maximum and mean use all valid pixels; median/P10/P90 use a bounded sample when needed.

Raster exports stream from disk without loading complete TIFFs or ZIPs into Streamlit memory. Download links expire after one hour or a server restart; refresh the Files tab to renew them. ZIP selections support up to 1,000 files. Files remain in unique server-local job folders; set `SATELLITE_OUTPUT_ROOT` to change the default `output_data` root.

Launch **app.py**, which serves the Streamlit pages and export routes together. `portal.py` contains the home page and is not the production entry point. The locked Streamlit version supports this ASGI entry point; `--server.baseUrlPath=your-prefix` is supported by the export links too.

The older `sentinel-2-l2a` catalog has known inconsistent COG offset metadata. Strict mode rejects ambiguous legacy corrections; advanced legacy controls let an informed user explicitly choose declared metadata or already-harmonized values. The selected policy is recorded in provenance. Collection 1 is recommended, while legacy data remains available for historical coverage gaps.

Sentinel-2 composites now contain **physical reflectance**, not unlabeled DN floats. Indices use each asset's scale and offset. Sentinel-1 exports remain amplitude data and require configured AWS credentials plus explicit requester-pays access; the application does not perform SAR calibration or terrain correction. Credentials are obtained from the standard AWS provider chain and are not written into job requests.

To resume a raster job from the terminal:

```bash
uv run --frozen python jobs.py output_data/OWNER_UUID/JOB_UUID
```

Recovery reuses validated complete date bundles. Expanding an incomplete catalog search requires a new search/job. Job receipts are private recovery capabilities; this is a local/trusted application, not an authenticated public service.

## Tests and performance

```bash
uv run --frozen python -m unittest discover -v
uvx ruff==0.17.2 check .
uv run --frozen python scripts/benchmark_pipeline.py
```

Tests are deterministic, offline and use temporary rasters. No test downloads satellite data or writes into existing output folders. The benchmark script can compare a historical processing module via `--baseline-source /path/to/original.py`.

An optional, explicit live smoke test uses a public London fixture and verifies real COG calibration (requires internet access):

```bash
uv run --frozen python scripts/smoke_catalog.py
```

The architecture report documents all settings, APIs, calibration and masking semantics, job recovery, measurements and remaining deployment limits:

- [Detailed architecture report and diagrams](docs/ARCHITECTURE.md)
- [Raw performance measurements](docs/benchmark_results.json)
- [Validation evidence and tested environment](docs/validation_results.json)
- [Results dashboard implementation, semantics and diagrams](docs/UX_IMPROVEMENTS.md)
- [Performance measurements with result summaries enabled](docs/ux_benchmark_results.json)
