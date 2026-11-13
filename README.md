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
5. Outputs and their processing manifest are saved in a unique server-local folder. Point observations can also be downloaded as CSV/JSON.

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
