import os
import time
from datetime import datetime, timezone

import geopandas as gpd

import processing_utils
from stac_utils import search_stac
from processing_utils import process_items


def format_rate(count, seconds):
    if not seconds:
        return "n/a"
    return f"{count / seconds:.2f}"


def build_benchmark_report(search_seconds, processing_seconds, total_seconds, items, bands, output_dir, report):
    unique_dates = sorted(
        {
            item.datetime.strftime("%Y-%m-%d")
            if item.datetime
            else item.properties.get("datetime", "unknown")[:10]
            for item in items
        }
    )
    missing_band_counts = report.get("missing_band_counts", {})
    skipped_rows = [row for row in report["date_details"] if not row["saved"]]

    lines = [
        "# Downloader Benchmark Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Configuration",
        f"- Date workers: `{processing_utils.DATE_WORKER_LIMIT}`",
        f"- Band workers: `{processing_utils.BAND_WORKER_LIMIT}`",
        f"- GDAL cache: `{os.environ.get('GDAL_CACHEMAX', 'unset')}`",
        f"- Output nodata: `{processing_utils.COMMON_NODATA}`",
        f"- Output directory: `{output_dir}`",
        f"- Selected bands: `{', '.join(bands)}`",
        "",
        "## Benchmark Results",
        f"- Search time: `{search_seconds:.2f}s`",
        f"- Processing time: `{processing_seconds:.2f}s`",
        f"- End-to-end time: `{total_seconds:.2f}s`",
        f"- Items returned: `{len(items)}`",
        f"- Unique dates returned: `{len(unique_dates)}`",
        f"- Dates saved: `{report['saved_dates']}`",
        f"- Dates skipped: `{report['skipped_dates']}`",
        f"- Output TIFF files: `{report['saved_dates']}`",
        f"- Items/sec during processing: `{format_rate(len(items), processing_seconds)}`",
        f"- Dates/sec during processing: `{format_rate(report['total_dates'], processing_seconds)}`",
        "",
        "## Processing Summary",
        f"- Dates with missing bands: `{report['dates_with_missing_bands']}`",
    ]

    if missing_band_counts:
        lines.append("- Missing band frequency:")
        for band_name, count in missing_band_counts.items():
            lines.append(f"  - `{band_name}` missing on `{count}` dates")
    else:
        lines.append("- Missing band frequency: `none`")

    if skipped_rows:
        lines.extend(
            [
                "",
                "## Skipped Dates",
                "| Date | Scenes | Reason | Missing Bands |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for row in skipped_rows:
            missing_bands = ", ".join(row["missing_bands"]) if row["missing_bands"] else "-"
            lines.append(
                f"| {row['date']} | {row['item_count']} | {row['reason_label']} | {missing_bands} |"
            )
    else:
        lines.extend(
            [
                "",
                "## Skipped Dates",
                "None.",
            ]
        )

    return "\n".join(lines) + "\n"

def test_downloader():
    print("Testing Downloader...")
    benchmark_start = time.time()
    gdf = gpd.read_file('test_data/test-polygon.geojson')
    bbox = list(gdf.total_bounds)
    aoi_geom = gdf.union_all()
    
    search_start = time.time()
    items = search_stac(
        bbox=bbox,
        start_date="2025-04-01",
        end_date="2025-04-27",
        collection="sentinel-2-l2a",
        max_cloud_cover=15
    )
    search_seconds = time.time() - search_start
    
    print(f"Found {len(items)} items")
    unique_dates = sorted(
        {
            item.datetime.strftime("%Y-%m-%d")
            if item.datetime
            else item.properties.get("datetime", "unknown")[:10]
            for item in items
        }
    )
    print(
        f"Benchmark: STAC search returned {len(items)} items across {len(unique_dates)} dates "
        f"in {search_seconds:.2f}s"
    )
    
    if items:
        # User specified "all bands except vv and vh" -> ['red', 'green', 'blue', 'nir', 'nir08', 'swir16', 'swir22', 'rededge1']
        bands = ['blue', 'green', 'red', "rededge1",'nir', 'nir08', 'swir16', 'swir22']
        output_dir = "test_output"
        files, t_sec, report = process_items(
            items,
            aoi_geom,
            collection="sentinel-2-l2a",
            bands=bands,
            out_dir=output_dir,
            return_report=True,
        )
        total_seconds = time.time() - benchmark_start
        print(f"Benchmark: processing saved {len(files)} composites in {t_sec:.2f}s")
        print(f"Benchmark: end-to-end runtime {total_seconds:.2f}s")
        print(f"Benchmark: processing throughput {format_rate(len(items), t_sec)} items/s")
        print(f"Benchmark: date throughput {format_rate(report['total_dates'], t_sec)} dates/s")
        print(
            "Summary:",
            {
                "total_dates": report["total_dates"],
                "saved_dates": report["saved_dates"],
                "skipped_dates": report["skipped_dates"],
                "missing_band_counts": report["missing_band_counts"],
            },
        )
        skipped_rows = [row for row in report["date_details"] if not row["saved"]]
        if skipped_rows:
            print("Skipped date details:")
            for row in skipped_rows:
                print(
                    {
                        "date": row["date"],
                        "reason": row["reason_label"],
                        "missing_bands": row["missing_bands"],
                    }
                )

        benchmark_report = build_benchmark_report(
            search_seconds=search_seconds,
            processing_seconds=t_sec,
            total_seconds=total_seconds,
            items=items,
            bands=bands,
            output_dir=output_dir,
            report=report,
        )
        report_path = os.path.join(output_dir, "benchmark_report.md")
        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write(benchmark_report)
        print(f"Benchmark report written to {report_path}")
        
        # from processing_utils import calculate_indices
        # t1 = time.time()
        # idx = calculate_indices(files, ["NDVI", "NDWI", "EVI", "SAVI"], "test_output")
        # print(f"Total calculate_indices time: {time.time()-t1}s")
        # print("Downloader test completed.")

if __name__ == "__main__":
    test_downloader()
