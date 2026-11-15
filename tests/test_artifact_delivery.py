import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
from unittest.mock import patch
import zipfile

from starlette.applications import Starlette

import artifact_delivery as delivery
from persistence import write_json
from result_artifacts import committed_files, summary_csv
from tests.fixtures import RasterCase


class DeliveryTests(RasterCase):
    def setUp(self):
        super().setUp()
        self.directory = self.root / "outputs"
        self.names = ["2025-04-01_10m_all_bands.tif", "2025-04-01_NDVI.tif"]
        self.directory.mkdir()
        self.payloads = {self.names[0]: b"A" * (delivery.CHUNK_SIZE * 3 + 11), self.names[1]: b"B" * 99}
        for name, data in self.payloads.items():
            (self.directory / name).write_bytes(data)
        row = {
            "date": "2025-04-01",
            "saved": True,
            "fingerprint": "fixture",
            "files": [self.names[0]],
            "index_files": [self.names[1]],
        }
        write_json(self.directory / "manifest.json", {"fingerprint": "fixture", "date_details": [row]})
        write_json(self.directory / ".dates/2025-04-01.json", row)
        self.now = 100.0
        self.registry = delivery.ExportRegistry(clock=lambda: self.now)
        self.addCleanup(patch.stopall)
        patch.object(delivery, "EXPORTS", self.registry).start()
        self.app = Starlette(routes=delivery.export_routes())

    def request(self, path, headers=None, method="GET"):
        messages = []

        async def run():
            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
                "method": method,
                "path": path,
                "raw_path": path.encode(),
                "root_path": "",
                "query_string": b"",
                "scheme": "http",
                "http_version": "1.1",
                "server": ("test", 80),
                "client": ("127.0.0.1", 1),
                "headers": headers or [],
            }

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                messages.append(message)

            await self.app(scope, receive, send)

        asyncio.run(run())
        return messages[0], [m.get("body", b"") for m in messages if m["type"] == "http.response.body"]

    def test_file_response_streams_bounded_chunks(self):
        token = self.registry.issue(self.directory, [self.names[0]])
        response, chunks = self.request(f"/exports/file/{token}")
        self.assertEqual(response["status"], 200)
        self.assertEqual(b"".join(chunks), self.payloads[self.names[0]])
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(max(map(len, chunks)), delivery.CHUNK_SIZE)
        self.assertIn((b"cache-control", b"private, no-store"), response["headers"])

    def test_download_supports_byte_ranges(self):
        token = self.registry.issue(self.directory, [self.names[0]])
        response, chunks = self.request(f"/exports/file/{token}", [(b"range", b"bytes=10-19")])
        self.assertEqual(response["status"], 206)
        self.assertEqual(b"".join(chunks), self.payloads[self.names[0]][10:20])

    def test_prefixed_routes_match_generated_links(self):
        self.app = Starlette(routes=delivery.export_routes(base_path="/research/tools/"))
        url = delivery.export_url(self.directory, self.names, bundle=True, base_path="research/tools")
        response, chunks = self.request(url)
        self.assertEqual(response["status"], 200)
        with zipfile.ZipFile(io.BytesIO(b"".join(chunks))) as archive:
            self.assertEqual(sorted(archive.namelist()), sorted(self.names))

    def test_head_does_not_send_file_body(self):
        token = self.registry.issue(self.directory, [self.names[0]])
        response, chunks = self.request(f"/exports/file/{token}", method="HEAD")
        self.assertEqual(response["status"], 200)
        self.assertEqual(b"".join(chunks), b"")

    def test_zip_stream_contains_only_granted_files(self):
        token = self.registry.issue(self.directory, self.names)
        response, chunks = self.request(f"/exports/bundle/{token}")
        self.assertEqual(response["status"], 200)
        self.assertLessEqual(max(map(len, chunks)), delivery.CHUNK_SIZE)
        with zipfile.ZipFile(io.BytesIO(b"".join(chunks))) as archive:
            self.assertEqual(sorted(archive.namelist()), sorted(self.names))
            for name in self.names:
                self.assertEqual(archive.read(name), self.payloads[name])
                self.assertEqual(archive.getinfo(name).compress_type, zipfile.ZIP_STORED)
        self.assertEqual(list(self.directory.glob("*.zip")), [])

    def test_expired_and_invalid_capabilities_are_rejected(self):
        token = self.registry.issue(self.directory, self.names)
        self.now += 3601
        for candidate in (token, "unknown"):
            response, _ = self.request(f"/exports/bundle/{candidate}")
            self.assertEqual(response["status"], 404)

    def test_grant_cannot_expose_uncommitted_or_escaping_files(self):
        (self.directory / "uncommitted.tif").write_bytes(b"private")
        for name in ("uncommitted.tif", "../outside.tif", ".raster-stage-x/partial.tif"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.registry.issue(self.directory, [name])

    def test_changed_marker_or_symlink_invalidates_link(self):
        token = self.registry.issue(self.directory, [self.names[0]])
        path = self.directory / self.names[0]
        path.unlink()
        outside = self.root / "outside.tif"
        outside.write_bytes(b"private")
        path.symlink_to(outside)
        response, _ = self.request(f"/exports/file/{token}")
        self.assertEqual(response["status"], 404)

    def test_concurrent_link_generation_reuses_grant(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            tokens = list(pool.map(lambda _: self.registry.issue(self.directory, self.names), range(20)))
        self.assertEqual(len(set(tokens)), 1)
        self.assertEqual(len(self.registry.grants), 1)

    def test_manifest_corruption_does_not_expose_arbitrary_files(self):
        (self.directory / "manifest.json").write_text("broken")
        self.assertEqual(committed_files(self.directory), {})
        with self.assertRaises(ValueError):
            self.registry.issue(self.directory, self.names)

    def test_csv_contains_only_committed_artifacts(self):
        files = committed_files(self.directory)
        text = summary_csv(files)
        self.assertIn(self.names[0], text)
        self.assertIn("coverage_percent", text)
        self.assertEqual(len(text.splitlines()), 3)
