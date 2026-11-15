"""Capability-scoped, same-origin raster downloads with bounded streaming buffers."""

from collections import deque
from dataclasses import dataclass
import io
from pathlib import Path
import secrets
import threading
import time
import zipfile

from starlette.responses import FileResponse, Response, StreamingResponse
from starlette.routing import Route

from result_artifacts import committed_files

CHUNK_SIZE = 64 * 1024
MAX_BUNDLE_FILES = 1000
HEADERS = {
    "Cache-Control": "private, no-store",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "Content-Encoding": "identity",
}


@dataclass(frozen=True)
class Grant:
    directory: Path
    filenames: tuple
    expires: float


class ExportRegistry:
    def __init__(self, ttl=3600, maximum=2048, clock=time.monotonic):
        self.ttl, self.maximum, self.clock = ttl, maximum, clock
        self.lock = threading.Lock()
        self.grants = {}
        self.keys = {}

    def issue(self, directory, filenames):
        directory = Path(directory).resolve()
        filenames = tuple(sorted(set(filenames)))
        if not filenames or len(filenames) > MAX_BUNDLE_FILES:
            raise ValueError(f"Select between 1 and {MAX_BUNDLE_FILES} raster files.")
        allowed = committed_files(directory)
        if any(name not in allowed for name in filenames):
            raise ValueError("Only committed raster outputs can be downloaded. Refresh the results.")
        key = (str(directory), filenames)
        now = self.clock()
        with self.lock:
            for token, grant in list(self.grants.items()):
                if grant.expires <= now:
                    self.grants.pop(token, None)
                    grant_key = (str(grant.directory), grant.filenames)
                    if self.keys.get(grant_key) == token:
                        self.keys.pop(grant_key, None)
            token = self.keys.get(key)
            if token in self.grants and self.grants[token].expires > now + min(60, self.ttl / 2):
                return token
            # Bound capabilities held by all active UI sessions in this process.
            while len(self.grants) >= self.maximum:
                old = next(iter(self.grants))
                grant = self.grants.pop(old)
                grant_key = (str(grant.directory), grant.filenames)
                if self.keys.get(grant_key) == old:
                    self.keys.pop(grant_key, None)
            token = secrets.token_urlsafe(32)
            self.grants[token] = Grant(directory, filenames, now + self.ttl)
            self.keys[key] = token
            return token

    def resolve(self, token):
        with self.lock:
            grant = self.grants.get(token)
            if grant is None or grant.expires <= self.clock():
                return None
        allowed = committed_files(grant.directory)
        if any(name not in allowed for name in grant.filenames):
            return None
        return grant


EXPORTS = ExportRegistry()


class _ZipSink(io.RawIOBase):
    def __init__(self):
        self.parts = deque()
        self.position = 0

    def writable(self):
        return True

    def seekable(self):
        return False

    def tell(self):
        return self.position

    def write(self, data):
        data = bytes(data)
        self.position += len(data)
        if data:
            self.parts.append(data)
        return len(data)

    def drain(self):
        while self.parts:
            yield self.parts.popleft()


def zip_chunks(grant):
    """TIFFs are already compressed: store their bytes without recompressing."""
    sink = _ZipSink()
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in grant.filenames:
            path = grant.directory / name
            # File descriptors are scoped to each member and close on disconnect.
            with path.open("rb") as source, archive.open(name, "w", force_zip64=True) as target:
                while chunk := source.read(CHUNK_SIZE):
                    target.write(chunk)
                    yield from sink.drain()
            yield from sink.drain()
    yield from sink.drain()


async def download_file(request):
    grant = EXPORTS.resolve(request.path_params["token"])
    if grant is None or len(grant.filenames) != 1:
        return Response(
            "Download expired or unavailable. Refresh the Files tab for a new link.",
            status_code=404,
            headers=HEADERS,
            media_type="text/plain",
        )
    name = grant.filenames[0]
    return FileResponse(grant.directory / name, filename=name, media_type="image/tiff", headers=HEADERS)


async def download_bundle(request):
    grant = EXPORTS.resolve(request.path_params["token"])
    if grant is None:
        return Response(
            "Download expired or unavailable. Refresh the Files tab for a new link.",
            status_code=404,
            headers=HEADERS,
            media_type="text/plain",
        )
    headers = {**HEADERS, "Content-Disposition": 'attachment; filename="satellite-outputs.zip"'}
    return StreamingResponse(zip_chunks(grant), media_type="application/zip", headers=headers)


def export_routes(base_path=""):
    prefix = "/" + base_path.strip("/") if base_path.strip("/") else ""
    return [
        Route(prefix + "/exports/file/{token}", download_file, methods=["GET", "HEAD"]),
        Route(prefix + "/exports/bundle/{token}", download_bundle, methods=["GET"]),
    ]


def export_url(directory, filenames, bundle=False, base_path=""):
    token = EXPORTS.issue(directory, filenames)
    prefix = "/" + base_path.strip("/") if base_path.strip("/") else ""
    return f"{prefix}/exports/{'bundle' if bundle else 'file'}/{token}"
