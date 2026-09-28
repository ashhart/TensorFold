"""Bounded image acquisition for OpenAI-compatible image_url parts."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import http.client
import io
import ipaddress
import queue
import socket
import ssl
import threading
import time
from typing import Any
from urllib.parse import urljoin, urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError

from tensorfold.server.errors import RequestError


_ALLOWED_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
_DNS_SLOTS = threading.BoundedSemaphore(8)


@dataclass(frozen=True)
class ImageLimits:
    max_bytes: int = 20 * 1024**2
    max_pixels: int = 16_000_000
    timeout_s: float = 10.0
    max_redirects: int = 3


@dataclass(frozen=True)
class PreparedImage:
    image: Image.Image
    sha256: str
    media_type: str
    source: str
    encoded_bytes: int


def _public_https(url: str, timeout: float) -> tuple[Any, str]:
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise RequestError("image_url must use HTTPS or a data:image URL")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise RequestError("image_url must not contain credentials or fragments")
    if parsed.port not in (None, 443):
        raise RequestError("image_url must use the standard HTTPS port 443")
    result: queue.Queue[Any] = queue.Queue(maxsize=1)
    resolve_deadline = time.monotonic() + timeout
    if not _DNS_SLOTS.acquire(timeout=max(0.001, timeout)):
        raise RequestError("image_url DNS resolver is busy")

    def resolve() -> None:
        try:
            try:
                result.put(
                    socket.getaddrinfo(
                        parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM
                    )
                )
            except OSError as exc:
                result.put(exc)
        finally:
            _DNS_SLOTS.release()

    try:
        threading.Thread(target=resolve, name="tensorfold-image-dns", daemon=True).start()
    except BaseException:
        _DNS_SLOTS.release()
        raise
    try:
        addresses = result.get(
            timeout=max(0.001, resolve_deadline - time.monotonic())
        )
    except queue.Empty as exc:
        raise RequestError("image_url host resolution timed out") from exc
    if isinstance(addresses, OSError):
        exc = addresses
        raise RequestError("image_url host could not be resolved") from exc
    public: list[str] = []
    for entry in addresses:
        address = ipaddress.ip_address(entry[4][0])
        if not address.is_global:
            raise RequestError("image_url resolves to a private or non-public address")
        public.append(str(address))
    return parsed, public[0]


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Verify TLS for the requested host while connecting only to a vetted IP."""

    def __init__(self, hostname: str, address: str, port: int, timeout: float) -> None:
        super().__init__(
            hostname,
            port=port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._address = address

    def connect(self) -> None:
        raw = socket.create_connection(
            (self._address, self.port), self.timeout, self.source_address
        )
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


def _read_https(url: str, limits: ImageLimits) -> tuple[bytes, str, str]:
    current = url
    deadline = time.monotonic() + limits.timeout_s
    for redirect in range(limits.max_redirects + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RequestError("image_url fetch timed out")
        parsed, address = _public_https(current, remaining)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RequestError("image_url fetch timed out")
        connection = _PinnedHTTPSConnection(
            parsed.hostname, address, parsed.port or 443, remaining
        )
        target = parsed.path or "/"
        if parsed.query:
            target += f"?{parsed.query}"
        try:
            connection.request(
                "GET",
                target,
                headers={
                    "Accept": ", ".join(_ALLOWED_TYPES),
                    "Host": parsed.netloc,
                    "User-Agent": "TensorFold/vision",
                },
            )
            response = connection.getresponse()
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            connection.close()
            raise RequestError("image_url could not be fetched") from exc
        if response.status in {301, 302, 303, 307, 308}:
            if redirect >= limits.max_redirects:
                connection.close()
                raise RequestError("image_url redirected too many times")
            location = response.getheader("Location")
            connection.close()
            if not location:
                raise RequestError("image_url redirect has no destination")
            current = urljoin(current, location)
            continue
        if not 200 <= response.status < 300:
            status = response.status
            connection.close()
            raise RequestError(f"image_url returned HTTP {status}")
        media_type = (
            response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
        )
        if media_type not in _ALLOWED_TYPES:
            connection.close()
            raise RequestError("image_url content type must be JPEG, PNG or WebP")
        declared = response.getheader("Content-Length")
        try:
            if declared and int(declared) > limits.max_bytes:
                connection.close()
                raise RequestError(f"image exceeds the {limits.max_bytes} bytes limit")
        except ValueError as exc:
            connection.close()
            raise RequestError("image_url returned an invalid Content-Length") from exc
        chunks: list[bytes] = []
        size = 0
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RequestError("image_url fetch timed out")
                if connection.sock is not None:
                    connection.sock.settimeout(max(0.001, remaining))
                chunk = response.read1(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limits.max_bytes:
                    raise RequestError(f"image exceeds the {limits.max_bytes} bytes limit")
                chunks.append(chunk)
        finally:
            connection.close()
        return (
            b"".join(chunks),
            media_type,
            f"https://{parsed.hostname}/<redacted>",
        )
    raise RequestError("image_url redirected too many times")


def _read_data_url(url: str, limits: ImageLimits) -> tuple[bytes, str, str]:
    header, separator, payload = url.partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(";base64"):
        raise RequestError("image data URL must be base64 encoded")
    media_type = header[5:-7].lower()
    if media_type not in _ALLOWED_TYPES:
        raise RequestError("image data URL must contain JPEG, PNG or WebP")
    if len(payload) > (limits.max_bytes + 2) // 3 * 4:
        raise RequestError(f"image exceeds the {limits.max_bytes} bytes limit")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as exc:
        raise RequestError("image data URL contains invalid base64") from exc
    if len(raw) > limits.max_bytes:
        raise RequestError(f"image exceeds the {limits.max_bytes} bytes limit")
    return raw, media_type, f"data:{media_type};base64,<redacted>"


def _decode(raw: bytes, media_type: str, limits: ImageLimits) -> Image.Image:
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            if getattr(opened, "is_animated", False):
                raise RequestError("animated images are unsupported")
            if opened.format != _ALLOWED_TYPES[media_type]:
                raise RequestError("image bytes do not match their declared media type")
            width, height = opened.size
            if width < 1 or height < 1 or width * height > limits.max_pixels:
                raise RequestError(f"image exceeds the {limits.max_pixels} pixels limit")
            opened.load()
            return ImageOps.exif_transpose(opened).convert("RGB")
    except RequestError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise RequestError("image bytes could not be decoded") from exc


def load_image_url(url: str, *, limits: ImageLimits | None = None) -> PreparedImage:
    """Fetch and decode one image without exposing its original bytes in logs or errors."""

    if not isinstance(url, str) or not url:
        raise RequestError("image_url.url must be a non-empty string")
    limits = limits or ImageLimits()
    if url.startswith("data:"):
        raw, media_type, source = _read_data_url(url, limits)
    else:
        raw, media_type, source = _read_https(url, limits)
    return PreparedImage(
        image=_decode(raw, media_type, limits),
        sha256=hashlib.sha256(raw).hexdigest(),
        media_type=media_type,
        source=source,
        encoded_bytes=len(raw),
    )


def image_url_from_messages(messages: list[dict[str, Any]]) -> str | None:
    """Return the sole normalized image URL, if any."""

    found: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                found.append(part["image_url"]["url"])
    return found[0] if found else None
