"""Bounded HTTPS image retrieval with validated, pinned public addresses."""

from __future__ import annotations

import http.client
import ipaddress
import queue
import socket
import ssl
import threading
import time
from urllib.parse import quote, urljoin, urlsplit

from tensorfold import __version__

MEDIA_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}     # a declared type, its format
_DNS_SLOTS = threading.BoundedSemaphore(4)
_REDIRECTS = {301, 302, 303, 307, 308}
_SPECIAL_V4 = (ipaddress.ip_network("192.0.0.0/24"), ipaddress.ip_network("168.63.129.16/32"))


class ImageInputError(ValueError):
    """An invalid, inaccessible, or oversized image input."""


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ImageInputError("image download timed out")
    return remaining


def _public_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if not address.is_global or address.is_multicast or address.is_reserved:
        return False
    if isinstance(address, ipaddress.IPv4Address) and any(address in block for block in _SPECIAL_V4):
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped or address.sixtofour or address.teredo:
            return False
    return True


def _resolve(host: str, port: int, deadline: float) -> list[tuple]:
    """Bound stalled DNS workers as well as each caller's wait."""
    _remaining(deadline)
    slots = _DNS_SLOTS
    if not slots.acquire(blocking=False):
        raise ImageInputError("image DNS resolver is busy; retry the request")
    result: queue.Queue = queue.Queue(maxsize=1)

    def resolve() -> None:
        try:
            result.put(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except Exception as exc:
            result.put(exc)
        finally:
            slots.release()

    try:
        threading.Thread(target=resolve, daemon=True).start()
    except RuntimeError:
        slots.release()
        raise ImageInputError("image DNS resolver is unavailable") from None
    try:
        addresses = result.get(timeout=_remaining(deadline))
    except queue.Empty:
        raise ImageInputError("image DNS lookup timed out") from None
    if isinstance(addresses, Exception):
        raise ImageInputError("image host could not be resolved") from None
    if not addresses or any(not _public_ip(item[4][0]) for item in addresses):
        raise ImageInputError("image URLs must resolve only to public internet addresses")
    return addresses


def _url(value: str, max_url_chars: int) -> tuple[str, str, int, str]:
    if len(value) > max_url_chars or any(ord(char) <= 32 or ord(char) == 127 for char in value):
        raise ImageInputError("image URL is too long or contains whitespace/control characters")
    try:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.fragment:
            raise ValueError
        if parsed.username is not None or parsed.password is not None or "\\" in value:
            raise ValueError
        host = parsed.hostname.encode("idna").decode("ascii")
        port = parsed.port if parsed.port is not None else 443
        if "%" in host or port != 443:
            raise ValueError
    except (ValueError, UnicodeError):
        raise ImageInputError("image URL must be HTTPS on port 443, without credentials or a fragment") from None
    if host.rstrip(".").lower() in {"localhost", "metadata.google.internal", "instance-data"}:
        raise ImageInputError("image URLs must use public internet hosts")
    target = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
    if parsed.query:
        target += "?" + quote(parsed.query, safe="/%?:@!$&'()*+,;=-._~")
    return host, port, target


def _close_socket(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()


def _request(host: str, port: int, target: str, addresses: list[tuple],
             max_bytes: int, deadline: float, media_types=MEDIA_TYPES) -> tuple[bytes | None, str | None, str | None]:
    """A watchdog bounds slow headers and TLS handshakes, not just individual reads."""
    timeout = _remaining(deadline)
    family, kind, protocol, _, address = addresses[0]
    sock = socket.socket(family, kind, protocol)
    live_socket = [sock]
    timer = threading.Timer(timeout, lambda: _close_socket(live_socket[0]))
    timer.daemon = True
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        timer.start()
        sock.settimeout(_remaining(deadline))
        sock.connect(address)
        context = ssl.create_default_context()        # TLS verified for the host, on the address checked for it
        sock = context.wrap_socket(sock, server_hostname=host, do_handshake_on_connect=False)
        live_socket[0] = sock
        sock.settimeout(_remaining(deadline))
        sock.do_handshake()
        connection.sock = sock
        connection.request("GET", target, headers={"Accept": ", ".join(media_types), "Accept-Encoding": "identity",
                                                   "User-Agent": f"TensorFold/{__version__}"})
        response = connection.getresponse()
        _remaining(deadline)
        if response.status in _REDIRECTS:
            location = response.getheader("Location")
            if not location:
                raise ImageInputError("image redirect has no destination")
            return None, location, None
        if response.status != 200:
            raise ImageInputError(f"image download returned HTTP {response.status}")
        if response.getheader("Content-Encoding", "identity").lower() != "identity":
            raise ImageInputError("compressed HTTP image responses are unsupported")
        media = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
        if media not in media_types:
            raise ImageInputError("image URL content type must be JPEG, PNG or WebP" if media_types is MEDIA_TYPES
                                  else f"media URL content type must be one of {', '.join(media_types)}")
        length = response.getheader("Content-Length")
        if length is not None:
            if not length.isascii() or not length.isdecimal() or int(length) > max_bytes:
                raise ImageInputError("image response exceeds the encoded byte limit")
        data = bytearray()
        while len(data) <= max_bytes:
            sock.settimeout(_remaining(deadline))
            chunk = response.read1(min(64 * 1024, max_bytes + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > max_bytes:
            raise ImageInputError("image response exceeds the encoded byte limit")
        _remaining(deadline)
        return bytes(data), None, media
    finally:
        timer.cancel()
        connection.close()
        _close_socket(sock)


def fetch_image(url: str, *, max_bytes: int, deadline: float, max_redirects: int,
                max_url_chars: int, media_types=MEDIA_TYPES) -> tuple[bytes, str]:
    """Resolve and validate each redirect, then connect directly to its checked address."""
    try:
        for redirect in range(max_redirects + 1):
            host, port, target = _url(url, max_url_chars)
            addresses = _resolve(host, port, deadline)
            data, location, media = _request(host, port, target, addresses, max_bytes, deadline, media_types)
            if location is None:
                return data, media
            if redirect == max_redirects:
                raise ImageInputError("image download has too many redirects")
            url = urljoin(url, location)
    except ImageInputError:
        raise
    except (OSError, ValueError, http.client.HTTPException):
        raise ImageInputError("image download failed or timed out") from None
    raise ImageInputError("image download failed")
