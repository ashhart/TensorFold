"""CPU-only image validation, canonicalization, and mocked HTTP boundary checks."""

from __future__ import annotations

import base64
import dataclasses
import io
import socket
import threading
import time
from urllib.parse import quote_from_bytes

import pytest

from tensorfold.vision import images, images_http
from tensorfold.vision.images import ImageInput, ImageInputError, ImageLimits, ImageSource, load_images, split_images

Image = pytest.importorskip("PIL.Image")


def encoded(mode="RGB", size=(2, 3), color=(10, 20, 30), format="PNG", **kwargs):
    output = io.BytesIO()
    Image.new(mode, size, color).save(output, format=format, **kwargs)
    return output.getvalue()


def data_url(data=None, media="image/png"):
    data = encoded() if data is None else data
    return f"data:{media};base64," + base64.b64encode(data).decode("ascii")


def limits(**kwargs):
    return dataclasses.replace(images.DEFAULT_LIMITS, **kwargs)


def test_split_preserves_order_detail_metadata_and_caller():
    first, second = data_url(), data_url(encoded(color=(30, 20, 10)))
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "name": "client", "content": [
            {"type": "text", "text": "before"},
            {"type": "image_url", "image_url": {"url": first, "detail": "high"}},
            {"type": "text", "text": "between"},
            {"type": "image_url", "image_url": {"url": second}},
            {"type": "text", "text": "after"},
        ]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call"}]},
    ]
    template, sources = split_images(messages)
    assert sources == [ImageSource(first, "high"), ImageSource(second)]
    assert template[1]["content"] == [
        {"type": "text", "text": "before"}, {"type": "image", "detail": "high"},
        {"type": "text", "text": "between"}, {"type": "image", "detail": "auto"},
        {"type": "text", "text": "after"},
    ]
    assert template[1]["name"] == "client"
    assert template[2] == messages[2]
    template[1]["content"][0]["text"] = "changed"
    assert messages[1]["content"][0]["text"] == "before"
    assert messages[1]["content"][1]["type"] == "image_url"


@pytest.mark.parametrize("role", ["system", "developer", "assistant", "tool"])
def test_images_require_user_role(role):
    with pytest.raises(ImageInputError, match="only in user"):
        split_images([{"role": role, "content": [{"type": "image_url", "image_url": {"url": data_url()}}]}])


@pytest.mark.parametrize("part", [
    {"type": "image_url", "image_url": "url"},
    {"type": "image_url", "image_url": {"url": ""}},
    {"type": "image_url", "image_url": {"url": "file:///image.png"}},
    {"type": "image_url", "image_url": {"url": "https://example.com/image", "detail": []}},
    {"type": "text", "text": 1},
    {"type": "text", "text": "x", "input_audio": {"data": "x"}},
    {"type": "input_audio", "input_audio": {"data": "x"}},
    {"type": "video_url", "video_url": {"url": "https://example.com/video"}},
    1,
])
def test_rejects_malformed_and_unsupported_parts(part):
    with pytest.raises(ImageInputError):
        split_images([{"role": "user", "content": [part]}])


def test_count_limit_precedes_loading(monkeypatch):
    monkeypatch.setattr(images, "_pillow", lambda: pytest.fail("should not decode"))
    parts = [{"type": "image_url", "image_url": {"url": data_url()}}] * 2
    with pytest.raises(ImageInputError, match="at most 1"):
        split_images([{"role": "user", "content": parts}], limits=limits(max_images=1))
    with pytest.raises(ImageInputError, match="at most 1"):
        load_images([ImageSource(data_url())] * 2, limits=limits(max_images=1))


@pytest.mark.parametrize("kwargs", [{"max_images": True}, {"max_redirects": False}, {"max_pixels": 1.5},
                                     {"timeout_seconds": float("inf")}, {"timeout_seconds": float("nan")},
                                     {"max_encoded_bytes": 0}])
def test_limit_configuration_requires_finite_positive_values(kwargs):
    with pytest.raises(ValueError):
        ImageLimits(**kwargs)


def test_hash_uses_pixels_dimensions_and_ignores_container_and_metadata():
    png = encoded(pnginfo=None)
    bmp = encoded(format="BMP")
    first, second = load_images([ImageSource(data_url(png)), ImageSource(data_url(bmp), "high")])
    assert first.content_hash == second.content_hash
    assert first.detail == "auto" and second.detail == "high"
    assert first.pixels == bytes([10, 20, 30]) * 6
    assert first.content_hash != ImageInput(3, 2, first.pixels).content_hash
    assert first.content_hash != load_images([ImageSource(data_url(encoded(color=(11, 20, 30))))])[0].content_hash


def test_decoded_pixels_are_immutable_and_pil_copies_are_independent():
    result = load_images([ImageSource(data_url())])[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.width = 10
    copy = result.to_pil()
    copy.putpixel((0, 0), (255, 255, 255))
    assert result.to_pil().getpixel((0, 0)) == (10, 20, 30)
    with pytest.raises(ImageInputError):
        ImageInput(1, 1, bytearray([0, 0, 0]))


def test_sniffs_bytes_instead_of_trusting_mime():
    result = load_images([ImageSource(data_url(encoded(), "text/plain"))])[0]
    assert (result.width, result.height) == (2, 3)
    with pytest.raises(ImageInputError, match="invalid or unsupported"):
        load_images([ImageSource(data_url(b"not an image", "image/png"))])


def test_percent_encoded_data_url():
    result = load_images([ImageSource("data:image/png," + quote_from_bytes(encoded()))])[0]
    assert result.pixels == bytes([10, 20, 30]) * 6


@pytest.mark.parametrize("url", ["data:image/png;base64,%%%", "data:image/png;base64", "data:image/png;base64,",
                                  "data:image/png;invalid,abc", "data:image/png;base64,é", "data:image/png,é"])
def test_invalid_data_urls(url):
    with pytest.raises(ImageInputError):
        load_images([ImageSource(url)])


def test_encoded_limits_apply_per_image_and_to_request():
    data = encoded()
    with pytest.raises(ImageInputError):
        load_images([ImageSource(data_url(data))], limits=limits(max_encoded_bytes=len(data) - 1))
    with pytest.raises(ImageInputError):
        load_images([ImageSource(data_url(data))] * 2, limits=limits(max_total_encoded_bytes=len(data)))
    assert len(load_images([ImageSource(data_url(data))], limits=limits(max_encoded_bytes=len(data)))) == 1


@pytest.mark.parametrize("bounds", [{"max_dimension": 2}, {"max_pixels": 5}, {"max_total_pixels": 5}])
def test_dimension_limit_precedes_pixel_decode(monkeypatch, bounds):
    url = data_url()
    monkeypatch.setattr(Image.Image, "load", lambda self: pytest.fail("should reject before decoding pixels"))
    with pytest.raises(ImageInputError, match="pixel limit"):
        load_images([ImageSource(url)], limits=limits(**bounds))


def test_total_pixel_limit():
    with pytest.raises(ImageInputError, match="pixel limit"):
        load_images([ImageSource(data_url())] * 2, limits=limits(max_total_pixels=11))


def test_exif_orientation_is_applied_before_hash():
    source = Image.new("RGB", (2, 3))
    source.putdata([(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15), (16, 17, 18)])
    exif = source.getexif()
    exif[274] = 6
    output = io.BytesIO()
    source.save(output, format="PNG", exif=exif)
    result = load_images([ImageSource(data_url(output.getvalue()))])[0]
    expected = source.transpose(Image.Transpose.ROTATE_270)
    assert (result.width, result.height) == expected.size
    assert result.pixels == expected.tobytes()
    assert result.content_hash == ImageInput(expected.width, expected.height, expected.tobytes()).content_hash


def test_alpha_is_composited_on_white_and_grayscale_becomes_rgb():
    transparent, grayscale = load_images([
        ImageSource(data_url(encoded("RGBA", (1, 1), (10, 20, 30, 0)))),
        ImageSource(data_url(encoded("L", (1, 1), 37))),
    ])
    assert transparent.pixels == bytes([255, 255, 255])
    assert grayscale.pixels == bytes([37, 37, 37])


def test_animation_is_rejected():
    output = io.BytesIO()
    first, second = Image.new("RGB", (2, 2), "red"), Image.new("RGB", (2, 2), "blue")
    first.save(output, format="GIF", save_all=True, append_images=[second])
    with pytest.raises(ImageInputError, match="single frame"):
        load_images([ImageSource(data_url(output.getvalue()))])


def test_pillow_missing_error_is_actionable(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImageInputError, match=r"install tensorfold\[vision\]"):
        load_images([ImageSource(data_url())])


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.1.1", "169.254.169.254",
                                       "100.100.100.200", "168.63.129.16", "192.0.0.192",
                                       "0.0.0.0", "224.0.0.1", "255.255.255.255", "::1",
                                       "::", "fe80::1", "fc00::1", "ff02::1", "::ffff:127.0.0.1",
                                       "2002:7f00:1::", "2001:db8::1"])
def test_private_or_special_addresses_are_rejected(address):
    assert not images_http._public_ip(address)


def test_dns_rejects_any_private_answer(monkeypatch):
    answers = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))
               for address in ("1.1.1.1", "127.0.0.1")]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: answers)
    with pytest.raises(ImageInputError, match="only to public"):
        images_http._resolve("example.com", 443, time.monotonic() + 1)


@pytest.mark.parametrize("url", ["file:///image.png", "ftp://example.com/image", "http://user:pass@example.com/image",
                                  "http://example.com/\nimage", "http://example.com\\@localhost/image",
                                  "http://[fe80::1%en0]/image", "http://localhost/image", "http://example.com:0/image"])
def test_remote_url_rejects_unsafe_syntax(url):
    with pytest.raises(ImageInputError):
        images_http._url(url, 4096)


def test_redirect_is_resolved_and_checked_again(monkeypatch):
    hosts = []

    def dns(host, port, **kwargs):
        hosts.append(host)
        address = "1.1.1.1" if len(hosts) == 1 else "169.254.169.254"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    monkeypatch.setattr(socket, "getaddrinfo", dns)
    monkeypatch.setattr(images_http, "_request", lambda *args: (None, "http://metadata.example/image"))
    with pytest.raises(ImageInputError, match="only to public"):
        images_http.fetch_image("http://example.com/image", max_bytes=100, deadline=time.monotonic() + 1,
                                max_redirects=3, max_url_chars=4096)
    assert hosts == ["example.com", "metadata.example"]


def test_redirect_cannot_access_filesystem(monkeypatch):
    monkeypatch.setattr(images_http, "_resolve", lambda *args: [])
    monkeypatch.setattr(images_http, "_request", lambda *args: (None, "file:///image.png"))
    with pytest.raises(ImageInputError, match="HTTP"):
        images_http.fetch_image("http://example.com/image", max_bytes=100, deadline=time.monotonic() + 1,
                                max_redirects=3, max_url_chars=4096)


class FakeSocket:
    def __init__(self, *args):
        self.connected = None
        self.closed = False

    def settimeout(self, value):
        assert value > 0

    def connect(self, address):
        self.connected = address

    def shutdown(self, how):
        pass

    def close(self):
        self.closed = True


class FakeResponse:
    status = 200

    def __init__(self, body, headers=None):
        self.body = io.BytesIO(body)
        self.headers = headers or {}

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read1(self, size):
        return self.body.read(size)


def mock_http(monkeypatch, response):
    sock = FakeSocket()
    requests = []

    class Connection:
        def __init__(self, host, port, timeout):
            self.sock = None

        def request(self, *args, **kwargs):
            assert self.sock is sock
            requests.append((args, kwargs))

        def getresponse(self):
            return response

        def close(self):
            pass

    monkeypatch.setattr(socket, "socket", lambda *args: sock)
    monkeypatch.setattr(images_http.http.client, "HTTPConnection", Connection)
    return sock, requests


def test_connection_uses_checked_ip_without_second_dns_lookup(monkeypatch):
    dns_calls = []

    def dns(*args, **kwargs):
        dns_calls.append(args)
        address = "1.1.1.1" if len(dns_calls) == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 80))]

    monkeypatch.setattr(socket, "getaddrinfo", dns)
    sock, requests = mock_http(monkeypatch, FakeResponse(b"image"))
    result = images_http.fetch_image("http://example.com/image", max_bytes=10, deadline=time.monotonic() + 1,
                                    max_redirects=3, max_url_chars=4096)
    assert result == b"image" and len(dns_calls) == 1
    assert sock.connected == ("1.1.1.1", 80) and sock.closed
    assert requests[0][0] == ("GET", "/image")


@pytest.mark.parametrize("body,headers", [(b"123456", {}), (b"", {"Content-Length": "6"}),
                                          (b"", {"Content-Encoding": "gzip"})])
def test_remote_body_is_bounded(monkeypatch, body, headers):
    sock, _ = mock_http(monkeypatch, FakeResponse(body, headers))
    addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 80))]
    with pytest.raises(ImageInputError):
        images_http._request("http", "example.com", 80, "/", addresses, 5, time.monotonic() + 1)
    assert sock.closed


def test_expired_download_does_not_reach_network(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: pytest.fail("no DNS after deadline"))
    with pytest.raises(ImageInputError, match="timed out"):
        images_http.fetch_image("http://example.com/image", max_bytes=10, deadline=time.monotonic() - 1,
                                max_redirects=3, max_url_chars=4096)


def test_dns_wait_is_bounded(monkeypatch):
    released = threading.Event()
    finished = threading.Event()

    def dns(*args, **kwargs):
        try:
            released.wait(1)
            return []
        finally:
            finished.set()

    monkeypatch.setattr(socket, "getaddrinfo", dns)
    try:
        with pytest.raises(ImageInputError, match="DNS lookup timed out"):
            images_http._resolve("example.com", 80, time.monotonic() + 0.02)
    finally:
        released.set()
        assert finished.wait(1)


def test_dns_worker_count_is_bounded(monkeypatch):
    slots = threading.BoundedSemaphore(1)
    assert slots.acquire(blocking=False)
    monkeypatch.setattr(images_http, "_DNS_SLOTS", slots)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: pytest.fail("resolver capacity is full"))
    with pytest.raises(ImageInputError, match="resolver is busy"):
        images_http._resolve("example.com", 80, time.monotonic() + 1)


def test_slow_headers_are_interrupted_by_total_timeout(monkeypatch):
    response = FakeResponse(b"image")
    sock, _ = mock_http(monkeypatch, response)
    stopped = threading.Event()
    sock.shutdown = lambda how: stopped.set()
    connection_type = images_http.http.client.HTTPConnection

    def slow_response(self):
        assert stopped.wait(1), "watchdog did not interrupt headers"
        return response

    monkeypatch.setattr(connection_type, "getresponse", slow_response)
    addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 80))]
    with pytest.raises(ImageInputError, match="timed out"):
        images_http._request("http", "example.com", 80, "/", addresses, 10, time.monotonic() + 0.02)
    assert sock.closed


def test_https_verifies_original_hostname_on_pinned_socket(monkeypatch):
    sock, _ = mock_http(monkeypatch, FakeResponse(b"image"))
    handshake = []
    sock.do_handshake = lambda: handshake.append(True)
    names = []

    class Context:
        def wrap_socket(self, raw, *, server_hostname, do_handshake_on_connect):
            assert raw is sock and not do_handshake_on_connect
            names.append(server_hostname)
            return sock

    monkeypatch.setattr(images_http.ssl, "create_default_context", Context)
    addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))]
    data, redirect = images_http._request("https", "example.com", 443, "/", addresses, 10, time.monotonic() + 1)
    assert data == b"image" and redirect is None
    assert names == ["example.com"] and handshake == [True]
    assert sock.connected == ("1.1.1.1", 443)


def test_redirect_count_is_bounded(monkeypatch):
    monkeypatch.setattr(images_http, "_resolve", lambda *args: [])
    monkeypatch.setattr(images_http, "_request", lambda *args: (None, "/image"))
    with pytest.raises(ImageInputError, match="too many redirects"):
        images_http.fetch_image("http://example.com/image", max_bytes=100, deadline=time.monotonic() + 1,
                                max_redirects=0, max_url_chars=4096)


def test_remote_input_uses_same_decoder_and_hash(monkeypatch):
    data = encoded()
    monkeypatch.setattr(images, "fetch_image", lambda *args, **kwargs: data)
    remote, inline = load_images([ImageSource("https://example.com/image"), ImageSource(data_url(data))], allow_urls=True)
    assert remote == inline


def test_remote_urls_are_off_by_default(monkeypatch):
    monkeypatch.setattr(images, "fetch_image", lambda *args, **kwargs: pytest.fail("fetched with URLs off"))
    part = {"type": "image_url", "image_url": {"url": "https://example.com/image"}}
    with pytest.raises(ImageInputError, match="--vision-urls"):
        split_images([{"role": "user", "content": [part]}])
    with pytest.raises(ImageInputError, match="--vision-urls"):
        load_images([ImageSource("https://example.com/image")])
    _, sources = split_images([{"role": "user", "content": [part]}], allow_urls=True)
    assert sources == [ImageSource("https://example.com/image")]
