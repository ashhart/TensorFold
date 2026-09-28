import base64
import copy
import io
import socket

import pytest
from PIL import Image

from tensorfold.server.errors import RequestError
from tensorfold.server.images import ImageLimits, load_image_url
from tensorfold.server.messages import normalize_messages, validate_modalities


def png_data_url(width=2, height=2):
    image = Image.new("RGB", (width, height), (12, 34, 56))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def image_messages(url=None):
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url or png_data_url()}},
        {"type": "text", "text": "Describe it."},
    ]}]


def test_one_user_image_is_preserved_without_mutating_input():
    messages = image_messages()
    original = copy.deepcopy(messages)
    normalized = normalize_messages(messages, allow_images=True)
    assert normalized == messages
    assert messages == original


@pytest.mark.parametrize("messages, error", [
    ([{"role": "system", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]}], "user"),
    ([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/b.png"}},
    ]}], "one image"),
    ([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": 123}},
    ]}], "url"),
    ([{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
    ]}], "audio"),
])
def test_image_contract_rejects_unsupported_shapes(messages, error):
    with pytest.raises(RequestError, match=error):
        normalize_messages(messages, allow_images=True)


def test_text_only_apps_still_reject_image_parts():
    with pytest.raises(RequestError, match="text"):
        normalize_messages(image_messages(), allow_images=False)


def test_top_level_modalities_are_capability_aware():
    validate_modalities({"modalities": ["text"]}, allow_images=True)
    with pytest.raises(RequestError, match="output"):
        validate_modalities({"modalities": ["text", "image"]}, allow_images=True)
    with pytest.raises(RequestError):
        validate_modalities({"modalities": ["text", "image"]}, allow_images=False)
    with pytest.raises(RequestError, match="audio"):
        validate_modalities({"modalities": ["audio"]}, allow_images=True)


def test_data_url_decodes_to_deterministic_rgb_and_digest():
    prepared = load_image_url(png_data_url())
    assert prepared.image.mode == "RGB"
    assert prepared.image.size == (2, 2)
    assert prepared.media_type == "image/png"
    assert len(prepared.sha256) == 64
    assert prepared.source == "data:image/png;base64,<redacted>"


def test_data_url_enforces_encoded_and_pixel_limits():
    with pytest.raises(RequestError, match="bytes"):
        load_image_url(png_data_url(), limits=ImageLimits(max_bytes=8))
    with pytest.raises(RequestError, match="pixels"):
        load_image_url(png_data_url(10, 10), limits=ImageLimits(max_pixels=50))


@pytest.mark.parametrize("url", [
    "file:///tmp/picture.png",
    "http://example.com/picture.png",
    "https://example.com:444/picture.png",
    "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
    "data:image/png;base64,not-base64",
])
def test_unsafe_or_invalid_image_sources_are_rejected(url):
    with pytest.raises(RequestError):
        load_image_url(url)


def test_https_private_network_target_is_rejected(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 443)),
    ])
    with pytest.raises(RequestError, match="private"):
        load_image_url("https://example.com/picture.png")
