"""CPU image inputs canonicalized to immutable, oriented RGB pixels."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import math
import struct
import time
import warnings
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote_to_bytes

from .images_http import MEDIA_TYPES, ImageInputError, fetch_image

_DETAILS = {"auto", "low", "high"}
_FORMATS = tuple(MEDIA_TYPES.values())
_MEDIA = {"image", "images", "image_url", "input_image", "audio", "input_audio", "video", "video_url"}


@dataclass(frozen=True, slots=True)
class ImageLimits:
    max_images: int = 4
    max_encoded_bytes: int = 10 * 1024 * 1024
    max_total_encoded_bytes: int = 20 * 1024 * 1024
    max_dimension: int = 8192
    max_pixels: int = 16 * 1024 * 1024
    max_total_pixels: int = 32 * 1024 * 1024
    timeout_seconds: float = 10.0
    total_timeout_seconds: float = 30.0
    max_redirects: int = 3
    max_url_chars: int = 4096
    max_visual_tokens: int = 4096        # the tokens a request's images share (CUDA Qwen: --vision-image-tokens)

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value < 0 or (name != "max_redirects" and value == 0)):
                raise ValueError(f"{name} must be finite and positive, except zero redirects")
            if name not in {"timeout_seconds", "total_timeout_seconds"} and not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")


DEFAULT_LIMITS = ImageLimits()


def _count_error(limits: ImageLimits) -> ImageInputError:
    return ImageInputError(
        f"a request supports at most {limits.max_images} images across the full message history, "
        "including prior turns; remove older image content, start a new conversation, or restart the server "
        "with --vision-max-images N to raise the count limit (other image limits still apply)")


@dataclass(frozen=True, slots=True)
class ImageSource:
    url: str
    detail: str = "auto"


@dataclass(frozen=True, slots=True)
class ImageInput:
    width: int
    height: int
    pixels: bytes
    detail: str = "auto"
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (not isinstance(self.pixels, bytes)
                or any(type(size) is not int or not 0 < size < 2**32 for size in (self.width, self.height))
                or len(self.pixels) != self.width * self.height * 3
                or not isinstance(self.detail, str) or self.detail not in _DETAILS):
            raise ImageInputError("image input requires RGB bytes, matching dimensions and valid detail")
        digest = hashlib.sha256(b"tensorfold-rgb-v1\0" + struct.pack(">II", self.width, self.height))
        digest.update(self.pixels)
        object.__setattr__(self, "content_hash", digest.hexdigest())

    def to_pil(self):
        """Return a fresh mutable RGB image; cached pixels remain immutable."""
        image, _ = _pillow()
        return image.frombytes("RGB", (self.width, self.height), self.pixels)


def _pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError:
        raise ImageInputError("image inputs require Pillow; install tensorfold[vision]") from None
    return Image, ImageOps


def _source(value: Any, limits: ImageLimits, allow_urls: bool) -> ImageSource:
    if not isinstance(value, dict) or not isinstance(value.get("url"), str) or not value["url"]:
        raise ImageInputError("image_url must contain a non-empty url string")
    detail = value.get("detail", "auto")
    if not isinstance(detail, str) or detail not in _DETAILS:
        raise ImageInputError("image detail must be auto, low or high")
    source = ImageSource(value["url"], detail)
    _check_source(source, limits, allow_urls)
    return source


def _check_source(source: ImageSource, limits: ImageLimits, allow_urls: bool = False) -> None:
    if not isinstance(source, ImageSource) or not isinstance(source.url, str) or not source.url:
        raise ImageInputError("image source must contain a non-empty URL")
    if not isinstance(source.detail, str) or source.detail not in _DETAILS:
        raise ImageInputError("image detail must be auto, low or high")
    if source.url.startswith("data:"):
        if len(source.url) > limits.max_encoded_bytes * 3 + 256:
            raise ImageInputError("image data URL exceeds the encoded byte limit")
    elif not source.url.startswith("https://"):
        raise ImageInputError("images require data URLs or public HTTPS URLs")
    elif not allow_urls:
        raise ImageInputError("image URLs are off on this server; send the image as a data URL, or start the server "
                              "with --vision-urls")
    elif len(source.url) > limits.max_url_chars:
        raise ImageInputError("image URL is too long")


def split_images(messages: list[dict[str, Any]], *, limits: ImageLimits = DEFAULT_LIMITS, allow_urls: bool = False
                 ) -> tuple[list[dict[str, Any]], list[ImageSource]]:
    """Preserve ordered parts, replacing user and tool-result image URLs with processor image markers."""
    if not isinstance(messages, list) or not messages:
        raise ImageInputError("messages must be a non-empty list")
    output, sources = [], []
    for message in messages:
        if not isinstance(message, dict):
            raise ImageInputError("each message must be an object")
        role = message.get("role")
        if not isinstance(role, str) or role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ImageInputError("invalid message role")
        if any(message.get(key) for key in _MEDIA):
            raise ImageInputError("images must be image_url parts in user or tool message content")
        content = message.get("content")
        if content is None or isinstance(content, str):
            output.append(dict(message))
            continue
        if not isinstance(content, list):
            raise ImageInputError("message content must be text or an array of content parts")
        parts = []
        for part in content:
            if not isinstance(part, dict):
                raise ImageInputError("each content part must be an object")
            kind = part.get("type")
            if kind == "text":
                if not isinstance(part.get("text"), str) or any(part.get(key) for key in _MEDIA):
                    raise ImageInputError("text parts must contain a text string without media")
                parts.append(dict(part))
            elif kind == "image_url":
                if role not in ("user", "tool"):
                    raise ImageInputError("image_url parts are supported only in user and tool messages")
                if any(part.get(key) for key in _MEDIA - {"image_url"}):
                    raise ImageInputError("image_url parts cannot contain other media")
                if len(sources) >= limits.max_images:
                    raise _count_error(limits)
                source = _source(part.get("image_url"), limits, allow_urls)
                sources.append(source)
                parts.append({"type": "image", "detail": source.detail})
            else:
                raise ImageInputError("content parts must be text or image_url; audio and video are unsupported")
        output.append({**message, "content": parts})
    return output, sources


def _data_bytes(url: str, max_bytes: int) -> bytes:
    header, separator, payload = url.partition(",")
    if not separator or len(header) > 256:
        raise ImageInputError("invalid image data URL")
    media = header[5:].split(";")
    if len(media) > 2 or (len(media) == 2 and media[1].lower() != "base64"):
        raise ImageInputError("image data URL supports only optional base64 encoding")
    max_chars = 4 * ((max_bytes + 2) // 3) if len(media) == 2 else max_bytes * 3
    if len(payload) > max_chars:
        raise ImageInputError("image data URL exceeds the encoded byte limit")
    if not payload.isascii():
        raise ImageInputError("invalid image data URL encoding")
    try:
        if len(media) == 2:
            data = base64.b64decode(payload, validate=True)
        else:
            data = unquote_to_bytes(payload)
    except (ValueError, binascii.Error):
        raise ImageInputError("invalid image data URL encoding") from None
    if not data or len(data) > max_bytes:
        raise ImageInputError("image is empty or exceeds the encoded byte limit")
    return data


def _decode(data: bytes, detail: str, limits: ImageLimits, remaining_pixels: int) -> ImageInput:
    image, image_ops = _pillow()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", image.DecompressionBombWarning)
            with image.open(io.BytesIO(data), formats=_FORMATS) as source:
                width, height = source.size
                if (min(width, height) <= 0 or max(width, height) > limits.max_dimension
                        or width * height > min(limits.max_pixels, remaining_pixels)):
                    raise ImageInputError("image dimensions exceed the decoded pixel limit")
                if getattr(source, "n_frames", 1) != 1:
                    raise ImageInputError("animated and multipage images are unsupported; send a single frame")
                source.verify()
            with image.open(io.BytesIO(data), formats=_FORMATS) as source:
                source.load()
                oriented = image_ops.exif_transpose(source)
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    rgba = oriented.convert("RGBA")
                    rgb = image.new("RGB", oriented.size, "white")
                    rgb.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    rgb = oriented.convert("RGB")
                return ImageInput(rgb.width, rgb.height, rgb.tobytes(), detail)
    except ImageInputError:
        raise
    except (OSError, ValueError, SyntaxError, image.DecompressionBombWarning, image.DecompressionBombError):
        raise ImageInputError("image bytes are invalid or unsupported; use JPEG, PNG or WebP") from None


def load_images(sources: list[ImageSource], *, limits: ImageLimits = DEFAULT_LIMITS, allow_urls: bool = False
                ) -> list[ImageInput]:
    """Bound encoded bytes and decoded pixels across all images in one request."""
    if not isinstance(sources, (list, tuple)) or len(sources) > limits.max_images:
        raise _count_error(limits)
    total_bytes, total_pixels = 0, 0
    deadline = time.monotonic() + limits.total_timeout_seconds
    output = []
    for source in sources:
        _check_source(source, limits, allow_urls)
        remaining = min(limits.max_encoded_bytes, limits.max_total_encoded_bytes - total_bytes)
        if remaining <= 0 or time.monotonic() >= deadline:
            raise ImageInputError("image request exceeds the total byte or time limit")
        if source.url.startswith("data:"):
            data = _data_bytes(source.url, remaining)
        else:
            data, _ = fetch_image(source.url, max_bytes=remaining,
                                  deadline=min(deadline, time.monotonic() + limits.timeout_seconds),
                                  max_redirects=limits.max_redirects, max_url_chars=limits.max_url_chars)
        total_bytes += len(data)
        decoded = _decode(data, source.detail, limits, limits.max_total_pixels - total_pixels)
        if time.monotonic() >= deadline:
            raise ImageInputError("image request exceeds the total time limit")
        total_pixels += decoded.width * decoded.height
        output.append(decoded)
    return output
