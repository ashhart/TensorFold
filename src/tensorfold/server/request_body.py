"""Bounded HTTP/1.1 request framing, shared by the CUDA and MLX handlers."""

from __future__ import annotations

import re
from typing import Any

from tensorfold.server.errors import RequestError

LIMIT = 32 * 1024**2
_METADATA_LIMIT = 65536


def read_body(handler: Any, *, limit: int = LIMIT) -> bytes:
    """Read one fixed-length or chunked body without consuming the next request."""

    def refuse(message: str) -> None:
        # Once framing is uncertain, unread bytes must not become another request.
        handler.close_connection = True
        raise RequestError(message)

    def check_size(size: int) -> None:
        if size > limit:
            refuse(f"request body exceeds the {limit // 1024**2} MiB limit")

    def exact(size: int) -> bytes:
        data = handler.rfile.read(size)
        if len(data) != size:
            refuse("incomplete request body")
        return data

    def line() -> bytes:
        data = handler.rfile.readline(_METADATA_LIMIT + 1)
        if len(data) > _METADATA_LIMIT or not data.endswith(b"\r\n"):
            refuse("invalid or oversized chunked request framing")
        return data[:-2]

    transfers = handler.headers.get_all("Transfer-Encoding", [])
    lengths = handler.headers.get_all("Content-Length", [])
    if transfers:
        if lengths:
            refuse("Content-Length and Transfer-Encoding cannot be combined")
        codings = [value.strip().lower() for header in transfers for value in header.split(",")]
        if codings != ["chunked"] or getattr(handler, "request_version", "HTTP/1.1") != "HTTP/1.1":
            refuse("unsupported request Transfer-Encoding; expected chunked over HTTP/1.1")
        body = bytearray()
        extensions = 0
        while True:
            header = line()
            size_text, separator, extension = header.partition(b";")
            size_text = size_text.rstrip(b" \t") if separator else size_text
            if not re.fullmatch(rb"[0-9a-fA-F]+", size_text):
                refuse("invalid chunk size")
            extensions += len(extension)
            if extensions > _METADATA_LIMIT:
                refuse("chunk extensions exceed the request framing limit")
            size = int(size_text, 16)
            check_size(len(body) + size)
            if size == 0:
                trailers = 0
                while trailer := line():
                    trailers += len(trailer) + 2
                    if trailers > _METADATA_LIMIT or not re.match(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+:", trailer):
                        refuse("invalid or oversized request trailers")
                return bytes(body)
            body.extend(exact(size))
            if exact(2) != b"\r\n":
                refuse("invalid chunk terminator")
    if not lengths:
        return b""
    values = [value.strip() for header in lengths for value in header.split(",")]
    if any(not re.fullmatch(r"[0-9]+", value) for value in values):
        refuse("invalid Content-Length")
    try:
        sizes = [int(value) for value in values]
    except ValueError:
        refuse("invalid Content-Length")
    if len(set(sizes)) != 1:
        refuse("conflicting Content-Length values")
    check_size(sizes[0])
    return exact(sizes[0])
