"""Makers a person adds or completes from Settings: a model family's or a hardware maker's name, the words that
recognise it, its logo. The page's own list (`ui/src/brand.ts`) covers the known makers; an entry here comes first,
so a new family or a missing logo needs no new TensorFold Web. Kept in one JSON file, the logos inside it as data URLs.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
from pathlib import Path

KINDS = ("model", "hardware")
MAX_BRANDS = 100
MAX_WORDS = 12
WORD_CHARS = 40
LABEL_CHARS = 40
LOGO_BYTES = 150 * 1024
LOGO_TYPES = ("image/png", "image/jpeg", "image/webp", "image/svg+xml")
ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
# an SVG shown as an image runs no script, but one carrying any is refused all the same
SVG_ACTIVE = re.compile(rb"<script|javascript:|\son[a-z]+\s*=|<foreignobject", re.IGNORECASE)


class Invalid(ValueError):
    """The entry cannot be kept as given; the message says why, for the page."""


def _logo(value: object) -> str:
    if not isinstance(value, str) or not value.startswith("data:"):
        raise Invalid("the logo must be an image file")
    head, _, data = value.partition(",")
    kind = head[5:].removesuffix(";base64")
    if kind not in LOGO_TYPES or not head.endswith(";base64"):
        raise Invalid("the logo must be a PNG, JPEG, WebP or SVG file")
    try:
        raw = base64.b64decode(data, validate=True)
    except ValueError as exc:
        raise Invalid("the logo file could not be read") from exc
    if len(raw) > LOGO_BYTES:
        raise Invalid(f"the logo is over {LOGO_BYTES // 1024} KB")
    if kind == "image/svg+xml" and SVG_ACTIVE.search(raw):
        raise Invalid("this SVG holds a script or an event handler")
    return value


def clean(brand_id: str, body: object, *, known: dict | None) -> dict:
    """One entry checked: label, kind and words; the logo kept from `known` when the body leaves it out."""
    if not ID.match(brand_id):
        raise Invalid("unknown maker")
    if not isinstance(body, dict):
        raise Invalid("expected an object")
    label = str(body.get("label") or "").strip()
    if not label or len(label) > LABEL_CHARS:
        raise Invalid(f"a name of 1 to {LABEL_CHARS} characters")
    kind = body.get("kind")
    if kind not in KINDS:
        raise Invalid("kind is model or hardware")
    words = body.get("words") or []
    if not isinstance(words, list) or len(words) > MAX_WORDS:
        raise Invalid(f"at most {MAX_WORDS} words")
    words = list(dict.fromkeys(str(w).strip().lower() for w in words if str(w).strip()))
    if any(len(w) > WORD_CHARS for w in words):
        raise Invalid(f"a word is at most {WORD_CHARS} characters")
    logo = known.get("logo") if known else None
    if "logo" in body:
        logo = None if body["logo"] is None else _logo(body["logo"])
    if not words and not logo:
        raise Invalid("give the words that recognise it, or a logo")
    return {"id": brand_id, "label": label, "kind": kind, "words": words, "logo": logo}


class Brands:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._cached: tuple[tuple[int, int], list[dict]] | None = None   # read once per version of the file

    def version(self) -> str:
        """The file's version (its size and time), for the page's cache: unchanged, the logos are not sent again."""

        try:
            st = self.path.stat()
        except OSError:
            return "none"
        return f"{st.st_mtime_ns:x}-{st.st_size:x}"

    def _read(self) -> list[dict]:
        try:
            st = self.path.stat()
            key = (st.st_mtime_ns, st.st_size)
            if self._cached and self._cached[0] == key:
                return [dict(b) for b in self._cached[1]]
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return []
        out = [b for b in data.get("brands", []) if isinstance(b, dict) and ID.match(str(b.get("id", "")))]
        self._cached = (key, out)
        return [dict(b) for b in out]

    def _write(self, brands: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"brands": brands}))
        os.replace(tmp, self.path)

    def all(self) -> list[dict]:
        return self._read()

    def save(self, brand_id: str, body: object) -> dict:
        """An entry is its id and its kind: NVIDIA makes models and hardware, and each has its own entry."""

        kind = body.get("kind") if isinstance(body, dict) else None

        def same(b: dict) -> bool:
            return b["id"] == brand_id and b.get("kind") == kind

        with self._lock:
            brands = self._read()
            known = next((b for b in brands if same(b)), None)
            entry = clean(brand_id, body, known=known)
            if not known and len(brands) >= MAX_BRANDS:
                raise Invalid(f"at most {MAX_BRANDS} makers")
            brands = [entry if same(b) else b for b in brands] if known else [*brands, entry]
            self._write(brands)
            return entry

    def remove(self, brand_id: str, kind: str | None = None) -> bool:
        """The entry of that id and kind; without a kind (an older page), every entry of that id."""

        with self._lock:
            brands = self._read()
            left = [b for b in brands if b["id"] != brand_id or (kind is not None and b.get("kind") != kind)]
            if len(left) == len(brands):
                return False
            self._write(left)
            return True
