"""Digest-only API keys and atomic reloads of a restricted file."""

from contextlib import contextmanager
import hashlib
import hmac
import ipaddress
import os
from pathlib import Path
import re
import signal
import stat
import threading
import time


_LABEL = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")


def _digest(key):
    if not isinstance(key, str) or not key or any(char.isspace() or ord(char) < 32 for char in key):
        raise ValueError("API keys must be nonempty text without whitespace")
    return hashlib.sha256(key.encode("utf-8")).digest()


class KeyStore:
    """Only digests and labels survive configuration or file parsing."""

    def __init__(self, keys=(), *, key_file=None, environment="", metrics_open=False, clock=time.monotonic):
        self._static = tuple((_digest(key), f"cli-{i}") for i, key in enumerate(keys, 1))
        self._static += tuple((_digest(key.strip()), f"env-{i}")
                              for i, key in enumerate(environment.split(","), 1) if key.strip())
        self.key_file = Path(key_file).expanduser() if key_file else None
        self.metrics_open = bool(metrics_open)
        self._clock = clock
        self._lock = threading.Lock()
        self._reload = threading.Event()
        self._checked = float("-inf")
        self._stamp = None
        self._file = ()
        self._valid = True
        if self.key_file is not None:
            try:
                self._file, self._stamp = self._read()
            except OSError:
                raise ValueError("API key file cannot be opened safely") from None

    @property
    def enabled(self):
        return bool(self._static or self.key_file is not None)

    def _read(self):
        fd = os.open(self.key_file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o044:
                raise ValueError("API key file must be a regular file unreadable by other users; use chmod 600")
            if info.st_size > 1 << 20:
                raise ValueError("API key file exceeds 1 MiB")
            with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as stream:
                text = stream.read((1 << 20) + 1)
            if len(text.encode("utf-8")) > 1 << 20:
                raise ValueError("API key file exceeds 1 MiB")
            keys = []
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                label, separator, key = line.partition(":")
                if separator:
                    label, key = label.strip(), key.strip()
                    if not _LABEL.fullmatch(label):
                        raise ValueError("API key labels need 1-64 letters, digits, dots, underscores or hyphens")
                else:
                    label, key = f"file-{len(keys) + 1}", line
                keys.append((_digest(key), label))
            if not keys:
                raise ValueError("API key file contains no keys")
            return tuple(keys), (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size, info.st_mode)
        finally:
            os.close(fd)

    def request_reload(self, *_):
        self._reload.set()

    def refresh(self):
        if self.key_file is None:
            return
        with self._lock:
            now = self._clock()
            forced = self._reload.is_set()
            if not forced and now - self._checked < 1:
                return
            self._reload.clear()
            self._checked = now
            try:
                info = self.key_file.stat()
                stamp = (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size, info.st_mode)
                if forced or not self._valid or stamp != self._stamp:
                    keys, stamp = self._read()
                    self._file, self._stamp, self._valid = keys, stamp, True
            except (OSError, UnicodeError, ValueError):
                was_valid = self._valid
                self._file, self._valid = (), False
                if was_valid:
                    print("[tensorfold] API key file reload refused; authenticated routes remain closed", flush=True)

    def match(self, headers):
        self.refresh()
        with self._lock:
            keys = self._static + self._file
            valid = self._valid
        raw = []
        for name in ("Authorization", "x-api-key"):
            values = headers.get_all(name, []) if hasattr(headers, "get_all") else [headers.get(name, "")]
            value = values[0] if len(values) == 1 else ""
            if name == "Authorization":
                scheme, separator, value = value.partition(" ")
                value = value.strip() if separator and scheme.lower() == "bearer" else ""
            try:
                raw.append(_digest(value))
            except ValueError:
                raw.append(hashlib.sha256(b"").digest())
        label = None
        for digest, candidate_label in keys:
            matched = False
            for candidate in raw:
                matched |= hmac.compare_digest(digest, candidate)
            if matched and label is None:
                label = candidate_label
        return label if valid else None

    @contextmanager
    def signals(self):
        supported = self.key_file is not None and hasattr(signal, "SIGHUP")
        supported = supported and threading.current_thread() is threading.main_thread()
        previous = signal.signal(signal.SIGHUP, self.request_reload) if supported else None
        try:
            yield
        finally:
            if supported:
                signal.signal(signal.SIGHUP, previous)


def configure(args):
    store = KeyStore(getattr(args, "api_key", ()) or (), key_file=getattr(args, "api_key_file", None),
                     environment=os.environ.pop("TENSORFOLD_API_KEY", ""),
                     metrics_open=getattr(args, "metrics_open", False))
    if hasattr(args, "api_key"):
        args.api_key = []
    if not store.enabled:
        try:
            loopback = ipaddress.ip_address(args.host).is_loopback
        except ValueError:
            loopback = str(args.host).lower() == "localhost"
        if not loopback:
            print("[tensorfold] warning: non-loopback server has no API key; requests are open", flush=True)
    return store
