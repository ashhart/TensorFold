"""TensorFold Web settings on disk: the engine's address, the memory budget, the chosen model, launch flags and the API key."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import secrets
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_ENGINE = "http://127.0.0.1:8080"


def config_dir() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "TensorFold Web"
    if platform.system() == "Windows":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "TensorFold Web"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "tensorfold-web"


@dataclass
class Settings:
    engine_url: str = DEFAULT_ENGINE
    remote: bool = False                 # the engine runs elsewhere; TensorFold Web only watches it
    model: str | None = None             # the checkpoint to serve
    budget_fraction: float | None = None # None: TensorFold's default (70%)
    context: int | None = None           # None: let TensorFold size it
    thinking: str = "default"            # "default", "on", "off"
    extra_args: list[str] = field(default_factory=list)
    share: bool = False                  # bind the engine on every interface; the API key then protects it
    api_key: str | None = None           # the engine's API key, generated when sharing is turned on
    cortheon: bool = False               # Cortheon's reasoning and web research in the engine (1.0.9 on; 1.0.11 for its fixes)
    theme: str = "system"
    energy_price: float | None = None    # electricity, per kWh, for the Bench's energy cost
    currency: str = "EUR"
    setup_hidden: bool = False           # the first-run checklist put away; Settings keeps the same checks

    def to_dict(self, *, mask: bool = False) -> dict:
        d = asdict(self)
        if mask and d.get("api_key"):
            d["api_key"] = "•" * 8
        d["api_key_set"] = bool(self.api_key)
        return d

    def engine_port(self) -> int:
        return urlsplit(self.engine_url).port or 8080


THINKING = ("default", "on", "off")
THEMES = ("system", "light", "dark")


def validate(patch: dict) -> dict:
    """The fields a client may set, with their types checked; raises ValueError naming the first bad one."""

    out: dict = {}
    for key, value in patch.items():
        if key not in Settings.__dataclass_fields__:
            continue                                   # unknown fields are ignored
        if key == "engine_url":
            out[key] = _engine_url(value)
        elif key == "api_key":
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("api_key must be a non-empty string, or null to forget it")
            out[key] = value.strip() if value else None
        elif key in ("remote", "share", "setup_hidden", "cortheon"):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
            out[key] = value
        elif key == "model":
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("model must be a repository id or null")
            out[key] = value
        elif key == "budget_fraction":
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not 0.3 <= float(value) <= 0.95):
                raise ValueError("budget_fraction must be between 0.3 and 0.95, or null")
            out[key] = None if value is None else float(value)
        elif key == "context":
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1024):
                raise ValueError("context must be at least 1024 tokens, or null")
            out[key] = value
        elif key == "thinking":
            if value not in THINKING:
                raise ValueError(f"thinking must be one of {', '.join(THINKING)}")
            out[key] = value
        elif key == "energy_price":
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not 0 < float(value) <= 10):
                raise ValueError("energy_price must be a price per kWh above 0 and at most 10, or null")
            out[key] = None if value is None else float(value)
        elif key == "currency":
            if not isinstance(value, str) or len(value.strip()) != 3 or not value.strip().isalpha():
                raise ValueError("currency must be a three-letter code, such as EUR or USD")
            out[key] = value.strip().upper()
        elif key == "theme":
            if value not in THEMES:
                raise ValueError(f"theme must be one of {', '.join(THEMES)}")
            out[key] = value
        elif key == "extra_args":
            if not isinstance(value, list) or not all(isinstance(a, str) for a in value):
                raise ValueError("extra_args must be a list of strings")
            from . import native  # whatever engine runs them: TensorFold Web sets these itself

            for a in value:
                flag = a.partition("=")[0]
                if flag in native.SECRET:
                    raise ValueError(f"{flag} would put a secret on the command line: {native.SECRET[flag]}")
                if native.managed(flag):
                    raise ValueError(f"{flag} is set by TensorFold Web: {native.managed(flag)}")
            out[key] = value
    return out


def _engine_url(value: object) -> str:
    """An http(s) address with a host and, when given, a numeric port; the trailing slash dropped."""

    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        raise ValueError("engine_url must start with http:// or https://")
    parts = urlsplit(value)
    if not parts.hostname:
        raise ValueError("engine_url needs a host, such as http://spark.local:8080")
    if parts.username or parts.password:
        raise ValueError("engine_url must not carry a user or password: put the key in the API key field")
    try:
        parts.port  # noqa: B018  (raises on a port that is not a number)
    except ValueError as exc:
        raise ValueError("engine_url has a port that is not a number") from exc
    if parts.path.rstrip("/").endswith("/engine") or "/engine/" in parts.path:
        raise ValueError("engine_url is a TensorFold Web's relay (/engine): give the engine's own address, such as "
                         "http://127.0.0.1:8080")
    return value.rstrip("/")


def new_api_key() -> str:
    return "tf-" + secrets.token_urlsafe(24)


def path() -> Path:
    return config_dir() / "settings.json"


def load() -> Settings:
    try:
        data = json.loads(path().read_text())
    except (OSError, ValueError):
        return Settings()
    return Settings(**{k: v for k, v in data.items() if k in Settings.__dataclass_fields__})


def secure_config_dir() -> Path:
    """The settings folder, made private on every start (not only when the service is installed): readable by this
    user only, and the files that hold conversations, runs and the engine's log narrowed to 0600 as well."""

    p = config_dir()
    p.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        os.chmod(p, 0o700)
        for f in [*p.glob("*.sqlite*"), *p.glob("*.log*"), *p.glob("*.json")]:
            with contextlib.suppress(OSError):
                if f.is_file():
                    os.chmod(f, 0o600)
    return p


def write_private(p: Path, text: str) -> None:
    """Write ``p`` atomically, readable by this user only from the first byte: a file created with the usual mode
    and restricted afterwards can be opened by someone else in between."""

    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}-{os.getpid()}-{threading.get_ident()}.tmp")   # one writer, one temporary file
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)                    # a leftover file of the same name keeps its old mode otherwise
        with os.fdopen(fd, "w") as f:
            f.write(text)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(p)


def save(s: Settings) -> None:
    """Atomic, and readable by this user only: the file may hold the engine's API key."""

    write_private(path(), json.dumps(asdict(s), indent=1))
