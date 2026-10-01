"""Versioned launch-agent profiles; paths, arguments and environment are data, never shell text."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from .safety import ControlError, absolute, atomic_write, private_read

_NAME = re.compile(r"[a-z][a-z0-9-]{0,47}\Z")
_RESERVED_NAMES = {"control-smoke"}
_ENV = re.compile(r"[A-Z_][A-Z0-9_]{0,127}\Z")
_RESERVED = {"--host", "--port", "--name", "--alias", "--backend", "--no-update-check"}
_SENSITIVE = re.compile(r"(?i)(token|password|api[-_]key|secret)")
_ENV_PREFIXES = ("TENSORFOLD_", "TF_", "MLX_", "HF_", "HUGGINGFACE_", "TORCH_", "CUDA_")
_ENV_NAMES = {"TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS"}


def name_of(name: str) -> str:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ControlError("profile name: 1–48 lowercase letters, digits or hyphens; start with a letter")
    if name in _RESERVED_NAMES:
        raise ControlError(f"profile name {name} is reserved")
    return name


def string(value: Any, label: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or any(ord(c) < 32 for c in value):
        raise ControlError(f"{label} must be nonempty text without control characters (max {maximum})")
    return value


@dataclass(frozen=True)
class Paths:
    home: Path = field(default_factory=Path.home)

    def __post_init__(self) -> None:
        object.__setattr__(self, "home", absolute(self.home))

    @property
    def root(self) -> Path:
        return absolute(self.home) / "Library/Application Support/TensorFold/control"

    @property
    def profiles(self) -> Path:
        return self.root / "profiles"

    def profile(self, name: str) -> Path:
        return self.profiles / f"{name_of(name)}.json"

    def plist(self, name: str) -> Path:
        return absolute(self.home) / "Library/LaunchAgents" / f"dev.tensorfold.{name_of(name)}.plist"

    def log_dir(self, name: str) -> Path:
        return absolute(self.home) / "Library/Logs/TensorFold" / name_of(name)

    def log(self, name: str) -> Path:
        return self.log_dir(name) / "server.log"

    def working(self, name: str) -> Path:
        return self.root / "work" / name_of(name)


@dataclass(frozen=True)
class Profile:
    name: str
    model: str
    python: str = field(default_factory=lambda: str(absolute(sys.executable)))
    host: str = "127.0.0.1"
    port: int = 8080
    backend: str = "mlx"
    args: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    environment_file: str | None = None
    allow_network: bool = False
    allow_download: bool = False
    log_bytes: int = 8 << 20
    log_backups: int = 4
    schema: int = 1

    def __post_init__(self) -> None:
        name_of(self.name)
        string(self.model, "model")
        if self.model.startswith("-"):
            raise ControlError("model may not start with a dash")
        python = absolute(string(self.python, "python"))
        if not Path(self.python).is_absolute():
            raise ControlError("python must be an absolute path to the serving virtual environment")
        object.__setattr__(self, "python", str(python))
        try:
            address = ipaddress.ip_address(self.host)
        except ValueError as exc:
            raise ControlError("host must be a numeric IPv4 or IPv6 address") from exc
        if not address.is_loopback and not self.allow_network:
            raise ControlError("non-loopback binding needs --allow-network; put authentication in front of it")
        if type(self.port) is not int or not 1024 <= self.port <= 65535:
            raise ControlError("port must be an integer from 1024 through 65535")
        if self.backend not in {"mlx", "auto"}:
            raise ControlError("macOS LaunchAgents support backend mlx or auto; remote CUDA is monitor-only")
        if type(self.schema) is not int or self.schema != 1:
            raise ControlError("unsupported profile schema (expected 1)")
        if type(self.allow_network) is not bool or type(self.allow_download) is not bool:
            raise ControlError("allow_network and allow_download must be booleans")
        if not isinstance(self.args, (tuple, list)) or len(self.args) > 128:
            raise ControlError("args must be a list of at most 128 literal arguments")
        for arg in self.args:
            string(arg, "serve argument")
            option = arg.split("=", 1)[0]
            if (option in _RESERVED or arg == "--"
                    or (option.startswith("--") and any(r.startswith(option) for r in _RESERVED))):
                raise ControlError(f"{option} is managed by the profile; do not put it in --arg")
            # max-tokens is harmless; authentication material must never reach the process argument list.
            if option.startswith("--") and _SENSITIVE.search(option) and option not in {"--max-tokens"}:
                raise ControlError("secret-bearing flags belong in a private environment file")
        object.__setattr__(self, "args", tuple(self.args))
        validate_env(self.environment, secrets=False)
        if self.environment_file is not None:
            string(self.environment_file, "environment_file")
            if not Path(self.environment_file).is_absolute():
                raise ControlError("environment_file must be absolute")
        if type(self.log_bytes) is not int or not 65536 <= self.log_bytes <= 64 << 20:
            raise ControlError("log_bytes must be 64 KiB through 64 MiB")
        if type(self.log_backups) is not int or not 1 <= self.log_backups <= 10:
            raise ControlError("log_backups must be 1 through 10")

    @property
    def label(self) -> str:
        return f"dev.tensorfold.{self.name}"

    @property
    def endpoint(self) -> str:
        address = ipaddress.ip_address(self.host)
        host = ("::1" if address.version == 6 else "127.0.0.1") if address.is_unspecified else self.host
        return f"http://{'[' + host + ']' if ':' in host else host}:{self.port}"

    def command(self) -> list[str]:
        return [self.python, "-u", "-m", "tensorfold", "serve", self.model, "--host", self.host,
                "--port", str(self.port), "--name", self.name, "--backend", self.backend,
                "--no-update-check", *self.args]

    def encode(self) -> bytes:
        return (json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    @classmethod
    def decode(cls, data: bytes) -> "Profile":
        try:
            values = json.loads(data)
            if not isinstance(values, dict):
                raise ValueError("expected object")
            return cls(**values)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ControlError(f"invalid service profile: {exc}") from exc


def validate_env(values: dict, *, secrets: bool) -> dict[str, str]:
    if not isinstance(values, dict) or len(values) > 64:
        raise ControlError("environment must be an object with at most 64 entries")
    for key, value in values.items():
        if not isinstance(key, str) or not _ENV.fullmatch(key):
            raise ControlError("environment names must be uppercase identifiers")
        if not (key.startswith(_ENV_PREFIXES) or key in _ENV_NAMES):
            raise ControlError(f"environment override not allowed: {key}")
        string(value, f"environment {key}", 16384)
        if not secrets and _SENSITIVE.search(key):
            raise ControlError(f"{key} must be in --env-file, not the profile")
    return dict(values)


def read_environment(profile: Profile) -> dict[str, str]:
    values = dict(profile.environment)
    if profile.environment_file:
        try:
            external = json.loads(private_read(Path(profile.environment_file), 65536))
        except (ValueError, UnicodeError) as exc:
            raise ControlError("environment file must be a private JSON object") from exc
        values.update(validate_env(external, secrets=True))
    values.update(PYTHONUNBUFFERED="1", TENSORFOLD_NO_LIVE="1", TENSORFOLD_NO_UPDATE_CHECK="1")
    if not profile.allow_download:
        values.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return values


class Store:
    def __init__(self, paths: Paths | None = None):
        self.paths = paths or Paths()

    def get(self, name: str) -> Profile:
        path = self.paths.profile(name)
        try:
            result = Profile.decode(private_read(path))
        except FileNotFoundError as exc:
            raise ControlError(f"no installed profile: {name}") from exc
        if result.name != name:
            raise ControlError(f"profile name disagrees with filename: {path}")
        return result

    def list(self) -> tuple[list[Profile], list[str]]:
        profiles, errors = [], []
        for path in sorted(self.paths.profiles.glob("*.json")):
            try:
                profiles.append(self.get(path.stem))
            except (ControlError, OSError) as exc:
                errors.append(str(exc))
        return profiles, errors

    def put(self, profile: Profile) -> None:
        atomic_write(self.paths.profile(profile.name), profile.encode())
