"""Read benchmark checkpoint and runtime evidence from installed files without downloading models."""

from __future__ import annotations

import ast
import hashlib
from importlib import metadata as packages
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any

from tensorfold import __version__, families, hub
from tensorfold.benchmark import hardware

_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                    "vocab.json", "merges.txt", "chat_template.jinja")
_TOKENIZER_LIMIT = 256 * 1024**2


def _snapshot(model_arg: str) -> tuple[Path | None, str | None, bool]:
    path = Path(model_arg).expanduser()
    if path.is_dir():
        return path, None, True
    if not _REPO.fullmatch(model_arg):
        return None, None, True
    try:
        return hub.cached(model_arg), model_arg, False
    except (OSError, ValueError, ImportError):
        return None, model_arg, False


def _config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        file = path / "config.json"
        if file.stat().st_size > 4 * 1024**2:
            return {}
        with file.open("rb") as handle:
            raw = handle.read(4 * 1024**2 + 1)
        if len(raw) > 4 * 1024**2:
            return {}
        found = json.loads(raw)
        return found if isinstance(found, dict) else {}
    except (OSError, ValueError, UnicodeError, RecursionError):
        return {}


def _file_hash(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        if path.stat().st_size > 4 * 1024**2:
            return None
        with path.open("rb") as handle:
            raw = handle.read(4 * 1024**2 + 1)
        return hashlib.sha256(raw).hexdigest() if len(raw) <= 4 * 1024**2 else None
    except OSError:
        return None


def _tokenizer_hash(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    total, count = 0, 0
    try:
        for name in _TOKENIZER_FILES:
            file = path / name
            if not file.is_file():
                continue
            size = file.stat().st_size
            total += size
            if total > _TOKENIZER_LIMIT:
                return None
            digest.update(name.encode("ascii") + b"\0" + str(size).encode("ascii") + b"\0")
            with file.open("rb") as handle:
                seen = 0
                for chunk in iter(lambda: handle.read(1024**2), b""):
                    seen += len(chunk)
                    if seen > size:
                        return None
                    digest.update(chunk)
            if seen != size:
                return None
            count += 1
    except OSError:
        return None
    return digest.hexdigest() if count else None


def _quantization(config: dict[str, Any]) -> str | None:
    text = config.get("text_config")
    for source in (config, text if isinstance(text, dict) else {}):
        for key in ("quantization", "quantization_config"):
            quant = source.get(key)
            if not isinstance(quant, dict) or not quant:
                continue
            method = str(quant.get("quant_method", "mlx" if "bits" in quant else "")).lower()
            if method not in ("mlx", "exl3", "modelopt", "compressed-tensors", "awq", "gptq", "bitsandbytes"):
                return None
            details = [method]
            for name in ("quant_algo", "format", "mode"):
                value = quant.get(name)
                if isinstance(value, str) and value.lower() in (
                    "nvfp4", "mxfp4", "mxfp8", "fp8", "fp4", "mixed_precision", "affine", "float-quantized",
                    "pack-quantized", "int-quantized", "w4a16_nvfp4",
                ):
                    details.append(value.lower())
            bits, group = quant.get("bits"), quant.get("group_size")
            if type(bits) is int and 1 <= bits <= 32:
                details.append(f"{bits}bit")
            if type(group) is int and 1 <= group <= 4096:
                details.append(f"g{group}")
            return "-".join(details)
    return "unquantized" if config else None


def _package_version(name: str) -> str | None:
    try:
        return hardware._version(packages.version(name))
    except packages.PackageNotFoundError:
        return None


def _draft_complete(path: Path) -> bool:
    tokenizer = ((path / "tokenizer.json").is_file() or (path / "tokenizer.model").is_file()
                 or ((path / "vocab.json").is_file() and (path / "merges.txt").is_file()))
    try:
        return tokenizer and (path / "config.json").is_file() and hub._cached_weights_complete(path)
    except (OSError, ValueError, TypeError, RecursionError):
        return False


def _cuda_version() -> str | None:
    # Reading the wheel's constant avoids importing torch or initializing a GPU context.
    try:
        file = Path(packages.distribution("torch").locate_file("torch/version.py"))
        if file.stat().st_size > 64 * 1024:
            return None
        for node in ast.parse(file.read_text()).body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            if any(isinstance(target, ast.Name) and target.id == "cuda" for target in targets):
                return hardware._version(ast.literal_eval(node.value))
    except (packages.PackageNotFoundError, OSError, ValueError, SyntaxError, TypeError, UnicodeError):
        pass
    return None


def collect(model_arg: str, backend: str, *, drafter: str = "auto") -> dict[str, Any]:
    """Return the model and runtime allowlists, never names or paths of local checkpoint directories."""

    if backend == "auto":
        backend = "mlx" if sys.platform == "darwin" else "cuda"
    if backend not in ("mlx", "cuda"):
        raise ValueError("Benchmark backend must be mlx or cuda")
    path, repo_id, local = _snapshot(model_arg)
    config = _config(path)
    family = None
    text = config.get("text_config")
    kind = config.get("model_type") or (text.get("model_type") if isinstance(text, dict) else None)
    if isinstance(kind, str):
        family = families.families().get(kind)
    draft_repo = None
    if drafter == "auto" and family is not None:
        candidate = getattr(family.package, "DRAFTER", "")
        if isinstance(candidate, str) and _REPO.fullmatch(candidate):
            found, _, _ = _snapshot(candidate)
            if found is not None and _draft_complete(found):
                draft_repo = candidate
    elif drafter not in ("", "none", "auto"):
        found, candidate, _ = _snapshot(drafter)
        if candidate and found is not None and _draft_complete(found):
            draft_repo = candidate
    revision = path.name if repo_id and path and path.parent.name == "snapshots" and _REVISION.fullmatch(path.name) else None
    system = "macos" if sys.platform == "darwin" else "linux" if sys.platform.startswith("linux") else "unknown"
    dependencies = {key: _package_version(package) for key, package in (
        ("mlx", "mlx"), ("mlx_lm", "mlx-lm"), ("torch", "torch"), ("triton", "triton"),
    )}
    dependencies.update({"cuda": _cuda_version() if backend == "cuda" else None,
                         "driver": hardware.driver_version() if backend == "cuda" else None})
    return {"model": {"repo_id": repo_id, "revision": revision,
                      "family": family.model_type if family else None,
                      "config_sha256": _file_hash(path / "config.json" if path else None),
                      "tokenizer_sha256": _tokenizer_hash(path), "quantization": _quantization(config),
                      "drafter_repo_id": draft_repo, "local": local},
            "runtime": {"tensorfold_version": __version__, "backend": backend,
                        "python_version": platform.python_version(), "platform": system,
                        "dependencies": dependencies}}
