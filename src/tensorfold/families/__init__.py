"""Model families TensorFold can load, picked by the checkpoint's ``model_type``.

Each family is a package in this folder (``tensorfold/families/<name>/``) holding everything specific to it:
its forward pass, its kernels, any draft head. The package's ``__init__`` says what it serves and how to load
it:

    MODEL_TYPES = ("nemotron_h",)          # config.json model_type values it takes
    TITLE = "Nemotron 3.5 Lightning"
    LANES = False                          # True: engine.lane_engine.LaneEngine; False: engine.family_engine.SerialEngine
    def load(model_dir, **options) -> (model, tokenizer)

and optionally:

    MODELS = ("owner/checkpoint",)                # Hugging Face checkpoints the family is tested with
    DRAFTER = "owner/draft-model"                 # a draft model it can use (``tensorfold pull`` it once)
    def check(model_dir) -> None                  # refuse an unsupported checkpoint before any weight is read
    MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200"}   # set before MLX starts (unless already set)
    def engine_settings(model) -> dict             # keyword arguments for the engine (e.g. max_rows)
    def kernel_version(model) -> str               # names the kernels in prefix-snapshot keys
    def setup(app, model, **options) -> None       # extras on the server app (e.g. a draft model)

``options`` are the CLI's family options (``lane_kernels``, ``drafter``, ``mtp_drafts``, ``mtp_head``, ...); a
family takes the ones it knows. A new family is a new package here; nothing else registers it. ``detect`` reads
config.json only, so the CLI knows what it is loading before it touches MLX or any weights. What the engines
call on a model is described in ``engine/family_engine.py`` and ``docs/recipes/adding-a-family.md``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class Family:
    model_type: str
    title: str
    module: str
    lanes: bool  # served by the lane engine

    @property
    def package(self) -> ModuleType:
        return importlib.import_module(self.module)


_found: dict[str, Family] | None = None


def families() -> dict[str, Family]:
    """Every family package in this folder, by the model_type values it serves."""

    global _found
    if _found is None:
        found: dict[str, Family] = {}
        for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
            if not info.ispkg:
                continue
            module = f"{__name__}.{info.name}"
            package = importlib.import_module(module)
            for kind in getattr(package, "MODEL_TYPES", ()):
                if kind in found:
                    raise ValueError(f"model_type {kind!r} claimed by {found[kind].module} and {module}")
                found[kind] = Family(kind, str(getattr(package, "TITLE", info.name)), module,
                                     bool(getattr(package, "LANES", False)))
        _found = found
    return _found


def read_config(model_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(model_dir) / "config.json").read_text())


def quantization(config: dict[str, Any]) -> tuple[int | None, int | None]:
    """(bits, group size) of a checkpoint's quantized weights, (None, None) when it has none."""

    for source in (config, config.get("text_config") or {}):
        found = source.get("quantization") or source.get("quantization_config")
        if isinstance(found, dict) and "bits" in found:
            return int(found["bits"]), int(found.get("group_size", 64))
    return None, None


def model_type(model_dir: str | Path) -> str:
    config = read_config(model_dir)
    return str(config.get("model_type") or config.get("text_config", {}).get("model_type") or "")


def detect(model_dir: str | Path) -> Family:
    kind = model_type(model_dir)
    family = families().get(kind)
    if family is None:
        raise ValueError(f"TensorFold has no family for model_type {kind!r} (known: {sorted(families())})")
    return family


def load(model_dir: str | Path, **options: Any) -> tuple[Any, Any]:
    family = detect(model_dir)
    return family.package.load(Path(model_dir), **options)


def kernel_version(family: Family, model: Any) -> str:
    """Fingerprint the active family and versioned kernels for safe prefix-snapshot reuse."""

    hook = getattr(family.package, "kernel_version", None)
    if hook is not None:
        return str(hook(model))
    digest = hashlib.sha256()
    modules = (family.module, getattr(family.package, "KERNEL_PACKAGE", ""),
               *getattr(family.package, "KERNEL_DEPENDENCIES", ()))
    for module_name in filter(None, modules):
        digest.update(module_name.encode())
        module = importlib.import_module(module_name)
        source = Path(str(module.__file__))
        paths = sorted(source.parent.rglob("*.py")) if source.name == "__init__.py" else [source]
        for path in paths:
            digest.update(path.relative_to(source.parent).as_posix().encode())
            digest.update(path.read_bytes())
    version = getattr(family.package, "KERNEL_VERSION", "")
    prefix = f"{family.model_type}-{version}-" if version else ""
    return prefix + digest.hexdigest()[:12]
