"""Model families TensorFold can load, picked by the checkpoint's ``model_type``.

Each family is a package in this folder (``tensorfold/families/<name>/``) holding everything specific to it:
its forward pass, its kernels, any draft head. The package's ``__init__`` says what it serves and how to load
it:

    MODEL_TYPES = ("nemotron_h",)          # config.json model_type values it takes
    TITLE = "Nemotron 3.5 Lightning"
    LANES = False                          # True: engine.lane_engine.LaneEngine; False: engine.family_engine.SerialEngine
    def load(model_dir, **options) -> (model, tokenizer)

and optionally:

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
    """The family's own name for the kernels a model runs, else a hash of the family package's sources: a prefix
    snapshot computed by other kernels has other bits, so snapshots are keyed by it."""

    hook = getattr(family.package, "kernel_version", None)
    if hook is not None:
        return str(hook(model))
    folder = Path(str(family.package.__file__)).parent
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]
