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


RECIPES_URL = "https://github.com/ashhart/TensorFold/blob/main/docs/recipes/README.md"
RUNBOOK_URL = "https://github.com/ashhart/TensorFold/blob/main/RUNBOOK.md"
OWN_MODEL_HELP = (f"To run a model or checkpoint TensorFold has no recipe for, write one with the recipe book "
                  f"({RECIPES_URL}: adding a family on a Mac, adding a CUDA family on NVIDIA GPUs), and read the "
                  f"runbook first ({RUNBOOK_URL}).")
MLX_QUANT = "mlx"


def _quantization_block(config: dict[str, Any]) -> dict[str, Any] | None:
    for source in (config, config.get("text_config") or {}):
        for key in ("quantization_config", "quantization"):
            found = source.get(key)
            if isinstance(found, dict) and found:
                return found
    return None


def quant_method(config: dict[str, Any]) -> str | None:
    """How a checkpoint stores its weights: ``"mlx"`` for MLX's affine quantization (``bits`` and ``group_size``,
    no ``quant_method``), ``"mlx-<mode>"`` for MLX's other modes, the config's ``quant_method`` otherwise
    (``"exl3"``, ``"modelopt"``, ``"gptq"``, ``"awq"``, ``"fp8"``, ...), or None for unquantized weights."""

    found = _quantization_block(config)
    if found is None:
        return None
    method = found.get("quant_method")
    if method:
        return str(method).lower()
    if "bits" in found:
        mode = str(found.get("mode") or "affine").lower()
        return MLX_QUANT if mode == "affine" else f"{MLX_QUANT}-{mode}"
    return None


def quantization(config: dict[str, Any]) -> tuple[int | None, int | None]:
    """(bits, group size) of MLX affine-quantized weights; (None, None) for any other format or none."""

    found = _quantization_block(config)
    if found is None or quant_method(config) != MLX_QUANT:
        return None, None
    return int(found["bits"]), int(found.get("group_size", 64))


def describe_quantization(config: dict[str, Any]) -> str:
    method = quant_method(config)
    if method is None:
        return "none (unquantized weights)"
    if method == MLX_QUANT:
        bits, group = quantization(config)
        return f"MLX {bits}-bit, groups of {group}"
    bits = (_quantization_block(config) or {}).get("bits")
    return f"{method}" + (f" ({bits}-bit)" if bits else "")


def backends_of(family: Family) -> tuple[str, ...]:
    """The backends a family has an engine for: ``mlx`` (``load``) and ``cuda`` (``cuda_engine``)."""

    package = family.package
    return tuple(b for b, member in (("mlx", "load"), ("cuda", "cuda_engine")) if hasattr(package, member))


def readable_quants(family: Family, backend: str) -> tuple[str | None, ...]:
    """The storage formats a family's engine reads on ``backend``. By default MLX's affine quantization, plus
    unquantized weights on a Mac (MLX's own kernels load them); a family lists more in its package as
    ``QUANT_METHODS = {"cuda": ("mlx", "exl3")}``."""

    declared = getattr(family.package, "QUANT_METHODS", {}) or {}
    default = (MLX_QUANT, None) if backend == "mlx" else (MLX_QUANT,)
    return tuple(declared.get(backend, default))


def require_readable(family: Family, config: dict[str, Any], backend: str) -> None:
    """Refuse, before any weight downloads, a checkpoint whose storage format the family's engine on ``backend``
    does not read, or MLX weights of another bit width or group size than its kernels take
    (``CUDA_QUANTIZATION = (bits, group)`` for the CUDA engine)."""

    method = quant_method(config)
    where = "NVIDIA GPUs (CUDA)" if backend == "cuda" else "Apple Silicon (MLX)"
    tested = ", ".join(getattr(family.package, "MODELS", ())) or "none listed"
    accepted = readable_quants(family, backend)
    if method not in accepted:
        names = {None: "unquantized weights", MLX_QUANT: "MLX-quantized weights"}
        reads = " or ".join(names.get(m, str(m)) for m in accepted)
        raise ValueError(f"{family.title} on {where} does not read this checkpoint's weights "
                         f"({describe_quantization(config)}); it reads {reads}. Tested checkpoints: {tested}. "
                         f"{OWN_MODEL_HELP}")
    expected = getattr(family.package, "CUDA_QUANTIZATION", None) if backend == "cuda" else None
    if expected is not None and method == MLX_QUANT and quantization(config) != tuple(expected):
        bits, group = expected
        raise ValueError(f"{family.title}'s CUDA kernels read MLX {bits}-bit weights in groups of {group}; this "
                         f"checkpoint has {describe_quantization(config)}. Tested checkpoints: {tested}. "
                         f"{OWN_MODEL_HELP}")


def model_type(model_dir: str | Path) -> str:
    config = read_config(model_dir)
    return str(config.get("model_type") or config.get("text_config", {}).get("model_type") or "")


def detect(model_dir: str | Path) -> Family:
    kind = model_type(model_dir)
    family = families().get(kind)
    if family is None:
        raise ValueError(f"TensorFold has no recipe for model_type {kind!r} yet (it has: "
                         f"{', '.join(sorted(families()))}; `tensorfold models` lists the tested checkpoints). "
                         f"{OWN_MODEL_HELP}")
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
