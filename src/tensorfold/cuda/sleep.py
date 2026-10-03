"""Rebuild dense Qwen from pinned local checkpoints without retaining its device objects."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import sys
from typing import Any, Callable

from tensorfold.cuda import precision, prompt_precision


class CheckpointIdentity:
    """Full content hashes use a bounded read buffer even for multi-gigabyte shards."""

    _SUFFIXES = {".safetensors", ".json", ".model", ".tiktoken", ".txt", ".jinja", ".jinja2", ".bpe", ".vocab"}

    def __init__(self, model_dir: Path, drafter: str = "") -> None:
        self.paths = tuple(dict.fromkeys(Path(path).resolve() for path in (model_dir, drafter) if path))
        self.manifest = self._read()

    def _read(self) -> dict[tuple[str, str], str]:
        manifest = {}
        for root in self.paths:
            if not (root / "config.json").is_file() or not any(root.glob("*.safetensors")):
                raise ValueError("sleep checkpoint is missing its config or weights")
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if any(part.startswith(".") for part in relative.parts) or path.suffix not in self._SUFFIXES:
                    continue
                if not path.is_file():
                    raise ValueError(f"sleep checkpoint file is missing: {relative}")
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
                manifest[(str(root), str(relative))] = digest.hexdigest()
        return manifest

    def verify(self) -> None:
        try:
            current = self._read()
        except OSError as exc:
            raise ValueError("sleep checkpoint files are missing or unreadable") from exc
        if current != self.manifest:
            raise ValueError("sleep checkpoint content changed; restore the original files or restart the server")


def clear_tensor_caches() -> None:
    """Extension modules stay loaded; their cached device buffers must not."""

    for name in ("tensorfold.cuda.kernels.attention", "tensorfold.cuda.nvfp4.linear"):
        module = sys.modules.get(name)
        if module is not None:
            module.clear_tensor_cache()


def _settings(engine: Any) -> dict[str, Any]:
    """Only scalar settings survive the old runtime; no bound method or device array does."""

    weights = getattr(engine, "w", None)
    settings = {name: getattr(engine, name, None) for name in (
        "context_window", "tp", "rank", "max_rows", "tree_rows", "allow_copy", "concurrent", "vision_enabled")}
    settings.update({name: getattr(weights, name, None) for name in ("precision", "quant", "fast_prefill")})
    settings["own"] = tuple(sorted((getattr(weights, "own", None) or {}).items()))
    settings["draft"] = getattr(engine, "draft", None) is not None
    settings["vision"] = getattr(engine, "vision", None) is not None
    settings["streams"] = getattr(getattr(engine, "scheduler", None), "max_streams", 1)
    return settings


class CudaSleep:
    """The stable app owns the live runtime; this adapter holds only its reload recipe."""

    def __init__(self, app: Any, factory: Callable, model_dir: Path, options: dict[str, Any], *,
                 identity: CheckpointIdentity | None = None) -> None:
        self.app, self.factory = app, factory
        self.model_dir = Path(model_dir).resolve()
        self.options = dict(options)
        if any(not isinstance(value, (str, int, float, bool, type(None))) for value in self.options.values()):
            raise ValueError("sleep reload options must contain paths and scalars only")
        if self.options.get("drafter"):
            self.options["drafter"] = str(Path(self.options["drafter"]).resolve())
        self.identity = identity or CheckpointIdentity(self.model_dir, self.options.get("drafter", ""))
        self.verify_identity()
        context = app.effective_context_window
        if not isinstance(context, int) or context <= 0:
            raise ValueError("sleep requires a positive served context window")
        self.options.update(context=context, context_explicit=True)
        app.context_window = context
        self.settings = _settings(app.engine)
        self.settings["context_window"] = context
        self.math = (precision.mode(), precision.asked(), prompt_precision.fp8())
        self._pending = None

    def verify_identity(self) -> None:
        self.identity.verify()

    @staticmethod
    def memory_snapshot() -> dict[str, int]:
        import torch

        return {"allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved())}

    def release(self) -> None:
        """Lifecycle has drained all admitted HTTP work before stopping this worker."""

        import torch

        self.app.engine.close()
        torch.cuda.synchronize()
        self.app.engine, self.app.vision = None, None
        self._free_allocations()

    @staticmethod
    def _free_allocations() -> None:
        import torch

        clear_tensor_caches()
        gc.collect()
        torch.cuda.empty_cache()

    def restore(self) -> None:
        """Keep a returned but unverified worker for cleanup if publishing is refused."""

        self.verify_identity()
        mode, asked, fp8 = self.math
        precision.set_mode(mode, asked=asked)
        prompt_precision.set_fp8(fp8)
        self._pending = self.factory(self.model_dir, **self.options)
        self.verify_identity()
        if _settings(self._pending) != self.settings:
            raise ValueError("restored runtime settings differ from the served context, precision or drafting settings")
        self.app.engine, self.app.vision = self._pending, getattr(self._pending, "vision", None)
        self._pending = None

    def cleanup(self) -> None:
        """Called after failed-load exception frames have gone, including partial constructor tensors."""

        import torch

        if self._pending is not None:
            self._pending.close()
        torch.cuda.synchronize()
        self._pending = None
        self.app.engine, self.app.vision = None, None
        self._free_allocations()
