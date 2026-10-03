"""Rebuild dense Qwen from pinned local checkpoints without retaining its device objects."""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
                 identity: CheckpointIdentity | None = None, cache_dir: Path | str | None = None) -> None:
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
        self.prefixes = None
        if cache_dir is not None:
            import torch

            import tensorfold
            from tensorfold.cuda.prefix_store import PrefixStore
            from tensorfold.families.qwen3_5.cuda import prefix_snapshot

            # Files belong to this loaded runtime and recipe, never another process or model revision.
            signature = {"session": uuid.uuid4().hex, "family": "qwen3_5", "backend": "cuda",
                         "runtime": tensorfold.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
                         "capability": torch.cuda.get_device_capability(), "settings": self.settings,
                         "math": self.math, "checkpoints": sorted(self.identity.manifest.items())}
            key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
            self.prefixes = PrefixStore(Path(cache_dir), key, prefix_snapshot)

    def verify_identity(self) -> None:
        self.identity.verify()

    def prepare(self) -> None:
        """Serialization is preflight: a failed write leaves the original runtime awake."""

        self.verify_identity()
        if self.prefixes is not None:
            import torch

            torch.cuda.synchronize()
            engine = self.app.engine
            self.prefixes.save(engine.multi.cache if engine.concurrent else engine.cache)

    def cache_snapshot(self) -> dict:
        return self.prefixes.snapshot() if self.prefixes is not None else {"mode": "discard"}

    def close(self) -> None:
        if self.prefixes is not None:
            self.prefixes.close()

    def _attach_prefixes(self, engine) -> None:
        if self.prefixes is None:
            return
        owner = engine.multi if engine.concurrent else engine
        room = None if engine.concurrent else engine.room
        owner.cache = self.prefixes.attach(owner.cache, device=engine.w.norm.device, room=room,
                                           admit=lambda size, prompt: self._prefix_room(engine, size, prompt))
        if room is not None:
            room.cache = owner.cache

    @staticmethod
    def _prefix_room(engine, size: int, prompt: list[int]) -> bool:
        """A lazy load must fit beside the new request's caches, or it is a cache miss."""

        if engine.concurrent:
            decoder = engine.multi
            if decoder.memory_gate is None:
                return True
            from tensorfold.cuda.streams import Stream

            rows = decoder._first(Stream(prompt, engine.context_window - len(prompt)))
            return decoder.memory_gate.fits(size + rows * decoder.row_bytes)
        import torch

        from tensorfold.cuda.capacity import GIB, available_bytes
        from tensorfold.cuda.memory_gate import torch_live

        return torch_live(torch, available_bytes)() >= size + 2 * GIB

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
        if self.prefixes is not None:
            self.prefixes.verify()
        mode, asked, fp8 = self.math
        precision.set_mode(mode, asked=asked)
        prompt_precision.set_fp8(fp8)
        self._pending = self.factory(self.model_dir, **self.options)
        self.verify_identity()
        if _settings(self._pending) != self.settings:
            raise ValueError("restored runtime settings differ from the served context, precision or drafting settings")
        self._attach_prefixes(self._pending)
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
