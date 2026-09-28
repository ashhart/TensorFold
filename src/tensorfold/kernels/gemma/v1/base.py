"""The Gemma kernels' builder: constants baked into the source, one compiled kernel a set, named by its hash."""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx


class Kernel:
    """A Metal kernel with ``consts`` in its source (MLX regex-parses template arguments on every call)."""

    def __init__(self, name: str, body: str, inputs: list[str], outputs: list[str], header: str = "") -> None:
        self.name, self.body, self.inputs, self.outputs, self.header = name, body, inputs, outputs, header
        self.compiled: dict[tuple, Any] = {}

    def source(self, consts: tuple) -> str:
        lines = "".join(f"  constexpr {'float' if isinstance(v, float) else 'int'} {k} = {v!r};\n"
                        for k, v in consts)
        return lines + self.body

    def __call__(self, consts: tuple = (), **kwargs: Any) -> Any:
        run = self.compiled.get(consts)
        if run is None:
            source = self.source(consts)
            digest = hashlib.sha256((self.header + source).encode()).hexdigest()[:16]
            run = self.compiled[consts] = mx.fast.metal_kernel(
                name=f"{self.name}_{digest}", input_names=self.inputs, output_names=self.outputs, source=source,
                header=self.header)
        return run(**kwargs)


__all__ = ["Kernel"]
