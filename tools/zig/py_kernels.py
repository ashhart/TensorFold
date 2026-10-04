"""Import our Python kernel modules for their source strings, with a stand-in for MLX when it is not installed."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class _Any:
    """Whatever a module touches on mlx at import time (dtypes, classes): never called for GPU work here."""

    def __getattr__(self, name: str) -> "_Any":
        return _Any()

    def __call__(self, *args: object, **kwargs: object) -> "_Any":
        return _Any()


def _stand_in() -> None:
    """Register a minimal `mlx` package: module-level code in our kernel files only names its attributes."""

    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    core.__getattr__ = lambda name: _Any()          # type: ignore[attr-defined]
    metal = types.SimpleNamespace(is_available=lambda: False)
    core.metal = metal                              # type: ignore[attr-defined]
    nn = types.ModuleType("mlx.nn")
    nn.QuantizedLinear = type("QuantizedLinear", (), {})   # type: ignore[attr-defined]
    nn.Module = type("Module", (), {})              # type: ignore[attr-defined]
    nn.Linear = type("Linear", (), {})              # type: ignore[attr-defined]
    mlx.core, mlx.nn = core, nn                     # type: ignore[attr-defined]
    sys.modules.update({"mlx": mlx, "mlx.core": core, "mlx.nn": nn})


def load(name: str) -> types.ModuleType:
    """`tensorfold.<name>` from this tree's src/, importing real MLX if present and the stand-in otherwise."""

    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        if "mlx" not in sys.modules:
            _stand_in()
    return importlib.import_module(f"tensorfold.{name}")
