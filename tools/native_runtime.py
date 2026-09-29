"""Require the validated Python oracle runtime before generating native fixtures."""
import importlib.metadata


def require_mlx():
    versions = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "mlx-lm")}
    if versions["mlx"] != "0.32.2" or versions["mlx-metal"] != "0.32.2":
        raise RuntimeError(f"Native parity requires MLX/MLX-Metal 0.32.2; got {versions}. "
                           "Update this repository's virtual environment before generating oracles.")
    return versions
