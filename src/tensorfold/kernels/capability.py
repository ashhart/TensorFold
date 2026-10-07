"""Whether custom GPU kernels can build and run on this machine's active backend."""

from __future__ import annotations

import functools

import mlx.core as mx

# One float in, one float out: the smallest kernel that proves the backend compiles and runs a custom kernel
_SOURCE = "out[0] = src[0];"


def custom_kernels() -> bool:
    """True when the active backend builds and runs a custom kernel; the probe runs once per process."""
    return _probe()


@functools.cache
def _probe() -> bool:
    kernel = getattr(mx.fast, "metal_kernel", None)
    if kernel is None or mx.default_device() == mx.cpu:
        return False
    try:
        compiled = kernel(name="tf_capability_probe", input_names=["src"], output_names=["out"], source=_SOURCE)
        out = compiled(
            inputs=[mx.array([1.0])],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[(1,)],
            output_dtypes=[mx.float32],
        )[0]
        mx.eval(out)
    except Exception:  # noqa: BLE001 - no custom-kernel backend: the gate stays closed
        return False
    return bool(out[0] == 1.0)


__all__ = ["custom_kernels"]
