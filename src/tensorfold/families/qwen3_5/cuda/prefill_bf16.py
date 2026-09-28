"""BF16 row inputs for prompt projections that skip the FP8 path: an EXL3 pack's, or affine formats past 4-bit g64."""

from . import glue


def add_rmsnorm(x, residual, weight, eps):
    saved, normalized, _ = glue.add_rmsnorm(x, residual, weight, eps)
    return saved, normalized


def swiglu(gate, up):
    return glue.swiglu(gate, up)[0]


def gated_norm(y, z, weight, eps):
    return glue.gated_norm(y, z, weight, eps)[0]


def gate_mul(o, qg, *, heads, head_dim):
    return glue.gate_mul(o, qg, heads=heads, head_dim=head_dim)[0]
