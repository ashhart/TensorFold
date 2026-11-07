"""Shared reference-only FP32 elementary functions and activation rounding sites."""
import numpy as np
from fp64_reference import bf


_ELEMENTARY_KERNELS = {}


def elementary(values, operation, backend='numpy'):
    values = np.asarray(values, dtype=np.float32)
    if backend == 'numpy':
        if operation == 'rsqrt':
            return (1 / np.sqrt(values.astype(np.float64))).astype(np.float32)
        return getattr(np, operation)(values)
    if backend != 'metal' or operation not in ('exp', 'exp2', 'cos', 'sin', 'rsqrt'):
        raise ValueError('Unsupported elementary function backend or operation')
    import mlx.core as mx
    if operation not in _ELEMENTARY_KERNELS:
        function = 'metal::exp2' if operation == 'exp2' else 'metal::precise::' + operation
        _ELEMENTARY_KERNELS[operation] = mx.fast.metal_kernel(
            name='lfm3_reference_elementary_' + operation, input_names=['X'], output_names=['Y'],
            source='uint i=thread_position_in_grid.x;if(i<COUNT)Y[i]=' + function + '(X[i]);')
    result, = _ELEMENTARY_KERNELS[operation](inputs=[mx.array(values)], template=[('COUNT', values.size)],
        grid=(values.size, 1, 1), threadgroup=(256, 1, 1), output_shapes=[values.shape],
        output_dtypes=[mx.float32])
    return np.array(result)


def fma(a, b, c):
    # FP64 multiply/add then FP32 round reproduces the captured sites; this is
    # not a proof of correctly rounded fmaf for every possible IEEE input.
    return (np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64)
            + np.asarray(c, dtype=np.float64)).astype(np.float32)



def swiglu(gate, up, elementary_backend='numpy'):
    gate, up = np.asarray(gate, dtype=np.float32), np.asarray(up, dtype=np.float32)
    # NumPy and Metal precise exp are separately implemented; this mode does not assume byte equality.
    sigmoid = np.float32(1) / (np.float32(1) + elementary(-gate, 'exp', elementary_backend))
    return bf(bf(gate * sigmoid).astype(np.float32) * up)
