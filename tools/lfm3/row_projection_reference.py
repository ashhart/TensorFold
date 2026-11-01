#!/usr/bin/env python3
"""Independent affine row-projection replay with the shared Metal helper's FP32 rounding sites."""
import argparse
import ctypes
import os
import json
from pathlib import Path
import numpy as np
from fp64_reference import Oracle, bf


def affine_rows(oracle, name, inputs):
    original = np.asarray(inputs)
    if not np.isfinite(original).all() or not np.array_equal(original, bf(original)):
        raise ValueError('Expected finite BF16-representable inputs')
    inputs = np.asarray(original, dtype=np.float32)
    bits, group = oracle.quant(name)
    words = oracle.tensor(name + '.weight')
    n = words.shape[0]
    k = words.shape[1] * 32 // bits
    if inputs.ndim != 2 or inputs.shape[1] != k or k % 512:
        raise ValueError('Expected row-by-K inputs with a complete 512-value lane pass')
    positions = np.arange(k) * bits
    indices, shifts = positions // 32, positions % 32
    padded = np.pad(words.astype(np.uint64), ((0, 0), (0, 1)))
    codes = padded[:, indices] >> shifts.astype(np.uint64)
    cross = shifts + bits > 32
    codes[:, cross] |= padded[:, indices[cross] + 1] << (32 - shifts[cross]).astype(np.uint64)
    codes = (codes & ((1 << bits) - 1)).astype(np.uint8)
    scales = oracle.tensor(name + '.scales').astype(np.float32)
    biases = oracle.tensor(name + '.biases').astype(np.float32)
    output = np.empty((len(inputs), n), dtype=np.float32)
    library = os.environ.get('LFM3_ROW_REPLAY_LIBRARY')
    if library:
        function = ctypes.CDLL(library).lfm3_row_replay
        float_pointer, byte_pointer = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8)
        function.argtypes = [float_pointer, byte_pointer, float_pointer, float_pointer, float_pointer] + [ctypes.c_int] * 5
        function.restype = ctypes.c_int
        inputs = np.ascontiguousarray(inputs)
        codes = np.ascontiguousarray(codes)
        scales, biases = np.ascontiguousarray(scales), np.ascontiguousarray(biases)
        status = function(inputs.ctypes.data_as(float_pointer), codes.ctypes.data_as(byte_pointer),
                 scales.ctypes.data_as(float_pointer), biases.ctypes.data_as(float_pointer),
                 output.ctypes.data_as(float_pointer), len(inputs), n, k, group, bits)
        if status != 0:
            raise MemoryError('CPU projection replay scratch allocation failed')
        return output
    for first in range(0, len(inputs), 8):
        x = inputs[first:first + 8]
        for column in range(0, n, 256):
            count = min(256, n - column)
            lanes = np.zeros((len(x), count, 32), dtype=np.float32)
            for at in range(0, k, 512):
                xx = x[:, at:at + 512].reshape(len(x), 1, 32, 16)
                qq = codes[column:column + count, at:at + 512].reshape(1, count, 32, 16)
                sums = np.zeros((len(x), 1, 32), dtype=np.float32)
                for i in range(16):
                    sums = sums + xx[..., i]
                if bits == 4:
                    place = np.tile(np.array([1, 16, 256, 4096], dtype=np.float32), 4)
                    xx, qq = xx / place, qq * place
                dot = np.zeros_like(lanes)
                for i in range(0, 16, 4):
                    term = xx[..., i] * qq[..., i]
                    for j in range(1, 4):
                        term = term + xx[..., i + j] * qq[..., i + j]
                    dot = dot + term
                groups = (np.arange(32) * 16 + at) // group
                scale = scales[column:column + count, groups][None]
                bias = biases[column:column + count, groups][None]
                bias_product = sums * bias
                fused = (scale.astype(np.float64) * dot.astype(np.float64) + bias_product).astype(np.float32)
                lanes = lanes + fused
            for distance in (1, 2, 4, 8, 16):
                lanes = lanes + lanes[..., np.arange(32) ^ distance]
            output[first:first + len(x), column:column + count] = lanes[..., 0]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--projection', required=True)
    parser.add_argument('--inputs', type=Path, required=True, help='BF16-representable numeric NumPy rows')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-raw', type=Path, help='Optional captured pre-BF16 FP32 NumPy output')
    args = parser.parse_args()
    result = affine_rows(Oracle(args.model_dir), args.projection, np.load(args.inputs, allow_pickle=False))
    np.save(args.output, result)
    receipt = {'shape': list(result.shape), 'projection': args.projection,
               'scope': 'Projection rounding replay; not a whole-model fidelity pass',
               'affine_fusion': ('Explicit CPU fmaf' if os.environ.get('LFM3_ROW_REPLAY_LIBRARY') else
                                 'FP64 multiply-add then FP32 round; not arbitrary IEEE fmaf proof'),
               'accepted_threshold': None}
    if args.native_raw:
        native = np.load(args.native_raw, allow_pickle=False)
        if native.shape != result.shape or native.dtype != np.float32:
            raise ValueError('Captured native output must have the same FP32 geometry')
        receipt.update(raw_equal=int(np.count_nonzero(native == result)), values=result.size,
                       bf16_equal=int(np.count_nonzero(bf(native) == bf(result))),
                       maximum_absolute_error=float(np.max(np.abs(native.astype(np.float64) - result))))
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')


def dense_rows(inputs, weight):
    inputs, weight = np.asarray(inputs), np.asarray(weight)
    if inputs.ndim != 2 or weight.ndim != 2 or inputs.shape[1] != weight.shape[1] or inputs.shape[1] % 32:
        raise ValueError('Expected compatible dense row matrices with complete 32-lane passes')
    if any(not np.isfinite(x).all() or not np.array_equal(x, bf(x)) for x in (inputs, weight)):
        raise ValueError('Dense replay requires finite BF16 operands')
    inputs, weight = np.ascontiguousarray(inputs, dtype=np.float32), np.ascontiguousarray(weight, dtype=np.float32)
    output = np.empty((len(inputs), len(weight)), dtype=np.float32)
    library = os.environ.get('LFM3_ROW_REPLAY_LIBRARY')
    if library:
        function = ctypes.CDLL(library).lfm3_dense_replay
        pointer = ctypes.POINTER(ctypes.c_float)
        function.argtypes = [pointer, pointer, pointer] + [ctypes.c_int] * 3
        function.restype = None
        function(inputs.ctypes.data_as(pointer), weight.ctypes.data_as(pointer), output.ctypes.data_as(pointer),
                 len(inputs), len(weight), inputs.shape[1])
        return output
    for row in range(0, len(inputs), 8):
        x = inputs[row:row + 8]
        for column in range(0, len(weight), 256):
            w = weight[column:column + 256]
            lanes = np.zeros((len(x), len(w), 32), dtype=np.float32)
            for at in range(0, inputs.shape[1], 32):
                lanes = (x[:, None, at:at + 32].astype(np.float64)
                         * w[None, :, at:at + 32].astype(np.float64) + lanes).astype(np.float32)
            for distance in (1, 2, 4, 8, 16):
                lanes = lanes + lanes[..., np.arange(32) ^ distance]
            output[row:row + len(x), column:column + len(w)] = lanes[..., 0]
    return output


if __name__ == '__main__':
    main()
