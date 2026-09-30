"""Packed CUDA format contracts and kernel address arithmetic without GPU runtimes."""

import ast
from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.cuda.kernels.affine import BITS, GROUPS, input_slice, packed_shape

ROOT = Path(__file__).parents[1] / "src/tensorfold"


class Array(np.ndarray):
    def to(self, dtype):
        return np.asarray(self).astype(dtype).view(Array)


def array(value):
    return np.asarray(value).view(Array)


class Pointer:
    def __init__(self, values, offsets=0):
        self.values, self.offsets = np.asarray(values).reshape(-1), offsets

    def __add__(self, offsets):
        return Pointer(self.values, self.offsets + offsets)


class Language:
    uint32, float32, bfloat16 = np.uint32, np.float32, np.float32
    program = (0, 0)

    @staticmethod
    def load(pointer, mask=True, other=0):
        offsets, mask = np.broadcast_arrays(pointer.offsets, mask)
        assert np.all((offsets[mask] >= 0) & (offsets[mask] < len(pointer.values)))
        result = np.full(offsets.shape, other, dtype=pointer.values.dtype)
        result[mask] = pointer.values[offsets[mask]]
        return array(result)

    @staticmethod
    def store(pointer, value, mask=True):
        offsets, value, mask = np.broadcast_arrays(pointer.offsets, value, mask)
        assert np.all((offsets[mask] >= 0) & (offsets[mask] < len(pointer.values)))
        pointer.values[offsets[mask]] = value[mask]

    @staticmethod
    def where(condition, a, b):
        return array(np.where(condition, a, b))

    @classmethod
    def program_id(cls, axis):
        return cls.program[axis]

    arange = staticmethod(lambda a, b: array(np.arange(a, b)))
    zeros = staticmethod(lambda shape, dtype: array(np.zeros(shape, dtype=dtype)))
    trans = staticmethod(lambda a: a.T)
    sum = staticmethod(lambda a, axis: array(np.sum(a, axis=axis)))
    dot = staticmethod(lambda a, b, **kw: array(np.sum(a[:, :, None] * b[None, :, :], axis=1)))


def kernels():
    tree = ast.parse((ROOT / "cuda/kernels/affine_kernels.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("codes", "matmul")]
    for function in functions:
        function.decorator_list = []
        for argument in function.args.args:
            argument.annotation = None
    namespace = {"tl": Language}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "affine_kernel_cpu_contract", "exec"), namespace)
    return namespace


def pack(values, bits):
    result = []
    for row in values:
        stream = sum(int(value) << (i * bits) for i, value in enumerate(row))
        result.append([(stream >> shift) & 0xFFFFFFFF for shift in range(0, len(row) * bits, 32)])
    return np.array(result, dtype=np.uint32)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("group", GROUPS)
def test_every_packed_format_crosses_words_and_rank_boundaries_exactly(bits, group):
    rng = np.random.default_rng(811)
    values = rng.integers(0, 1 << bits, size=(5, group * 4), dtype=np.uint32)
    words = pack(values, bits)
    assert packed_shape(words.shape, (5, 4), (5, 4), bits, group) == values.shape
    decode = kernels()["codes"]
    rows, columns = array(np.arange(8)[:, None]), array(np.arange(group * 4)[None, :])
    decoded = decode(Pointer(words), rows, columns, rows < 5, words.shape[1], bits)
    np.testing.assert_array_equal(decoded[:5], values)
    assert not decoded[5:].any()
    for rank in (0, 1):
        (a, b), (g0, g1) = input_slice(group * 4, bits, group, rank)
        part = words[:, a:b].copy()
        got = decode(Pointer(part), rows, array(np.arange(group * 2)[None, :]), rows < 5, b - a, bits)
        np.testing.assert_array_equal(got[:5], values[:, rank * group * 2:(rank + 1) * group * 2])
        assert (g0, g1) == (rank * 2, rank * 2 + 2)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("group", GROUPS)
def test_kernel_rows_match_solo_with_masked_output_and_partial_row_tiles(bits, group):
    rng = np.random.default_rng(93)
    k, n, m = group * 2, 35, 19
    values = rng.integers(0, 1 << bits, size=(n, k), dtype=np.uint32)
    packed = pack(values, bits)
    scales = np.full((n, 2), 0.25, dtype=np.float32)
    biases = np.full((n, 2), -0.5, dtype=np.float32)
    x = rng.integers(-2, 3, size=(m, k)).astype(np.float32) * 0.5
    kernel = kernels()["matmul"]
    def run(inputs):
        out = np.empty((len(inputs), n), dtype=np.float32)
        for row in range((len(inputs) + 15) // 16):
            for column in range((n + 31) // 32):
                Language.program = row, column
                kernel(Pointer(inputs), Pointer(packed), Pointer(scales), Pointer(biases), Pointer(out),
                       len(inputs), n, k, bits, group)
        return out
    actual = run(x)
    expected = np.sum(x[:, None, :] * (values.astype(np.float32) * 0.25 - 0.5)[None, :, :], axis=2)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[17:18], run(x[17:18]))


@pytest.mark.parametrize("bits,group,shape", [(1, 64, (4, 16)), (8, 16, (4, 64)), (3, 32, (4, 4))])
def test_bad_layout_is_rejected_before_dispatch(bits, group, shape):
    with pytest.raises(ValueError):
        packed_shape(shape, (4, 2), (4, 2), bits, group)


def test_parallel_split_refuses_partial_groups():
    with pytest.raises(ValueError):
        input_slice(384, 5, 128, 0)


def test_qlinear_preserves_packed_width_and_scale_precision():
    tree = ast.parse((ROOT / "families/qwen3_5/cuda/weights.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QLinear")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"dataclass": dataclass, "cached_property": cached_property, "torch": SimpleNamespace(bfloat16="bf16")}
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    exec(compile(module, "affine_weight_metadata", "exec"), namespace)
    QLinear = namespace["QLinear"]
    metadata = SimpleNamespace(shape=(17, 4), dtype="fp16")
    weight = SimpleNamespace(shape=(17, 24))
    q = QLinear(weight, metadata, metadata, bits=3, gs=64)
    assert (q.n, q.k, q.fast) == (17, 256, False)
    assert not QLinear(weight, metadata, metadata, bits=4, gs=64).fast
    metadata.dtype = "bf16"
    assert QLinear(weight, metadata, metadata, bits=4, gs=64).fast


def test_loader_resolves_each_module_and_keeps_metadata_precision():
    tree = ast.parse((ROOT / "families/qwen3_5/cuda/weights.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QLinear")
    load = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load")
    linear = next(node for node in load.body if isinstance(node, ast.FunctionDef) and node.name == "qlinear")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    class Tensor:
        def __init__(self, shape, dtype):
            self.shape, self.ndim, self.dtype = shape, len(shape), dtype
        def contiguous(self):
            return self
        def view(self, dtype):
            return Tensor(self.shape, dtype)
    raw = {"quantization": {"bits": 4, "group_size": 64, "language_model.lm_head": {"bits": 8, "group_size": 128},
                            "model.q_proj": {"bits": 3, "group_size": 32}, "model.a_proj": False}}
    tensors = {}
    for path, bits, group, dtype in [("lm_head", 8, 128, "fp32"), ("model.q_proj", 3, 32, "fp16")]:
        tensors["language_model." + path + ".weight"] = Tensor((17, 256 * bits // 32), "u32")
        for suffix in (".scales", ".biases"):
            tensors["language_model." + path + suffix] = Tensor((17, 256 // group), dtype)
    tensors["language_model.model.a_proj.weight"] = Tensor((17, 256), "fp32")
    torch = SimpleNamespace(int32="i32", uint32="u32", bfloat16="bf16", float16="fp16", float32="fp32")
    namespace = {"dataclass": dataclass, "cached_property": cached_property, "torch": torch, "raw": raw,
                 "prefix": "language_model.", "t": tensors, "get": lambda name: tensors.pop("language_model." + name),
                 "tiled": False}
    module = ast.fix_missing_locations(ast.Module(body=[future, cls, linear], type_ignores=[]))
    exec(compile(module, "affine_loader_contract", "exec"), namespace)
    head, query, dense = (namespace["qlinear"](path) for path in ("lm_head", "model.q_proj", "model.a_proj"))
    assert (head.bits, head.gs, head.scales.dtype, head.k) == (8, 128, "fp32", 256)
    assert (query.bits, query.gs, query.scales.dtype, query.k) == (3, 32, "fp16", 256)
    assert (dense.layout, dense.k, dense.weight.dtype) == ("dense", 256, "fp32")
    assert not tensors


def test_generic_memory_counts_eight_bit_words_without_four_bit_padding(tmp_path):
    from tensorfold.families.qwen3_5.cuda.affine_memory import weight_transform

    config = {"quantization": {"bits": 8, "group_size": 32}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    prefix = "language_model.model.layers.0.self_attn.q_proj"
    entries, offset = {}, 0
    for suffix, dtype, shape, item in [("weight", "U32", [35, 64], 4),
                                      ("scales", "BF16", [35, 8], 2), ("biases", "BF16", [35, 8], 2)]:
        size = int(np.prod(shape)) * item
        entries[prefix + "." + suffix] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(entries).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    transform = weight_transform(tmp_path)
    assert transform(prefix + ".weight", entries[prefix + ".weight"]) == (35 * 64 * 4, 0)


def test_mixed_prefill_tiles_raw_fast_weights_and_preserves_fp8_dispatch():
    tree = ast.parse((ROOT / "families/qwen3_5/cuda/prefill.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_mm")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    seen = []
    def tile(weight):
        if weight.fast and weight.layout == "mlx":
            return SimpleNamespace(fast=True, layout="tiled", bits=weight.bits, n=256)
        return weight
    def project(x, weight, *, f32=False):
        if weight.fast:
            assert weight.layout == "tiled"
        seen.append((x, weight, f32))
        return "projection"
    class QLinear(SimpleNamespace):
        pass
    namespace = {"tile": tile, "matmul": project, "matmul_partial": lambda x, w: project(x, w, f32=True),
                 "shared": SimpleNamespace(prefill_matmul8=lambda x, w, **kw: (project(x, w, **kw), "fp8"),
                                           prefill_matmul=lambda x, w, f32=False, tile=None: project(x, w, f32=f32),
                                           prompt_tile=lambda rows, n: 9),     # bf16 rows on the prompt matmul
                 "QLinear": QLinear}
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    exec(compile(module, "mixed_prefill_dispatch", "exec"), namespace)
    rows = SimpleNamespace(shape=(4096, 256))
    four = QLinear(fast=True, layout="mlx", bits=4)
    other = QLinear(fast=False, layout="mlx", bits=8)
    exl3 = SimpleNamespace(prefill=lambda x: ("exl3", x))           # an EXL3 pack's projection takes bf16 rows
    assert namespace["_mm"](rows, exl3) == ("exl3", rows)
    assert namespace["_mm"](rows, four) == "projection"
    assert seen[-1][0].shape[0] == 4096 and seen[-1][1].layout == "tiled"
    assert namespace["_mm"](rows, four, f32=True) == "projection" and seen[-1][2]
    namespace["_mm"](rows, other)
    assert seen[-1][1] is other
    assert namespace["_mm"](("bytes", "sums", "scales"), four) == ("projection", "fp8")


def test_local_input_splits_group32_inputs_that_are_only64_columns_wide():
    tree = ast.parse((ROOT / "families/qwen3_5/cuda/distributed.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_rank", "local_input")]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {}
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    exec(compile(module, "affine_activation_split", "exec"), namespace)
    class Matrix(np.ndarray):
        def contiguous(self):
            return self.copy()
    x = np.arange(4 * 64).reshape(4, 64).view(Matrix)
    for rank in (0, 1):
        np.testing.assert_array_equal(namespace["local_input"](x, rank, group_size=32),
                                      x[:, rank * 32:(rank + 1) * 32])
    with pytest.raises(ValueError):
        namespace["local_input"](x, 0)
    with pytest.raises(ValueError):
        namespace["local_input"](x, 0, group_size=128)
