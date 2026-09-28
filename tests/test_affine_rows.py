"""CPU checks exercise packed extraction, launch invariants and row-backend format dispatch without MLX."""

from __future__ import annotations

import ast
import ctypes
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from tensorfold.kernels.qwen.dense.v1 import affine_rows

KERNELS = Path(affine_rows.__file__).parent


class Array:
    def __init__(self, shape, dtype="bf16"):
        self.shape, self.dtype = tuple(shape), dtype
        self.ndim, self.size = len(self.shape), math.prod(self.shape)

    def reshape(self, *shape):
        if -1 in shape:
            shape = tuple(self.size // -math.prod(shape) if d == -1 else d for d in shape)
        assert math.prod(shape) == self.size
        return Array(shape, self.dtype)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, end, step = index.indices(self.shape[0])
            return Array((len(range(start, end, step)), *self.shape[1:]), self.dtype)
        raise AssertionError(index)

    def __add__(self, other):
        return Array(self.shape, self.dtype)


def concatenate(arrays, axis=0):
    shape = list(arrays[0].shape)
    shape[axis] = sum(a.shape[axis] for a in arrays)
    return Array(shape, arrays[0].dtype)


class Linear:
    def __init__(self, n=8, k=128, bits=4, gs=64, mode="affine", dtype="bf16"):
        self.bits, self.group_size, self.mode = bits, gs, mode
        self.weight = Array((n, k * bits // 32), "u32")
        self.scales, self.biases = Array((n, k // gs), dtype), Array((n, k // gs), dtype)

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)


@pytest.fixture
def mx(monkeypatch):
    fake = ModuleType("mlx.core")
    fake.bfloat16, fake.float16, fake.float32, fake.uint32 = "bf16", "f16", "f32", "u32"
    fake.int32 = "i32"
    fake.array, fake.contiguous, fake.concatenate = Array, lambda value: value, concatenate
    fake.eval, fake.clear_cache = lambda *args: None, lambda: None
    fake.zeros = lambda shape, dtype: Array(shape, dtype)
    nn, mlx = ModuleType("mlx.nn"), ModuleType("mlx")
    nn.QuantizedLinear = Linear
    mlx.core, mlx.nn = fake, nn
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)
    inputs = ModuleType("tensorfold.kernels.inputs")
    inputs.mx, inputs.MIN_ELEMENTS = fake, 8
    load_definitions("../../../inputs.py", inputs.__dict__, {"padded"})
    monkeypatch.setitem(sys.modules, inputs.__name__, inputs)
    return fake


@pytest.fixture(scope="module")
def unpack(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler checks the Metal header's exact packed-bit expression on the CPU")
    folder = tmp_path_factory.mktemp("affine-packed-cpu")
    source, library = folder / "packed.cpp", folder / "packed.so"
    switches = "".join(f"case {bits}: return code_at<{bits}>(words + index / 32 * {bits}, index % 32);"
                       for bits in affine_rows.BITS)
    shim = ("#include <cstdint>\n#include <cstring>\nusing uint = std::uint32_t;\n#define device\n#define thread\n"
            "template <typename T, typename U> T as_type(U u) { T t; std::memcpy(&t, &u, sizeof t); return t; }\n")
    source.write_text(shim + affine_rows._HEADER
                      + '\nextern "C" uint decode(const uint* words, int index, int bits) { switch(bits) {'
                      + switches + "default: return 0;} }\n")
    subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-O1", str(source), "-o", str(library)],
                   check=True, capture_output=True, timeout=30)
    compiled = ctypes.CDLL(str(library))
    compiled.decode.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_int)
    compiled.decode.restype = ctypes.c_uint32
    return compiled.decode


@pytest.mark.parametrize("bits", affine_rows.BITS)
@pytest.mark.parametrize("group", affine_rows.GROUP_SIZES)
def test_shader_extracts_exact_codes_across_uint32_and_group_boundaries(unpack, bits, group):
    values = [(i * 29 + i // 7) % (1 << bits) for i in range(group * 3)]
    packed = sum(value << (i * bits) for i, value in enumerate(values))
    count = len(values) * bits // 32
    words = (ctypes.c_uint32 * count)(*((packed >> (32 * i)) & 0xFFFFFFFF for i in range(count)))
    assert [unpack(words, i, bits) for i in range(len(values))] == values


@pytest.mark.parametrize("bits", affine_rows.BITS)
@pytest.mark.parametrize("group", affine_rows.GROUP_SIZES)
def test_declared_layout_matches_packed_words_and_groups(mx, bits, group):
    module = Linear(n=7, k=384, bits=bits, gs=group)
    assert affine_rows.fits(module)
    assert affine_rows.shape(module.weight, module.scales, module.biases, group, bits) == (7, 384)


@pytest.mark.parametrize("bits,group,mode", [(1, 64, "affine"), (7, 64, "affine"), (8, 16, "affine"),
                                           (4, 64, "mxfp4"), (8, 64, "mxfp8")])
def test_unimplemented_formats_are_not_advertised(bits, group, mode):
    assert not affine_rows.readable(bits, group, mode)


def test_fits_refuses_tiled_weights_missing_groups_and_mismatched_dtypes(mx):
    module = Linear(bits=3, gs=32)
    module._lane_tiled = True
    assert not affine_rows.fits(module)
    module._lane_tiled = False
    module.scales = Array((8, 3))
    assert not affine_rows.fits(module)
    module.scales = Array((8, 4), "f16")
    assert not affine_rows.fits(module)
    module.biases = Array((8, 4), "f16")
    assert affine_rows.fits(module)


def _recording(monkeypatch):
    calls = []

    def factory(k, n, bits, group_size, ops, rt):
        def kernel(**kwargs):
            calls.append({"shape": (k, n, bits, group_size), "ops": ops, "rt": rt, **kwargs})
            return [Array(kwargs["output_shapes"][0])]
        return kernel

    monkeypatch.setattr(affine_rows, "_kernel", factory)
    return calls


@pytest.mark.parametrize("bits", affine_rows.BITS)
def test_every_width_launches_256_threads_and_one_kernel_per_row_bucket(mx, monkeypatch, bits):
    calls = _recording(monkeypatch)
    module = Linear(n=37, k=512, bits=bits, gs=128)
    widths = (1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 33, 128)
    for rows in widths:
        result = affine_rows.qmm(Array((1, rows, 512)), module.weight, module.scales, module.biases, 128, bits)
        assert result.shape == (1, rows, 37) and result.dtype == "bf16"
    assert all(call["shape"] == (512, 37, bits, 128) for call in calls)
    assert all(call["threadgroup"] == (32 * affine_rows.SG, 1, 1) for call in calls) and 32 * affine_rows.SG <= 256
    for rows, call in zip(widths, calls):
        ops, rt = affine_rows.launch(rows)
        assert (call["ops"], call["rt"]) == (ops, rt) and ops * rt <= 64
        assert call["grid"] == (-(-37 // (affine_rows.SG * ops)) * 32 * affine_rows.SG, -(-rows // rt), 1)
        assert rows <= rt or rt == 8


@pytest.mark.parametrize("bits", affine_rows.BITS)
@pytest.mark.parametrize("group", affine_rows.GROUP_SIZES)
def test_tiny_kernel_inputs_are_padded_without_changing_logical_layout(mx, monkeypatch, bits, group):
    calls = _recording(monkeypatch)
    module = Linear(n=1, k=group, bits=bits, gs=group)
    result = affine_rows.qmm(Array((1, group)), module.weight, module.scales, module.biases, group, bits)
    assert result.shape == (1, 1)
    assert all(array.size >= 8 for array in calls[0]["inputs"])
    assert calls[0]["shape"] == (group, 1, bits, group)
    if module.weight.size >= 8:
        assert calls[0]["inputs"][1] is module.weight
    else:
        assert calls[0]["inputs"][1].shape == (8,)


@pytest.mark.parametrize("x", [Array((2, 127)), Array((2, 128), "f32"), Array((128,)), Array((0, 128))])
def test_bad_inputs_fail_before_a_kernel_is_requested(mx, monkeypatch, x):
    monkeypatch.setattr(affine_rows, "_kernel", lambda: pytest.fail("invalid inputs reached Metal"))
    module = Linear(bits=8, gs=128)
    with pytest.raises(ValueError):
        affine_rows.qmm(x, module.weight, module.scales, module.biases, 128, 8)


def load_definitions(filename, namespace, names):
    tree = ast.parse((KERNELS / filename).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), filename, "exec"), namespace)
    return namespace


@pytest.fixture
def backend(mx, monkeypatch):
    calls = []
    simd = ModuleType("tensorfold.kernels.qwen.dense.v1.simd_qmm")
    simd.MAX_ROWS, simd.GROUP, simd.mma_one_row = 65536, 64, set()
    simd.check = lambda w, s, b, group_size: calls.append(("check", group_size)) or False
    simd.fragments = lambda x: x
    simd.qmm_fragments = lambda x, w, s, b: calls.append(("fragments",)) or Array((x.size // x.shape[-1], w.shape[0]))
    simd.qmm = lambda x, w, s, b, gs: calls.append(("fast", gs)) or Array((*x.shape[:-1], w.shape[0]))
    monkeypatch.setitem(sys.modules, simd.__name__, simd)
    # `from package import simd_qmm` reads the package attribute first, set once the real module was imported
    monkeypatch.setattr(sys.modules["tensorfold.kernels.qwen.dense.v1"], "simd_qmm", simd, raising=False)
    monkeypatch.setattr(affine_rows, "qmm", lambda x, w, s, b, gs, bits:
                        calls.append(("generic", gs, bits)) or Array((*x.shape[:-1], w.shape[0])))
    namespace = {"mx": mx, "FRAGMENT_ROWS": 8, "_ATTR": "_stacks", "BACKEND": None}
    names = {"Backend", "Stack", "_stackable", "project", "project_stack", "logits", "simd_qmm_backend"}
    load_definitions("row_matmul.py", namespace, names)
    namespace["BACKEND"] = namespace["simd_qmm_backend"]()
    return NS(api=namespace, calls=calls, simd=simd)


@pytest.mark.parametrize("bits,gs,rows,kind", [(4, 64, 1, ("fast", 64)), (4, 32, 4, ("fast", 32)),
                                            (4, 64, 7, ("fragments",)), (4, 128, 1, ("generic", 128, 4)),
                                            (3, 32, 7, ("generic", 32, 3)), (8, 64, 16, ("generic", 64, 8))])
def test_existing_fast_paths_and_new_formats_dispatch_separately(backend, bits, gs, rows, kind):
    module = Linear(bits=bits, gs=gs)
    assert backend.api["project"](module, Array((1, rows, 128))).shape == (1, rows, 8)
    assert backend.calls == [kind]


def test_prepare_keeps_four_bit_checks_and_warms_generic_formats(backend):
    modules = [Linear(bits=4), Linear(bits=6, gs=32), Linear(bits=8, gs=128)]
    backend.api["BACKEND"].prepare([(m.weight, m.scales, m.biases, m.group_size, m.bits) for m in modules])
    assert backend.calls == [("check", 64), ("generic", 32, 6), ("generic", 128, 8)]
    assert backend.simd.mma_one_row == {(8, 128, 64)}


def test_backend_preserves_legacy_five_argument_four_bit_callbacks(backend):
    calls = []

    def qmm(x, weight, scales, biases, group):
        calls.append((x, weight, scales, biases, group))
        return "legacy"

    target = backend.api["Backend"]("legacy", qmm, 16, lambda module: True)
    assert target("x", "w", "s", "b", 64) == "legacy"
    assert target("x", "w", "s", "b", 32, 4) == "legacy"
    assert calls == [("x", "w", "s", "b", 64), ("x", "w", "s", "b", 32)]


def test_stacks_do_not_combine_bit_widths_or_quantization_groups(backend):
    stackable, target = backend.api["_stackable"], backend.api["BACKEND"]
    assert stackable([Linear(bits=6), Linear(bits=6)], target)
    assert not stackable([Linear(bits=4), Linear(k=64, bits=8)], target)
    assert not stackable([Linear(bits=6, gs=32), Linear(bits=6, gs=64)], target)
    assert not stackable([Linear(dtype="bf16"), Linear(dtype="f16")], target)


def test_stack_keeps_explicit_bits_and_detects_changed_member_format(backend):
    members = [Linear(bits=5), Linear(bits=5)]
    stack = backend.api["Stack"](members)
    assert stack.bits == 5 and stack.group_size == 64 and stack.valid()
    backend.api["project_stack"](stack, Array((1, 3, 128)))
    assert backend.calls == [("generic", 64, 5)]
    members[0].bits = 4
    assert not stack.valid()


def test_direct_stack_construction_rejects_mixed_formats(backend):
    with pytest.raises(ValueError, match="share their bit width"):
        backend.api["Stack"]([Linear(bits=4), Linear(k=64, bits=8)])


def test_mixed_gdn_projections_are_forwarded_in_convolution_layout(mx, monkeypatch):
    stub = ModuleType("tensorfold.kernels.qwen.dense.v1.row_streams")
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    modules = [Linear(n=width, bits=bits) for width, bits in zip((16, 8, 2, 2), (2, 3, 5, 8))]
    calls = []
    namespace = {"mx": mx, "row_matmul": NS(GROUPS={"in": names}), "stack_of": lambda *args: None,
                 "project": lambda module, x: calls.append(module.bits) or Array((1, 2, module.weight.shape[0])),
                 "_recur": lambda gdn, y, *args: y,
                 "gdn_post": lambda rec, y, *args, **kwargs: rec}
    load_definitions("row_forward.py", namespace, {"_gdn"})
    gdn = NS(**dict(zip(names, modules)), conv_kernel_size=4, conv_dim=16, norm=NS(weight=None, eps=1e-6))
    rows = NS(single=True, windows=[None], parents=[(-1, 0)], chains=[True], records=[[]])
    out = namespace["_gdn"](gdn, Array((1, 2, 128)), [None], rows)
    assert calls == [2, 3, 5, 8] and out.shape == (1, 2, 28)
