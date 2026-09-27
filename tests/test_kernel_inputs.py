"""Every input of a custom kernel stays on one side of MLX's constant/device size line (8 elements) at every row and
stream count, so each kernel name has one source (tiny random shapes)."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.engine import gpu_sampling  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.kernels import inputs  # noqa: E402
from tensorfold.kernels.nemotron.lightning.v1 import kernels as K  # noqa: E402
from tensorfold.kernels.nemotron.lightning.v1 import rows  # noqa: E402


def test_padding():
    assert inputs.ints([3, 4]).tolist() == [3, 4, 0, 0, 0, 0, 0, 0]
    assert inputs.ints(range(9)).tolist() == list(range(9))
    assert inputs.floats([0.5, 2.0]).tolist() == [0.5, 2.0] + [0.0] * 6
    assert inputs.floats([1.0] * 9).dtype == mx.float32
    assert inputs.padded(mx.array([[1.5, 2.5]])).tolist() == [1.5, 2.5] + [0.0] * 6
    big = mx.arange(8)
    assert inputs.padded(big) is big


@pytest.fixture
def sizes(monkeypatch):
    """Each custom-kernel call's input sizes, by kernel name."""

    seen: dict[str, list[list[int]]] = {}
    real = mx.fast.metal_kernel

    def recording(**spec):
        kernel = real(**spec)

        def call(*, inputs, **kwargs):
            seen.setdefault(spec["name"], []).append([int(a.size) for a in inputs])
            return kernel(inputs=inputs, **kwargs)

        return call

    monkeypatch.setattr(mx.fast, "metal_kernel", recording)
    monkeypatch.setattr(K, "_kernels", {})
    monkeypatch.setattr(rows, "_kernels", {})
    monkeypatch.setattr(gpu_sampling, "_kernel", None)
    monkeypatch.setattr(gpu_sampling, "_kernel_ids", None)
    return seen


def _one_side(seen):
    assert seen
    for name, calls in seen.items():
        for i, column in enumerate(zip(*calls)):
            small = {n < inputs.MIN_ELEMENTS for n in column}
            assert len(small) == 1, f"{name} input {i}: sizes {sorted(set(column))} cross {inputs.MIN_ELEMENTS}"


class _Table:
    def __init__(self, experts, dims, hidden):
        self.fc1 = nn.QuantizedLinear(dims, hidden, bias=False, group_size=64, bits=4)
        self.fc2 = nn.QuantizedLinear(hidden, dims, bias=False, group_size=64, bits=4)
        for linear, n, k, seed in ((self.fc1, hidden, dims, 1), (self.fc2, dims, hidden, 2)):
            w = (mx.random.normal((experts, n, k), key=mx.random.key(seed)) * 0.05).astype(mx.bfloat16)
            linear.weight, linear.scales, linear.biases = mx.quantize(w, group_size=64, bits=4)


def test_nemotron_kernels(sizes):
    table = _Table(experts=12, dims=640, hidden=192)
    x = (mx.random.normal((5, 640), key=mx.random.key(3)) * 0.5).astype(mx.bfloat16)
    ids = mx.argsort(-mx.random.uniform(shape=(5, 12), key=mx.random.key(4)), axis=-1)[:, :6].astype(mx.uint32)
    for m in (1, 2, 5):
        for grouped in (False, True):
            mx.eval(rows.experts(table, x[:m], ids[:m], grouped=grouped))

    dims, eps = 512, mx.array([1e-5], dtype=mx.float32)
    weight = mx.ones((dims,), dtype=mx.bfloat16)
    for m in (1, 2, 3):
        h = mx.random.normal((m, dims)).astype(mx.bfloat16)
        routed = mx.random.normal((m, 6, dims)).astype(mx.bfloat16)
        mx.eval(K.add_norm_moe(h, routed, mx.random.uniform(shape=(m, 6)), h, weight, eps))
        mx.eval(K.add_norm(h, h, weight, eps, group_sums=True))

    heads, head_dim, groups, state = 4, 8, 2, 32
    conv_dim = heads * head_dim + 2 * groups * state
    params = (mx.random.normal((4, conv_dim)) * 0.5, mx.zeros((conv_dim,)), mx.zeros((heads,)), mx.ones((heads,)),
              mx.zeros((heads,)), mx.array([0.0, 1e4]))
    for lengths in ((1,), (2,), (9,), (1, 1), (4, 5), (1,) * 9):
        n = len(lengths)
        proj = mx.random.normal((sum(lengths), heads * head_dim + conv_dim + heads)).astype(mx.bfloat16)
        conv = mx.random.normal((n, 3, conv_dim)).astype(mx.bfloat16)
        ssm = mx.random.normal((n, heads, head_dim, state)) * 0.1
        mx.eval(K.mamba_scan(proj, conv, ssm, lengths, *params, heads=heads, head_dim=head_dim, groups=groups,
                             state_dim=state))
    _one_side(sizes)


def test_sampler(sizes):
    sampling = Sampling(seed=5, temperature=1.0, top_k=20, top_p=0.95)
    ids = mx.arange(0, 512, 2).astype(mx.uint32)
    for count in (1, 3, 4, 8, 9):
        logits = mx.random.normal((count, 256))
        mx.eval(gpu_sampling.sample(logits, sampling, list(range(count))))
        mx.eval(gpu_sampling.sample(logits, sampling, mx.arange(count)))
        mx.eval(gpu_sampling.sample(logits, sampling, list(range(count)), ids=ids))
    _one_side(sizes)


def test_rows_with_their_own_settings_draw_what_they_draw_alone():
    settings = [Sampling(seed=1234, temperature=1.0, top_k=20, top_p=0.95), None,
                Sampling(seed=99, temperature=0.7, top_k=0, top_p=0.8), Sampling(seed=7, temperature=1.3)]
    logits = mx.random.normal((7, 300), key=mx.random.key(5)) * 3
    rows = [settings[r % 4] for r in range(7)]
    positions = [40 + r for r in range(7)]
    together = gpu_sampling.sample_rows(logits, rows, positions)
    alone = [gpu_sampling.sample(logits[r:r + 1], rows[r], [positions[r]]) for r in range(7)]
    assert together.tolist() == mx.concatenate(alone).tolist()
