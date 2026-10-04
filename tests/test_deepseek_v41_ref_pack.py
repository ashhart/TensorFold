"""The reference's pack reader: tensors by name through symlinked shards, and every dense format to fp32 exactly."""

from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")

from dsv41_ref_pack import RefPack
from safetensors.torch import save_file

MODEL = os.environ.get("TF_DSV41_MODEL")
needs_model = pytest.mark.skipif(not MODEL, reason="set TF_DSV41_MODEL to the checkpoint")


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int32)


def _e4m3_values(q: torch.Tensor) -> torch.Tensor:
    """e4m3fn bytes to float64 from the bit fields: sign, 4-bit exponent (bias 7), 3-bit mantissa, subnormals."""

    b = q.contiguous().view(torch.uint8).to(torch.int64)
    sign = 1.0 - 2.0 * (b >> 7).to(torch.float64)
    e, m = ((b >> 3) & 0xF).to(torch.float64), (b & 7).to(torch.float64)
    mag = torch.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * torch.pow(2.0, e - 7))
    return sign * mag


def _e2m1_values(words: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 to float64 from the bit fields (low nibble = even element): sign, 2-bit exponent, 1-bit mantissa."""

    b = words.contiguous().view(torch.uint8).to(torch.int64)
    c = torch.stack([b & 0xF, b >> 4], dim=-1).reshape(b.shape[0], -1)
    sign = 1.0 - 2.0 * (c >> 3).to(torch.float64)
    e, m = ((c >> 1) & 3).to(torch.float64), (c & 1).to(torch.float64)
    return sign * torch.where(e == 0, m / 2, (1 + m / 2) * torch.pow(2.0, e - 1))


def _oracle(values: torch.Tensor, scale: torch.Tensor, block_rows: int, first: int = 0) -> torch.Tensor:
    """W[o, i] = v[o, i] * 2^(S[(first + o) // block_rows, i // 32] - 127), one 32-input block column at a time."""

    s = scale.contiguous().view(torch.uint8).to(torch.float64)
    owner = (torch.arange(values.shape[0]) + first) // block_rows
    out = torch.empty_like(values)
    for c in range(s.shape[1]):
        cols = slice(32 * c, 32 * c + 32)
        out[:, cols] = values[:, cols] * torch.pow(2.0, s[owner, c] - 127)[:, None]
    return out.to(torch.float32)


def _e4m3(gen: torch.Generator, n: int, k: int) -> torch.Tensor:
    b = torch.randint(0, 256, (n, k), generator=gen, dtype=torch.int32)
    return torch.where((b & 0x7F) == 0x7F, b - 1, b).to(torch.uint8).view(torch.float8_e4m3fn)   # no NaN codes


def _e8m0(gen: torch.Generator, *shape: int) -> torch.Tensor:
    s = torch.randint(0, 255, shape, generator=gen, dtype=torch.uint8)   # every finite byte, 2^-127 included
    s.view(-1)[0] = 0
    return s


@pytest.fixture
def pack(tmp_path):
    """A two-shard checkpoint whose second shard is a symlink into a sibling directory, as the pack's are."""

    gen = torch.Generator().manual_seed(0)
    first = {
        "a.wq.weight": _e4m3(gen, 70, 96), "a.wq.scale": _e8m0(gen, 3, 3).view(torch.float8_e8m0fnu),
        "a.norm.weight": torch.randn(96, generator=gen).to(torch.bfloat16),
        "a.sink": torch.randn(8, generator=gen),
    }
    second = {
        "lm_head.weight": _e4m3(gen, 40, 64), "lm_head.weight_scale": _e8m0(gen, 40, 2),
        "x.experts.0.w1.weight": torch.randint(-128, 128, (48, 64), generator=gen, dtype=torch.int8),
        "x.experts.0.w1.scale": _e8m0(gen, 48, 4).view(torch.float8_e8m0fnu),
    }
    snap, sibling = tmp_path / "snap", tmp_path / "sibling"
    snap.mkdir()
    sibling.mkdir()
    save_file(first, str(snap / "one.safetensors"))
    save_file(second, str(sibling / "two.safetensors"))
    (snap / "two.safetensors").symlink_to(sibling / "two.safetensors")
    where = {**{k: "one.safetensors" for k in first}, **{k: "two.safetensors" for k in second}}
    (snap / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": where}))
    return RefPack(snap), {**first, **second}


def test_tensor_reads_the_stored_bytes_through_a_symlinked_shard(pack):
    ref, stored = pack
    assert not ref.path("lm_head.weight").is_symlink() and ref.path("lm_head.weight").parent.name == "sibling"
    for name, want in stored.items():
        got = ref.tensor(name)
        assert got.dtype == want.dtype and torch.equal(got.view(torch.uint8), want.view(torch.uint8)), name
    assert torch.equal(ref.tensor("a.wq.weight", slice(5, 9)).view(torch.uint8), stored["a.wq.weight"][5:9].view(torch.uint8))


@pytest.mark.parametrize("rows", [None, slice(0, 32), slice(33, 70), slice(31, 33)])
def test_fp8_block_equals_the_per_block_formula(pack, rows):
    ref, stored = pack
    first = 0 if rows is None else rows.start
    w = stored["a.wq.weight"][rows or slice(None)]
    want = _oracle(_e4m3_values(w), stored["a.wq.scale"], 32, first)
    assert torch.equal(_bits(ref.dense_fp32("a.wq", rows)), _bits(want))


@pytest.mark.parametrize("rows", [None, slice(7, 23)])
def test_mxfp8_equals_the_per_block_formula(pack, rows):
    ref, stored = pack
    take = rows or slice(None)
    want = _oracle(_e4m3_values(stored["lm_head.weight"][take]), stored["lm_head.weight_scale"][take], 1)
    assert torch.equal(_bits(ref.dense_fp32("lm_head", rows)), _bits(want))


@pytest.mark.parametrize("rows", [None, slice(16, 48)])
def test_mxfp4_equals_the_per_block_formula(pack, rows):
    ref, stored = pack
    take = rows or slice(None)
    words, scale = stored["x.experts.0.w1.weight"][take], stored["x.experts.0.w1.scale"][take]
    got = ref.dense_fp32("x.experts.0.w1", rows)
    assert got.shape == (words.shape[0], 128) and torch.equal(_bits(got), _bits(_oracle(_e2m1_values(words), scale, 1)))


def test_low_nibble_is_the_even_element(tmp_path):
    save_file({"m.weight": torch.full((2, 16), 0x21, dtype=torch.uint8).view(torch.int8),
               "m.scale": torch.full((2, 1), 127, dtype=torch.uint8)}, str(tmp_path / "m.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"m.weight": "m.safetensors", "m.scale": "m.safetensors"}}))
    row = RefPack(tmp_path).dense_fp32("m")[0]
    assert row[0::2].eq(0.5).all() and row[1::2].eq(1.0).all()


def test_plain_floats_widen_and_unscaled_fp8_is_refused(pack):
    ref, stored = pack
    assert torch.equal(ref.dense_fp32("a.norm"), stored["a.norm.weight"].to(torch.float32))
    assert torch.equal(ref.dense_fp32("a.sink"), stored["a.sink"])
    with pytest.raises(ValueError, match="no scale"):
        ref.dense_fp32("a.wq.weight")


def _raw(name: str, rows: slice | None = None) -> torch.Tensor:
    root = os.path.realpath(MODEL)
    with open(os.path.join(root, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][name]
    with safetensors.safe_open(os.path.join(root, shard), framework="pt") as f:
        return f.get_tensor(name) if rows is None else f.get_slice(name)[rows]


@needs_model
@pytest.mark.parametrize("name", ["layers.0.attn.wq_a", "layers.20.attn.wo_b"])
def test_real_fp8_block_matches(name):
    want = _oracle(_e4m3_values(_raw(f"{name}.weight")), _raw(f"{name}.scale"), 32)
    assert torch.equal(_bits(RefPack(MODEL).dense_fp32(name)), _bits(want))


@needs_model
def test_real_lm_head_rows_match():
    rows = slice(0, 64)
    want = _oracle(_e4m3_values(_raw("lm_head.weight", rows)), _raw("lm_head.weight_scale", rows), 1)
    got = RefPack(MODEL).dense_fp32("lm_head", rows)
    assert got.shape == (64, 5120) and torch.equal(_bits(got), _bits(want))


@needs_model
def test_real_dspark_expert_matches():
    name = "mtp.0.ffn.experts.0.w1"
    want = _oracle(_e2m1_values(_raw(f"{name}.weight")), _raw(f"{name}.scale"), 1)
    got = RefPack(MODEL).dense_fp32(name)
    assert got.shape == (2304, 5120) and torch.equal(_bits(got), _bits(want))
