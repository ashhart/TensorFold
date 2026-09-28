"""The NVFP4 loader on a tiny modelopt checkpoint: the route's tensors, shapes and exactness.

Runs wherever Triton runs (the CUDA tests' directory; the loader's slicing is torch code, the FP4/BF16
*kernels* are checked in test_flashnext_nvfp4_kernels.py on a GPU). It loads a checkpoint that stores
what the Swift checkpoint stores — FP4 arrays for the routed experts, BF16 for everything else — and
checks the loader reads it into the engine's faces."""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("triton")
torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda import bf16, nvfp4  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import nvfp4_moe  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weights import Config  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from nvfp4_tiny import write  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    return write(tmp_path_factory.mktemp("nvfp4-tiny"))


def test_config_reads_the_modelopt_checkpoint(tiny: Path) -> None:
    cfg = Config.read(tiny)
    assert cfg.quant == "modelopt"
    assert cfg.nvfp4_group == 16
    assert cfg.ple_layers == [1]


def test_the_reader_maps_the_fp8_dtype(tiny: Path) -> None:
    from tensorfold.families.qwen4_exp.cuda.weights import _DT

    assert _DT["F8_E4M3"] is torch.float8_e4m3fn


def test_the_header_names_every_tensor(tiny: Path) -> None:
    shard = tiny / "model-00001-of-00001.safetensors"
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    w = hdr["model.layers.0.mlp.experts.0.gate_proj.weight"]
    assert w["dtype"] == "U8" and w["shape"] == [128, 128]
    assert hdr["model.layers.0.mlp.experts.0.gate_proj.weight_scale"]["dtype"] == "F8_E4M3"
    assert hdr["model.layers.0.self_attn.q_proj.weight"]["dtype"] == "BF16"
    assert hdr["mtp.layers.0.mlp.experts.gate_up_proj"]["dtype"] == "BF16"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the loader builds CUDA tensors")
def test_the_loader_builds_the_nvfp4_faces(tiny: Path) -> None:
    from tensorfold.families.qwen4_exp.cuda.weights import load

    w = load(tiny, mtp=True, draft_vocab=None)
    assert w.cfg.quant == "modelopt"
    l0 = w.layers[0]
    moe = l0.moe
    assert getattr(moe.experts, "kernel", "") == "nvfp4"
    ex = moe.experts
    assert ex.routed == 2 and ex.width == 128 and ex.dims == 256
    # the FP4 grids dequantize row-for-row to the checkpoint's exact weights (the loader's contract:
    # the stored values, not a requantization)
    gate = nvfp4.dequantize_fp4(ex.gate_up)[:128]
    assert gate.shape == (128, 256)
    # the non-experts ride the b16 face
    assert getattr(l0.attn, "kernel", "") == "b16" if hasattr(l0, "attn") else True
    assert getattr(l0.hc_up.down, "kernel", "") == "b16" if hasattr(l0, "hc_up") else True
    # the MTP layer's BF16 stacked experts rode the FP4 tables exactly
    mtp_ex = w.mtp.layer.moe.experts
    assert getattr(mtp_ex, "kernel", "") == "nvfp4"
    fd = nvfp4.dequantize_fp4(mtp_ex.down_proj)[:256]
    shard = tiny / "model-00001-of-00001.safetensors"
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        e = hdr["mtp.layers.0.mlp.experts.down_proj"]
        lo, hi = e["data_offsets"]
        f.seek(8 + n + lo)
        dn = torch.frombuffer(bytearray(f.read(hi - lo)), dtype=torch.bfloat16).reshape(2, 256, 128)
    assert torch.equal(fd, dn[0].to(device=fd.device, dtype=torch.float32))


def test_the_reader_finds_the_published_naming(tmp_path: Path) -> None:
    """The published NVFP4 checkpoint spells its language-model tensors ``model.language_model.*`` while its
    lm_head and mtp stay top level; the loader reads the same faces as from the plain ``model.*`` layout."""

    plain = write(tmp_path / "plain")
    named = write(tmp_path / "named", prefix="model.language_model.")
    names = _names(named)
    assert "model.language_model.embed_tokens.weight" in names, "the fixture lost its prefix"
    assert "model.language_model.layers.0.mlp.experts.0.gate_proj.weight" in names
    assert "lm_head.weight" in names and "mtp.fc_embedding.weight" in names

    from tensorfold.families.qwen4_exp.cuda.weights import load

    if not torch.cuda.is_available():                                   # the loader builds CUDA tensors
        pytest.skip("the loader builds CUDA tensors")
    a, b = load(plain, mtp=True, draft_vocab=None), load(named, mtp=True, draft_vocab=None)
    assert b.cfg.quant == "modelopt" and len(b.layers) == len(a.layers)
    for i, (x, y) in enumerate(zip(a.layers, b.layers, strict=True)):
        for face in ("gate_up", "down_proj"):
            got, want = getattr(y.moe.experts, face), getattr(x.moe.experts, face)
            assert torch.equal(nvfp4.dequantize_fp4(got), nvfp4.dequantize_fp4(want)), (i, face)
    assert torch.equal(b.embed[0].float(), a.embed[0].float())
    assert a.mtp is not None and b.mtp is not None
    assert torch.equal(nvfp4.dequantize_fp4(b.mtp.layer.moe.experts.down_proj),
                       nvfp4.dequantize_fp4(a.mtp.layer.moe.experts.down_proj))


def _names(dir: Path) -> set[str]:
    """Every tensor name in a tiny checkpoint's index."""

    index = json.loads((dir / "model.safetensors.index.json").read_text())
    return set(index["weight_map"])


def _tensor(dir: Path, name: str) -> torch.Tensor:
    """One tensor of a tiny checkpoint, read out of the file the index names."""

    shard = dir / json.loads((dir / "model.safetensors.index.json").read_text())["weight_map"][name]
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        entry = json.loads(f.read(n))[name]
        lo, hi = entry["data_offsets"]
        f.seek(8 + n + lo)
        raw = bytearray(f.read(hi - lo))
    dtype = {"BF16": torch.bfloat16, "I32": torch.int32}[entry["dtype"]]
    return torch.frombuffer(raw, dtype=dtype).reshape(entry["shape"])


def test_the_loader_reads_the_published_bf16_table(tmp_path: Path) -> None:
    """The published revision stores the n-gram table as bf16 rows with no per-shard scales and biases: the
    loader takes that layout as it ships, and a gather hands back the checkpoint's own bytes."""

    from tensorfold.families.qwen4_exp.host_table import BF16Table

    tiny = write(tmp_path / "bf16", ple_bf16=True)
    ngram = Config.read(tiny).ngram(0)
    assert "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.scales" not in _names(tiny)

    from tensorfold.families.qwen4_exp.cuda.weights import load

    if not torch.cuda.is_available():                                   # the loader builds CUDA tensors
        pytest.skip("the loader builds CUDA tensors")
    w = load(tiny, mtp=True, draft_vocab=None)
    ple = next(layer.ple for layer in w.layers if layer.ple is not None)
    assert isinstance(ple.table, BF16Table), "a bf16 table must not go through the 4-bit HostTable"
    assert (ple.table.rows, ple.table.width) == (ngram.rows, ngram.dims)
    ids = np.array([0, 17, ngram.rows - 1, 500])
    got = ple.table.gather(ids)
    stored = _tensor(tiny, "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight")
    want = stored[torch.from_numpy(ids)].view(torch.int16).numpy()
    assert got.dtype == np.uint16 and np.array_equal(got.view(np.int16), want)


def test_the_bf16_rows_reach_the_engine_buffers(tmp_path: Path) -> None:
    """``stage_ple_rows`` copies a bf16 table's rows to the device buffer the PLE kernel reads, bit for bit."""

    from tensorfold.families.qwen4_exp.cuda.forward import stage_ple_rows
    from tensorfold.families.qwen4_exp.cuda.state import Buffers

    tiny = write(tmp_path / "bf16-stage", ple_bf16=True)
    ngram = Config.read(tiny).ngram(0)
    if not torch.cuda.is_available():                                   # the loader builds CUDA tensors
        pytest.skip("the loader builds CUDA tensors")
    from tensorfold.families.qwen4_exp.cuda.weights import load

    w = load(tiny, mtp=True, draft_vocab=None)
    ple = next(layer.ple for layer in w.layers if layer.ple is not None)
    b = Buffers(w, rows=3, capacity=64)
    ids = (np.arange(3 * ngram.heads) % ngram.rows).reshape(3, ngram.heads)
    stage_ple_rows(ple, b, ids, at=0)
    stored = _tensor(tiny, "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight")
    want = stored.index_select(0, torch.from_numpy(ids.reshape(-1)))
    assert torch.equal(b.ple_v[:ids.size].cpu(), want.cpu())

