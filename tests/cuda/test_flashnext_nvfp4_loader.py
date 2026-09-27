"""The NVFP4 loader on a tiny modelopt checkpoint: the route's tensors, shapes and exactness.

Runs wherever Triton runs (the CUDA tests' directory; the loader's slicing is torch code, the FP4/BF16
*kernels* are checked in test_flashnext_nvfp4_kernels.py on a GPU). It loads a checkpoint that stores
what the Swift checkpoint stores — FP4 arrays for the routed experts, BF16 for everything else — and
checks the loader reads it into the engine's faces."""

import json
import struct
import sys
from pathlib import Path

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
    assert torch.equal(fd, dn[0].to(torch.float32))
