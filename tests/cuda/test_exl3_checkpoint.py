"""A real EXL3 checkpoint against ExLlamaV3 itself: the decoder (numpy and CUDA) bit for bit versus
``exllamav3``'s ``reconstruct``, for every bit width the checkpoint holds, and the layer's forward versus
``LinearEXL3``.

Needs ``TENSORFOLD_EXL3_MODEL=<checkpoint dir>`` (and ``exllamav3`` importable for the comparison): skipped
otherwise, so the default suite stays runnable on the box that has neither.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.cuda.exl3 import linear

MODEL = os.environ.get("TENSORFOLD_EXL3_MODEL", "")
SMALL = 8_000_000                      # elements: keep the numpy reference's memory sane

pytestmark = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(),
                                reason="set TENSORFOLD_EXL3_MODEL to an EXL3 checkpoint")


def _checkpoint() -> fmt.Checkpoint:
    return fmt.scan(MODEL)


def _raw_trellis(prefix: str, group: fmt.Exl3Tensor) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(str(Path(MODEL) / group.files[0]), framework="pt") as f:
        return f.get_tensor(f"{prefix}.trellis")


def _one_per_width() -> dict[float, str]:
    """One group per bit width the checkpoint holds, the smallest of each, so the whole set stays cheap."""

    best: dict[float, tuple[int, str]] = {}
    for prefix, group in _checkpoint().groups.items():
        if group.k * group.n > SMALL and group.bits in best:
            continue
        size = group.k * group.n
        if group.bits not in best or size < best[group.bits][0]:
            best[group.bits] = (size, prefix)
    return {bits: prefix for bits, (_, prefix) in sorted(best.items())}


def _reconstruct(trellis: torch.Tensor, bits: float, codebook: str) -> torch.Tensor:
    """ExLlamaV3's own device dequantization of the same trellis (it writes into a caller's buffer)."""

    from exllamav3.ext import exllamav3_ext as ext

    out = torch.empty((16 * trellis.shape[0], 16 * trellis.shape[1]), dtype=torch.half, device="cuda")
    ext.reconstruct(out, trellis.cuda(), float(bits), codebook == "mcg", codebook == "mul1")
    return out


def test_every_width_in_the_checkpoint_decodes_bit_for_bit_vs_exllamav3():
    widths = _one_per_width()
    assert len(widths) >= 2, f"expected several widths in {MODEL}, found {sorted(widths)}"
    for bits, prefix in widths.items():
        group = _checkpoint().groups[prefix]
        trellis = _raw_trellis(prefix, group)
        theirs = _reconstruct(trellis, bits, group.codebook)
        mine = linear.unpack_cuda(trellis, group.codebook)
        assert torch.equal(mine.view(torch.int16), theirs.view(torch.int16)), f"{prefix} ({bits} bits, CUDA)"
        numpy = fmt.unpack(trellis.numpy(), bits, group.codebook)
        assert np.array_equal(numpy.view(np.int16), theirs.cpu().numpy().view(np.int16)), \
            f"{prefix} ({bits} bits, numpy)"


def test_the_layer_decodes_the_checkpoint_to_the_same_weight():
    bits, prefix = sorted(_one_per_width().items())[0]
    group = _checkpoint().groups[prefix]
    layer = linear.Exl3Linear.load(MODEL, prefix)
    theirs = _reconstruct(_raw_trellis(prefix, group), bits, group.codebook).cpu()
    mine = layer.unpack().cpu()
    assert torch.equal(mine.view(torch.int16), theirs.view(torch.int16))


@pytest.mark.parametrize("rows", [1, 2, 3, 16, 17, 64, 128])
def test_the_layer_forward_matches_exllamav3(rows: int):
    exl3 = pytest.importorskip("exllamav3.modules.quant.exl3")
    bits, prefix = sorted(_one_per_width().items())[0]
    group = _checkpoint().groups[prefix]
    layer = linear.Exl3Linear.load(MODEL, prefix)
    theirs = exl3.LinearEXL3(None, layer.k, layer.n, suh=layer.suh, svh=layer.svh,
                             trellis=_raw_trellis(prefix, group).cuda(),
                             mcg=torch.tensor(0, dtype=torch.int32, device="cuda")
                             if layer.codebook != "mul1" else None,
                             mul1=torch.tensor(0, dtype=torch.int32, device="cuda")
                             if layer.codebook == "mul1" else None)
    x = torch.randn((rows, layer.k), device="cuda").half()
    out = torch.empty_like(x[..., :layer.n])
    theirs.bc.run(x, out)
    mine = layer(x, out_dtype=torch.float32).float()
    delta = (mine - out.float()).norm() / out.float().norm()
    assert delta.item() < 5e-2, f"{prefix} at {rows} rows: {delta.item():.2e}"
