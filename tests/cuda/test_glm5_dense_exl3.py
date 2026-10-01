"""GLM-5.3-Flash's dense EXL3 pack on a GPU: rank shards of a synthetic group through ``tensorfold.cuda.exl3``'s
Exl3Linear give the full layer's output (concatenated for a row cut, summed for a column cut) on the decode kernel
(1 to 128 rows) and the prompt GEMM (more rows), and a ``Dense3`` stack with a BF16 part writes each part into its own
columns through ``qmm.matmul``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.cuda.exl3.linear import Exl3Linear
from tensorfold.families.glm5_next.cuda import dense_exl3, qmm

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


def _group(kt, nt, bits=4, seed=0):
    rng = np.random.default_rng(seed)
    trellis = torch.from_numpy(rng.integers(-2**15, 2**15, size=(kt, nt, fmt.tile_words(bits))).astype(np.int16))
    suh = torch.from_numpy(rng.choice([-1.0, 1.0], kt * 16) * rng.uniform(0.5, 1.5, kt * 16)).half()
    svh = torch.from_numpy(rng.uniform(0.01, 0.02, nt * 16)).half()
    return trellis, suh, svh


def _out(layer, x):
    y = torch.empty((x.shape[0], layer.n), dtype=torch.float32, device="cuda")
    dense_exl3.run(layer, x, y)
    return y


@pytest.mark.parametrize("bits", [4, 6])
@pytest.mark.parametrize("rows", [1, 7, 128, 300])
def test_shards_give_the_full_output(bits, rows):
    tr, suh, svh = _group(32, 32, bits)
    full = Exl3Linear.from_tensors(tr, suh, svh, "mul1")
    x = (torch.randn(rows, full.k, device="cuda") * 0.5).bfloat16()
    ref = _out(full, x)
    row = torch.cat([_out(Exl3Linear.from_tensors(*dense_exl3.cut(tr, suh, svh, "row", r, 2), "mul1"), x)
                     for r in (0, 1)], 1)
    torch.testing.assert_close(row, ref, rtol=1e-5, atol=1e-5)
    h = full.k // 2
    col = sum(_out(Exl3Linear.from_tensors(*dense_exl3.cut(tr, suh, svh, "col", r, 2), "mul1"),
                   x[:, r * h:(r + 1) * h].contiguous()) for r in (0, 1))
    torch.testing.assert_close(col, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("rows", [3, 256])
def test_stack_with_a_bf16_part(rows):
    tr, suh, svh = _group(16, 16)
    lin = Exl3Linear.from_tensors(tr, suh, svh, "mul1")
    b16 = qmm.make_b16(torch.randn(256, lin.k, device="cuda").bfloat16() * 0.02)
    stack = dense_exl3.Dense3([lin, b16], [lin.n, b16.n], lin.k)
    x = (torch.randn(rows, lin.k, device="cuda") * 0.5).bfloat16()
    y = qmm.matmul(x, stack)
    assert y.shape == (rows, lin.n + 256) and y.dtype == torch.bfloat16
    torch.testing.assert_close(y[:, :lin.n].float(), _out(lin, x), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(y[:, lin.n:], qmm.matmul(x, b16))
