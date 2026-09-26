"""Packed checkpoint partitioning and fp32 row-parallel arithmetic."""

import os
import socket
import json
from pathlib import Path

import pytest
import torch

from tensorfold.families.qwen3_5.cuda.distributed import (
    gather_rank_partials, local_input, output_rows, row_partial, split_input, split_layer,
    split_output, sum_rank_partials,
)
from tensorfold.families.qwen3_5.cuda.weights import Attention, Config, GDN, Layer, QLinear


def _qlinear(n: int, k: int, seed: int = 1, device: str = "cpu") -> QLinear:
    g = torch.Generator(device=device).manual_seed(seed)
    words = torch.randint(0, 2**31 - 1, (n, k // 8), generator=g, dtype=torch.int32, device=device)
    scales = (torch.rand((n, k // 64), generator=g, device=device) * 0.02).to(torch.bfloat16)
    biases = (torch.rand((n, k // 64), generator=g, device=device) * 0.02 - 0.01).to(torch.bfloat16)
    return QLinear(words, scales, biases)


def _dequantize(q: QLinear) -> torch.Tensor:
    words = q.weight.to(torch.int64) & 0xFFFFFFFF
    bits = torch.arange(8, dtype=torch.int64, device=words.device) * 4
    vals = ((words[:, :, None] >> bits) & 15).reshape(q.n, q.k).float()
    return vals * q.scales.float().repeat_interleave(64, 1) + q.biases.float().repeat_interleave(64, 1)


def test_group_aligned_splits_keep_packed_words_unchanged():
    q = _qlinear(128, 256)
    x = torch.randn(7, q.k).to(torch.bfloat16)
    for rank in (0, 1):
        qi = split_input(q, rank)
        assert qi.k == 128
        assert torch.equal(qi.weight, q.weight[:, rank * 16:(rank + 1) * 16])
        assert torch.equal(_dequantize(qi), _dequantize(q)[:, rank * 128:(rank + 1) * 128])
        assert torch.equal(local_input(x, rank), x[:, rank * 128:(rank + 1) * 128])
        qo = split_output(q, rank)
        assert torch.equal(qo.weight, q.weight[rank * 64:(rank + 1) * 64])


def test_segmented_qkv_output_keeps_each_rank_head_local():
    q = _qlinear(384, 128)
    for rank in (0, 1):
        rows = output_rows(384, rank, segments=(128, 128, 128))
        shard = split_output(q, rank, segments=(128, 128, 128))
        expected = torch.cat([q.weight[i * 128 + rank * 64:i * 128 + (rank + 1) * 64]
                              for i in range(3)])
        assert torch.equal(shard.weight, expected)
        assert torch.equal(shard.weight, q.weight[rows])


def test_split_layer_maps_attention_gdn_and_mlp():
    c = Config(hidden=128, intermediate=256, layers=2, heads=4, kv_heads=2,
               head_dim=64, vocab=128, k_heads=4, v_heads=4, dk=32, dv=32,
               conv_kernel=4, interval=2, eps=1e-6, rope_dims=32,
               rope_theta=10000.0, eos=(1,))
    norm = torch.ones(c.hidden, dtype=torch.bfloat16)
    g = GDN(qkv=_qlinear(384, 128), z=_qlinear(128, 128), b=_qlinear(4, 128),
            a=_qlinear(4, 128), out=_qlinear(128, 128),
            conv=torch.arange(384 * 4, dtype=torch.bfloat16).reshape(384, 4),
            A_log=torch.arange(4).float(), dt_bias=torch.arange(4).float(),
            norm=torch.ones(32, dtype=torch.bfloat16))
    a = Attention(q=_qlinear(512, 128), k=_qlinear(128, 128), v=_qlinear(128, 128),
                  o=_qlinear(128, 256), q_norm=torch.ones(64), k_norm=torch.ones(64))
    for rank in (0, 1):
        for linear, gdn, attn in ((True, g, None), (False, None, a)):
            full = Layer(linear, norm, norm, gdn, attn,
                         _qlinear(256, 128), _qlinear(256, 128), _qlinear(128, 256))
            shard = split_layer(full, c, rank)
            assert shard.layer.gate.n == shard.layer.up.n == shard.layer.down.k == 128
            assert shard.layer.input_norm is norm
            if linear:
                assert shard.layer.gdn.qkv.n == 192
                assert shard.layer.gdn.conv.shape == (192, 4)
                assert shard.layer.gdn.out.k == 64
                assert torch.equal(shard.layer.gdn.A_log, g.A_log[rank * 2:(rank + 1) * 2])
            else:
                assert shard.layer.attn.q.n == 256
                assert shard.layer.attn.k.n == shard.layer.attn.v.n == 64
                assert shard.layer.attn.o.k == 128


def test_real_mlx_projection_shards_when_checkpoint_is_available():
    model = os.environ.get("TENSORFOLD_MLX_MODEL")
    if not model:
        pytest.skip("set TENSORFOLD_MLX_MODEL to validate the stored checkpoint")
    from safetensors import safe_open

    model = Path(model)
    c = Config.read(model)
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]

    def read(name):
        key = "language_model." + name
        with safe_open(str(model / index[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def projection(name):
        word = read(name + ".weight")
        return QLinear(word.view(torch.int32), read(name + ".scales"), read(name + ".biases"))

    qkv = projection("model.layers.0.linear_attn.in_proj_qkv")
    kd, vd = c.k_heads * c.dk, c.v_heads * c.dv
    assert qkv.n == 2 * kd + vd and qkv.k == c.hidden
    for rank in (0, 1):
        rows = output_rows(qkv.n, rank, segments=(kd, kd, vd))
        shard = split_output(qkv, rank, segments=(kd, kd, vd))
        assert torch.equal(shard.weight, qkv.weight[rows])
        assert torch.equal(shard.scales, qkv.scales[rows])
        assert torch.equal(shard.biases, qkv.biases[rows])

    out = projection("model.layers.0.linear_attn.out_proj")
    assert out.k == vd
    for rank in (0, 1):
        shard = split_input(out, rank)
        assert shard.k == vd // 2
        assert torch.equal(shard.weight, out.weight[:, rank * vd // 16:(rank + 1) * vd // 16])


def test_rank_sum_rounds_only_after_fp32_add():
    a = torch.tensor([[1.00390625]], dtype=torch.float32)
    b = torch.tensor([[1.0039065]], dtype=torch.float32)
    expected = (a + b).to(torch.bfloat16)
    assert torch.equal(sum_rank_partials((a, b)), expected)
    assert not torch.equal(a.to(torch.bfloat16) + b.to(torch.bfloat16), expected)


@pytest.mark.parametrize("n,k", [(128, 256), (5120, 6144)])
def test_cuda_partial_is_row_invariant_and_accurate(n, k):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("triton")
    q = _qlinear(n, k, seed=n + k, device="cuda")
    x = torch.randn((128, k), device="cuda").to(torch.bfloat16)
    rank0, rank1 = (split_input(q, rank) for rank in (0, 1))
    x0, x1 = (local_input(x, rank) for rank in (0, 1))
    alone0 = torch.cat([row_partial(x0[i:i + 1], rank0) for i in range(128)])
    alone1 = torch.cat([row_partial(x1[i:i + 1], rank1) for i in range(128)])
    for rows in (1, 16, 64, 128):
        p0, p1 = row_partial(x0[:rows], rank0), row_partial(x1[:rows], rank1)
        assert torch.equal(p0, alone0[:rows])
        assert torch.equal(p1, alone1[:rows])
        assert torch.equal(sum_rank_partials((p0, p1)),
                           sum_rank_partials((alone0[:rows], alone1[:rows])))
    perm = torch.randperm(128, device="cuda")
    assert torch.equal(row_partial(x0[perm], rank0), alone0[perm])
    assert torch.equal(row_partial(x1[perm], rank1), alone1[perm])
    full = sum_rank_partials((row_partial(x0[:16], rank0), row_partial(x1[:16], rank1))).float()
    ref = x[:16].float() @ _dequantize(q).T
    assert (full - ref).abs().max().item() <= ref.abs().max().item() * 2**-7


def _nccl_worker(rank: int, port: int):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        local = torch.full((16, 128), rank + 0.25, dtype=torch.float32, device=f"cuda:{rank}")
        got = gather_rank_partials(local)
        expected = torch.full_like(got, 1.5)
        assert torch.equal(got, expected)
    finally:
        dist.destroy_process_group()


def _gloo_worker(rank: int, port: int):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        partial = torch.full((3, 128), rank + 0.25, dtype=torch.float32)
        got = gather_rank_partials(partial)
        assert torch.equal(got, torch.full_like(got, 1.5))
    finally:
        dist.destroy_process_group()


def test_two_process_gloo_rank_order():
    from torch.multiprocessing import spawn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    spawn(_gloo_worker, args=(port,), nprocs=2)


def test_two_process_nccl_rank_order():
    if os.environ.get("TENSORFOLD_TEST_NCCL") != "1":
        pytest.skip("set TENSORFOLD_TEST_NCCL=1 to run the NCCL process test")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")
    from torch.multiprocessing import spawn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    spawn(_nccl_worker, args=(port,), nprocs=2)
