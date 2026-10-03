"""Prompt attention: the CUDA kernel gives _attend's bits on every served shape, at any depth, row offset and chunking."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.kernels.prefill_attention import attention, triton_attention  # noqa: E402

# 27B, Qwen3.6-35B-A3B, Nemotron, GLM-5.3 per-head keys (TF_GLM_LATENT=0), a small group
SHAPES = [(24, 4, 256), (16, 2, 256), (32, 2, 128), (32, 32, 256), (12, 4, 128)]
CHUNKS = [(0, 1), (0, 100), (0, 333), (17, 7), (64, 64), (1000, 129), (4133, 300)]


def _caches(gen, keys, kv_heads, dim):
    k = (torch.randn(keys, kv_heads, dim, generator=gen, device="cuda") * 1.5).bfloat16()
    v = torch.randn(keys, kv_heads, dim, generator=gen, device="cuda").bfloat16()
    return k, v


@pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 7,
                    reason="sm_80 kernel vs Triton bits; Volta routes prompts to volta_attention")
@pytest.mark.parametrize("heads,kv_heads,dim", SHAPES)
def test_cuda_prompt_attention_has_the_triton_bits(heads, kv_heads, dim):
    gen = torch.Generator(device="cuda").manual_seed(heads * dim + kv_heads)
    scale = dim ** -0.5
    for p0, w in CHUNKS + ([(70001, 257)] if heads == 24 else []):
        k, v = _caches(gen, p0 + w + 50, kv_heads, dim)
        k[p0 + w:] = float("nan")                                   # past the chunk's keys: never read
        v[p0 + w:] = float("nan")
        q = (torch.randn(w, heads, dim, generator=gen, device="cuda") * 1.5).bfloat16()
        got = attention(q, k, v, p0, scale=scale)
        want = triton_attention(q, k, v, p0, scale=scale)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (p0, w)


@pytest.mark.parametrize("heads,kv_heads,dim", SHAPES[:3])
def test_cuda_prompt_attention_rows_do_not_depend_on_chunking(heads, kv_heads, dim):
    gen = torch.Generator(device="cuda").manual_seed(9)
    total, p0 = 900, 2000
    k, v = _caches(gen, p0 + total, kv_heads, dim)
    q = torch.randn(total, heads, dim, generator=gen, device="cuda").bfloat16()
    whole = attention(q, k, v, p0, scale=dim ** -0.5)
    for size in (1, 15, 16, 17, 64, 100, 512):
        parts = [attention(q[a:a + size].contiguous(), k, v, p0 + a, scale=dim ** -0.5) for a in range(0, total, size)]
        assert torch.equal(whole, torch.cat(parts)), size
