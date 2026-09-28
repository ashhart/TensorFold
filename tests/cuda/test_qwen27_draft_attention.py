"""DFlash2's block attention reads each stream's context in place and matches masked attention over the same keys."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda.draft_attention import block_attention  # noqa: E402

D = 128


def _reference(q, k, v, keys, values, length, window, scale, causal):
    """Each stream's block over [its context | its block] with DFlash2's window mask, in fp32."""

    group, outs = q.shape[0] // k.shape[0], []
    for j, (kc, vc) in enumerate(zip(keys, values)):
        rows, s = slice(j * length, (j + 1) * length), kc.shape[1]
        kk = torch.cat((kc, k[:, rows]), 1).float().repeat_interleave(group, 0)
        vv = torch.cat((vc, v[:, rows]), 1).float().repeat_interleave(group, 0)
        qi = torch.arange(length, device=q.device)[:, None]
        ki = torch.arange(s + length, device=q.device)[None, :]
        block = ki >= s
        if causal:
            block = block & (ki <= s + qi)
        allowed = ((ki < s) & (s + qi - ki < window + 1)) | block
        att = (q[:, rows].float() @ kk.transpose(1, 2)) * scale
        o = att.masked_fill(~allowed, float("-inf")).softmax(-1) @ vv
        outs.append(o.transpose(0, 1).reshape(length, -1))
    return torch.cat(outs)


@pytest.mark.parametrize("heads,kv_heads,length", [(32, 8, 16), (16, 4, 16), (8, 2, 8)])
@pytest.mark.parametrize("causal", [False, True])
def test_block_attention_matches_masked_attention(heads, kv_heads, length, causal):
    gen = torch.Generator(device="cuda").manual_seed(heads + length + int(causal))
    lens, window = [1, 40, 63, 63, 130], 63             # empty-ish, short, full and past-window contexts
    lens = [min(n, window) for n in lens]
    streams = len(lens)
    q = torch.randn(heads, streams * length, D, generator=gen, device="cuda").bfloat16()
    k = torch.randn(kv_heads, streams * length, D, generator=gen, device="cuda").bfloat16()
    v = torch.randn(kv_heads, streams * length, D, generator=gen, device="cuda").bfloat16()
    keys = [torch.randn(kv_heads, n, D, generator=gen, device="cuda").bfloat16() for n in lens]
    values = [torch.randn(kv_heads, n, D, generator=gen, device="cuda").bfloat16() for n in lens]
    scale = D ** -0.5
    out = block_attention(q, k, v, keys, values, length, window, scale, causal)
    ref = _reference(q, k, v, keys, values, length, window, scale, causal)
    assert out.shape == (streams * length, heads * D)
    assert (out.float() - ref).abs().max().item() < 2e-2


def test_a_stream_gets_the_same_bits_alone_and_among_others():
    gen = torch.Generator(device="cuda").manual_seed(1)
    heads, kv_heads, length, window = 32, 8, 16, 2047
    lens = [2047, 500, 2047]
    q = torch.randn(heads, 3 * length, D, generator=gen, device="cuda").bfloat16()
    k = torch.randn(kv_heads, 3 * length, D, generator=gen, device="cuda").bfloat16()
    v = torch.randn(kv_heads, 3 * length, D, generator=gen, device="cuda").bfloat16()
    keys = [torch.randn(kv_heads, n, D, generator=gen, device="cuda").bfloat16() for n in lens]
    values = [torch.randn(kv_heads, n, D, generator=gen, device="cuda").bfloat16() for n in lens]
    together = block_attention(q, k, v, keys, values, length, window, D ** -0.5)
    rows = slice(length, 2 * length)
    alone = block_attention(q[:, rows].contiguous(), k[:, rows].contiguous(), v[:, rows].contiguous(), keys[1:2],
                            values[1:2], length, window, D ** -0.5)
    assert torch.equal(together[rows], alone)
