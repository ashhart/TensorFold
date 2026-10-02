"""The Volta tree attention matches an fp32 softmax attention over each row's visible keys, and a row never depends on
its window."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:
    pytest.skip("Volta only", allow_module_level=True)

from tensorfold.cuda.kernels import attention as shared  # noqa: E402


@pytest.mark.parametrize("w,p,d", [(1, 0, 256), (1, 2400, 256), (16, 511, 256), (5, 1500, 128), (128, 513, 256)])
def test_matches_fp32_attention(w, p, d):
    h, hk = 24, 4
    gen = torch.Generator(device="cuda").manual_seed(w * 7 + p)
    q = torch.randn((w, h, d), generator=gen, device="cuda").bfloat16()
    kn = torch.randn((w, hk, d), generator=gen, device="cuda").bfloat16()
    vn = torch.randn((w, hk, d), generator=gen, device="cuda").bfloat16()
    kc = torch.randn((p + 5, hk, d), generator=gen, device="cuda").bfloat16()
    vc = torch.randn((p + 5, hk, d), generator=gen, device="cuda").bfloat16()
    parents = [-1] + [(i - 1) // 2 for i in range(1, w)]
    plan = shared.plan([parents], [p], h // hk, "cuda")
    offs = torch.tensor(shared.offsets([(kc, vc)], "cuda"), dtype=torch.int64, device="cuda").view(-1, 2)
    scale = d ** -0.5
    out = shared.attention(q, kn, vn, offs, plan, scale=scale).float()
    for node in range(w):
        path, cur = [], node
        while cur >= 0:
            path.append(cur)
            cur = parents[cur]
        path = path[::-1]
        keys = torch.cat((kc[:p], kn[path]), 0).float().repeat_interleave(h // hk, dim=1)     # (T, H, D)
        vals = torch.cat((vc[:p], vn[path]), 0).float().repeat_interleave(h // hk, dim=1)
        s = torch.einsum("hd,thd->ht", q[node].float(), keys) * scale
        ref = torch.einsum("ht,thd->hd", s.softmax(-1), vals)
        err = (out[node] - ref).abs().max().item()
        assert err < 2e-2, (node, err)


def test_a_row_never_depends_on_its_window():
    """A row's partial is the same alone, in a chain and in a tree, committed keys or not: drafts equal serial."""

    h, hk, d, p = 24, 4, 256, 1000
    gen = torch.Generator(device="cuda").manual_seed(11)
    q = torch.randn((12, h, d), generator=gen, device="cuda").bfloat16()
    kn = torch.randn((12, hk, d), generator=gen, device="cuda").bfloat16()
    vn = torch.randn((12, hk, d), generator=gen, device="cuda").bfloat16()
    kc = torch.randn((p + 12, hk, d), generator=gen, device="cuda").bfloat16()
    vc = torch.randn((p + 12, hk, d), generator=gen, device="cuda").bfloat16()
    offs = torch.tensor(shared.offsets([(kc, vc)], "cuda"), dtype=torch.int64, device="cuda").view(-1, 2)
    scale = d ** -0.5
    chain = shared.attention(q, kn, vn, offs, shared.plan([list(range(-1, 11))], [p], h // hk, "cuda"), scale=scale)
    alone = shared.attention(q[:1], kn[:1], vn[:1], offs, shared.plan([[-1]], [p], h // hk, "cuda"), scale=scale)
    assert torch.equal(chain[0], alone[0])
    kc[p:p + 3], vc[p:p + 3] = kn[:3], vn[:3]              # the chain's first three keys, committed
    later = shared.attention(q[3:4], kn[3:4], vn[3:4], offs, shared.plan([[-1]], [p + 3], h // hk, "cuda"), scale=scale)
    assert torch.equal(chain[3], later[0])
