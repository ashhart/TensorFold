"""Grouped-expert kernels, decode and prefill: a pair's bits ignore the call's other rows; outputs track float64."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import experts  # noqa: E402

DEV = "cuda"

# (group size, SwiGLU, limit, experts incl. shared, D, NI, shared slots): Flash Next, GLM and Nemotron in small
CASES = [(32, True, 0.0, 41, 256, 96, 1), (64, True, 10.0, 19, 512, 128, 1), (64, False, 0.0, 12, 384, 192, 2)]


def mlx(e: int, n: int, k: int, gs: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    words = torch.randint(-(2 ** 31), 2 ** 31 - 1, (e, n, k // 8), generator=g, device=DEV,
                          dtype=torch.int64).to(torch.int32)
    scales = (torch.rand((e, n, k // gs), generator=g, device=DEV) * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((e, n, k // gs), generator=g, device=DEV) * 0.02).to(torch.bfloat16)
    return words, scales, biases


def dequant(words, scales, biases, gs: int) -> torch.Tensor:
    w = words.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=DEV, dtype=torch.int64) * 4
    q = ((w[..., None] >> shifts) & 0xF).reshape(*words.shape[:-1], -1).double()
    return q * scales.double().repeat_interleave(gs, -1) + biases.double().repeat_interleave(gs, -1)


def build(case, seed: int):
    gs, swiglu, limit, e, d, ni, _ = case
    up = [mlx(e, ni, d, gs, seed + i) for i in range(2 if swiglu else 1)]
    down = mlx(e, d, ni, gs, seed + 7)
    return experts.make(up, down, gs, limit=limit), up, down


def picks_for(rows: int, case, seed: int, *, hot: int | None = None) -> torch.Tensor:
    """Each row: distinct routed experts (the first ``hot`` if given, so every row shares them), then the shared."""

    _, _, _, e, _, _, shared = case
    routed = e - shared
    k = min(6, routed)
    g = torch.Generator().manual_seed(seed)
    rows_ = []
    for _ in range(rows):
        r = torch.arange(k) if hot is not None else torch.randperm(routed, generator=g)[:k]
        rows_.append(torch.cat([r, torch.arange(routed, e)]))
    return torch.stack(rows_).to(torch.int32).to(DEV)


def run(ex, x, picks, *, prefill: bool = False, y_dtype=torch.float32):
    rows, slots = picks.shape
    plan = experts.Plan(rows, slots, ex.count, DEV, prefill=prefill)
    experts.route(picks.contiguous(), plan)
    act = torch.empty((rows * slots, ex.width), dtype=torch.bfloat16, device=DEV)
    y = torch.empty((rows * slots, ex.dims), dtype=y_dtype, device=DEV)
    experts.gate_up(x, ex, plan, act, rows)
    experts.down(act, ex, plan, y, rows)
    return act, y, plan


@pytest.mark.parametrize("gs", [32, 64])
def test_pack_round_trip(gs):
    w = mlx(3, 96, 256, gs, 61)
    back = experts.unpack(experts.pack(*w, gs), gs)
    assert all(torch.equal(a, b) for a, b in zip(back, w))


@pytest.mark.parametrize("case", CASES)
def test_pairs_do_not_depend_on_the_window(case):
    ex, _, _ = build(case, 11)
    rows = 40
    x = (torch.randn((rows, ex.dims), device=DEV) * 0.5).to(torch.bfloat16)
    picks = picks_for(rows, case, 3)
    slots = picks.shape[1]
    act, y, _ = run(ex, x, picks)
    alone = [run(ex, x[r:r + 1], picks[r:r + 1]) for r in range(rows)]
    assert torch.equal(act, torch.cat([a[0] for a in alone])) and torch.equal(y, torch.cat([a[1] for a in alone]))
    for m in (2, 3, 8, 9, 16, 17, 33):
        a_m, y_m, _ = run(ex, x[:m], picks[:m])
        assert torch.equal(a_m, act[:m * slots]) and torch.equal(y_m, y[:m * slots]), m
    perm = torch.randperm(rows, generator=torch.Generator().manual_seed(5)).to(DEV)
    a_p, y_p, _ = run(ex, x[perm], picks[perm])
    idx = (perm[:, None] * slots + torch.arange(slots, device=DEV)).reshape(-1)
    assert torch.equal(a_p, act[idx]) and torch.equal(y_p, y[idx])


@pytest.mark.parametrize("case", CASES)
def test_an_expert_shared_by_many_rows(case):
    """Every row picks the same experts: items of 16 pairs, several a expert, same bits as alone."""

    ex, _, _ = build(case, 21)
    rows = 37
    x = (torch.randn((rows, ex.dims), device=DEV) * 0.5).to(torch.bfloat16)
    picks = picks_for(rows, case, 4, hot=0)
    _, y, plan = run(ex, x, picks)
    slots = picks.shape[1]
    alone = torch.cat([run(ex, x[r:r + 1], picks[r:r + 1])[1] for r in range(rows)])
    assert torch.equal(y, alone)
    assert int(plan.counts[1]) == slots
    assert int(plan.counts[0]) == slots * 3                    # 37 pairs an expert: items of 16, 16 and 5


def want_plan(picks: torch.Tensor, e: int, tile: int):
    flat = picks.reshape(-1).tolist()
    items, members = [], []
    for ex in range(e):
        pairs = [p for p, v in enumerate(flat) if v == ex]
        for j in range(0, len(pairs), tile):
            items.append([ex, len(members) + j, min(tile, len(pairs) - j)])
        members += pairs
    return items, members, len(set(flat))


@pytest.mark.parametrize("rows,prefill", [(23, False), (23, True), (300, False), (300, True), (1500, True)])
def test_plan_groups_pairs_by_expert(rows, prefill):
    """One block up to 1,024 pairs, then 1,024-pair blocks in turn, in items of 16 pairs (decode) or 64 (prefill)."""

    slots, e = 7, 50
    g = torch.Generator().manual_seed(9)
    picks = torch.stack([torch.randperm(e, generator=g)[:slots] for _ in range(rows)]).to(torch.int32)
    picks[:, 0] = 3                                            # one expert with a pair in every row
    plan = experts.Plan(rows, slots, e, DEV, prefill=prefill)
    tile = plan.tile
    experts.route(picks.to(DEV), plan)
    want_items, want_members, distinct = want_plan(picks, e, tile)
    n = int(plan.counts[0])
    assert n == len(want_items) and int(plan.counts[1]) == distinct
    assert plan.items[:n].tolist() == want_items
    assert plan.members.tolist() == want_members


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("case", CASES)
def test_matches_fp64(case, prefill):
    """Against float64 over the dequantized weights (rounded to bf16 first for the prefill form)."""

    gs, swiglu, limit = case[0], case[1], case[2]
    ex, up, down = build(case, 31)
    rows = 5
    x = (torch.randn((rows, ex.dims), device=DEV) * 0.5).to(torch.bfloat16)
    picks = picks_for(rows, case, 6)
    act, y, _ = run(ex, x, picks, prefill=prefill)
    slots = picks.shape[1]
    pe = picks.reshape(-1).long()
    xr = x.double().repeat_interleave(slots, 0)

    def weights(m):
        w = dequant(*m, gs)
        return w.to(torch.bfloat16).double() if prefill else w

    projs = [torch.einsum("pk,pnk->pn", xr, weights(m)[pe]) for m in up]
    if swiglu:
        g, u = (t.float().to(torch.bfloat16).double() for t in projs)
        if limit:
            g, u = g.clamp(max=limit), u.clamp(-limit, limit)
        ref = (g * torch.sigmoid(g)).to(torch.bfloat16).double() * u
    else:
        ref = projs[0].float().to(torch.bfloat16).double().clamp(min=0) ** 2
    assert (act.double() - ref).abs().max() <= 2 ** -6 * ref.abs().max()
    yref = torch.einsum("pk,pnk->pn", act.double(), weights(down)[pe])
    assert (y.double() - yref).abs().max() <= 1e-5 * yref.abs().max()


def test_graph_replay_follows_the_picks():
    case = CASES[1]
    ex, _, _ = build(case, 41)
    rows = 9
    picks = picks_for(rows, case, 7).contiguous()
    slots = picks.shape[1]
    plan = experts.Plan(rows, slots, ex.count, DEV)
    x = (torch.randn((rows, ex.dims), device=DEV) * 0.5).to(torch.bfloat16)
    act = torch.empty((rows * slots, ex.width), dtype=torch.bfloat16, device=DEV)
    y = torch.empty((rows * slots, ex.dims), dtype=torch.float32, device=DEV)

    def step():
        experts.route(picks, plan)
        experts.gate_up(x, ex, plan, act, rows)
        experts.down(act, ex, plan, y, rows)

    step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for seed in (8, 9):
        picks.copy_(picks_for(rows, case, seed))
        g.replay()
        _, want, _ = run(ex, x, picks)
        assert torch.equal(y, want)


@pytest.mark.parametrize("case", CASES)
def test_prefill_rows_do_not_depend_on_the_chunk(case):
    """Prefill form past 1,024 pairs: any chunk or row order keeps each row's bits; bf16 down is fp32 rounded."""

    ex, _, _ = build(case, 71)
    rows = 200
    x = (torch.randn((rows, ex.dims), device=DEV) * 0.5).to(torch.bfloat16)
    picks = picks_for(rows, case, 13)
    slots = picks.shape[1]
    assert rows * slots > experts.SMALL
    act, y, _ = run(ex, x, picks, prefill=True)
    for a, b in ((0, 1), (77, 78), (199, 200), (0, 7), (5, 22), (0, 64), (100, 200)):
        a1, y1, _ = run(ex, x[a:b], picks[a:b], prefill=True)
        assert torch.equal(a1, act[a * slots:b * slots]) and torch.equal(y1, y[a * slots:b * slots]), (a, b)
    perm = torch.randperm(rows, generator=torch.Generator().manual_seed(5)).to(DEV)
    a_p, y_p, _ = run(ex, x[perm], picks[perm], prefill=True)
    idx = (perm[:, None] * slots + torch.arange(slots, device=DEV)).reshape(-1)
    assert torch.equal(a_p, act[idx]) and torch.equal(y_p, y[idx])
    _, y16, _ = run(ex, x, picks, prefill=True, y_dtype=torch.bfloat16)
    assert torch.equal(y16, y.to(torch.bfloat16))
    _, y_hot, _ = run(ex, x[:90], picks_for(90, case, 4, hot=0), prefill=True)        # an expert in every row
    alone = torch.cat([run(ex, x[r:r + 1], picks_for(90, case, 4, hot=0)[r:r + 1], prefill=True)[1]
                       for r in range(0, 90, 29)])
    assert torch.equal(alone, y_hot.view(90, slots, -1)[0:90:29].reshape(-1, ex.dims))


def test_many_units_a_warp_keep_the_bits():
    """A call with far more units than resident warps (each warp takes several in turn) against rows alone."""

    case = (64, True, 0.0, 257, 2048, 128, 1)
    ex, _, _ = build(case, 81)
    rows = 64
    x = (torch.randn((rows, ex.dims), device=DEV) * 0.5).to(torch.bfloat16)
    g = torch.Generator().manual_seed(12)
    picks = torch.stack([torch.cat([torch.randperm(256, generator=g)[:8], torch.tensor([256])])
                         for _ in range(rows)]).to(torch.int32).to(DEV)
    _, y, plan = run(ex, x, picks)
    assert int(plan.counts[0]) * (ex.dims // experts.COLS) > 8192
    for r in (0, 31, 63):
        assert torch.equal(run(ex, x[r:r + 1], picks[r:r + 1])[1], y[9 * r:9 * r + 9]), r
