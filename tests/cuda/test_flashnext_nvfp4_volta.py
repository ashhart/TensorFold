"""Flash Next's NVFP4 checkpoint on sm_70, one GPU and four ranks as threads: shards are slices, outputs exact."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:
    pytest.skip("Volta only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import volta as vk  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import commit, forward  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.ranks import share  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weights import load  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from nvfp4_tiny import write  # noqa: E402

TINY = dict(hidden=512, heads=4, kv_heads=2, hd=64, nk=16, nv=48, moe_width=320, shared_width=320, experts=4,
            ple_bf16=True)
PROMPT = [5, 17, 99, 250, 31, 7, 64, 30, 11, 12, 13]
WORLD = 4


class _Hub:
    def __init__(self, world: int) -> None:
        self.world = world
        self.slots: list = [None] * world
        self.barrier = threading.Barrier(world, timeout=300)


class _ThreadComm:
    """``comm.NCCL``'s all_gather for ranks that are threads of one process on one GPU."""

    def __init__(self, hub: _Hub, rank: int) -> None:
        self.hub, self.rank, self.world = hub, rank, hub.world

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        torch.cuda.current_stream().synchronize()
        self.hub.slots[self.rank] = send
        self.hub.barrier.wait()
        n = send.numel()
        flat = recv.view(-1)
        for r in range(self.world):
            flat[r * n:(r + 1) * n].copy_(self.hub.slots[r].reshape(-1))
        torch.cuda.current_stream().synchronize()
        self.hub.barrier.wait()

    def barrier(self) -> None:
        self.hub.barrier.wait()


def _run_ranks(fn, engines: list) -> list:
    """fn(rank, engine) on every rank at once (threads); results in rank order."""

    hub = _Hub(len(engines))
    for r, e in enumerate(engines):
        e.w.comm = _ThreadComm(hub, r)
    results: list = [None] * len(engines)
    errors: list = []

    streams = [torch.cuda.Stream() for _ in engines]   # a rank's own stream, as its own process would have

    def body(r: int) -> None:
        try:
            with torch.no_grad(), torch.cuda.stream(streams[r]):
                results[r] = fn(r, engines[r])
                torch.cuda.current_stream().synchronize()
        except BaseException as exc:        # noqa: BLE001  (reported below; unblock the other ranks)
            errors.append(exc)
            hub.barrier.abort()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(len(engines))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    return results


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    return write(tmp_path_factory.mktemp("flashnext-volta"), **TINY)


@pytest.fixture(scope="module")
def models(tiny):
    single = load(tiny, draft_vocab=128)
    ranks = [load(tiny, tp=(r, WORLD), draft_vocab=128) for r in range(WORLD)]
    engines = [Engine(w, capacity=512, max_rows=8, prefill_rows=16) for w in ranks]
    _run_ranks(lambda r, e: (prefill(e, PROMPT[:3], None), serial_decode(e, 1, 2, None)), engines)   # build kernels
    return single, ranks


def _tensor(model_dir: Path, name: str) -> torch.Tensor:
    from safetensors import safe_open

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(model_dir / index[name]), framework="pt", device="cuda") as f:
        return f.get_tensor(name)


def test_shards_are_the_checkpoints_slices(tiny, models):
    """Each rank's experts, 16-bit linears and key/value heads are the checkpoint's own slices."""

    from tensorfold.cuda.kernels.qmmf_volta import VoltaLinear

    single, ranks = models
    cfg = single.cfg
    base = "model.layers.0."

    def stack(proj: str, part: str) -> torch.Tensor:
        return torch.stack([_tensor(tiny, f"{base}mlp.experts.{e}.{proj}.{part}") for e in range(cfg.experts)])

    for r, w in enumerate(ranks):
        lo, hi = share(cfg.moe_width, r, WORLD)
        assert w.cfg.moe_width == hi - lo and w.cfg.heads == 1 and w.cfg.kv_heads == 1
        got = w.layers[0].moe.experts.routed_experts

        def checkpoint(proj, rows):
            s2 = torch.stack([_tensor(tiny, f"{base}mlp.experts.{e}.{proj}.weight_scale_2").reshape(())
                              for e in range(cfg.experts)])
            wd, sc = stack(proj, "weight"), stack(proj, "weight_scale")
            if rows:
                return wd[:, lo:hi].contiguous(), sc[:, lo:hi].contiguous(), s2
            return wd[:, :, lo // 2:hi // 2].contiguous(), sc[:, :, lo // 16:hi // 16].contiguous(), s2

        want = vk.experts(checkpoint("gate_proj", True), checkpoint("up_proj", True), checkpoint("down_proj", False))
        for name in ("up", "up_s", "up_a", "down", "down_s", "down_a"):
            assert torch.equal(getattr(got, name), getattr(want, name)), (r, name)
        o = _tensor(tiny, "model.layers.1.self_attn.o_proj.weight").to(torch.bfloat16)
        full = VoltaLinear.from_bf16(o)
        shard = w.layers[1].attn.o
        g = cfg.heads // WORLD * cfg.head_dim // 64
        for name in ("words", "alpha"):
            assert torch.equal(getattr(shard, name), getattr(full.groups(r * g, (r + 1) * g), name)), (r, name)
        cols = slice(r * g * 64, (r + 1) * g * 64)
        assert torch.equal(shard.dense().double() * shard.alpha[:shard.n, None].double(), o[:, cols].double())
        k = _tensor(tiny, "model.layers.1.self_attn.k_proj.weight").to(torch.bfloat16)
        proj = w.layers[1].attn.proj
        rows = slice(2 * cfg.head_dim, 3 * cfg.head_dim)           # q (one head, 2 x head_dim) then k
        kv = r * cfg.kv_heads // WORLD
        values = proj.dense()[rows].double() * proj.alpha[rows, None].double()
        assert torch.equal(values, k[kv * cfg.head_dim:(kv + 1) * cfg.head_dim].double()), r


@pytest.mark.parametrize("sampling", [None, Sampling(seed=1234, top_k=20, top_p=0.95)])
def test_four_ranks_draft_the_serial_tokens_and_repeat_bit_for_bit(models, sampling):
    _, ranks = models
    engines = [Engine(w, capacity=512, max_rows=8, prefill_rows=16) for w in ranks]

    def body(r, e):
        first = prefill(e, PROMPT, sampling)
        out = {"serial": serial_decode(e, first, 24, sampling).tokens}
        for name, depth, conf in (("d1", 1, 0.0), ("d3", 3, 0.0), ("d5c", 5, 0.3)):
            prefill(e, PROMPT, sampling)
            got = mtp_decode(e, first, 24, sampling, depth=depth, confidence=conf)
            out[name], out[name + "_min_rows"] = got.tokens, min(got.widths)
        prefill(e, PROMPT, sampling)
        out["again"] = mtp_decode(e, first, 24, sampling, depth=3).tokens
        return out

    got = _run_ranks(body, engines)
    assert all(g == got[0] for g in got[1:])
    a = got[0]
    for name in ("d1", "d3", "d5c", "again"):
        assert a[name] == a["serial"], name
    assert a["d3_min_rows"] >= 2


def test_four_rank_windows_equal_serial_steps(models):
    _, ranks = models
    engines = [Engine(w, capacity=512, max_rows=16, prefill_rows=16) for w in ranks]
    nxt = [101, 33, 200, 5, 77, 150, 9, 10]

    def body(r, e):
        w = e.w
        forward(w, e.st, e.buf, PROMPT)
        commit(w, e.st, e.buf, len(PROMPT), len(PROMPT))
        serial, steps = e.st.clone(), []
        for t in nxt:
            steps.append(forward(w, serial, e.buf, [t])[0].clone())
            commit(w, serial, e.buf, 1, 1)
        bad = []
        for R in (2, 4, 8):
            st = e.st.clone()
            lg = forward(w, st, e.buf, nxt[:R])
            bad += [(R, row) for row in range(R) if not torch.equal(lg[row], steps[row])]
        return bad

    for r, bad in enumerate(_run_ranks(body, engines)):
        assert not bad, (r, bad)


def test_four_rank_logits_agree_with_one_gpu_to_rounding(models):
    single, ranks = models
    e1 = Engine(single, capacity=512, max_rows=16, prefill_rows=16)
    ref = forward(single, e1.st, e1.buf, PROMPT)[:len(PROMPT)].float().clone()
    engines = [Engine(w, capacity=512, max_rows=16, prefill_rows=16) for w in ranks]
    parts = _run_ranks(lambda r, e: forward(e.w, e.st, e.buf, PROMPT)[:len(PROMPT)].float().clone(), engines)
    got = torch.cat(parts, dim=1)
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=1)
    assert float(cos.min()) > 0.999, cos


def test_one_gpu_drafts_the_serial_tokens(models):
    single, _ = models
    e = Engine(single, capacity=512, max_rows=8, prefill_rows=16)
    first = prefill(e, PROMPT, None)
    ref = serial_decode(e, first, 24, None).tokens
    for depth in (2, 4):
        prefill(e, PROMPT, None)
        assert mtp_decode(e, first, 24, None, depth=depth).tokens == ref


def test_host_rows_return_the_stored_rows():
    from tensorfold.cuda.kernels.qmmf_volta import HostRows

    gen = torch.Generator().manual_seed(3)
    table = (torch.randn(5000, 256, generator=gen) * 3).to(torch.bfloat16)
    rows = HostRows(table)
    ids = torch.tensor([0, 4999, 17, 17, 2500, 1], dtype=torch.int32, device="cuda")
    assert rows.nbytes() == 0
    assert torch.equal(rows.rows(ids).cpu(), table[ids.cpu().long()])


MODEL = os.environ.get("TENSORFOLD_FLASHNEXT_NVFP4", "")


def test_real_fp8_ngram_rows_are_the_stored_rows():
    """The real checkpoint's FP8 n-gram rows, memory-mapped on the host, are bf16(e4m3 x scale) of the stored bytes."""

    if not MODEL:
        pytest.skip("set TENSORFOLD_FLASHNEXT_NVFP4 to read the real checkpoint's n-gram table")
    from safetensors import safe_open

    from tensorfold.families.qwen4_exp.host_table import open_table, shard_keys

    d = Path(MODEL)
    index = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    base = next(n.rsplit(".ngram_embedding.", 1)[0] for n in index if ".ngram_embedding.shard_" in n)
    shards = sorted({n.rsplit(".", 1)[0] for n in index if n.startswith(base + ".ngram_embedding.shard_")})
    keys = shard_keys(base + ".ngram_embedding", len(shards), index)
    scale_name = base + ".ngram_embedding.weight_scale"

    def scale(field: str) -> float:
        name = base + ".ngram_embedding." + field
        if name not in index:
            return 1.0
        with safe_open(str(d / index[name]), framework="pt") as f:
            return float(f.get_tensor(name).float().reshape(-1)[0])

    table = open_table(d, [(index[k + ".weight"], k) for k in keys], scale)
    per = table.starts[1] - table.starts[0]
    ids = np.array([0, 1, per - 1, per, per + 1, 3 * per + 17, table.rows - 1], dtype=np.int64)
    got = table.gather(ids)
    s = scale("weight_scale")
    for i, row in zip(ids, got):
        shard = int(np.searchsorted(table.starts, i, side="right") - 1)
        key = keys[shard] + ".weight"
        with safe_open(str(d / index[key]), framework="pt") as f:
            stored = f.get_slice(key)[int(i - table.starts[shard]):int(i - table.starts[shard]) + 1]
        want = (stored.float() * s).to(torch.bfloat16).view(torch.int16).numpy().reshape(-1)
        assert np.array_equal(row.view(np.int16), want), int(i)
    assert scale_name in index or s == 1.0


@pytest.mark.parametrize("keys", [1, 63, 600, 2100])
def test_volta_attention_rows_are_their_own_and_match_the_triton_partials(keys):
    """The sm_70 attention kernel: a row is the same alone or among others and matches the Triton kernel to rounding."""

    from tensorfold.families.qwen4_exp.cuda import attention as am

    gen = torch.Generator(device="cuda").manual_seed(keys)
    h, hk, d, rows = 6, 1, 256, 5
    cap = keys + rows + 8
    kc = (torch.randn((cap, hk, d), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    vc = (torch.randn((cap, hk, d), generator=gen, device="cuda")).to(torch.bfloat16)
    q = (torch.randn((rows, h, d), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    pos0 = torch.tensor([keys], dtype=torch.int32, device="cuda")

    def run(volta: bool, sl=slice(None), qsa_ids=None):
        scratch = am.AttnScratch(rows, h, d, cap, "cuda", budget=2048, ratio=4)
        if qsa_ids is not None:
            scratch.qsa = True
            scratch.sparse.fill_(1)
            scratch.nk.fill_(qsa_ids.shape[1])
            scratch.ids[:, :qsa_ids.shape[1]] = qsa_ids
        am._on_volta.cache_clear()
        old = am._on_volta
        am._on_volta = lambda index: volta                                                     # noqa: E731
        try:
            qq = q[sl]
            p = torch.tensor([keys + (sl.start or 0)], dtype=torch.int32, device="cuda")
            return am.attention(qq, kc, vc, p, scratch, qq.shape[0], d ** -0.5, out=torch.empty_like(qq),
                                context=cap).clone()
        finally:
            am._on_volta = old

    full = run(True)
    for r in range(rows):
        assert torch.equal(run(True, slice(r, r + 1))[0], full[r]), r
    ref = run(False)
    assert (full.float() - ref.float()).abs().max().item() <= 2 ** -6 * max(1.0, ref.float().abs().max().item())
    ids = torch.randperm(keys + 1, generator=gen, device="cuda")[:min(keys + 1, 300)].sort().values.to(torch.int32)
    ids = ids[None].repeat(rows, 1).contiguous()
    sparse = run(True, qsa_ids=ids)
    sparse_ref = run(False, qsa_ids=ids)
    assert (sparse.float() - sparse_ref.float()).abs().max().item() <= 2 ** -6 * max(
        1.0, sparse_ref.float().abs().max().item())
