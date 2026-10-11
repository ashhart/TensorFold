#!/usr/bin/env python3
"""Sessions over 4 interleaved slots on the reference store, pool and NVMe tier; every step's decisions go to JSON."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile


def _stub_triton() -> None:
    """Stand-in triton modules that only decorate, so the kernel modules import on a host without triton."""

    try:
        import triton  # noqa: F401
        return
    except ImportError:
        pass
    import types

    class _Any(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _anything

    def _anything(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _anything

    for name in ("triton", "triton.language", "triton.language.extra", "triton.language.extra.cuda",
                 "triton.language.extra.libdevice", "triton.runtime", "triton.compiler", "triton.backends",
                 "triton.backends.compiler", "triton.tools", "triton.tools.tensor_descriptor"):
        sys.modules[name] = _Any(name)
    sys.modules["triton"].language = sys.modules["triton.language"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-src", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    # split KV (a reference tree with the split pool): W ranks, this one's decisions and rows
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--rank", type=int, default=0)
    # the pool's pages (split's admission counts W x the scarcer residue: a little more pool keeps it admitting)
    ap.add_argument("--pages", type=int, default=28)
    a = ap.parse_args()
    sys.path.insert(0, a.py_src)
    _stub_triton()
    import numpy as np
    import torch

    from tensorfold.families.deepseek_v41.cuda import pool as PO
    from tensorfold.families.deepseek_v41.cuda import replay as RP
    from tensorfold.families.deepseek_v41.cuda import sessdisk as SD
    from tensorfold.families.deepseek_v41.cuda import sessions as SE
    from tensorfold.families.deepseek_v41.cuda import topology as TO
    from tensorfold.families.deepseek_v41.cuda.csa2.stores import Records
    from tensorfold.families.deepseek_v41.cuda.protocol import Bounded
    from tensorfold.families.deepseek_v41.cuda.slots import LOOKBACK

    cfg = json.load(open(a.config))
    topo = TO.build(cfg)
    tc = TO.text_config(cfg)
    D, hc, hd, W = int(topo.hidden), int(tc["hc_mult"]), int(topo.head_dim), int(topo.window)
    dec = int(topo.decoder_start)
    layers = [ly.index for ly in topo.layers]
    carries = [ly.index for ly in topo.layers if ly.role == "full" and ly.ratio == 2]
    ds_blocks = len(topo.dspark)
    TAG = SE.Tag(prefill="fast", grid=1, ced="replay").code()

    # 28-page pool, 4 slots of 16 pages, RAM tier of four snapshots (admissions spill), NVMe entries >= 1,024 tokens
    page, pages, slots_n, capacity = PO.PAGE, a.pages, 4, 4096
    disk_min = 1024
    valid = [0] * slots_n          # drafter.valid: rows before it are not the slot's (drafter.reset / restore)
    stash_from = [0] * slots_n     # the CED stash's first position (replay.empty at a gap: here the slot's start)

    def bounded(slot: int, pos: int) -> Bounded:
        """The served snapshot at ``pos`` in replay mode, drafter on."""

        arrays: dict = {}
        rows = torch.arange(max(0, pos - W), pos)
        rec = Records.new(W, "cpu")
        ring = [L for L in layers if L < dec]
        raws = [rec.raw(rows % W) for _ in ring]
        for name in raws[0]:
            arrays[f"swa_{name}"] = np.stack([r[name] for r in raws])
        m = pos - max(stash_from[slot], pos - RP.KEEP)
        z = (lambda *s, dt=torch.float32: torch.zeros(s, dtype=dt))  # noqa: E731
        sh = RP.Stash(pos - m, z(m, hc * D, dt=torch.bfloat16), z(m, D, dt=torch.bfloat16), z(m, 4), z(m, 4), z(m, 16),
                      [0] * m)
        arrays.update(sh.arrays())
        arrays["carry"] = np.stack([np.zeros((2 * hd,), dtype=np.float32) for _ in carries])
        arrays["lookback"] = np.asarray([-1] * LOOKBACK, dtype=np.int64)
        arrays["rings"] = np.asarray(ring, dtype=np.int64)
        ds = torch.arange(max(0, pos - W, valid[slot]), pos)
        if len(ds):
            draws = [Records.new(W, "cpu").raw(ds % W) for _ in range(ds_blocks)]
            for name in draws[0]:
                arrays[f"dspark_{name}"] = np.stack([r[name] for r in draws])
        return Bounded(pos, arrays, {"ced": "replay", "window": W, "ring": W, "ds_rows": int(len(ds))})

    tmp = tempfile.mkdtemp(prefix="session-slots-golden-")
    pool = PO.Pool(topo, pages * page, page, split=a.split, rank=a.rank) if a.split > 1 else PO.Pool(topo, pages * page, page)
    for _ in range(slots_n):
        pool.new_slot(capacity)
    # the RAM budget: four prompt snapshots of 128+ tokens' bounded state, plus a little
    ram = int(4.5 * bounded(0, 1024).nbytes())
    disk = SD.DiskTier(tmp, a.rank, SE.compat_extra(SE.Tag(), topo), budget_gib=64.0, min_tokens=disk_min, direct=False)
    store = SE.Store(pool, ram_bytes=ram, disk=disk)
    hist: list[list[int]] = [[] for _ in range(slots_n)]
    names: dict[str, str] = {}

    def name_of(key: str) -> str:
        return names[key]

    def state() -> dict:
        ram_e = sorted(name_of(e.key) for e in store.entries.values() if e.in_ram)
        disk_e = sorted(name_of(k) for k in disk.keys())
        free = pool.alloc.free_total if a.split > 1 else pool.alloc.n_free  # a split pool's every residue
        return {"ram": ram_e, "disk": disk_e, "available": pool.available(), "free": free}

    steps: list[dict] = []

    def chat_ids(chat: int, n: int, branch: int = 0) -> list[int]:
        # a chat's ids: its own stream; a branch rewrites its last 100 tokens (a regenerated message)
        ids = [chat * 100_000 + i for i in range(n)]
        if branch:
            cut = n - 100
            ids[cut:] = [chat * 100_000 + 50_000 + branch * 1000 + i for i in range(n - cut)]
        return ids

    def admit(slot: int, chat: int, n: int, max_new: int, reply: int, branch: int = 0, label: str = "") -> None:
        prompt = chat_ids(chat, n, branch)
        hit = store.find(prompt, TAG)
        tier, key, cached = "", "", 0
        if hit is not None:
            tier, key = hit
            cached = store.length(tier, key)
        need = pool.need_pages(n, max_new, capacity)
        shared = cached // page if tier == "ram" else 0
        want = need - shared
        spills = []
        if pool.available() < want:
            spills = store.plan_spill(want, keep=[key])
            assert spills is not None, "the scenario never makes the served engine wait"
        spilled = sorted(name_of(k) for k in spills)
        for k in spills:
            store.evict(k)
        sl = pool.slots[slot]
        assert not sl.pages
        sl.reserve(need)
        valid[slot] = 0
        stash_from[slot] = 0
        if tier:
            snap = store.restore(tier, key, sl)
            assert snap.pos == cached
            n_ds = int(snap.arrays["dspark_values"].shape[1]) if "dspark_values" in snap.arrays else 0
            valid[slot] = cached - n_ds if n_ds else cached
            stash_from[slot] = cached - (len(snap.arrays["ced_ids"]) if "ced_ids" in snap.arrays else 0)
        hist[slot] = list(prompt[:cached])
        s_at = SE.snapshot_point(n)
        save_at = s_at if s_at > cached and s_at >= 64 else None
        saved = None
        if save_at is not None:
            sl.ensure(save_at)
            hist[slot].extend(prompt[cached:save_at])
            b = bounded(slot, save_at)
            k = SE.entry_key(TAG, hist[slot])
            names.setdefault(k, f"c{chat}b{branch}@{save_at}")
            e = store.save(sl, b, hist[slot], TAG, "prompt")
            saved = {"at": save_at, "nbytes": b.nbytes(), "ds_rows": b.meta["ds_rows"], "dup": e is None}
        sl.ensure(n - 1)
        hist[slot].extend(prompt[len(hist[slot]):n - 1])
        # the reply: the verify windows commit the prompt's last token and the reply's tokens
        sl.ensure(n + reply)
        steps.append({"op": "admit", "label": label, "slot": slot, "chat": chat, "branch": branch, "n": n,
                      "max_new": max_new, "reply": reply, "tier": tier or "none", "cached": cached, "need": need,
                      "want": want, "spills": spilled, "saved": saved, **state()})

    def finish(slot: int) -> None:
        pool.slots[slot].release()
        hist[slot] = []
        steps.append({"op": "finish", "slot": slot, **state()})

    def wait_probe(chat: int, n: int, max_new: int) -> None:
        """Admission of a request that does not fit: plan_spill None (it waits); nothing changes."""

        prompt = chat_ids(chat, n)
        hit = store.find(prompt, TAG)
        key = hit[1] if hit else ""
        cached = store.length(*hit) if hit else 0
        need = pool.need_pages(n, max_new, capacity)
        want = need - (cached // page if hit and hit[0] == "ram" else 0)
        spills = store.plan_spill(want, keep=[key]) if pool.available() < want else []
        steps.append({"op": "probe", "chat": chat, "n": n, "max_new": max_new, "need": need, "want": want,
                      "wait": spills is None, **state()})

    # turn 1 of four chats, interleaved: slots fill, the RAM tier trims (parks >= 1,024 tokens, drops shorter ones)
    admit(0, 1, 1500, 200, 40, label="c1 t1")
    admit(3, 6, 100, 50, 10, label="c6 t1 (under one window)")
    finish(3)
    admit(1, 2, 700, 100, 30, label="c2 t1 (short: dropped on trim)")
    admit(2, 3, 1300, 200, 50, label="c3 t1")
    finish(1)
    admit(1, 4, 1100, 300, 20, label="c4 t1")
    finish(0)
    finish(2)
    # turn 2 of chat 1 resumes (RAM or NVMe) while chat 4 decodes; chat 2 resumes nothing (dropped)
    admit(0, 1, 1700, 200, 30, label="c1 t2")
    admit(2, 2, 900, 100, 20, label="c2 t2")
    wait_probe(5, 2000, 3000)
    finish(1)
    admit(1, 3, 1600, 200, 20, label="c3 t2")
    # an identical resend of chat 3 turn 2 in another slot while slot 1 holds it: a RAM hit's shared pages
    admit(3, 3, 1600, 100, 10, label="c3 t2 resend")
    finish(3)
    finish(0)
    finish(2)
    # a regenerated message: chat 1 again with its tail rewritten (resumes below the branch point)
    admit(0, 1, 1700, 200, 30, branch=1, label="c1 t2 regenerate")
    admit(2, 4, 1400, 1200, 20, label="c4 t2 (big quota: spills)")
    finish(1)
    admit(1, 1, 2000, 200, 10, label="c1 t3")
    finish(0)
    finish(2)
    admit(3, 2, 1200, 100, 10, label="c2 t3")
    finish(1)
    finish(3)

    out = {"python": "8474f31" if a.split < 2 else f"kvsplit split {a.split} rank {a.rank}", "tag": TAG, "page": page, "pages": pages, "slots": slots_n, "capacity": capacity,
           "slack": pool.slack, "ram_bytes": ram, "disk_min": disk_min,
           "measure": {"layers": len(layers), "decoder_start": dec, "window": W, "carries": len(carries),
                       "head_dim": hd, "hidden": D, "hc": hc, "ds_blocks": ds_blocks, "keep": RP.KEEP,
                       "row_bytes": int(Records.new(1, "cpu").values.shape[1] + Records.new(1, "cpu").scales.shape[1]),
                       "lookback": LOOKBACK},
           "stats": store.stats, "steps": steps}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{len(steps)} steps, stats {store.stats}, ram budget {ram} B -> {a.out}")


if __name__ == "__main__":
    main()
