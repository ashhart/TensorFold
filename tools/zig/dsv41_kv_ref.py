#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
"""KV pool set-up for the references and captures: one forward slot in a replicated or split pool, bound as serving."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

MODES = ("pool", "split")


def _zero(store) -> None:
    import torch

    for attr in ("values", "scales", "data"):
        t = getattr(store, attr, None)
        if isinstance(t, torch.Tensor):
            t.zero_()


def ranges(rows: list[int]) -> list[list[int]]:
    """Sorted row indices as [a, b) runs."""

    out: list[list[int]] = []
    for r in rows:
        if out and out[-1][1] == r:
            out[-1][1] = r + 1
        else:
            out.append([r, r + 1])
    return out


class KV:
    """The pool, the forward's pool slot ``ps`` and the page holder ``other``, bound to ``fw``."""

    def __init__(self, fw, pack: Path, mode: str, rank: int, world: int, capacity: int, device,
                 scramble: bool = True) -> None:
        from tensorfold.families.deepseek_v41.cuda import pool as PO
        from tensorfold.families.deepseek_v41.cuda import topology as TO
        kvpool = PO.kvpool

        if mode not in MODES:
            raise ValueError(f"KV mode {mode!r}: {MODES}")
        self.PO, self.kvpool = PO, kvpool
        self.mode, self.rank, self.world = mode, int(rank), int(world)
        self.topo = TO.load(pack)
        self.index_kv = os.environ.get("TF_DSV41_INDEX_KV", "bf16").strip() or "bf16"
        self.split = self.world if mode == "split" else 0
        page = PO.PAGE
        maxp = kvpool.pages_for(capacity, page)
        npages = 2 * maxp + 4                   # the slot, the holder's interleaved pages and its restore shifts
        whole = max(1, self.split)
        npages = -(-npages // whole) * whole
        # split= / rank= only for a split pool (a replicated pool is the default: older Pools have neither keyword)
        kw = {"split": self.split, "rank": self.rank} if self.split else {}
        self.pool = PO.Pool(self.topo, npages * page, index_kv=self.index_kv, device=device, **kw)
        self.ps = self.pool.new_slot(capacity)
        self.other = self.pool.new_slot(2 * capacity + 4 * page)
        self.scramble = bool(scramble)
        self.page = page
        fw.bind([self.ps])
        self.kvx = getattr(fw, "kvx", None)

    # -- pages -----------------------------------------------------------------------------------------------------
    def _hold_one(self) -> None:
        self.other.ensure((len(self.other.pages) + 1) * self.page)

    def grow(self, end: int) -> None:
        need = self.kvpool.pages_for(int(end), self.page)
        while len(self.ps.pages) < need:
            if self.scramble:
                self._hold_one()
            self.ps.ensure((len(self.ps.pages) + 1) * self.page)

    def fresh(self, fw) -> None:
        self.ps.release()
        self.other.release()
        for t in self.pool.phys.values():
            t.zero_()
        for s in fw.slots:
            for st in s.swa.values():
                _zero(st)
            for c in s.carry.values():
                c.zero_()
        fw.reset(0)

    # -- description -----------------------------------------------------------------------------------------------
    def shifts(self, fw) -> dict[int, int]:
        out = {}
        for blk in fw.blocks:
            if blk.mode == "full":
                per = self.pool.family(f"comp.{blk.index}").page_rows(self.page)
                out[blk.index] = per.bit_length() - 1
        return out

    def describe(self, fw) -> dict:
        p = self.pool
        d = {"mode": self.mode, "page": p.page, "split": self.split, "rank": self.rank, "world": self.world,
             "index_kv": self.index_kv, "pool_tokens": p.npages * p.page, "npages": p.npages, "null": p.null,
             "scramble": self.scramble, "slot_pages": [int(x) for x in self.ps.pages],
             "holder_pages": [int(x) for x in self.other.pages], "table_len": int(self.ps.table.numel()),
             "shift": {str(L): s for L, s in self.shifts(fw).items()},
             "families": {f.name: {"layer": f.layer, "ratio": f.ratio, "row_bytes": f.row_bytes, "split": getattr(f, "split", False),
                                   "dtype": str(p.phys[f.name].dtype).replace("torch.", ""),
                                   "shape": list(p.phys[f.name].shape)} for f in p.fams}}
        if self.split:
            d.update(lnull=p.lnull, discard=p.discard, local_pages=p.local,
                     slot_local=[int(p.local_page(x)) for x in self.ps.pages])
        if self.kvx is not None:
            d.update(compact=self.kvx.compact, var=bool(self.kvx.var), dense_max=int(self.kvx.dense_max))
        return d

    # -- the state dump --------------------------------------------------------------------------------------------
    def dump(self, fw, put, sd: Path) -> dict:
        """The forward slot's pool state by role (families, tables, rows in logical order) as state.json's "kv"."""

        import torch

        p, ps, pos = self.pool, self.ps, int(fw.slot.pos)
        info = self.describe(fw)
        rows_info = {}
        for blk in fw.blocks:
            if blk.mode != "full":
                continue
            L = blk.index
            comp, ik = p.phys[f"comp.{L}"], p.phys[f"index_k.{L}"]
            f = p.family(f"comp.{L}")
            per = f.page_rows(p.page)
            shift = per.bit_length() - 1
            n = pos // f.ratio
            put(f"s.kv.comp.L{L}", comp)
            put(f"s.kv.ik.L{L}", ik)
            r = torch.arange(n, dtype=torch.long, device=comp.device)
            tab = ps.table.to(torch.long)
            rk = (tab[r >> shift] << shift) | (r & (per - 1))
            put(f"s.kv.ik.L{L}.rows", ik.index_select(0, rk))
            if self.split:
                rc = r[((r >> shift) % self.world) == self.rank]
                ctab = ps.ctable.to(torch.long)
                pc = (ctab[rc >> shift] << shift) | (rc & (per - 1))
            else:
                rc, pc = r, rk
            put(f"s.kv.comp.L{L}.rows", comp.index_select(0, pc))
            owned = ranges(rc.cpu().tolist())
            one = {"layer": L, "ratio": f.ratio, "shift": shift, "rows": n, "comp_count": int(rc.numel()),
                   "comp_logical": owned, "ik_count": n}
            (sd / f"s.kv.comp.L{L}.rows.json").write_text(json.dumps(one) + "\n")
            rows_info[str(L)] = one
        put("s.kv.pt", ps.table)
        if self.split:
            put("s.kv.ct", ps.ctable)
        info["rows"] = rows_info
        info["pos"] = pos
        return info


class Park:
    """The session round trip on ``kv``'s slot: save and park at the prompt's end, restore onto other physical pages."""

    def __init__(self, kv: KV, root: Path) -> None:
        from tensorfold.families.deepseek_v41.cuda import sessdisk as SD
        from tensorfold.families.deepseek_v41.cuda import sessions as SS
        from tensorfold.families.deepseek_v41.cuda.residency import Policy

        self.kv, self.SS = kv, SS
        self.tag = SS.Tag()
        ident = SD.compat_ident(image="dsv41-kv-ref", knobs=SD.knobs_from_env(),
                                layout=SS.compat_extra(self.tag, kv.topo, kv.split), extra={"page": kv.pool.page})
        pol = Policy.from_env()
        self.disk = SD.DiskTier(root, kv.rank, ident, budget_gib=16.0, min_tokens=1, supersede=pol.supersede)
        self.disk.staging_bytes = pol.staging_mib << 20
        if pol.async_io:
            self.disk.start_async(pol.inflight_mib << 20)
        self.store = SS.Store(kv.pool, disk=self.disk, policy=pol)
        self.key = None
        self.info: dict = {"policy": pol.describe(), "root": str(root)}

    def save_park(self, fw, ids) -> None:
        import torch

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        snap = fw.snapshot(0)
        e = self.store.save(self.kv.ps, snap, list(ids), self.tag.code(), "prompt")
        if e is None:
            raise RuntimeError("park: the store kept no entry")
        self.key = e.key
        t1 = time.perf_counter()
        self.store.park(e.key)
        flush = getattr(self.disk, "flush", None)
        if callable(flush) and getattr(self.disk, "io", None) is not None:
            flush()
        t2 = time.perf_counter()
        ent = self.disk.index.get(e.key)
        self.info.update(key=e.key, pos=int(snap.pos), save_s=round(t1 - t0, 4), park_s=round(t2 - t1, 4),
                         file_bytes=int(getattr(ent, "size", 0) or 0), pages=len(e.pages),
                         bounded_bytes=int(snap.nbytes()))

    def restore(self, fw, prompt_next) -> None:
        """Empty the slot, find the entry for ``prompt_next`` (a strict prefix), restore it and the bounded state."""

        import torch

        kv = self.kv
        kv.ps.release()
        fw.reset(0)
        if kv.scramble:
            kv._hold_one()
        got = self.store.find(list(prompt_next), self.tag.code())
        if got is None or got[1] != self.key:
            raise RuntimeError(f"park: the store finds {got} for the prompt, not ('disk', {self.key})")
        t0 = time.perf_counter()
        snap = self.store.restore(got[0], got[1], kv.ps)
        fw.restore(0, snap)
        torch.cuda.synchronize()
        self.info.update(tier=got[0], restore_s=round(time.perf_counter() - t0, 4),
                         restored_pages=[int(x) for x in kv.ps.pages])
        if kv.split:
            self.info["restored_local"] = [int(kv.pool.local_page(x)) for x in kv.ps.pages]

    def close(self) -> None:
        try:
            self.store.close()
        except Exception:                       # noqa: BLE001 - the reference's result is written already
            pass


def install_glue(cap) -> None:
    """Record the split exchange's non-Triton steps (gather, all_gather / var_gather) into ``cap``'s op log as glue."""

    from tensorfold.families.deepseek_v41.cuda import kvsplit as KS

    if getattr(KS, "_capture_glue", False):
        return
    real_fast, real_var, real_torch = KS.fast_gather, KS.var_gather, KS.torch

    def fast(comm, send, recv):
        if not cap.recording:
            return real_fast(comm, send, recv)
        row = {"kind": "glue", "ext": "glue", "func": "kvsplit.all_gather", "name": "kvsplit.all_gather",
               "world": int(getattr(comm, "world", 1)), "rank": int(getattr(comm, "rank", 0)), "peer": "zeros"}
        return cap.record(row, [send, recv], {}, lambda: real_fast(comm, send, recv))

    def var(comm, send, recv, lens):
        if not cap.recording:
            return real_var(comm, send, recv, lens)
        row = {"kind": "glue", "ext": "glue", "func": "kvsplit.var_gather", "name": "kvsplit.var_gather",
               "world": int(getattr(comm, "world", 1)), "rank": int(getattr(comm, "rank", 0)), "peer": "zeros"}
        return cap.record(row, [send, recv, lens], {}, lambda: real_var(comm, send, recv, lens))

    class TorchGlue:
        """``torch`` as kvsplit sees it, ``index_select`` recorded."""

        def __getattr__(self, name):
            return getattr(real_torch, name)

        @staticmethod
        def index_select(*args, **kwargs):
            if not cap.recording:
                return real_torch.index_select(*args, **kwargs)
            row = {"kind": "glue", "ext": "glue", "func": "kvsplit.gather", "name": "kvsplit.gather"}
            return cap.record(row, list(args), kwargs, lambda: real_torch.index_select(*args, **kwargs))

    KS.fast_gather, KS.var_gather, KS.torch = fast, var, TorchGlue()
    KS._capture_glue = True
