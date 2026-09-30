"""A MoE layer's routed experts in ONE copy that two kernels share:

* cuda-exl3's stacked layout, one stack per expert width: gate|up fused [n, D/16, 2I/16, 16K] and down
  [n, I/16, D/16, 16K] - prompt chunks run its grouped GEMM (weights read once a chunk, tensor cores);
* TensorFold's universal experts kernel reads each expert's gate and up as column blocks of the fused stack through
  its trellis row stride (``prepare(..., gu_stride=)``) - decode windows, row-invariant as ever.

Needs a native ``cuda_exl3`` build (and vLLM's ``moe_align_block_size`` op) for the prompt GEMM; ``available()`` says
whether they load, else the engine keeps TensorFold's own experts layout for prompts and decode alike.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from ..config import Config

_OPS: list = []
_LOCK = __import__("threading").Lock()


def available() -> bool:
    try:
        import cuda_exl3.ops as cops
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size  # noqa: F401

        return cops.backend() == "native"
    except Exception:                                  # noqa: BLE001
        return False


def _ops():
    with _LOCK:                                      # ranks as threads (tests) arrive together
        if not _OPS:
            import cuda_exl3.ops  # noqa: F401  registers torch.ops.cuda_exl3_C
            from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

            ops = torch.ops.cuda_exl3_C
            _OPS.extend([ops, moe_align_block_size, "out" in str(ops.exl3_moe_gemm.default._schema)])
        return tuple(_OPS)


def _block_m(rows: int, experts: int) -> int:
    """cuda-exl3's row-block ladder (one of its GEMM's BM tiers)."""
    per = rows / max(experts, 1)
    return 16 if per < 16 else 32 if per < 48 else 64 if per < 96 else 128


class SharedExperts:
    def __init__(self, r, cfg: Config, layer: int, device="cuda") -> None:
        from tensorfold.cuda.exl3 import experts as tfx

        self.tfx = tfx
        base = f"model.layers.{layer}.mlp.experts"
        E, D = cfg.n_routed_experts, cfg.hidden_size
        shape = lambda n: r._file(n).get_slice(n).get_shape()   # noqa: E731  (full shapes: the index, no reads)
        kw = [shape(f"{base}.{e}.gate_proj.trellis")[-1] for e in range(E)]
        groups: dict[int, list[int]] = {}
        for e, k in enumerate(kw):
            groups.setdefault(k, []).append(e)
        self.dims, self.count = D, E
        gate, up, down = [None] * E, [None] * E, [None] * E
        self.groups = []
        g13s = g13v = g2s = g2v = None
        for k, ids in sorted(groups.items()):
            n = len(ids)
            fused = w2 = None
            s13 = torch.empty((n, 2, D), dtype=torch.float16, device=device)
            for j, e in enumerate(ids):
                p = f"{base}.{e}"
                tg, tu, td = (r.get(f"{p}.{m}_proj.trellis", device) for m in ("gate", "up", "down"))
                if tg.shape[-1] != k or tu.shape[-1] != k or td.shape[-1] != k:
                    raise ValueError(f"layer {layer} expert {e}: gate/up/down widths differ")
                if fused is None:
                    it = tg.shape[1]
                    fused = torch.empty((n, tg.shape[0], 2 * it, k), dtype=torch.int16, device=device)
                    w2 = torch.empty((n, *td.shape), dtype=torch.int16, device=device)
                    v13 = torch.empty((n, 2 * it * 16), dtype=torch.float16, device=device)
                    s2 = torch.empty((n, 1, it * 16), dtype=torch.float16, device=device)
                    v2 = torch.empty((n, D), dtype=torch.float16, device=device)
                fused[j, :, :it].copy_(tg)
                fused[j, :, it:].copy_(tu)
                w2[j].copy_(td)
                del tg, tu, td
                h = lambda m, s: r.get(f"{p}.{m}_proj.{s}", device).half()   # noqa: E731
                s13[j, 0], s13[j, 1] = h("gate", "suh"), h("up", "suh")
                v13[j, :it * 16], v13[j, it * 16:] = h("gate", "svh"), h("up", "svh")
                s2[j, 0], v2[j] = h("down", "suh"), h("down", "svh")
            I = it * 16
            gmap = torch.full((E,), -1, dtype=torch.int32, device=device)
            gmap[torch.tensor(ids, device=device)] = torch.arange(n, dtype=torch.int32, device=device)
            st = {"E": n, "H": D, "I": I, "w13_trellis": fused, "w13_suh": s13, "w13_svh": v13, "w2_trellis": w2,
                  "w2_suh": s2, "w2_svh": v2}
            self.groups.append(SimpleNamespace(st=st, gmap=gmap, K=k))
            for j, e in enumerate(ids):
                gate[e] = (fused[j, :, :it], s13[j, 0], v13[j, :I])
                up[e] = (fused[j, :, it:], s13[j, 1], v13[j, I:])
                down[e] = (w2[j], s2[j, 0], v2[j])
        self.width = I
        self.cb = 2 if r.codebook(f"{base}.0.gate_proj") == "mul1" else 1
        self.ex = tfx.prepare(gate, up, down, self.cb, device=device, gu_stride=2 * (I // 16))
        torch.cuda.empty_cache()

    # decode windows: the row-invariant kernel
    def scratch(self, rows: int, slots: int, device="cuda"):
        return self.tfx.Scratch(self.ex, rows, slots, device=device)

    def decode(self, x, pick, wts, scratch, out, R: int) -> None:
        self.tfx.routed(x, pick, wts, self.ex, scratch, out, R)

    # prompt chunks: cuda-exl3's grouped GEMM, one width group after another accumulating into one output
    def prefill(self, x: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
        ops, align, has_out = _ops()
        tokens, T = x.shape[0], pick.shape[1]
        out = None
        for g in self.groups:
            st = g.st
            H, I = st["H"], st["I"]
            bm = _block_m(tokens * T, self.count)
            sid, eid, nrows = align(pick, bm, self.count, expert_map=g.gmap, pad_sorted_ids=True)
            sid, eid, nrows = sid.int(), eid.int(), nrows.int()
            rows = min(eid.numel() * bm, sid.numel())
            eid = eid[: rows // bm]
            a13 = torch.empty((2, rows, H), dtype=torch.half, device=x.device)
            ops.exl3_moe_had_in(x, a13, st["w13_suh"], sid, eid, nrows, bm, T, tokens * T, True)
            inter = ops.exl3_moe_gemm(a13, st["w13_trellis"], st["w13_suh"], st["w13_svh"], eid, nrows, [I, I],
                                      self.cb, bm, torch.bfloat16, sid, None, tokens, T)
            a2 = torch.empty((1, rows, I), dtype=torch.half, device=x.device)
            ops.exl3_moe_glu_had_in(inter, a2, st["w2_suh"], eid, nrows, bm)
            args = (a2, st["w2_trellis"], st["w2_suh"], st["w2_svh"], eid, nrows, [H], self.cb, bm, torch.bfloat16,
                    sid, wts, tokens, T)
            if has_out:
                out = ops.exl3_moe_gemm(*args, out)
            else:
                y = ops.exl3_moe_gemm(*args)
                out = y if out is None else out.add_(y)
        return out
