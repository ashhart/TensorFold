"""The NVFP4 MoE (NVIDIA ModelOpt's FP4 format) for the Flash Next CUDA engine.

The Swift 1.5 Flash Next NVFP4 checkpoint quantizes ONLY the routed experts (the 512 experts of the main
layers and of the MTP layer): per expert and projection, packed E2M1 nibbles (uint8 [N, K/2]), fp8e4m3
block scales ([N, K/16]) and a second per-tensor scale. Everything else — the shared expert, the router,
the hyper-connections, DeltaNet, attention, PLE, embeddings, lm_head and the rest of the MTP head — is
stored BF16 (``exclude_modules`` in the checkpoint's quantization config); ``weights.py`` feeds those BF16
tensors here as plain [out, in] tensors.

``MoE4`` is the engine-facing table, on the qmm ``Experts`` face (``count`` / ``width`` / ``dims``): the
routed experts as one stacked FP4 grid per projection — gate and up rows joined row-wise per expert into
an ``[E, 2*NI, K]`` grid so expert ``e`` owns rows ``[e*2NI, (e+1)*2NI)`` (gate rows first, then up) and
down rows into ``[E, D, NI]`` — and the shared expert as an identity-scaled FP4 table
(``nvfp4.fp4_from_bf16``: BF16 values ride the same kernel as scale-1 rows). Expert ids follow the qmm
convention: 0..E-1 routed, E the shared one.

The serving path is the single row-invariant ``nvfp4.matmul``: which rows go to which expert is data, and
a row's bits depend only on its own input, never on the window it was drafted in — the same contract
``tests/cuda/test_flashnext_nvfp4.py`` checks kernel-first, the way ``adding-a-cuda-family.md`` asks.
``gateup_out`` / ``down_out`` take the engine's ``Plan`` (item list, members and counts on the device) and
write each row back at its own member index:

    gate/up: act = silu(bf16(x @ gate.T)) * (x @ up.T)      (the checkpoint's gate and up, one grid)
    down:    y   = act @ down.T                             (fp32 sums: the combine adds the slots in order)

``moe`` is the full step (router, selection, the grouped experts, the shared expert with its sigmoid
gate) into the qmm ``MoEBuffers`` contract: ``y`` [R, slots, D] fp32 with slot k the shared expert,
``wts`` the routed weights then the shared gate. The flat grouped kernels (`nvfp4_grouped`) resolve the
plan on the device so a CUDA-graph capture accepts the step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import nvfp4, nvfp4_grouped

import os


def _moe_backend() -> str:
    """``grouped`` (default) or ``python`` host item loop via ``TENSORFOLD_NVFP4_MOE``."""

    return (os.environ.get("TENSORFOLD_NVFP4_MOE") or "grouped").strip().lower()



@dataclass
class Expert4:
    """One expert: gate/up rows in one grid (gate rows first), and the down grid."""

    gu: nvfp4.FP4         # [2*NI, K]
    down: nvfp4.FP4       # [K, NI]


@dataclass
class MoE4:
    """One layer's experts (the engine's ``MoEW.experts``). The routed experts ride two stacked grids —
    ``gate_up`` [E, 2NI, D] (expert e's rows at e*2NI, gate first then up) and ``down_proj`` [E, D, NI]
    (expert e's rows at e*D) — with a leading expert axis over the tiles; ``shared`` is the BF16 expert
    as identity-scaled tables. Expert id ``count - 1`` is the shared one; ``width`` / ``dims`` are the
    qmm names for NI / K (the MoE buffers take them)."""

    gate_up: nvfp4.FP4    # packed tiles [E, 2NI/BN, D/64, 32, BN], fp8 scales [E, D/16, 2NI]
    down_proj: nvfp4.FP4  # packed tiles [E, D/BN, NI/64, 32, BN], fp8 scales [E, NI/16, D]
    shared: Expert4       # the BF16 shared expert as identity-scaled tables
    kernel: str = "nvfp4"

    # The grouped step resolves its plan's item list, its member gather and its member scatter on the device
    # (``nvfp4_grouped``), so the engine can capture a decode step again. The grid it launches is fixed in the
    # plan's capacity, so the routing decides how many items are live, never how many programs run.
    capturable = True  # retry graphs after contiguous scale2 fix; was False for Spark zero-token garbage

    # The shared expert's output and split-K partials, kept between steps: an allocation made inside a capture
    # is a block the graph pool may hand back while the kernel reading it still runs, which is what killed a
    # cached load (and, before the capture, the 42nd layer). Allocating them once takes that decision out.
    shared_buffers: dict = field(default_factory=dict, repr=False)

    def shared_out(self, x: torch.Tensor, fp: nvfp4.FP4, *, f32: bool = False):
        """``(out, part)`` for the shared expert, sized once from the first (uncaptured) step."""

        sk = nvfp4.split_for(fp.n, fp.k)
        key = (int(x.shape[0]), fp.n, f32)
        got = self.shared_buffers.get(key)
        if got is None:
            out = torch.empty((int(x.shape[0]), fp.n), dtype=torch.float32 if f32 else torch.bfloat16,
                              device=x.device)
            part = torch.empty((sk, int(x.shape[0]), fp.n), dtype=torch.float32, device=x.device) if sk > 1 \
                else None
            got = (out, part)
            self.shared_buffers[key] = got
        return got

    @property
    def routed(self) -> int:
        return int(self.gate_up.weight.shape[0])

    @property
    def count(self) -> int:
        return self.routed + 1

    @property
    def width(self) -> int:
        return self.down_proj.k                              # NI: a gate row's inputs, down's outputs

    @property
    def dims(self) -> int:
        return self.down_proj.n                              # K: x's inputs, down's outputs

    def nbytes(self) -> int:
        return self.gate_up.nbytes() + self.down_proj.nbytes() \
            + self.shared.gu.nbytes() + self.shared.down.nbytes()

    def _expert(self, e: int) -> Expert4:
        """One expert as per-expert FP4 tables (the CPU/test path; the serving path slices the stacked
        grids per program without materialising these). Exact by construction: the stacked grids' tile
        and scale rows for expert ``e`` are the per-expert table's, row for row (checked in the tests).

        Slices are made contiguous: Triton pointer math in ``nvfp4.matmul`` assumes a dense tile slab,
        and ``weight[e]`` on a stacked 5-D grid is a view that is not contiguous along the tile axis."""

        gu = nvfp4.FP4(self.gate_up.weight[e].contiguous(), self.gate_up.scale[e].contiguous(),
                       2 * self.width, self.gate_up.k,
                       scale2=_slice(self.gate_up.scale2, e), packed=self.gate_up.packed)
        down = nvfp4.FP4(self.down_proj.weight[e].contiguous(), self.down_proj.scale[e].contiguous(),
                         self.dims, self.width,
                         scale2=_slice(self.down_proj.scale2, e), packed=self.down_proj.packed)
        return Expert4(gu, down)

    def gateup_rows(self, x: torch.Tensor, e: int) -> torch.Tensor:
        """x [M, K] bf16 -> act [M, NI] bf16 for one expert (the checkpoint's silu(gate) * up)."""

        ex = self._expert(e)
        ni = self.width
        gu = nvfp4.matmul(x, ex.gu)
        gate = gu[:, :ni].to(torch.float32)
        up = gu[:, ni:].to(torch.float32)
        return ((gate / (1.0 + torch.exp(-gate))).to(torch.bfloat16).to(torch.float32)
                * up).to(torch.bfloat16)

    def down_rows(self, act: torch.Tensor, e: int) -> torch.Tensor:
        """act [M, NI] bf16 -> y [M, K] bf16 for one expert."""

        return nvfp4.matmul(act, self._expert(e).down)

    def gateup(self, x: torch.Tensor, groups: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        """x [M, K] bf16, top-k routed slots. ``groups`` [U] the distinct expert ids (in increasing
        order), ``perm`` [U, maxm] flat row*32 + slot codes per group (-1 after the last): act [M, NI]
        bf16. The shared expert (id ``count - 1``) rides every row: the caller folds its output in
        (``gateup_rows(x, count - 1)``)."""

        act = torch.empty((x.shape[0], self.width), dtype=torch.bfloat16, device=x.device)
        for e, rows in self._rows(groups, perm):
            act[rows] = self.gateup_rows(x[rows], e)
        return act

    def down(self, act: torch.Tensor, groups: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        """act [M, NI] bf16, same grouping: y [M, K] bf16."""

        y = torch.empty((act.shape[0], self.dims), dtype=torch.bfloat16, device=act.device)
        for e, rows in self._rows(groups, perm):
            y[rows] = self.down_rows(act[rows], e)
        return y

    @staticmethod
    def _dest(codes: torch.Tensor, slots: int) -> torch.Tensor:
        """The grouping's codes (``row * 32 + slot``, the ``qmm.Group`` convention) as the engine buffer's
        flat ``row * slots + slot`` index (the qmm kernels' ``dest``; ``slots`` is the buffer's)."""

        return (codes // 32) * slots + (codes % 32)

    def _rows(self, groups: torch.Tensor, perm: torch.Tensor):
        """(expert id, row indices) per group, the shared expert's id skipped."""

        for u in range(int(groups.shape[0])):
            e = int(groups[u])
            if e == self.count - 1:
                continue                                     # the shared expert: the caller's path
            codes = perm[u]
            codes = codes[codes >= 0]
            if codes.numel():
                yield e, codes // 32



    def gateup_out(self, x: torch.Tensor, plan, act: torch.Tensor, top_k: int) -> torch.Tensor:
        """The gate/up step for every plan item into act [R, slots, NI] bf16; the shared expert's slot is
        filled for every row from its BF16 tables. The plan's item list stays on the device.

        Set ``TENSORFOLD_NVFP4_MOE=python`` to force the host item loop (Spark IMA diagnosis)."""

        ni = self.width
        slots = act.shape[1]
        flat = act.reshape(-1, ni)
        if _moe_backend() == "python":
            self._gateup_python(x, plan, flat, slots)
        else:
            nvfp4_grouped.gateup(self.gate_up, x, flat, plan, ni, slots)
        out, part = self.shared_out(x, self.shared.gu)
        g = nvfp4.matmul(x, self.shared.gu, out=out, part=part)
        gate = g[:, :ni].to(torch.float32)
        up = g[:, ni:].to(torch.float32)
        act[:, top_k] = ((gate / (1.0 + torch.exp(-gate))).to(torch.bfloat16).to(torch.float32)
                         * up).to(torch.bfloat16)
        return act


    def down_out(self, act: torch.Tensor, plan, y: torch.Tensor, top_k: int) -> torch.Tensor:
        """The down step for every plan item into y [R, slots, D] (the combine adds the slots in order; a
        prefill buffer's bf16 slots round each sum once, so the fp32 sum is rounded into the slot's own
        dtype — a decode plan's slots are fp32). The shared expert's slot for every row."""

        d, ni = self.dims, self.width
        flat = y.reshape(-1, d)
        if _moe_backend() == "python":
            self._down_python(act, plan, flat)
        else:
            nvfp4_grouped.down(self.down_proj, act.reshape(-1, ni), flat, plan, slots=y.shape[1])
        out, part = self.shared_out(act[:, top_k], self.shared.down, f32=True)
        y[:, top_k] = nvfp4.matmul(act[:, top_k], self.shared.down, f32=True, out=out, part=part).to(y.dtype)
        return y

    def _gateup_python(self, x, plan, flat, slots: int) -> None:
        n = int(plan.counts[0].item())
        items = plan.items[:n].detach().cpu().tolist()
        members = plan.members.detach().cpu()
        for e, first, count in items:
            e, first, count = int(e), int(first), int(count)
            if count == 0 or e >= self.routed:
                continue
            dest = members[first:first + count].to(device=x.device, dtype=torch.long)
            flat[dest] = self.gateup_rows(x[dest // slots], e)

    def _down_python(self, act, plan, flat) -> None:
        ni = self.width
        n = int(plan.counts[0].item())
        items = plan.items[:n].detach().cpu().tolist()
        members = plan.members.detach().cpu()
        act_flat = act.reshape(-1, ni)
        for e, first, count in items:
            e, first, count = int(e), int(first), int(count)
            if count == 0 or e >= self.routed:
                continue
            dest = members[first:first + count].to(device=act.device, dtype=torch.long)
            flat[dest] = nvfp4.matmul(act_flat[dest].contiguous(), self._expert(e).down, f32=True).to(flat.dtype)


def _slice(factor: torch.Tensor | None, e: int) -> torch.Tensor | None:
    """Expert ``e``'s per-tensor factor, or None where the table carries none — a BF16 pattern grid, where the
    kernel applies its own default. Indexing None is what raised on a real serve's first draft step.

    Contiguous: a stacked ``scale2`` can be an expand-view (stride 0); Triton's ``S2 + rn`` assumes unit
    stride, so a non-contiguous row would multiply the wrong lanes."""

    if factor is None:
        return None
    row = factor[e]
    return row if row.is_contiguous() else row.contiguous()


def _stacked_bf16(bits: torch.Tensor, rows: torch.Tensor) -> nvfp4.FP4:
    """Per-expert bf16-pattern grids [E, N, K] (uint16) and identity row scales [E, N, K/16] fp32 -> one table
    with a leading expert axis over the tiles (``n`` rows per expert; the tile count and the scale's leading
    axis carry the expert axis — ``_tile_bits`` keeps leading dims). The exact-BF16 form: patterns and fp32
    scales, not the checkpoint's packed bytes."""

    e, n, k = bits.shape
    return nvfp4.FP4(nvfp4._tile_bits(bits.contiguous()), rows.permute(0, 2, 1).contiguous(), n, k)


def moe4_from_checkpoint(gate: tuple, up: tuple, down: tuple,
                         shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> MoE4:
    """One layer from the checkpoint's stacked per-expert arrays: gate/up/down each a
    ([E, N, K/2] uint8 words, [E, N, K/16] fp8e4m3 block scales, [E] fp32 per-tensor scales) stack (the loader
    gathers the per-expert tensors into stacks), shared the BF16 (gate, up, down) [out, in] tensors. Gate and
    up rows join row-wise per expert, so the stacked gate/up table keeps expert e's rows contiguous (gate
    first, then up) with its own per-tensor scale beside each half. Nothing is widened here: the checkpoint's
    bytes go to the device and the kernels decode them, which is what keeps the model the size of its file."""

    gw, gs, g2 = gate
    uw, us, u2 = up
    dw, ds, d2 = down
    e, ni = int(gw.shape[0]), int(gw.shape[1])
    gu = nvfp4.stacked_fp4(torch.cat([gw, uw], dim=1), torch.cat([gs, us], dim=1),
                           torch.cat([g2[:, None].expand(e, ni), u2[:, None].expand(e, ni)], dim=1))
    return MoE4(gu, nvfp4.stacked_fp4(dw, ds, d2[:, None].expand(e, int(dw.shape[1]))), expert4_from_bf16(*shared))


def expert4_from_bf16(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> Expert4:
    """The shared expert from BF16 weights (torch linear layout [out, in]): identity-scaled FP4 tables."""

    gu = nvfp4.fp4_from_bf16(torch.cat([gate, up], dim=0).contiguous())
    return Expert4(gu, nvfp4.fp4_from_bf16(down.contiguous()))


def moe4_from_bf16(gate_up: torch.Tensor, down: torch.Tensor,
                   shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> MoE4:
    """The stacked BF16 experts (the MTP layer's, ``mtp.layers.0.mlp.experts`` in the checkpoint's
    exclusions): gate_up [E, 2NI, K] and down [E, K, NI] ride the FP4 kernels as identity-scaled tables —
    bit-exact (the bf16 values are the stored operands, scale 1), the shared expert its own tables."""

    gu = _stacked_bf16(*_one_bf16(gate_up))
    dn = _stacked_bf16(*_one_bf16(down))
    return MoE4(gu, dn, expert4_from_bf16(*shared))


def _one_bf16(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A [E, N, K] bf16 stack -> the code-bit grid [E, N, K] (uint16 patterns) and identity row scales
    [E, N, K/16] fp32 — the ``_fp4_stack`` face, so the stacked tables are built the same way."""

    e, n, k = t.shape
    bits = t.contiguous().view(torch.uint16)
    rows = torch.ones((e, n, k // nvfp4.GS), dtype=torch.float32, device=t.device)
    return bits, rows


def moe4_from_experts(gate: list, up: list, down: list, shared: tuple) -> MoE4:
    """Per-expert FP4 pairs (the CPU/test path): gate/up/down each a list of (words, weight_scale,
    weight_scale_2) over the routed experts, shared the BF16 (gate, up, down)."""

    def stack(items):
        return (torch.stack([t[0] for t in items]),
                torch.stack([t[1] for t in items]),
                torch.stack([torch.as_tensor(t[2]).reshape(()) for t in items]))

    return moe4_from_checkpoint(stack(gate), stack(up), stack(down), shared)


# -- the engine's MoE step -------------------------------------------------------------------------------

def moe(x: torch.Tensor, xs: torch.Tensor, router_rows: torch.Tensor, ex: MoE4, buf, cfg) -> None:
    """Route rows x [R, D] and run their experts into buf (the qmm ``moe`` contract, NVFP4 edition):
    buf.y [R, slots, D] fp32 (slot k: the shared expert), buf.wts [R, slots] (routed weights, then the
    shared gate's sigmoid). ``xs`` (the 32-group sums) is unused: NVFP4 has no biases, the format is
    purely multiplicative."""

    from tensorfold.cuda import moe as moe_mod

    rows = x.shape[0]
    top_k = int(cfg.num_experts_per_tok)
    moe_mod.router(x, router_rows, buf.logits[:rows])
    moe_mod.select(buf.logits[:rows], buf, top_k, ex.routed)
    ex.gateup_out(x, buf.plan, buf.act[:rows], top_k)
    ex.down_out(buf.act[:rows], buf.plan, buf.y[:rows], top_k)
    sg = buf.logits[:rows, ex.routed].to(torch.bfloat16).to(torch.float32)
    buf.wts[:rows, top_k] = (1.0 / (1.0 + torch.exp(-sg))).to(torch.bfloat16).to(torch.float32)
    return buf
