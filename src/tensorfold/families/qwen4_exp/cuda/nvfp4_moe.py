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
``gateup_out`` / ``down_out`` take the rows grouped per expert the way the engine's MoE step leaves them
(``buf.plan``: ``items`` of (expert, first, count) over ``members``, whose entries are the flat
``row * slots + slot`` indices the buffers use — what ``moe.select`` builds) and write each row back at
its own index:

    gate/up: act = silu(bf16(x @ gate.T)) * (x @ up.T)      (the checkpoint's gate and up, one grid)
    down:    y   = act @ down.T                             (fp32 sums: the combine adds the slots in order)

``moe`` is the full step (router, selection, the grouped experts, the shared expert with its sigmoid
gate) into the qmm ``MoEBuffers`` contract: ``y`` [R, slots, D] with slot k the shared expert, ``wts``
the routed weights then the shared gate. The grouped loop is one ``nvfp4.matmul`` per plan item — the
same per-row arithmetic as the one-expert path; the flat batched kernel joins when it measures faster on
Spark. The loop reads the item list on the host, so this MoE step does not go inside a captured graph
yet (``forward._mm``'s single-expert kernels do).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import nvfp4


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

    gate_up: nvfp4.FP4    # tiles [E, 2NI/BN, D/64, 64, BN], scales [E, D/16, 2NI]
    down_proj: nvfp4.FP4  # tiles [E, D/BN, NI/64, 64, BN], scales [E, NI/16, D]
    shared: Expert4       # the BF16 shared expert as identity-scaled tables
    kernel: str = "nvfp4"

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
        and scale rows for expert ``e`` are the per-expert table's, row for row (checked in the tests)."""

        gu = nvfp4.FP4(self.gate_up.weight[e], self.gate_up.scale[e], 2 * self.width, self.gate_up.k)
        down = nvfp4.FP4(self.down_proj.weight[e], self.down_proj.scale[e], self.dims, self.width)
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

    def items(self, plan):
        """(expert id, slot indices) per plan item, the shared expert's items skipped: a plan member is
        already the buffers' flat ``row * slots + slot`` index."""

        for e, first, count in plan.items[:int(plan.counts[0])].tolist():
            if e != self.count - 1 and count:
                yield e, plan.members[first:first + count]

    def gateup_out(self, x: torch.Tensor, plan, act: torch.Tensor, top_k: int) -> torch.Tensor:
        """The gate/up step writing at the plan's member indices into act [R, slots, NI] bf16 (the
        engine's buffers); the shared expert's slot filled for every row from its BF16 tables."""

        ni = self.width
        slots = act.shape[1]
        flat = act.reshape(-1, ni)
        for e, dest in self.items(plan):
            flat[dest] = self.gateup_rows(x[dest // slots], e)
        g = nvfp4.matmul(x, self.shared.gu)
        gate = g[:, :ni].to(torch.float32)
        up = g[:, ni:].to(torch.float32)
        act[:, top_k] = ((gate / (1.0 + torch.exp(-gate))).to(torch.bfloat16).to(torch.float32)
                         * up).to(torch.bfloat16)
        return act

    def down_out(self, act: torch.Tensor, plan, y: torch.Tensor, top_k: int) -> torch.Tensor:
        """The down step writing at the plan's member indices into y [R, slots, D] (the combine adds the
        slots in order; a prefill buffer's bf16 slots round each sum once); the shared expert's slot for
        every row. The down outputs are fp32 sums."""

        d, ni = self.dims, self.width
        slots = y.shape[1]
        flat = y.reshape(-1, d)
        for e, dest in self.items(plan):
            flat[dest] = nvfp4.matmul(act.reshape(-1, ni)[dest], self._expert(e).down, f32=True)
        y[:, top_k] = nvfp4.matmul(act[:, top_k], self.shared.down, f32=True)
        return y


def _stacked(bits: torch.Tensor, rows: torch.Tensor) -> nvfp4.FP4:
    """Per-expert code-bit grids [E, N, K] (uint16) and row scales [E, N, K/16] fp32 -> one FP4 table with
    a leading expert axis over the tiles (``n`` rows per expert; the tile count and the scale's leading
    axis carry the expert axis — ``_tile_bits`` keeps leading dims)."""

    e, n, k = bits.shape
    return nvfp4.FP4(nvfp4._tile_bits(bits.contiguous()), rows.permute(0, 2, 1).contiguous(), n, k)


def _fp4_stack(w: torch.Tensor, s: torch.Tensor, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stacked checkpoint arrays (words [E, N, K/2] uint8, weight_scale [E, N, K/16] fp8e4m3,
    weight_scale_2 [E] fp32) -> code bits [E, N, K] uint16 and row scales [E, N, K/16] fp32 (the
    per-tensor factors folded into the row scales — the FP4 table's form). A scalar ``s2``: the same
    per-tensor scale on every expert (the 512 tiny per-expert scalars read as one stack)."""

    bits = nvfp4.e2m1_bits(w)
    e = int(w.shape[0])
    if isinstance(s2, torch.Tensor) and s2.dim() == 0:
        rows = nvfp4.row_scales(s, float(s2)).reshape(e, s.shape[-2], s.shape[-1])
    else:
        rows = torch.stack([nvfp4.row_scales(s[i], float(s2[i])) for i in range(e)])
    return bits, rows


def moe4_from_checkpoint(gate: tuple, up: tuple, down: tuple,
                         shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> MoE4:
    """One layer from the checkpoint's stacked per-expert arrays: gate/up/down each a
    ([E, N, K/2] uint8, [E, N, K/16] fp8e4m3, [E] fp32) stack (the loader gathers the per-expert tensors
    into stacks), shared the BF16 (gate, up, down) [out, in] tensors. Gate and up rows join row-wise per
    expert, so the stacked gate/up grid keeps expert e's rows contiguous (gate first, then up)."""

    gb, gr = _fp4_stack(*gate)
    ub, ur = _fp4_stack(*up)
    db, dr = _fp4_stack(*down)
    gu_bits = torch.cat([gb, ub], dim=1)                       # [E, 2NI, K]: gate rows, then up rows
    gu_rows = torch.cat([gr, ur], dim=1)                       # [E, 2NI, K/16]
    return MoE4(_stacked(gu_bits, gu_rows), _stacked(db, dr), expert4_from_bf16(*shared))


def expert4_from_bf16(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> Expert4:
    """The shared expert from BF16 weights (torch linear layout [out, in]): identity-scaled FP4 tables."""

    gu = nvfp4.fp4_from_bf16(torch.cat([gate, up], dim=0).contiguous())
    return Expert4(gu, nvfp4.fp4_from_bf16(down.contiguous()))


def moe4_from_bf16(gate_up: torch.Tensor, down: torch.Tensor,
                   shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> MoE4:
    """The stacked BF16 experts (the MTP layer's, ``mtp.layers.0.mlp.experts`` in the checkpoint's
    exclusions): gate_up [E, 2NI, K] and down [E, K, NI] ride the FP4 kernels as identity-scaled tables —
    bit-exact (the bf16 values are the stored operands, scale 1), the shared expert its own tables."""

    gu = _stacked(*_one_bf16(gate_up))
    dn = _stacked(*_one_bf16(down))
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
    buf.y [R, slots, D] (slot k: the shared expert; fp32 slots, bf16 on a prefill plan), buf.wts
    [R, slots] (routed weights, then the shared gate's sigmoid). ``xs`` (the 32-group sums) is unused:
    NVFP4 has no biases, the format is purely multiplicative."""

    from tensorfold.cuda import moe as moe_mod

    rows = x.shape[0]
    top_k = int(cfg.num_experts_per_tok)
    moe_mod.router(x, router_rows, buf.logits[:rows])
    moe_mod.select(buf.logits[:rows], buf, top_k, ex.routed)
    plan = buf.plan
    ex.gateup_out(x, plan, buf.act[:rows], top_k)
    ex.down_out(buf.act[:rows], plan, buf.y[:rows], top_k)
    sg = buf.logits[:rows, ex.routed].to(torch.bfloat16).to(torch.float32)
    buf.wts[:rows, top_k] = (1.0 / (1.0 + torch.exp(-sg))).to(torch.bfloat16).to(torch.float32)
    return buf
