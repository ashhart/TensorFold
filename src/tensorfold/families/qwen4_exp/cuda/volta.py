"""Flash Next's NVFP4 checkpoint on sm_70: its linears and experts on ``qmmf_volta``'s exact fp16 expansions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorfold.cuda.kernels.qmmf_volta import VoltaExperts, VoltaLinear


def linear(w: torch.Tensor, cols: slice | None = None) -> VoltaLinear:
    """A bf16 [N, K] weight as a 16-bit Volta linear; ``cols``: a row-parallel shard of whole 64-input groups."""

    full = VoltaLinear.from_bf16(w.to(torch.bfloat16).contiguous())
    if cols is None:
        return full
    if cols.start % 64 or cols.stop % 64:
        raise ValueError(f"input columns [{cols.start}, {cols.stop}) are not whole 64-input groups")
    return full.groups(cols.start // 64, cols.stop // 64).copy()


@dataclass
class VoltaMoE:
    """One layer's routed NVFP4 experts on the Volta kernel, the bf16 shared expert and the router as 16-bit linears."""

    routed_experts: VoltaExperts
    router: VoltaLinear                  # [experts + 1, D]: the routed experts' rows, then the shared expert's gate
    shared_gu: VoltaLinear               # [2 NI, D]: gate rows, then up rows
    shared_down: VoltaLinear             # [D, NI]
    kernel: str = "volta"

    capturable = True                    # the plan stays on the device

    @property
    def routed(self) -> int:
        return self.routed_experts.count

    @property
    def count(self) -> int:
        return self.routed + 1

    @property
    def width(self) -> int:
        return self.routed_experts.width

    @property
    def dims(self) -> int:
        return self.routed_experts.dims

    def nbytes(self) -> int:
        return (self.routed_experts.nbytes() + self.router.nbytes() + self.shared_gu.nbytes()
                + self.shared_down.nbytes())


def experts(gate: tuple, up: tuple, down: tuple) -> VoltaExperts:
    """Stacked checkpoint arrays (words, e4m3 scales, per-expert scales) as the Volta kernel's tiles."""

    return VoltaExperts.make(gate, up, down)


def experts_from_bf16(gate_up: torch.Tensor, down: torch.Tensor) -> VoltaExperts:
    """The MTP layer's bf16 experts (gate_up [E, 2 NI, D], down [E, D, NI]) quantized to NVFP4: they only draft."""

    from tensorfold.cuda.nvfp4 import experts as nvx

    ni = gate_up.shape[1] // 2
    return VoltaExperts.make(nvx.quantize(gate_up[:, :ni], chunk=2), nvx.quantize(gate_up[:, ni:], chunk=2),
                             nvx.quantize(down, chunk=2))


def nvfp4_rows(w: torch.Tensor, chunk: int = 4096) -> VoltaLinear:
    """A bf16 [N, K] draft-only weight as NVFP4 by ModelOpt's recipe, a chunk of rows at a time (the whole's bits)."""

    from tensorfold.cuda.nvfp4.experts import E2M1

    n, k = w.shape
    g = (w.abs().amax().float() / (6.0 * 448.0)).clamp_min(1e-30)
    mags = torch.tensor(E2M1, device=w.device)
    words = torch.empty((n, k // 2), dtype=torch.uint8, device=w.device)
    scales = torch.empty((n, k // 16), dtype=torch.uint8, device=w.device)
    for a in range(0, n, chunk):
        blocks = w[a:a + chunk].float().view(-1, k // 16, 16)
        s = (blocks.abs().amax(-1) / 6.0 / g).clamp(max=448.0).to(torch.float8_e4m3fn)
        step = (s.float() * g)[..., None]
        v = torch.where(step > 0, blocks / step.clamp_min(1e-30), torch.zeros_like(blocks))
        code = ((v.abs()[..., None] - mags).abs().argmin(-1) + 8 * (v < 0).to(torch.int64)).view(-1, k).to(torch.uint8)
        words[a:a + chunk] = code[:, 0::2] | (code[:, 1::2] << 4)
        scales[a:a + chunk] = s.view(torch.uint8)
    return VoltaLinear.from_nvfp4(words, scales.view(torch.float8_e4m3fn), float(g))


def make(routed: VoltaExperts, router: torch.Tensor, shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
         down_cols: slice | None) -> VoltaMoE:
    """A layer: ``shared`` (gate, up, down) bf16 with gate/up already this rank's rows and down its whole rows."""

    gate, up, down = shared
    return VoltaMoE(routed, linear(router), linear(torch.cat([gate, up])), linear(down, down_cols))


def moe(x: torch.Tensor, ex: VoltaMoE, buf, top_k: int, prompt: bool) -> None:
    """Route rows x [R, D] and run their experts into ``buf``, the shared expert last; ``prompt``: the prompt GEMM."""

    from tensorfold.cuda import moe as moe_mod

    rows = x.shape[0]
    f32 = buf.y.dtype == torch.float32
    buf.logits[:rows].copy_(ex.router.prefill(x, f32=True) if prompt else ex.router.matmul(x, f32=True))
    moe_mod.select_rows(buf.logits[:rows], buf, top_k, ex.routed)
    act = buf.act.view(-1, ex.width)
    ex.routed_experts.run(x, buf.pick[:rows], act, buf.y.view(-1, ex.dims)[:rows * buf.slots])
    g = ex.shared_gu.prefill(x) if prompt else ex.shared_gu.matmul(x)
    gate, upv = g[:, :ex.width].float(), g[:, ex.width:].float()
    buf.act[:rows, top_k] = ((gate / (1.0 + torch.exp(-gate))).to(torch.bfloat16).float() * upv).to(torch.bfloat16)
    shared = buf.act[:rows, top_k]
    buf.y[:rows, top_k] = (ex.shared_down.prefill(shared, f32=f32) if prompt
                           else ex.shared_down.matmul(shared, f32=f32))
