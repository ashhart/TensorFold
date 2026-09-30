"""DFlash2 drafts for full GLM-5.3: TensorFold's GLM-5.3-Flash drafter (glm5_next/cuda/dflash2.py) over this family.

The drafter reads the target's hidden rows after the tap layers (its ``target_layer_ids``, the output of each layer),
keeps its own sliding-window context from the committed rows' taps, and drafts a block of mask rows after the pending
token. Only the output head differs from the Flash family: here it is the bf16 lm_head share of each rank (the router
kernel), and each rank's top-k candidates merge over the ranks exactly as there. Drafts never have to be exact - the
target verifies every one - so the drafter's 4-bit weights and ring-free projections are fine.

Checkpoint: e.g. incoai/GLM-5.3-DFlash2 (6 layers, block 8, taps 5/19/33/47/61/75, trained against BF16 GLM-5.3).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tensorfold.families.glm5_next.cuda import glue
from tensorfold.families.glm5_next.cuda.dflash2 import Drafter

from . import fused


class GlmDrafter(Drafter):
    def __init__(self, draft_dir: str | Path, w: fused.Weights, capacity: int) -> None:
        shim = SimpleNamespace(device=w.device, rank=w.rank, world=w.world, comm=w.comm, embed=w.embed, head=None,
                               draft_head=None, vocab_offset=w.vocab_off)
        super().__init__(draft_dir, shim, capacity=capacity)
        self.fw = w
        V = w.lm_head.shape[0]
        self.logits = torch.empty((self.block - 1, V), dtype=torch.float32, device=w.device)

    def _block_compute(self) -> None:
        """[pending, mask x (block - 1)] at the committed length; each rank's top-k over its lm_head share, merged."""
        n = self.block
        x = torch.empty((n, self.D), dtype=torch.bfloat16, device=self.dev)
        glue.embed(self.ids, self.fw.embed, self.D, 1, x)
        cos, sin = self._rotary(n)
        idx = self.pos_dev + self.ar[:n]
        for i in range(len(self.layers)):
            x = self._layer(i, x, cos, sin, idx)
        h, _ = self._norm(x[1:], self.norm)
        glue.router(h, self.fw.lm_head, self.logits)
        vals, local = torch.topk(self.logits, self.top_k, dim=-1)
        gids = (local + self.fw.vocab_off).to(torch.int32)
        packed = torch.cat([vals, gids.view(torch.float32)], dim=1).contiguous()
        if self.world > 1:
            got = torch.empty((self.world * packed.numel(),), dtype=torch.float32, device=self.dev)
            self.fw.comm.all_gather(packed.view(-1), got)
            self.packed = got.view(self.world, n - 1, 2 * self.top_k)
        else:
            self.packed = packed.view(1, n - 1, 2 * self.top_k)
        self.proj = F.linear(h, self.hproj).float()
