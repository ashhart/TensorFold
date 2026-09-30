"""On-policy DFlash2 training data from the serving target (rank 0 writes; ``TF_GLM53_CAPTURE_DIR``): per request one
``seq-*.safetensors`` in ``dflash2_train.py``'s layout - ids int32 [T], pos int32 [T], the target's tap rows as
aux_fp8 e4m3 [T, taps * D] + aux_scale f32 [T, taps] (per row and tap, absmax / 448), and soft labels for the rows the
target sampled from: topk_ids int32 [T, K] (-1: none), topk_logits f16 [T, K], topk_lse f32 [T] (row t predicts
ids[t + 1]). T = the committed positions (every row whose taps exist)."""

from __future__ import annotations

import os

import torch

TOPK = int(os.environ.get("TF_GLM53_CAPTURE_TOPK", "32"))


class Capture:
    def __init__(self, out_dir: str, taps: int, hidden: int) -> None:
        self.dir, self.taps, self.D = out_dir, taps, hidden
        os.makedirs(out_dir, exist_ok=True)
        self.n = len([f for f in os.listdir(out_dir) if f.startswith("seq-")])
        self.reset()

    def reset(self) -> None:
        self.aux_q: list[torch.Tensor] = []
        self.aux_s: list[torch.Tensor] = []
        self.soft: dict[int, tuple] = {}
        self.rows = 0

    def add_taps(self, taps: torch.Tensor) -> None:
        """Committed rows' taps [n, taps * D] (bf16) at the next positions."""
        v = taps.float().view(taps.shape[0], self.taps, self.D)
        sc = (v.abs().amax(-1, keepdim=True) / 448.0).clamp(min=1e-12)
        self.aux_q.append((v / sc).to(torch.float8_e4m3fn).reshape(taps.shape[0], -1).cpu())
        self.aux_s.append(sc.squeeze(-1).cpu())
        self.rows += taps.shape[0]

    def add_logits(self, row: int, logits: torch.Tensor) -> None:
        """The full target logits the sampler used at position ``row`` (predicting row + 1)."""
        lg = logits.float().view(-1)
        vals, ids = torch.topk(lg, TOPK)
        self.soft[row] = (ids.int().cpu(), vals.half().cpu(), torch.logsumexp(lg, 0).cpu())

    def finish(self, ids: list[int], meta: dict) -> str | None:
        from safetensors.torch import save_file

        T = self.rows
        if T < 64 or not self.aux_q:
            self.reset()
            return None
        tid = torch.full((T, TOPK), -1, dtype=torch.int32)
        tlg = torch.zeros((T, TOPK), dtype=torch.float16)
        tls = torch.zeros((T,), dtype=torch.float32)
        for r, (i, v, l) in self.soft.items():
            if r < T - 1:                                # the last committed row's successor is not in ids[:T]
                tid[r], tlg[r], tls[r] = i, v, l
        tens = {"ids": torch.tensor(ids[:T], dtype=torch.int32), "pos": torch.arange(T, dtype=torch.int32),
                "aux_fp8": torch.cat(self.aux_q), "aux_scale": torch.cat(self.aux_s).float(),
                "topk_ids": tid, "topk_logits": tlg, "topk_lse": tls}
        fn = os.path.join(self.dir, f"seq-{self.n:06d}.safetensors")
        save_file(tens, fn, metadata={k: str(v) for k, v in meta.items()} | {"n_rows": str(T)})
        os.chmod(fn, 0o644)
        self.n += 1
        self.reset()
        return fn
