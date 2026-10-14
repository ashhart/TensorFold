"""glm.head fixtures: forward.compute's tail on both ranks (stream mean, final norm, head half), then sample_rows."""
from __future__ import annotations

import threading
from pathlib import Path

import torch

from glm_oracle_layer import CAP, DECODE_ROWS, PREFILL_ROWS, Comm, Hub, _load, save

CASES = (("d1", 1, False), ("d8", 8, False), ("p300", 300, True))   # a prompt chunk's head reads its last row only
SEED = 7171


def _tail(w, x: torch.Tensor, R: int, prefill: bool) -> dict:
    """compute()'s lines after the layers, on the engine's own buffers."""
    from tensorfold.families.glm5_next.cuda import forward, glue

    c = w.cfg
    b = forward.Buffers(w, PREFILL_ROWS if prefill else DECODE_ROWS, CAP, prefill=prefill)
    b.x[:R].copy_(x)
    glue.stream_mean(b.x[:R], b.hidden[:R])
    glue.rmsnorm(b.hidden[:R], w.norm, c.eps, b.fnormed[:R], b.fxs[:R])
    if prefill:
        lg = forward.mm(b, b.fnormed[R - 1:R], w.head, b.fxs[R - 1:R], b.logits[:1])
    else:
        lg = forward.mm(b, b.fnormed[:R], w.head, b.fxs[:R], b.logits[:R])
    torch.cuda.synchronize()
    return {"hidden": b.hidden[:R].clone(), "fnormed": b.fnormed[:R].clone(), "fxs": b.fxs[:R].clone(),
            "logits": lg.clone()}


def _sample(W, logits: list[torch.Tensor]) -> list[int]:
    """decode.sample_rows greedy on both ranks at once (its all_gather of the packed candidates)."""
    from tensorfold.families.glm5_next.cuda import decode

    hub, res, errs = Hub(), [None, None], []

    def body(rank: int) -> None:
        try:
            W[rank].comm = Comm(hub, rank)
            res[rank] = decode.sample_rows(W[rank], logits[rank], list(range(logits[rank].shape[0])), None)
        except BaseException as e:   # the peer must not wait at the barrier forever
            errs.append(e)
            hub.bar.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for w in W:
        w.comm = None
    if errs:
        raise errs[0]
    assert res[0] == res[1], "the ranks chose differently"
    return res[0]


def write(out: Path, model_dir: Path) -> dict:
    W = [_load(model_dir, rank, 0) for rank in (0, 1)]
    c = W[0].cfg
    assert torch.equal(W[0].norm, W[1].norm) and W[0].norm.dtype == torch.bfloat16
    heads = []
    for rank, w in enumerate(W):
        assert w.head.weight.dtype == torch.bfloat16 and w.head.n == c.vocab // 2 and w.vocab_offset == rank * w.head.n
        heads.append({"w": save(out, f"hd_head{rank}", w.head.weight), "n": w.head.n, "k": w.head.k,
                      "offset": w.vocab_offset})
    man = {"params": {"eps": c.eps, "hidden": c.hidden, "streams": c.streams, "vocab": c.vocab},
           "norm": save(out, "hd_norm", W[0].norm), "heads": heads, "cases": {}}
    g = torch.Generator().manual_seed(SEED)
    for tag, R, prefill in CASES:
        x = torch.randn((R, c.streams * c.hidden), generator=g).to(torch.bfloat16).cuda()
        t = [_tail(w, x, R, prefill) for w in W]
        for key in ("hidden", "fnormed", "fxs"):
            assert torch.equal(t[0][key], t[1][key]), f"{tag}: the ranks' {key} differ"
        tok = _sample(W, [r["logits"] for r in t])
        e = {"rows": R, "prefill": prefill, "x": save(out, f"hd_{tag}_x", x),
             "tok": save(out, f"hd_{tag}_tok", torch.tensor(tok, dtype=torch.int32)),
             "logits": [save(out, f"hd_{tag}_logits{rank}", r["logits"]) for rank, r in enumerate(t)]}
        for key in ("hidden", "fnormed", "fxs"):
            e[key] = save(out, f"hd_{tag}_{key}", t[0][key])
        wins = sum(int(v >= W[0].head.n) for v in tok)
        print(f"  head {tag}: {R} rows, {len(tok)} tokens ({wins} from rank 1)")
        man["cases"][tag] = e
    return man
