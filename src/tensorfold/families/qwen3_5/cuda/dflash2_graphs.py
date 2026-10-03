"""DFlash2's draft step as CUDA graphs: several streams' blocks in one capture, each context read in place."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .glue import embedding
from .qmm_fast import matmul, matmul_rows


@torch.no_grad()
def graph_blocks(d, snaps: list, pendings: list[int], max_nodes: int, block: int | None = None) -> list:
    """``launch_blocks`` from a graph for this many streams that reads each context in place."""

    out: list = [None] * len(snaps)
    live = [i for i, snap in enumerate(snaps) if snap[2] > 0]
    if not live or max_nodes < 1:
        return out
    length = min(block or d.block, max_nodes + 1)
    S, L = len(live), d.layers
    graphs = d.__dict__.setdefault("_graphs", {})
    g = graphs.get((S, length))
    if g is None:
        host = torch.zeros(3 * L * S + 2 * S, dtype=torch.int64).pin_memory()
        g = {"host": host, "h": host.numpy()}
    h = g["h"]
    for j, i in enumerate(live):
        kcs, vcs, _, end = snaps[i]
        for layer in range(L):
            kc, vc = kcs[layer], vcs[layer]
            if not kc.is_contiguous() or not vc.is_contiguous() or (kc.data_ptr() | vc.data_ptr()) % 16:
                return d.launch_blocks(snaps, pendings, max_nodes, block)
            h[2 * S * layer + 2 * j], h[2 * S * layer + 2 * j + 1] = kc.data_ptr(), vc.data_ptr()
            h[2 * L * S + S * layer + j] = kc.shape[1]
        h[3 * L * S + j], h[3 * L * S + S + j] = end, pendings[i]
    if "graph" not in g:
        g["dev"] = g["host"].to(d.device)
        graphs[(S, length)] = g = _capture_blocks(d, g, S, length)
    else:
        g["dev"].copy_(g["host"], non_blocking=True)
    if g["calls"]:
        g["peer"].before_replay(g["calls"])
    g["graph"].replay()
    values, global_ids, projected = g["out"]
    if d.world == 2:
        values, global_ids = _merge_ranks(d, values, global_ids)
    shared = [global_ids, torch.cat((values, projected), dim=1), None]     # read back once, by the first finish
    for j, i in enumerate(live):
        out[i] = (shared, j * (length - 1), length - 1, int(pendings[i]))
    return out


def _merge_ranks(d, values: torch.Tensor, global_ids: torch.Tensor):
    import torch.distributed as dist

    both_values = torch.empty((2, *values.shape), dtype=values.dtype, device=values.device)
    both_ids = torch.empty((2, *global_ids.shape), dtype=global_ids.dtype, device=values.device)
    dist.all_gather_into_tensor(both_values, values.contiguous())
    dist.all_gather_into_tensor(both_ids, global_ids.contiguous())
    merged = torch.cat((both_values[0], both_values[1]), dim=1)
    values, pick = torch.topk(merged, k=16, dim=-1, sorted=False)
    return values, torch.cat((both_ids[0], both_ids[1]), dim=1).gather(1, pick)


def _capture_blocks(d, g: dict, S: int, length: int) -> dict:
    import gc

    L, dev = d.layers, g["dev"]
    lens = [torch.zeros(S, dtype=torch.int32, device=d.device) for _ in range(L)]
    static = [(dev[2 * S * i:2 * S * (i + 1)], lens[i]) for i in range(L)]
    pos, pend = dev[3 * L * S:3 * L * S + S], dev[3 * L * S + S:3 * L * S + 2 * S]
    masks = torch.full((S, length - 1), d.mask_id, dtype=torch.int32, device=d.device)
    steps = torch.arange(length, device=d.device, dtype=torch.float32)

    def step():
        for i in range(L):                  # the lengths as int32, refreshed with the pointers
            lens[i].copy_(dev[2 * L * S + S * i:2 * L * S + S * (i + 1)])
        tokens = torch.cat([pend[:, None].to(torch.int32), masks], dim=1).reshape(-1)
        phase = (pos.float()[:, None] + steps[None, :]).reshape(-1)[:, None] * d.inv_freq[None, :]
        cos, sin = phase.cos().contiguous(), phase.sin().contiguous()
        x = embedding(tokens, d.target_embed)
        for layer in range(L):
            x = d._layer_fast(layer, x, cos, sin, None, length, static=static[layer])
        h = F.rms_norm(x.view(S, length, -1)[:, 1:].reshape(-1, d.hidden), (d.hidden,),
                       d.weights["norm.weight"], d.eps)
        projected = d._lin(h, "candidate_selector.hidden_projection.weight").float()
        logits = _sub_logits(d, h)
        values, local_ids = torch.topk(logits.float(), k=16, dim=-1, sorted=False)
        return values, d.head_ids[local_ids], projected

    peer = None
    if d.world == 2:
        from tensorfold.cuda import p2p

        peer = p2p.peer()
    step()                                   # kernels compiled and two ranks' sums met outside the capture
    torch.cuda.synchronize()
    if "_pool" not in d.__dict__:
        d._pool = torch.cuda.graph_pool_handle()
    graph = torch.cuda.CUDAGraph()
    if peer is not None:
        peer.begin_capture()
    enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.graph(graph, pool=d._pool):
            out = step()
    finally:
        calls = peer.end_capture() if peer is not None else 0
        if enabled:
            gc.enable()
    g.update(graph=graph, out=out, calls=calls, peer=peer, static=static, lens=lens, masks=masks, steps=steps)
    return g


def _sub_logits(d, h: torch.Tensor) -> torch.Tensor:
    if d.sub_parts is not None:
        h = h.contiguous()
        return torch.cat([part(h)[:, lo:hi] for part, lo, hi in d.sub_parts], dim=1)
    if d.head_cols is not None:
        return d.sub_head(h.contiguous()).index_select(1, d.head_cols)
    if d.sub_rows is not None:
        return matmul_rows(h, d.sub_rows)
    return matmul(h, d.sub_head)
