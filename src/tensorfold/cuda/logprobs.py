"""Raw target log probabilities with the same FP32 reduction tree at every batch size."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _parts(X, P, STRIDE: tl.constexpr, V: tl.constexpr, T: tl.constexpr, B: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    cols = part * B + tl.arange(0, B)
    x = tl.load(X + row * STRIDE + cols, cols < V, other=-float("inf")).to(tl.float32)
    peak = tl.max(x, 0)
    mass = tl.sum(tl.exp(x - tl.where(peak == -float("inf"), 0.0, peak)), 0)
    tl.store(P + (row * T + part) * 2, peak)
    tl.store(P + (row * T + part) * 2 + 1, mass)


@tr.jit
def _finish(P, L, T: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    at = tl.arange(0, B)
    peak = tl.load(P + (row * T + at) * 2, at < T, other=-float("inf"))
    mass = tl.load(P + (row * T + at) * 2 + 1, at < T, other=0.0)
    maximum = tl.max(peak, 0)
    total = tl.sum(mass * tl.exp(peak - maximum), 0)
    tl.store(L + row, maximum + tl.log(total))


@torch.no_grad()
def capture(logits, tokens, positions, probabilities, rows=None):
    """Only accepted target rows reach the collector; source logits are read-only."""

    if probabilities is None or not tokens:
        return
    # Bound all vocabulary-sized sorting temporaries, including selected source rows.
    width = max(1, (128 * 1024**2) // (48 * logits.shape[1]))
    if len(tokens) > width:
        for start in range(0, len(tokens), width):
            end = start + width
            capture(logits[start:end] if rows is None else logits, tokens[start:end], positions[start:end],
                    probabilities, None if rows is None else rows[start:end])
        return
    if rows is not None:
        logits = logits.index_select(0, torch.tensor(rows, dtype=torch.long, device=logits.device))
    if not logits.is_cuda or logits.ndim != 2 or logits.shape[0] != len(tokens) or logits.stride(1) != 1:
        raise ValueError("probabilities need CUDA target rows and one accepted token per row")
    n, vocab = logits.shape
    tiles = tr.cdiv(vocab, 1024)
    parts = torch.empty((n, tiles, 2), dtype=torch.float32, device=logits.device)
    lse = torch.empty((n,), dtype=torch.float32, device=logits.device)
    _parts[(n, tiles)](logits, parts, logits.stride(0), vocab, tiles, 1024, num_warps=4)
    _finish[(n,)](parts, lse, tiles, tr.next_power_of_2(tiles), num_warps=4)
    ids = torch.tensor(tokens, dtype=torch.long, device=logits.device)[:, None]
    chosen = (logits.gather(1, ids).float()[:, 0] - lse).cpu().tolist()
    labels = getattr(probabilities, "labels", None)
    if labels:                                   # a decision: the label logits at the prompt's last position
        picked = logits.index_select(1, torch.tensor(labels, dtype=torch.long, device=logits.device)).float()
        for row, pos in enumerate(positions):
            probabilities.add_labels(pos, picked[row].cpu().tolist(), float(lse[row].item()))
    count = min(probabilities.top, vocab)
    if count:
        values = logits.float()
        bits = values.view(torch.int32).to(torch.int64)
        bits = torch.where(values == 0, 0, bits)
        ordered = torch.where(bits < 0, ~bits, bits ^ 0x80000000) - 0x80000000
        token_ids = torch.arange(vocab, dtype=torch.int64, device=logits.device)
        keys = (ordered << 32) | (0xFFFFFFFF - token_ids)
        top_ids = keys.topk(count, dim=-1, sorted=True).indices
        scores = (logits.gather(1, top_ids).float() - lse[:, None]).cpu().tolist()
        alternatives = top_ids.cpu().tolist()
    else:
        alternatives, scores = [[] for _ in tokens], [[] for _ in tokens]
    probabilities.add(positions, tokens, chosen, alternatives, scores)
