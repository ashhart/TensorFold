"""Two ranks of the reduced DeepSeek-V4.1 checkpoint (layers 0-7, DSpark tapping their last three) serving fixed
prompts serial and drafted, greedy and keyed: each rank prints one JSON line a request with the hashes of its logits
shards (prompt head, every verify forward, every DSpark block), the reply, the drafts kept and the committed cache.
Run over NCCL on two machines (rank 1 first), then with ``--pair`` as two threads on one GPU: each rank's lines must
be identical.
"""
# TF_DSV41_MODEL=PACK PYTHONPATH=src:tests/cuda python tests/cuda/dsv41_tp_check.py --rank 1 --master HOST
# TF_DSV41_MODEL=PACK PYTHONPATH=src:tests/cuda python tests/cuda/dsv41_tp_check.py --pair
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import DEFAULT_DRAFTS, sample
from tensorfold.families.deepseek_v41.cuda.engine import DeepSeekV41Engine

LAYERS = 8
CONTEXT = 4096
PROMPTS = (300, 2100)               # one prompt chunk; past the first 2048-row chunk
REPLY = 48
SAMPLINGS = {"greedy": None, "keyed": Sampling(seed=5, temperature=1.0, top_k=20, top_p=0.95)}
_HEADS: dict[int, list[str]] = {}   # id(weights) -> its request's prompt-head hash, set by the first sampled rows


def _bits(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def _cache(e: DeepSeekV41Engine) -> str:
    """The committed backbone state: position, token tail, ring slots of the last window, cache entries, tails."""

    st, n = e.e.st, e.w.cfg.num_hidden_layers
    h = hashlib.sha256(json.dumps([st.pos, st.history]).encode())
    win = st.rings.shape[1]
    parts = [st.rings[:n, [p % win for p in range(max(0, st.pos - win), st.pos)]]]
    for layer, r in st.ratio.items():
        parts += [st.comp[layer][:st.pos // r], st.index_k[layer][:st.pos // r]]
    parts += [st.tail_valid, st.tail[st.tail_valid.bool()]]
    for t in parts:
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _hook_sampling() -> None:
    """Hash the logits of a request's first sampled rows, which in ``_run`` are prefill's head row."""

    target = sample.target_rows

    def rows(w, logits, positions, sampling):
        head = _HEADS.get(id(w))
        if head is not None and not head:
            head.append(_bits(logits))
        return target(w, logits, positions, sampling)

    sample.target_rows = rows


def _record(e: DeepSeekV41Engine, lines: list[str]) -> None:
    """Every request this rank runs adds a line: its logits, DSpark block and reply hashes and the cache's."""

    head: list[str] = []
    steps: list[str] = []
    blocks: list[str] = []
    _HEADS[id(e.e.w)] = head
    forward, propose, run = e.e.forward, e.e.propose, e._run

    def fwd(tokens):
        out = forward(tokens)
        steps.append(_bits(out))
        return out

    def prop(*a):
        out = propose(*a)
        blocks.append(_bits(e.e.dbuf.dlog))
        return out

    def ran(prompt, req, hit, on_tokens):
        stats = run(prompt, req, hit, on_tokens)
        lines.append(json.dumps({"rank": e.rank, "prompt": len(prompt), "draft": req.draft,
                                 "sampling": "greedy" if req.sampling is None else "keyed", "cached": req.cached,
                                 "reply": stats["sha256"], "accepted": stats["accepted"], "head": head[0],
                                 "steps": list(steps), "blocks": list(blocks), "cache": _cache(e)}))
        head.clear()
        steps.clear()
        blocks.clear()
        return stats

    e.e.forward, e.e.propose, e._run = fwd, prop, ran


def _prompts(model: Path) -> list[list[int]]:
    """BOS and the first tokens of DeepSeek's MIT ``inference/model.py`` shipped in the checkpoint."""

    from tokenizers import Tokenizer

    text = (model / "inference" / "model.py").read_text()
    ids = Tokenizer.from_file(str(model / "tokenizer.json")).encode(text, add_special_tokens=False).ids
    bos = Config.read(model, LAYERS).bos_token_id
    return [[bos] + ids[:n - 1] for n in PROMPTS]


def _lead(e: DeepSeekV41Engine, prompts: list[list[int]]) -> None:
    """Rank 0: each prompt and sampling serial, then drafted (which keeps and, on the second sampling, resumes)."""

    try:
        for prompt in prompts:
            for name, sampling in SAMPLINGS.items():
                serial: list[int] = []
                drafted: list[int] = []
                e.generate(prompt, REPLY, sampling, serial.extend, draft=False)
                e.generate(prompt, REPLY, sampling, drafted.extend)
                if drafted != serial:
                    raise AssertionError(f"{len(prompt)}-token prompt, {name}: the drafted reply is not the serial one")
    finally:
        e.shutdown()


def _engine(model: Path, rank: int, **kw) -> DeepSeekV41Engine:
    return DeepSeekV41Engine(model, rank=rank, policy=(DEFAULT_DRAFTS, None), context=CONTEXT,
                             context_explicit=True, layers=LAYERS, **kw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1))
    parser.add_argument("--master", default="")
    parser.add_argument("--port", type=int, default=29631)
    parser.add_argument("--pair", action="store_true", help="both ranks as two threads on this GPU")
    parser.add_argument("--model", type=Path, default=os.environ.get("TF_DSV41_MODEL"))
    args = parser.parse_args()
    if args.model is None or not (args.model / "config.json").is_file():
        parser.error("--model (or TF_DSV41_MODEL) must name the checkpoint")
    if args.pair == (args.rank is not None) or (args.rank is not None and not args.master):
        parser.error("either --pair, or --rank with --master")
    prompts = _prompts(args.model)
    _hook_sampling()
    lines: list[list[str]] = [[], []]
    if args.pair:
        from dsv41_pair import pair, run_pair

        comms = pair()
        engines = run_pair(*(lambda c, r=r: _engine(args.model, r, master="", port=0, comm=c, graphs=False)
                             for r in (0, 1)), comms)
        for e in engines:
            _record(e, lines[e.rank])
        run_pair(lambda _: _lead(engines[0], prompts), lambda _: engines[1].follow(), comms)
    else:
        e = _engine(args.model, args.rank, master=args.master, port=args.port)
        _record(e, lines[args.rank])
        if args.rank == 0:
            _lead(e, prompts)
        else:
            e.follow()
    for rank in (0, 1) if args.pair else (args.rank,):
        for line in lines[rank]:
            print(line)
        print(json.dumps({"rank": rank, "requests": len(lines[rank]),
                          "digest": hashlib.sha256("\n".join(lines[rank]).encode()).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
