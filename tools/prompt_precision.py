"""KL and top-1 of a decode forward against an fp32 dequantized matmul, plus perplexity.

Eight sequences of 4096 tokens from wikitext-2, CPython, and chats. The reference
replaces each quantized linear that the forward calls with an fp32 matmul of the
dequantized weight. KL is KL(reference || path) over the vocabulary at every row.
"""

from __future__ import annotations

import argparse
import sys
import sysconfig
import tempfile
from pathlib import Path

import numpy as np

LENGTH = 4096
WIKI_COUNT, CODE_COUNT, CHAT_COUNT = 4, 2, 2
CHATS = (
    "User: How does a hash map grow?\nAssistant: It allocates a larger bucket array and reinserts every key.\n",
    "User: What does a lock guard on return?\nAssistant: Its destructor releases the lock as the stack unwinds.\n",
    "User: Why copy a buffer before a retry?\n"
    "Assistant: The first call can consume the bytes. The retry needs its own copy.\n",
)


def encode(tokenizer, text: str) -> list[int]:
    out = tokenizer.encode(text)
    return list(out.ids if hasattr(out, "ids") else out)


def windows(ids: list[int], length: int, count: int) -> list[list[int]]:
    got = [ids[s:s + length] for s in range(0, len(ids) - length + 1, length)]
    if len(got) < count:
        raise SystemExit(f"need {count} windows of {length}, found {len(got)} in {len(ids)} tokens")
    return got[:count]


def repeat_to(text: str, chars: int) -> str:
    block = text if text.endswith("\n") else text + "\n"
    copies = max(1, (chars + len(block) - 1) // len(block))
    return block * copies


def code_text(limit: int = 2_000_000) -> str:
    root = Path(sysconfig.get_paths()["stdlib"])
    parts: list[str] = []
    used = 0
    for path in sorted(root.rglob("*.py")):
        if any(part in path.parts for part in ("test", "tests", "idlelib", "site-packages", "__pycache__")):
            continue
        piece = path.read_text(errors="ignore")
        parts.append(piece)
        used += len(piece)
        if used >= limit:
            break
    if not parts:
        raise SystemExit(f"no Python sources under {root}")
    return "\n".join(parts)


def sequences(tokenizer, wikitext: Path, length: int, wiki: int, code: int, chat: int) -> list[tuple[str, list[int]]]:
    if not wikitext.is_file():
        raise SystemExit(f"missing wikitext file {wikitext}")
    rows: list[tuple[str, list[int]]] = []
    if wiki:
        rows += [("wikitext", w) for w in windows(encode(tokenizer, wikitext.read_text()), length, wiki)]
    if code:
        rows += [("code", w) for w in windows(encode(tokenizer, code_text()), length, code)]
    if chat:
        rows += [("chat", w) for w in windows(encode(tokenizer, repeat_to("".join(CHATS), length * 8)), length, chat)]
    return rows


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


class Score:
    def __init__(self) -> None:
        self.kl = 0.0
        self.rows = 0
        self.top = 0
        self.nll: dict[str, list[float]] = {}
        self.nll_ref: dict[str, list[float]] = {}

    def add(self, ref: np.ndarray, path: np.ndarray, targets: np.ndarray | None, source: str) -> None:
        if ref.shape != path.shape:
            raise SystemExit(f"logit shape {path.shape} against reference {ref.shape}")
        r = log_softmax(ref.astype(np.float64))
        p = log_softmax(path.astype(np.float64))
        prob = np.exp(r)
        self.kl += float((prob * (r - p)).sum())
        self.rows += int(ref.shape[0])
        self.top += int((np.argmax(r, axis=-1) == np.argmax(p, axis=-1)).sum())
        if targets is None or len(targets) == 0:
            return
        n = len(targets)
        pick = np.arange(n)
        self.nll.setdefault(source, [0.0, 0.0])
        self.nll_ref.setdefault(source, [0.0, 0.0])
        self.nll[source][0] += float((-p[pick, targets]).sum())
        self.nll[source][1] += n
        self.nll_ref[source][0] += float((-r[pick, targets]).sum())
        self.nll_ref[source][1] += n

    def line(self) -> str:
        kl = self.kl / self.rows
        top = 100.0 * self.top / self.rows

        def ppl(bucket: dict[str, list[float]], source: str) -> str:
            total = bucket.get(source)
            if not total or total[1] == 0:
                return "n/a"
            return f"{float(np.exp(total[0] / total[1])):.4f}"

        def delta(source: str) -> str:
            got, ref = self.nll.get(source), self.nll_ref.get(source)
            if not got or not ref or got[1] == 0 or ref[1] == 0:
                return "n/a"
            base = float(np.exp(ref[0] / ref[1]))
            if base == 0:
                return "n/a"
            return f"{(float(np.exp(got[0] / got[1])) - base) / base * 100:+.3f}%"

        return (f"KL {kl:.5f}  top-1 {top:.2f}%  rows {self.rows}  "
                f"ppl wikitext {ppl(self.nll, 'wikitext')} (ref {ppl(self.nll_ref, 'wikitext')}, {delta('wikitext')})  "
                f"ppl code {ppl(self.nll, 'code')} (ref {ppl(self.nll_ref, 'code')}, {delta('code')})")


def counts(n: int) -> tuple[int, int, int]:
    if n == 8:
        return WIKI_COUNT, CODE_COUNT, CHAT_COUNT
    if n < 1:
        raise SystemExit("--sequences starts at 1")
    return n, 0, 0


def self_test() -> int:
    wiki, code, chat = counts(8)
    if (wiki, code, chat) != (4, 2, 2) or counts(1) != (1, 0, 0):
        raise SystemExit("sequence split")
    ids = list(range(LENGTH * 3))
    got = windows(ids, LENGTH, 2)
    if len(got) != 2 or got[0][0] != 0 or got[1][0] != LENGTH:
        raise SystemExit("windows")
    vocab = 5
    rows = 4
    base = np.zeros((rows, vocab), dtype=np.float32)
    base[:, 1] = 4
    score = Score()
    score.add(base, base.copy(), np.array([1, 1, 1, 1]), "wikitext")
    if score.kl != 0 or score.top != rows:
        raise SystemExit("identical distributions")
    shifted = base.copy()
    shifted[:, 1] = 0
    shifted[:, 2] = 4
    score.add(base, shifted, np.array([1, 1, 1, 1]), "code")
    if score.rows != 8 or score.top != 4 or "top-1 50.00%" not in score.line():
        raise SystemExit("shifted top-1")
    text = repeat_to("abc ", 20)
    if len(text) < 20 or not text.startswith("abc"):
        raise SystemExit("repeat")
    print("self-test ok")
    return 0


def as_rows(logits) -> np.ndarray:
    import mlx.core as mx

    arr = np.array(logits.astype(mx.float32))
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise SystemExit(f"expected one stream, got {arr.shape}")
        arr = arr[0]
    if arr.ndim != 2:
        raise SystemExit(f"expected rows by vocabulary, got {arr.shape}")
    return arr


def clear_cache() -> None:
    import mlx.core as mx

    fn = getattr(mx, "clear_cache", None)
    if fn is not None:
        fn()


def chunk_logits(step, ids: np.ndarray, cache, width: int) -> np.ndarray:
    import mlx.core as mx

    pieces = []
    for start in range(0, len(ids), width):
        logits = step(ids[start:start + width], cache)
        mx.eval(logits)
        pieces.append(as_rows(logits))
        clear_cache()
    return np.concatenate(pieces, axis=0)


def fp32_weight(weight, scales, biases, group: int, bits: int):
    import mlx.core as mx

    return mx.dequantize(weight, scales.astype(mx.float32), biases.astype(mx.float32),
                         group_size=group, bits=bits).astype(mx.float32)


def install_fp32_flash() -> None:
    import mlx.core as mx
    from mlx.nn.layers.quantized import QuantizedLinear

    def call(self, x):
        weight = fp32_weight(self.weight, self.scales, self.biases, int(self.group_size), int(self.bits))
        y = mx.matmul(x.astype(mx.float32), weight.T)
        if "bias" in self:
            y = y + self["bias"].astype(mx.float32)
        return y

    QuantizedLinear.__call__ = call


def install_fp32_glm() -> None:
    import mlx.core as mx

    from tensorfold.families.glm5_next import kda, linear, mla, mlp, model, mtp
    from tensorfold.families.glm5_next.linear import Dense, Q, QSplit

    def project(x, q, *, rows_exact: bool = False):
        del rows_exact
        if isinstance(q, QSplit):
            return mx.concatenate([project(x, part) for part in q.parts], axis=-1)
        if isinstance(q, Dense):
            return mx.matmul(x.astype(mx.float32), q.weight.astype(mx.float32).T)
        weight = fp32_weight(q.weight, q.scales, q.biases, int(q.group), int(q.bits))
        return mx.matmul(x.astype(mx.float32), weight.T)

    for module in (linear, mla, mlp, kda, model, mtp):
        module.project = project


def load_family(family: str, model_dir: Path, ple_on_ssd: bool, ssd_experts: float | None):
    if family == "flash":
        from tensorfold.families.qwen4_exp.runtime import load

        return load(model_dir, drafts=0, ple_on_ssd=ple_on_ssd, ssd_experts=ssd_experts)
    from tensorfold.families.glm5_next.runtime import load

    return load(model_dir, drafts=0, ssd_experts=ssd_experts)


def bind(family: str, model):
    """Path step, reference step, and the call that installs the fp32 linears."""

    if family == "flash":
        inner = model.model

        def path(chunk, cache):
            return model.head(model.hidden(chunk, cache))

        def reference(chunk, cache):
            return inner(chunk.reshape(1, -1), cache)

        def arm() -> None:
            inner.__dict__.pop("fused", None)
            install_fp32_flash()

        return path, reference, arm

    def path(chunk, cache):
        return model.head(model.hidden(chunk, cache))

    def reference(chunk, cache):
        hidden = model.model.hidden(chunk, cache)
        q = model.model.lm_head
        import mlx.core as mx

        flat = hidden.reshape(-1, hidden.shape[-1])
        weight = fp32_weight(q.weight, q.scales, q.biases, int(q.group), int(q.bits))
        return mx.matmul(flat.astype(mx.float32), weight.T).reshape(*hidden.shape[:-1], weight.shape[0])

    def arm() -> None:
        model.model.hc_fused_ok = lambda: False
        install_fp32_glm()

    return path, reference, arm


def store_path(rows: np.ndarray, directory: Path, index: int) -> np.ndarray:
    path = directory / f"path-{index}.npy"
    np.save(path, rows)
    mapped = np.load(path, mmap_mode="r")
    return mapped


def score_pair(path_rows: np.ndarray, ref_rows: np.ndarray, ids: np.ndarray, source: str, score: Score,
               block: int = 32) -> None:
    if len(path_rows) != len(ids) or len(ref_rows) != len(ids):
        raise SystemExit(f"{source}: {len(path_rows)} path rows, {len(ref_rows)} reference rows, {len(ids)} tokens")
    for start in range(0, len(ids), block):
        stop = min(len(ids), start + block)
        pred = min(stop - start, len(ids) - start - 1)
        targets = ids[start + 1:start + 1 + pred] if pred else None
        score.add(np.array(ref_rows[start:stop]), np.array(path_rows[start:stop]), targets, source)


def run(args: argparse.Namespace) -> int:
    import os

    if args.dense:
        os.environ["TF_FLASH_DENSE" if args.family == "flash" else "TF_GLM_DENSE"] = args.dense
    model, tokenizer = load_family(args.family, args.model, args.ple_on_ssd, args.ssd_experts)
    wiki, code, chat = counts(args.sequences)
    rows = sequences(tokenizer, args.wikitext, args.length, wiki, code, chat)
    width = args.width or int(model.fused_rows)
    path_step, ref_step, arm = bind(args.family, model)
    score = Score()
    with tempfile.TemporaryDirectory(prefix="tf-precision-") as tmp:
        folder = Path(tmp)
        stored = []
        for index, (source, ids) in enumerate(rows):
            tokens = np.asarray(ids, dtype=np.int64)
            got = chunk_logits(path_step, tokens, model.make_cache(), width)
            stored.append((source, tokens, store_path(got, folder, index)))
            del got
            clear_cache()
            print(f"path {index} {source} rows {len(tokens)}", flush=True)
        arm()
        for index, (source, tokens, path_rows) in enumerate(stored):
            ref = chunk_logits(ref_step, tokens, model.make_cache(), width)
            score_pair(path_rows, ref, tokens, source, score)
            del ref
            clear_cache()
            print(f"seq {index} {source}  {score.line()}", flush=True)
    print(score.line(), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a decode forward against an fp32 dequantized matmul.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--family", choices=("flash", "glm"), default="flash")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--wikitext", type=Path, default=Path.home() / "tf-data" / "wikitext-2-raw" / "wiki.test.raw")
    parser.add_argument("--length", type=int, default=LENGTH)
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--ple-on-ssd", action="store_true")
    parser.add_argument("--ssd-experts", type=float, default=None)
    parser.add_argument("--dense", choices=("rows", "matrix"))
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.model is None:
        parser.error("--model is required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
