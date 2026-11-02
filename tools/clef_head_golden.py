"""Write zig/src/core/joint_schema/fixtures/clef_tiny.json: a generated tiny Clef head's FP64 and torch BF16 logits."""
# usage: python3 -I tools/clef_head_golden.py CLEF_DIR OUT_JSON (CLEF_DIR holds joint_schema_model.py; CPU only)
import copy
import importlib.util
import json
import math
import os
import sys

SOURCES = {"repo": "Cloudflare/clef-flash", "revision": "fb4417d31c1d", "file": "joint_schema_model.py", "torch": "2.15"}
# Clef-Flash's ratios (hidden = 4 width, feedforward = 4 width, 2 + 4 layers) at width 32 with 4 heads of 8
CONFIG = {"hidden_size": 128, "width": 32, "routing_layers": 2, "layers": 4, "heads": 4, "feedforward": 128}
VOCAB, ROWS, ZERO_TOKEN_ROW = 40, 48, 29
# the release's own scalar values; the ln 100 clamp is checked on its own (CLAMP)
SCALARS = {"prior_logit_scale": 2.296875, "joint_logit_scale": 2.296875, "residual_gate": -2.0}
CLAMP = 5.0
QUESTIONS = [(0, (5, 9), [(12, 15), (15, 19)]), (1, (20, 23), [(24, 26), (26, 29), (29, 30)]),
             (2, (31, 35), [(36, 38), (38, 40), (40, 42), (42, 45)])]
NORMS = [0.0, 1e-13, 1e-9, 1e-5, 1.0]
COSINE_PAIRS = [(0.0, 1.0, False), (1e-13, 1.0, False), (1e-9, 1e-9, False), (1e-5, 1e-5, True), (1.0, 1.0, False)]
MASK = (1 << 64) - 1


def splitmix64(x: int) -> int:
    """SplitMix64's output for state x (zig/src/core/joint_schema/tiny.zig computes the same)."""
    z = (x + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def fnv1a(name: str) -> int:
    """FNV-1a, 32 bits, of the name's UTF-8 bytes."""
    h = 0x811C9DC5
    for b in name.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def grid(name: str, n: int, levels: int, exp: int, offset: float = 0.0) -> list[float]:
    """n values offset + k 2^-exp, k in [-levels, levels]: exact in BF16 for levels under 128."""
    return [offset + (splitmix64((fnv1a(name) << 32) ^ i) % (2 * levels + 1) - levels) * 2.0 ** -exp for i in range(n)]


def recipe(name: str, shape: list[int]) -> tuple[int, int, float]:
    """A head tensor's levels, exponent and offset, from its name and shape alone."""
    module = name.rsplit(".", 1)[0].split(".")[-1]
    if "norm" in module:
        return (31, 7, 1.0) if name.endswith(".weight") else (31, 8, 0.0)
    if name == "type_embedding.weight":
        return 127, 8, 0.0
    if name.endswith("bias"):
        return 127, 10, 0.0
    return 127, (shape[-1].bit_length() + 11) // 2, 0.0


def tensor_values(name: str, shape: list[int]) -> list[float]:
    """The generated values of one head tensor."""
    if name in SCALARS:
        return [SCALARS[name]]
    n = 1
    for d in shape:
        n *= d
    return grid(name, n, *recipe(name, shape))


def inputs():
    """The hidden rows, the output-embedding table (row 0 zero) and the token ids (one option holds only token 0)."""
    h = CONFIG["hidden_size"]
    hidden = grid("hidden", ROWS * h, 127, 4)
    table = [0.0] * h + grid("lm_head", (VOCAB - 1) * h, 127, 7)
    ids = [1 + splitmix64((fnv1a("ids") << 32) ^ i) % (VOCAB - 1) for i in range(ROWS)]
    ids[ZERO_TOKEN_ROW] = 0
    return hidden, table, ids


def bits(t) -> list[int]:
    """BF16 tensor values as their 16-bit patterns."""
    import torch
    return (t.contiguous().view(-1).view(torch.int16).to(torch.int32) & 0xFFFF).tolist()


def near_zero(torch, functional):
    """Both normalize forms on vectors of norm 0, 1e-13, 1e-9, 1e-5 and 1, in FP64 and torch BF16."""
    w = CONFIG["width"]
    base = torch.tensor(grid("near_zero_a", w, 127, 7), dtype=torch.float64)
    other = torch.tensor(grid("near_zero_b", w, 127, 7), dtype=torch.float64)

    def scaled(v, norm):
        return (v * (norm / v.norm())).to(torch.bfloat16)

    out = {"normalize": [], "cosine": []}
    for norm in NORMS:
        v = scaled(base, norm)
        out["normalize"].append({"norm": norm, "x": bits(v), "fp64": functional.normalize(v.double(), dim=-1).tolist(),
                                 "bf16": bits(functional.normalize(v, dim=-1))})
    for nf, no, parallel in COSINE_PAIRS:
        f, o = scaled(base, nf), scaled(base if parallel else other, no)
        out["cosine"].append({"norms": [nf, no], "f": bits(f), "o": bits(o),
                              "fp64": functional.cosine_similarity(f.double(), o.double(), dim=-1).item(),
                              "bf16": bits(functional.cosine_similarity(f, o, dim=-1).reshape(1))[0]})
    return out


def main() -> int:
    """Build the tiny head from the generator, run it in FP64 and BF16 on the CPU, and write the fixture."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
    clef_dir, out_path = sys.argv[1], sys.argv[2]
    spec = importlib.util.spec_from_file_location("joint_schema_model", os.path.join(clef_dir, "joint_schema_model.py"))
    jsm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = jsm
    spec.loader.exec_module(jsm)
    import torch
    import torch.nn.functional as functional
    torch.set_grad_enabled(False)
    head = jsm.JointSchemaHead(**CONFIG)
    state = {name: torch.tensor(tensor_values(name, list(p.shape)), dtype=torch.float64).reshape(p.shape)
             for name, p in head.state_dict().items()}
    head.load_state_dict(state, strict=True)
    hidden, table, ids = inputs()
    h = CONFIG["hidden_size"]
    record = jsm.EncodedRecord(
        input_ids=tuple(ids), record_id="tiny",
        questions=tuple(jsm.EncodedQuestion(f"q{k}", t, span, tuple(opts), tuple(str(j) for j in range(len(opts))))
                        for k, (t, span, opts) in enumerate(QUESTIONS)))
    x64 = torch.tensor(hidden, dtype=torch.float64).reshape(1, ROWS, h)
    e64 = torch.tensor(table, dtype=torch.float64).reshape(VOCAB, h)
    id_t = torch.tensor([ids])
    mask = torch.ones(1, ROWS, dtype=torch.long)
    with torch.inference_mode():
        fp64 = copy.deepcopy(head).double().eval()(x64, id_t, mask, [record], e64)[0]
        bf16_head = copy.deepcopy(head).to(torch.bfloat16).eval()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            bf16 = bf16_head(x64.bfloat16(), id_t, mask, [record], e64.bfloat16())[0]
        paths = sorted({e.name for e in prof.events() if any(k in e.name for k in ("attention", "multi_head", "sdpa", "flash"))})
        cases = near_zero(torch, functional)
        capped = torch.tensor(CLAMP, dtype=torch.bfloat16).clamp(max=math.log(100.0)).exp()
        cases["clamp"] = {"p": CLAMP, "fp64": math.exp(min(CLAMP, math.log(100.0))), "bf16": bits(capped.reshape(1))[0]}
    out = {
        "sources": dict(SOURCES, torch_seen=torch.__version__, device="cpu", attention_ops=paths),
        "config": CONFIG, "vocab": VOCAB, "rows": ROWS, "scalars": SCALARS,
        "sums": {name: sum(tensor_values(name, list(p.shape))) for name, p in head.state_dict().items()},
        "ids": ids,
        "questions": [{"type": t, "span": list(span), "options": [list(o) for o in opts]} for t, span, opts in QUESTIONS],
        "fp64": [x for q in fp64 for x in q.tolist()],
        "bf16": [x for q in bf16 for x in bits(q)],
        "near_zero": cases,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {out_path}: {len(out['fp64'])} logits; attention ops {paths}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
