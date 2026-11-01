"""Oracle A for Gemma 4 E4B: transformers BF16 on the CPU, greedy tokens and the full logits at every step."""
# Per prompt: chat template, 512-token prefill, greedy decode; fp32 logits to OUT/<name>.logits.npy, ids to .json

from __future__ import annotations

import argparse
import json
import struct
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

STOPS = {1, 106, 50}
TEXT = "model.language_model."


def dequantized(raw: np.memmap, base: int, header: dict, name: str, bits: int = 4) -> torch.Tensor:
    """A weight of the MLX 4/64 checkpoint back to bf16: bf16(q * scale + bias) in fp32, as the XPU gathers do."""
    stem = name.removesuffix(".weight")

    def get(n: str, dtype) -> np.ndarray:
        a, b = header[n]["data_offsets"]
        return raw[base + a : base + b].view(dtype).reshape(header[n]["shape"])

    w, s, bi = get(name, np.uint32), get(stem + ".scales", np.uint16), get(stem + ".biases", np.uint16)
    out = np.empty((w.shape[0], w.shape[1] * 8), dtype=np.uint16)
    for r in range(0, w.shape[0], 4096):
        blk = w[r : r + 4096]
        q = ((blk[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(blk.shape[0], -1).astype(np.float32)
        sc = (s[r : r + 4096].astype(np.uint32) << 16).view(np.float32).repeat(64, axis=1)
        bs = (bi[r : r + 4096].astype(np.uint32) << 16).view(np.float32).repeat(64, axis=1)
        v = (q * sc + bs).view(np.uint32)
        out[r : r + 4096] = ((v + 0x7FFF + ((v >> 16) & 1)) >> 16).astype(np.uint16)
    return torch.from_numpy(out).view(torch.bfloat16)


def load_model(src: Path, quant: Path | None = None) -> Gemma4ForCausalLM:
    config = json.loads((src / "config.json").read_text())["text_config"]
    cfg = Gemma4TextConfig(**config)
    cfg._attn_implementation = "sdpa"
    try:
        from transformers.initialization import no_init_weights
    except ImportError:
        from transformers.modeling_utils import no_init_weights
    torch.set_default_dtype(torch.bfloat16)
    with no_init_weights():
        model = Gemma4ForCausalLM(cfg)
    torch.set_default_dtype(torch.float32)
    path = (quant or src) / ("model.safetensors" if quant else "model-00001-of-00001.safetensors")
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    raw = np.memmap(path, mode="r", dtype=np.uint8)
    sd = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, t in header.items():
            if not name.startswith(TEXT) or name.endswith((".scales", ".biases")):
                continue
            a, b = t["data_offsets"]
            if t["dtype"] == "U32":
                arr = dequantized(raw, 8 + n, header, name)
            else:
                arr = torch.from_numpy(raw[8 + n + a : 8 + n + b].view(np.uint16).reshape(t["shape"])).view(torch.bfloat16)
            sd["model." + name[len(TEXT) :]] = arr
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if m != "lm_head.weight"]
    if missing or unexpected:
        raise SystemExit(f"load: missing {missing[:5]} unexpected {unexpected[:5]}")
    model.tie_weights()
    return model.eval()


def klass(name: str) -> str:
    parts = name.split(".")
    if "layers" in parts:
        parts = parts[parts.index("layers") + 2 :]
    return ".".join(parts).removesuffix(".weight")


# With --ranges: the largest |x| entering and leaving every linear class, to OUT/ranges.json (the fp16 range question).
def attach_ranges(model: torch.nn.Module, table: dict) -> None:
    def hook(mod, args, out, name=None):
        e = table.setdefault(name, {"in": 0.0, "out": 0.0})
        e["in"] = max(e["in"], float(args[0].detach().abs().max()))
        e["out"] = max(e["out"], float(out.detach().abs().max()))

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            mod.register_forward_hook(lambda m, a, o, name=klass(name): hook(m, a, o, name))


@torch.no_grad()
def run(model, ids: list[int], steps: int, chunk: int, forced: list[int] | None = None):
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    x = torch.tensor([ids])
    logits = None
    for at in range(0, len(ids), chunk):
        end = min(len(ids), at + chunk)
        out = model(input_ids=x[:, at:end], past_key_values=cache, use_cache=True, logits_to_keep=1 if end == len(ids) else 0)
        cache = out.past_key_values
        if end == len(ids):
            logits = out.logits[0, -1].float()
    gen, rows = [], []
    for i in range(len(forced) if forced is not None else steps):
        rows.append(logits.numpy())
        tok = forced[i] if forced is not None else int(logits.argmax())
        gen.append(tok)
        if forced is None and tok in STOPS:
            break
        if forced is not None and i == len(forced) - 1:
            break
        out = model(input_ids=torch.tensor([[tok]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[0, -1].float()
    return gen, np.stack(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("prompts", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--only", help="comma-separated prompt names")
    p.add_argument("--ranges", action="store_true")
    p.add_argument("--force", type=Path, help="a reference directory: its prompt and generated ids are used as they are (teacher forcing)")
    p.add_argument("--threads", type=int, default=22)
    p.add_argument("--quant", type=Path, help="serve the weights of this MLX 4/64 checkpoint, dequantized (oracle B: same weights as the engine)")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, args.quant)
    table: dict = {}
    if args.ranges:
        attach_ranges(model, table)
    only = set(args.only.split(",")) if args.only else None
    for pr in json.loads(args.prompts.read_text()):
        if only and pr["name"] not in only:
            continue
        kw = {"enable_thinking": pr.get("enable_thinking", False)}
        text = tok.apply_chat_template(pr["messages"], tools=pr.get("tools"), add_generation_prompt=True, tokenize=False, **kw)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        forced = None
        if args.force:
            meta = json.loads((args.force / f"{pr['name']}.json").read_text())
            assert meta["prompt_ids"] == ids, "the forced prompt differs from the rendered one"
            forced = meta["generated_ids"]
        t0 = time.time()
        gen, rows = run(model, ids, args.steps, args.chunk, forced)
        np.save(args.out / f"{pr['name']}.logits.npy", rows)
        (args.out / f"{pr['name']}.json").write_text(
            json.dumps({"name": pr["name"], "prompt_ids": ids, "generated_ids": gen, "reply": tok.decode(gen), "chat_text": text}, ensure_ascii=False) + "\n"
        )
        print(f"{pr['name']:16s} prompt {len(ids):5d} tokens, {len(gen):3d} generated, {time.time() - t0:6.1f}s: {tok.decode(gen)[:90]!r}", flush=True)
        if args.ranges:
            (args.out / "ranges.json").write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
