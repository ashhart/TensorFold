#!/usr/bin/env python3
"""exllamav3's answer key for ling_check.py on a full EXL3 pack that no single GPU holds, streamed module by module."""

# python3 ling_ref_stream.py PACK OUT_DIR [--texts ling_texts.json] [--max-len 512] [--names a,b]

# Each module (embedding, 72 blocks, final norm, head; no MTP) loads once, runs every sequence, then unloads.

# Calls as ling_layers_ref.py makes them ({"attn_mode": "flash_attn_nc"}, no cache); pack tokenizer, no BOS or template.

# Writes <name>.ids (int32), <name>.ref_logits.f32 [T, vocab] and the floor's prefix runs <name>@<k> (k = T//2, T-17).

# Also names.txt and meta.json (floor KL and top-1, true-token NLL, versions); exllamav3 v1.6.6, TF32 override off.

import argparse
import json
import time
from pathlib import Path

import torch


def kl_stats(ref, got):
    kls = []
    for i in range(0, ref.shape[0], 256):
        lp = torch.log_softmax(ref[i:i + 256].float(), -1)
        lq = torch.log_softmax(got[i:i + 256].float(), -1)
        kls.append((lp.exp() * (lp - lq)).sum(-1))
    kl = torch.cat(kls)
    return {"kl": float(kl.mean()), "kl_max": float(kl.max()), "kl_argmax": int(kl.argmax()),
            "top1": float((ref.argmax(-1) == got.argmax(-1)).float().mean()),
            "byte_equal_rows": int((ref == got).all(-1).sum())}


def true_nll(logits, ids):
    total, T = 0.0, ids.numel()
    for i in range(0, T - 1, 256):
        j = min(i + 256, T - 1)
        lp = torch.log_softmax(logits[i:j].float(), -1)
        total += float(-lp.gather(1, ids[i + 1:j + 1, None].long()).sum())
    return total / (T - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--texts", default=str(Path(__file__).resolve().parent / "ling_texts.json"))
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--names", default="")
    ap.add_argument("--dry", action="store_true", help="tokenise and list the modules, no GPU")
    a = ap.parse_args()

    from exllamav3 import Config, Model, Tokenizer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    config = Config.from_directory(a.model)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    V = config.vocab_size
    texts = json.loads(Path(a.texts).read_text())
    if a.names:
        texts = {k: v for k, v in texts.items() if k in a.names.split(",")}
    seqs = {}   # key -> ids [1, T]; keys: name, name@k (prefixes for the noise floor)
    for name, text in texts.items():
        ids = tok.encode(text, add_bos=False, encode_special_tokens=True)[:, : a.max_len]
        T = ids.shape[1]
        seqs[name] = ids
        for k in (T // 2, T - 17):
            seqs[f"{name}@{k}"] = ids[:, :k]
        print(f"[ref-stream] {name}: {T} tokens", flush=True)
    print(f"[ref-stream] {len(model.modules)} modules: first block {model.first_block_idx}, last block "
          f"{model.last_kv_module_idx}, head {model.logit_layer_idx}", flush=True)
    if a.dry:
        return
    dev = torch.device("cuda:0")
    params = {k: {"attn_mode": "flash_attn_nc"} for k in seqs}
    x = {k: model.prepare_inputs(v, params[k]) for k, v in seqs.items()}
    for i, m in enumerate(model.modules[: model.logit_layer_idx + 1]):
        t1 = time.perf_counter()
        m.load(dev)
        with torch.inference_mode():
            for k in seqs:
                x[k] = m.forward(m.prepare_for_device(x[k], params[k]), params[k])
                if i == model.logit_layer_idx:
                    x[k] = x[k].float()[0, :, :V].cpu()
        m.unload()
        config.stc.release_arena() if hasattr(config, "stc") and hasattr(config.stc, "release_arena") else None
        torch.cuda.empty_cache()
        print(f"[ref-stream] module {i + 1}/{model.logit_layer_idx + 1} {type(m).__name__} "
              f"{time.perf_counter() - t1:.1f}s", flush=True)
    logits = x
    meta = {"model": a.model, "layers": config.num_hidden_layers, "vocab": V, "streamed": True,
            "torch": torch.__version__, "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "texts": {}}
    try:
        meta["exllamav3"] = Path("/opt/exllamav3.commit").read_text().strip()
    except OSError:
        pass
    for name in texts:
        ids = seqs[name]
        T = ids.shape[1]
        ids[0].to(torch.int32).numpy().tofile(out / f"{name}.ids")
        logits[name].contiguous().numpy().tofile(out / f"{name}.ref_logits.f32")
        noise = {}
        for k in (T // 2, T - 17):
            logits[f"{name}@{k}"].contiguous().numpy().tofile(out / f"{name}@{k}.ref_logits.f32")
            noise[f"prefix_{k}"] = kl_stats(logits[name][:k], logits[f"{name}@{k}"])
        nll = true_nll(logits[name], ids[0])
        meta["texts"][name] = {"tokens": T, "nll": nll, "noise": noise}
        print(f"[ref-stream] {name}: {T} tokens, nll {nll:.4f}, floor "
              + ", ".join(f"{p}: KL {v['kl']:.5f} top1 {v['top1']:.4f}" for p, v in noise.items()), flush=True)
    meta["wall_s"] = time.perf_counter() - t0
    (out / "names.txt").write_text("\n".join(texts.keys()) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[ref-stream] wrote {out} in {meta['wall_s']:.0f}s, peak {meta['peak_gib']:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
