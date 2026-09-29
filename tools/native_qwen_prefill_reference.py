"""Trace the actual Qwen family prefill; compare every layer/cache with Zig."""
import argparse
import json
from pathlib import Path

import numpy as np
from native_runtime import require_mlx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--model", type=Path, default=Path("build/models/Qwen3.8-27B-MLX-4bit"))
    parser.add_argument("--length", type=int, help="Repeat four token IDs to this length")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--simd", action="store_true")
    parser.add_argument("--final-only", action="store_true", help="Do not hook/evaluate intermediate layers")
    parser.add_argument("--image", action="append", type=Path, default=[])
    parser.add_argument("--prompt", default="Write a short Python function that computes the Fibonacci sequence.")
    parser.add_argument("--generate", type=int, default=0, help="Check serial continuation with seed 5678, temperature 0.7, top-k 12, top-p 0.8")
    args = parser.parse_args()
    if args.length is not None and not 1 <= args.length <= 262144:
        parser.error("--length must be 1..262144")
    if not 0 <= args.generate <= 128:
        parser.error("--generate must be 0..128")
    versions = require_mlx()
    if args.compare:
        report = json.loads((args.directory / "report.json").read_text())
        if report.get("versions") != versions:
            raise ValueError("Stale reference runtime; regenerate the Python prefill trace")
        native = json.loads((args.compare / "run.json").read_text())
        if native["prompt_tokens"] != report["tokens"]:
            raise ValueError("Trace prompt mismatch")
        if native.get("metal_backend") != ("simd" if report["simd"] else "tensor") or native.get("prefill_mode") != "regular":
            raise ValueError("Trace backend or prefill mode mismatch")
        if native.get("mlx_version") != versions["mlx"]:
            raise ValueError("Native and Python MLX versions differ")
        continuation_matches = native["tokens"] == report.get("generated", [])
        expected = {f"{start}-{layer}-{label}.npy" for start in report["starts"]
                    for layer in range(64) for label in ("hidden", "cache0", "cache1")}
        expected.update(f"{start}-64-logits.npy" for start in report["starts"])
        expected.update(f"{start}-16-{label}.npy" for start in report["starts"] for label in ("q", "k", "v", "a", "b", "g", "beta", "qkv", "conv"))
        expected.update(f"{start}-3-{label}.npy" for start in report["starts"] for label in ("q", "attention"))
        if report.get("final_only"):
            expected = {name for name in expected if name.endswith(("-cache0.npy", "-cache1.npy", "-logits.npy"))}
        expected.update(f"{len(report['tokens']) + step}-64-decode-logits.npy"
                        for step in range(max(0, len(report.get("generated", [])) - 1)))
        native_names = {x.name for x in args.compare.glob("*.npy")}
        if {x.name for x in args.directory.glob("*.npy")} != expected or not expected <= native_names or (not report.get("final_only") and native_names != expected):
            raise ValueError("Incomplete prefill trace")
        failures = []
        for name in sorted(expected, key=lambda x: (int(x.split("-")[0]), int(x.split("-")[1]), x)):
            a, b = np.load(args.directory / name), np.load(args.compare / name)
            if not np.array_equal(a.view(np.uint32), b.view(np.uint32)):
                failures.append(name)
                detail = (f"{np.count_nonzero(a != b)}/{a.size}, max {np.max(np.abs(a-b))}"
                          if a.shape == b.shape else f"shape {a.shape}/{b.shape}")
                if len(failures) <= 12:
                    print(name, detail)
                if len(failures) <= 12 and a.shape == b.shape and np.count_nonzero(a != b) < 10:
                    at = np.flatnonzero(a != b)
                    print("  indices", at.tolist(), "python", a.flat[at].tolist(), "native", b.flat[at].tolist())
        print(f"Compared {len(expected)} prefill/decode arrays; {len(failures)} differ; "
              f"{len(native['tokens'])} continuation tokens {'match' if continuation_matches else 'DIFFER'}")
        raise SystemExit(bool(failures) or not continuation_matches)
    args.directory.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt
    if args.native:
        import subprocess
        run = args.directory / "run.json"
        run.unlink(missing_ok=True)
        command = [str(args.native.resolve()), "run", str(args.model), "--max-tokens", str(args.generate), "--trace-dir", str(args.directory), "--report", str(run),
                   "--seed", "5678", "--temperature", "0.7", "--top-k", "12", "--top-p", "0.8"]
        command += (["--tokens", ",".join(str(1000 + (i % 4) * 37) for i in range(args.length))]
                    if args.length else ["--prompt", prompt])
        if args.simd:
            command.append("--metal-simd")
        for image in args.image:
            command.extend(["--image", str(image)])
        subprocess.run(command, check=True)
        return
    import mlx.core as mx
    (args.directory / "report.json").unlink(missing_ok=True)
    from tensorfold.families.qwen3_5 import load
    if json.loads((args.model / "config.json").read_text()).get("model_type") == "prism_hadamard_qwen35":
        from tensorfold.families.bonsai import pack
        from tensorfold.families.qwen3_5 import lane_family
        from mlx_lm.utils import load_tokenizer
        model = pack.build(args.model, form="packed" if args.simd else "lanes")
        family = lane_family(model, lanes=not args.simd, drafter="", drafter_bits=4, title="Ternary Bonsai 2", use=str(args.model))
        tokenizer = load_tokenizer(args.model)
    else:
        family, tokenizer = load(args.model, lane_kernels="off" if args.simd else "on", vision=bool(args.image))
    tokens = ([1000 + (i % 4) * 37 for i in range(args.length)] if args.length else
              tokenizer.encode(prompt, add_special_tokens=False))
    cache = family.make_cache()
    encoded = None
    if args.image:
        from tensorfold.vision.images import ImageSource, load_images
        import base64
        images = load_images([ImageSource("data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()) for path in args.image])
        rendered = "<|vision_start|><|image_pad|><|vision_end|>" * len(images) + prompt
        prepared = family.vision.prepare(rendered, images)
        tokens = list(prepared.token_ids)
        encoded = family.encode_vision(prepared, cache)
    layers = family.core.layers
    indices = {id(layer): i for i, layer in enumerate(layers)}
    cls = type(layers[0])
    original = cls.__call__
    from mlx_lm.models import qwen3_5, qwen3_next
    from mlx_lm.models.gated_delta import compute_g
    original_gdn = qwen3_5.gated_delta_update
    original_attention = qwen3_next.scaled_dot_product_attention
    current = None

    def save(layer, label, x):
        mx.eval(x)
        np.save(args.directory / f"{start}-{layer}-{label}.npy", np.asarray(x.astype(mx.float32)))

    def traced(self, *a, **kw):
        nonlocal current
        current = indices[id(self)]
        if current == 16:
            gdn = self.linear_attn
            qkv = gdn.in_proj_qkv(self.input_layernorm(a[0]))
            state = kw["cache"][0]
            if state is None:
                state = mx.zeros((1,3,10240),dtype=mx.bfloat16)
            save(16,"qkv",qkv)
            save(16,"conv",gdn.conv1d(mx.concatenate([state,qkv],axis=1)))
        out = original(self, *a, **kw)
        save(indices[id(self)], "hidden", out)
        return out

    def gdn(q,k,v,a,b,alog,dt,*args,**kw):
        if current == 16:
            for label, value in zip(("q", "k", "v", "a", "b", "g", "beta"),
                                    (q,k,v,a,b,compute_g(alog,a,dt),mx.sigmoid(b))):
                save(16,label,value)
        return original_gdn(q,k,v,a,b,alog,dt,*args,**kw)

    def attention(q,k,v,*args,**kw):
        out = original_attention(q,k,v,*args,**kw)
        if current == 3:
            save(3,"q",q)
            save(3,"attention",out)
        return out

    if not args.final_only:
        cls.__call__ = traced
        qwen3_5.gated_delta_update = gdn
        qwen3_next.scaled_dot_product_attention = attention
    starts = list(range(0, len(tokens), 2048))
    try:
        for start in starts:
            inputs = mx.array([tokens[start:start+2048]], dtype=mx.uint32)
            hidden = (family.prefill_vision(inputs, cache, encoded, start, min(start+2048, len(tokens)))
                      if encoded is not None else family.prefill(inputs, cache))
            save(64, "logits", family.head(hidden[:, -1:]))
            for i, item in enumerate(cache):
                for j, value in enumerate(item.state):
                    if hasattr(item, "keys"):
                        value = value[:, :, :item.offset]
                    save(i, f"cache{j}", value)
            print(f"Traced prefill {start}..{min(start+2048, len(tokens))}", flush=True)
    finally:
        cls.__call__ = original
        qwen3_5.gated_delta_update = original_gdn
        qwen3_next.scaled_dot_product_attention = original_attention
    generated = []
    if args.generate:
        from tensorfold.engine.exact_sampling import Sampling
        settings = Sampling(5678, 0.7, 12, 0.8)
        for step in range(args.generate):
            logits = family.head(hidden[:, -1:])
            token = int(family.sample(logits, settings, [len(tokens) + step])[0])
            generated.append(token)
            if token in (248044, 248046):
                break
            if step + 1 < args.generate:
                hidden = family.hidden(mx.array([[token]], dtype=mx.uint32), cache)
                start = len(tokens) + step
                save(64, "decode-logits", family.head(hidden[:, -1:]))
        print("Continuation:", generated, flush=True)
    (args.directory / "report.json").write_text(json.dumps({"starts": starts, "tokens": tokens, "generated": generated, "versions": versions, "simd": args.simd, "final_only": args.final_only}) + "\n")


if __name__ == "__main__":
    main()
