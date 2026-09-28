"""Compare Flash intermediates using the original fused blocks and native trace files."""
import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--tokens", default="1000,1037,1074,1111")
    p.add_argument("--length", type=int, help="Repeat the token pattern to this prompt length")
    p.add_argument("--compare", type=Path)
    p.add_argument("--native", type=Path, help="Run native tracing with the same generated prompt")
    p.add_argument("--gdn-layer", type=int, help="Trace one recurrent block across all prefill chunks")
    args = p.parse_args()
    if args.compare:
        failures = 0
        paths = sorted(args.directory.glob("*.npy"))
        # Accept the explicit plus sign from older Zig signed-position traces.
        native_files = list(args.compare.glob("*.npy"))
        native_paths = {x.name.replace("+", "0"): x for x in native_files}
        token_count = args.length or len(args.tokens.split(","))
        expected = 242 if args.gdn_layer is None else 2 * ((token_count + 15) // 16)
        if len(paths) != expected or len(native_files) != expected or {x.name for x in paths} != native_paths.keys():
            raise ValueError(f"Expected matching complete traces with {expected} arrays")
        for path in paths:
            a, b = np.load(path), np.load(native_paths[path.name])
            if not np.array_equal(a, b):
                failures += 1
                print(path.name, "different", np.count_nonzero(a != b), "of", a.size,
                      "max", np.max(np.abs(a - b)))
        print(f"Compared {len(paths)} intermediates; {failures} differ")
        raise SystemExit(bool(failures))
    tokens = np.array([int(x) for x in args.tokens.split(",")], dtype=np.int32)
    if args.length:
        tokens = np.resize(tokens, args.length)
    if args.native:
        import subprocess
        args.directory.mkdir(parents=True, exist_ok=True)
        command = [str(args.native.resolve()), "run", str(args.model), "--tokens",
                        ",".join(map(str, tokens)), "--no-drafts", "--max-tokens", "0",
                        "--trace-dir", str(args.directory)]
        if args.gdn_layer is not None:
            command += ["--trace-gdn", str(args.gdn_layer)]
        subprocess.run(command, check=True)
        return
    import mlx.core as mx
    from tensorfold.families.qwen4_exp.model import load
    from tensorfold.families.qwen4_exp.decode import FusedDecode
    from tensorfold.kernels.qwen.flash_next.v1 import embed, hc
    from tensorfold.families.qwen4_exp import decode
    from types import SimpleNamespace
    decode.DENSE = "rows"
    K = SimpleNamespace(embed_rows=embed.embed_rows, hc_norm=hc.hc_norm, hc_project=decode.hc_project)
    embed.PleTables = lambda embedding: embedding
    embed.ple_lookup = lambda ids, tables: tables(ids)
    model, _ = load(args.model, lazy=True)
    fused = FusedDecode(model)
    for layer in model.layers:
        if "ple" in layer:
            layer.ple.ple_embedding.__dict__.pop("fused_tables", None)
    cache = model.make_cache()
    if args.gdn_layer is not None:
        args.directory.mkdir(parents=True, exist_ok=True)
        original = fused._gdn
        def traced(index, x, c):
            out = original(index, x, c)
            if index == args.gdn_layer:
                for label, value in (("mixed", x), ("branch", out)):
                    mx.eval(value)
                    np.save(args.directory / f"{start:06}-{index:02}-{label}.npy", np.asarray(value.astype(mx.float32)))
            return out
        fused._gdn = traced
        for start in range(0, len(tokens), 16):
            mx.eval(fused(tokens[start:start + 16][None], cache))
            if start % 512 == 0:
                print(f"Prefill {start + min(16, len(tokens) - start)}/{len(tokens)}", flush=True)
        return
    for start in range(0, ((len(tokens) - 1) // 16) * 16, 16):
        mx.eval(fused(tokens[start:start + 16][None], cache))
        if start % 512 == 0:
            print(f"Prefill {start + 16}/{len(tokens)}", flush=True)
    tokens = tokens[((len(tokens) - 1) // 16) * 16:]
    h = K.embed_rows(tokens, model.model.embed_tokens, tile=fused.streams)
    args.directory.mkdir(parents=True, exist_ok=True)
    def save(i, label, value):
        mx.eval(value)
        np.save(args.directory / f"{i:02}-{label}.npy", np.asarray(value.astype(mx.float32)))
    pending = ("none", (), None)
    for i, layer in enumerate(model.layers):
        if "ple" in layer:
            h = fused._write_back(h, pending)
            pending = ("none", (), None)
            h = h + fused._ple(layer.ple, h, tokens[None], cache[i])
        save(i, "input", fused._write_back(h, pending))
        kind, branch, inject = pending
        hn, ssp = K.hc_norm(h, streams=fused.streams, write_back=kind, branch=branch, inject=inject)
        ahc = fused.layers[i]["attn_hc"]
        mixed, inj = K.hc_project(hn, ssp, ahc.down, ahc.up, ahc.scale,
                                eps=fused.eps, streams=fused.streams, low=ahc.low)
        save(i, "mixed", mixed)
        out = fused._gdn(i, mixed, cache[i]) if layer.is_linear else fused._attention(i, mixed, cache[i])
        save(i, "branch", out)
        hm, ssp = K.hc_norm(hn, streams=fused.streams, write_back="plain", branch=(out,), inject=inj)
        mhc = fused.layers[i]["mlp_hc"]
        mixed, inj2 = K.hc_project(hm, ssp, mhc.down, mhc.up, mhc.scale,
                                 eps=fused.eps, streams=fused.streams, low=mhc.low)
        save(i, "moe-input", mixed)
        kind, branch = fused._moe(i, mixed)
        h, pending = hm, (kind, branch, inj2)
        save(i, "output", fused._write_back(h, pending))
        print(f"Traced layer {i}", flush=True)
    from tensorfold.families.qwen4_exp.decode import project
    hn, ssp = K.hc_norm(h, streams=fused.streams, write_back=pending[0], branch=pending[1], inject=pending[2])
    mix = fused.mixer
    mixed = K.hc_project(hn, ssp, mix.down, mix.up, mix.scale,
                         eps=fused.eps, streams=fused.streams, low=mix.low)[0]
    save(47, "head-mixed", mixed)
    save(47, "logits", project(mixed, model.lm_head))


if __name__ == "__main__":
    main()
