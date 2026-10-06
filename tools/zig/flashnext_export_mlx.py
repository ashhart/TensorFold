"""Write a dump's pack_mlx.safetensors: the projections the Zig prompt path multiplies, in MLX's 6-bit g32 layout."""
import sys
from pathlib import Path

import mlx.core as mx

from tensorfold.families.qwen4_exp.runtime import load

model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
rt, _ = load(model_dir, drafts=3)
fused, pack = rt.fused, {}


def put(name, lin):
    bits, group = getattr(lin, "bits", None), getattr(lin, "group_size", None)
    assert bits == 6 and group == 32, (name, bits, group)
    pack[name + ".mw"], pack[name + ".ms"], pack[name + ".mb"] = lin.weight, lin.scales, lin.biases


for i, (layer, entry) in enumerate(zip(rt.model.layers, fused.layers)):
    if layer.is_linear:
        proj, _, g = entry["gdn"]
        put(f"L{i}.gdn.in", proj)
        put(f"L{i}.gdn.out", g.out_proj)
    else:
        proj, *_, a = entry["attn"]
        put(f"L{i}.att.proj", proj)
        put(f"L{i}.att.o", a.o_proj)
for layer in rt.model.layers:
    if "ple" in layer:
        put("ple.kv", fused.ple_parts[id(layer.ple)][0])
mx.eval(list(pack.values()))
mx.save_safetensors(str(out), pack)
print("saved", len(pack), "tensors", sum(v.nbytes for v in pack.values()) / 1e9, "GB")
