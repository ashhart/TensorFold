"""Write a dump's pack_mtp_mlx.safetensors: the MTP head's prompt-time projections in MLX's 6-bit g32 layout."""
import sys
from pathlib import Path

import mlx.core as mx

from tensorfold.families.qwen4_exp.runtime import load

model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
rt, _ = load(model_dir, drafts=3)
pack = {}


def put(name, lin):
    bits, group = getattr(lin, "bits", None), getattr(lin, "group_size", None)
    assert bits == 6 and group == 32, (name, bits, group)
    pack[name + ".mw"], pack[name + ".ms"], pack[name + ".mb"] = lin.weight, lin.scales, lin.biases


head, entry = rt.mtp, rt.mtp_fused.layers[0]
put("mtp.att.proj", entry["attn"][0])
put("mtp.fce", head.fc_embedding)
put("mtp.fch", head.fc_hidden)
mx.eval(list(pack.values()))
mx.save_safetensors(str(out), pack)
print("saved", len(pack), "tensors", sum(v.nbytes for v in pack.values()) / 1e9, "GB")
