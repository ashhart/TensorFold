"""engram_fp8.py <export dir>: Engram tables BF16 to FP8 E4M3 rows with one BF16 scale a row, after nvfp4.py.

Value = row x scale, scale = row max / 448; config.json gets engram_fp8_embedding: true. The run that made
atakans/Hy4-flash-NVFP4 used this file before its lint changes (context managers, imports one a line).
"""

import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def read(path):
    with open(path) as f:
        return json.load(f)


def write(value, path):
    with open(path, "w") as f:
        json.dump(value, f, indent=2)


d = sys.argv[1]
idx_path = os.path.join(d, "model.safetensors.index.json")
idx = read(idx_path)
wm = idx["weight_map"]
keys = [k for k in wm if k.endswith("multi_head_embedding.embedding.weight")]
report = {}
for shard in sorted({wm[k] for k in keys}):
    path = os.path.join(d, shard)
    with safe_open(path, "pt") as f:
        meta = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118 (safe_open has no __iter__)
    for k in [k for k in keys if wm[k] == shard]:
        w = tensors[k]
        q = torch.empty(w.shape, dtype=torch.float8_e4m3fn)
        s = torch.empty(w.shape[0], dtype=torch.bfloat16)
        err = ref = 0.0
        step = 1 << 22
        for i in range(0, w.shape[0], step):
            x = w[i : i + step].cuda().float()
            sc = (x.abs().amax(1) / 448).clamp(min=1e-12).to(torch.bfloat16)
            y = (x / sc.float()[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
            err += (y.float() * sc.float()[:, None] - x).pow(2).sum().item()
            ref += x.pow(2).sum().item()
            q[i : i + len(x)] = y.cpu()
            s[i : i + len(x)] = sc.cpu()
        sk = k.removesuffix("weight") + "weight_scale_inv"
        tensors[k], tensors[sk], wm[sk] = q, s, shard
        report[k] = {"rows": w.shape[0], "rel_rms_error": (err / ref) ** 0.5}
        print(k, report[k], flush=True)
    save_file(tensors, path + ".tmp", metadata=meta)
    os.replace(path + ".tmp", path)
idx["metadata"]["total_size"] = sum(
    os.path.getsize(os.path.join(d, f)) for f in set(wm.values())
)
write(idx, idx_path)
cfg_path = os.path.join(d, "config.json")
cfg = read(cfg_path)
(cfg.get("text_config") or cfg)["engram_fp8_embedding"] = True
write(cfg, cfg_path)
write(report, os.path.join(d, "engram-fp8.json"))
print("ENGRAM_DONE", flush=True)
