"""The release's Engram hash multipliers and the reference's hash ids for a few tokens (fixtures/engram_hashes.json)."""

import json, sys
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.hy_v4_flash.modeling_hy_v4_flash import HYV4FlashEngramLayer

ckpt = sys.argv[1]
cfg = AutoConfig.from_pretrained(ckpt).text_config
wm = json.load(open(f"{ckpt}/model.safetensors.index.json"))["weight_map"]
ids = torch.tensor([[120000, 3717, 120001, 3462, 341, 220, 1162, 843, 220, 1498, 30, 120025, 5, 5, 5, 119999, 120900, 0]])
out = {"tokens": ids[0].tolist(), "layers": {}}
for layer_id in cfg.engram_layer_ids:
    key = f"model.layers.{layer_id - 1}.engram_layer.lookup_table"
    with safe_open(f"{ckpt}/{wm[key]}", "pt") as f:
        table = f.get_tensor(key)
    with torch.device("meta"):
        mod = HYV4FlashEngramLayer(cfg, layer_id)
    mod.lookup_table = table
    mod.head_vocab_sizes = torch.tensor(mod.config_head_vocab_sizes)
    mod.hash_multipliers = mod.draw_hash_multipliers(int(table.max()) + 1)
    ngrams = torch.stack([ids] + [torch.nn.functional.pad(ids, (o, 0), value=cfg.engram_pad_id)[:, : ids.shape[1]] for o in range(1, cfg.engram_max_ngram_size)], -1)
    out.setdefault("compressed", {str(t): int(table[t]) if t < table.shape[0] else None for t in ids[0].tolist() + [cfg.engram_pad_id]})
    out["layers"][str(layer_id)] = {"compressed_vocab": int(table.max()) + 1, "table_rows": table.shape[0],
                                    "multipliers": mod.hash_multipliers.tolist(), "hash_ids": mod.hash_ngrams(ngrams)[0].tolist()}
json.dump(out, sys.stdout)
