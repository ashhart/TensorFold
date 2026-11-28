# GigaChat 4

Host admission only: nothing is served, and no GPU kernel exists for this family yet. Tracking issue: #5336.

| Checkpoint | Revision | Stored | Licence |
| --- | --- | --- | --- |
| [ai-sage/GigaChat4-37B-A4B](https://huggingface.co/ai-sage/GigaChat4-37B-A4B) | `1bf9a4b2b8d7d0f970e48d6458be77d49e07bcfc` | FP8 E4M3, 128x128 blocks, F32 scales; BF16 elsewhere | Noncommercial License for individuals 1.0 |
| [ai-sage/GigaChat4-37B-A4B-bf16](https://huggingface.co/ai-sage/GigaChat4-37B-A4B-bf16) | `e8f40e01f289d9bd71b10a21ca3fbad8a8f0d6dc` | BF16 | Noncommercial License for individuals 1.0 |
| [ai-sage/GigaChat4-708B-A41B](https://huggingface.co/ai-sage/GigaChat4-708B-A41B) | `2baf8b0cf71f1528c14e7a274877d9c8fbf22697` | FP8 E4M3, 128x128 blocks, F32 scales; BF16 elsewhere | MIT |

The 37B weights are for noncommercial use only, under their own licence
([text](../../LICENSES/GigaChat4-Noncommercial-1.0.txt)); TensorFold's Apache-2.0 licence does not change that.
Every receipt that names a 37B checkpoint names this licence.

## What the host package checks

`zig/src/families/gigachat4/` reads `config.json` and refuses, by key, any layout the engine does not run. It maps
every tensor of the model shards with its dtype and shape: the FP8 releases' 36,700 and 95,616 tensors and the BF16
copy's 18,742, matched against the releases' safetensors headers, and reports missing, retyped, reshaped and unknown
tensors by name. It derives the layer plan (Gated DeltaNet or MLA with a DSA indexer per layer, dense or MoE
feed-forward, the stored MTP layer after the decoder) and the memory floors for one device.

```sh
zig build test-gigachat4-host
```

With the 37B release's shards (or copies of their headers) in a directory, `TF_GIGACHAT4_HEADERS=DIR` also checks the
map against them, and `TF_GIGACHAT4_CONFIG_37B=FILE` checks its `config.json`.

## Not in this package

Kernels, the MTP and DFlash2 drafters, serving and multi-rank. The `dflash_draft/` drafter is not a model shard and is
not in the map.
