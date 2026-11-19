# SenseNova-Loop-9B on native Metal

SenseNova-Loop-9B (`sensenova/SenseNova-Loop-9B`, `model_type: sensenova_loop`) runs natively on Apple Silicon in both full BF16 and AWQ 4-bit (`group_size=128`) weights through the Metal `qwen27` lane backend.

## Architecture and execution schedule

The checkpoint expands `32` physical decoder layers into `56` executed steps at runtime (`4` stem layers `0..3`, `24` looped trunk layers `4..27` run twice in sequence across two iterations, and `4` tail layers `28..31`). Across the `56` steps, `42` are Gated DeltaNet linear-attention layers and `14` are grouped-query full-attention layers (`16` query heads, `4` KV heads, `head_dim=256`, `rotary_dim=64`), with distinct KV/recurrent slots per loop step and shared weights across the trunk's two passes.

Each trunk layer runs a two-stream manifold-constrained hyper-connection (`mHC`) pre/post routing block (`Sinkhorn-Knopp`, `20` iterations, `pre_scale=1.0`), fused across adjacent steps via `sn_mhc_post_and_pre_norm`. Sibling linear projections (`qkv` + `zba` on linear-attention layers and `q` + `k` + `v` on non-tail attention layers) are stacked with `OutSplits`, reducing executed dense dispatches from `281` to `226` per token.

- **Weights and formats:** BF16 safetensors (`26,249 MB` executed linear traffic/step) and standard AutoAWQ GEMM `W4A16` `g128` (`7,398 MB` executed linear traffic/step, lossless FP32 scale/zero metadata).
- **AWQ g128 export config:** Converted from `sensenova/SenseNova-Loop-9B` (`rev 3e1ba6e8ac98090f9f7c03a55f26269eb3850fb6`) with `"quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128, "zero_point": true, "version": "gemm"}`, keeping `mhc_*`, `linear_attn.{in_proj_a,in_proj_b,conv1d,A_log,dt_bias,norm}`, RMSNorms, `embed_tokens`, `model.visual`, and `loop_gate` in BF16 (`modules_to_not_convert`), calibrating AWQ input activation scales across both loop passes of layers `4..27`.
- **Context length:** Static YaRN scaling enabled at `factor=2.0` (`262,144` base to `524,288` max context), plus `--rope-yarn` factor selection on SenseNova checkpoints.
- **Vision inputs:** Supports the checkpoint's `model.visual.` tower through `--vision`.
- **Prompt chunking and decoding:** Uses `2048`-row linear-attention prompt chunks on Apple M3 Ultra and M5 Ultra (via single-chain prompt planning on `wide_prompt`), Copy-Draft (`--no-drafts` disables drafting), and fused tail greedy argmax.

## Launch

```bash
./zig-out/native/bin/tensorfold-native serve sensenova/SenseNova-Loop-9B \
  --host 127.0.0.1 --port 18090 --parallel 4 --context 36864
```

Serve an AWQ W4A16 `g128` export with the identical flags. On `256 GB` or `512 GB` unified-memory Macs, both formats comfortably hold four concurrent `36K` streams (`25.7 GiB` active memory in BF16, `15.1 GiB` in AWQ at startup with `--prompt-cache-gib 0`).

## Verification receipts

Measured against `sensenova/SenseNova-Loop-9B` (Apache-2.0, Hugging Face revision `3e1ba6e8ac98090f9f7c03a55f26269eb3850fb6`) and an AWQ W4A16 `g128` conversion on Apple M5 Ultra `512 GB` (macOS `27.2.1`, Apple Metal `32023.922`, Zig `0.17.0`). Prompt and concurrent throughput below were measured with `--parallel 4 --context 36864 --prompt-cache-gib 0`.

### Oracle parity (`tf-sensenova-oracle`)

- **BF16 vs Hugging Face PyTorch trace:** `embed_tokens cos=1.0000000`, `exec3_out cos=0.9999639`, `exec27_h_after_collapse cos=0.9998821`, `exec51_h_after_collapse cos=0.9999005`, `exec55_out cos=0.9998916`, `final_norm cos=0.9998723`, `logits_bf16 cos=0.9999594`, top-1 `4/4`, top-16 overlap `16/16`, `6/6` greedy generations exact (`4/4`, `4/4`, `42/42`, `64/64`, `64/64`, `64/64`).
- **AWQ W4A16 g128 vs quantized PyTorch trace:** `embed_tokens cos=1.0000000`, `exec3_out cos=0.9999746`, `exec27_h_after_collapse cos=0.9999127`, `exec51_h_after_collapse cos=0.9998882`, `exec55_out cos=0.9999162`, `final_norm cos=0.9998916`, `logits_bf16 cos=0.9999633`, top-1 `4/4`, top-16 overlap `16/16`, `6/6` greedy generations exact (`4/4`, `42/42`, `22/22`, `64/64`, `96/96`, `64/64`).

### Decode, concurrency, and cold prefill (Apple M5 Ultra)

| Check | Format | Prompt / Mode | Decode (`tok/s`) | TTFT / Prefill | Exactness |
| --- | --- | --- | --- | --- | --- |
| `tools/bench_openai.py` (`256` tok, `3` reps) | BF16 | `fibonacci-raw` `T=1.0` / `T=0` | `29.06` / `28.94` | `0.090s` / `0.092s` | `complete` |
| `tools/bench_openai.py` (`256` tok, `3` reps) | BF16 | `gpu-chat-no-think` `T=1.0` / `T=0` | `29.04` / `29.02` | `0.090s` / `0.090s` | `complete` |
| `tools/bench_openai.py` (`256` tok, `3` reps) | AWQ g128 | `fibonacci-raw` `T=1.0` / `T=0` | `55.51` / `54.32` | `0.058s` / `0.060s` | `complete` |
| `tools/bench_openai.py` (`256` tok, `3` reps) | AWQ g128 | `gpu-chat-no-think` `T=1.0` / `T=0` | `55.10` / `54.30` | `0.058s` / `0.058s` | `complete` |
| `tools/bench_concurrent.py` (`c=4`, `T=0`) | BF16 | `code` / `chat` (`agg`, `per-stream`) | `106.7` (`27.0`) / `109.1` (`27.9`) | `0` failed | `alone 12/12`, `serial 1/1` |
| `tools/bench_concurrent.py` (`c=4`, `T=0`) | AWQ g128 | `code` / `chat` (`agg`, `per-stream`) | `200.1` (`51.0`) / `200.0` (`51.5`) | `0` failed | `alone 12/12`, `serial 1/1` |
| `tools/bench_cache.py --require-cold` (`2,048` tok) | BF16 / AWQ | Raw token array, median `4` runs | — | `0.594s` (`3,445`) / `0.735s` (`2,786` tok/s) | `qualified: true` |
| `tools/bench_cache.py --require-cold` (`8,192` tok) | BF16 / AWQ | Raw token array, median `4` runs | — | `2.345s` (`3,493`) / `2.885s` (`2,840` tok/s) | `qualified: true` |
| `tools/bench_cache.py --require-cold` (`32,768` tok) | BF16 / AWQ | Raw token array, median `4` runs | — | `10.227s` (`3,204`) / `12.543s` (`2,613` tok/s) | `qualified: true` |

### Metal kernel contracts

- **Apple M5 Ultra (`applegpu_g17`, `tensor_units=true`):**
  - `--only sensenova`: `24 kernels: 24 pass gates 1-4, 0 fail, 0 known findings; 45 controls: 45 red as planted, 0 missed; 0 limit problems`.
  - `--only qwen27`: `40 kernels: 40 pass gates 1-4, 0 fail, 0 known findings; 6 controls: 6 red as planted, 0 missed; 0 limit problems`.
- **Apple M3 Ultra (`applegpu_g15`, `tensor_units=false`, macOS `26.6.2`):**
  - `--only sensenova`: `17 kernels: 17 pass gates 1-4, 0 fail, 0 known findings; 37 controls: 37 red as planted, 0 missed; 0 limit problems`.
  - `--only qwen27`: `30 kernels: 30 pass gates 1-4, 0 fail, 0 known findings; 3 controls: 3 red as planted, 0 missed; 0 limit problems`.
