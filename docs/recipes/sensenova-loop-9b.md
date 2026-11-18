# SenseNova-Loop-9B on native Metal

SenseNova-Loop-9B (`sensenova/SenseNova-Loop-9B`, `sensenova_loop` / `sensenova_loop_text`) runs natively on Apple Silicon in both full BF16 and AWQ 4-bit (`group_size=128`) weights through the Metal `qwen27` lane backend.

## Architecture and execution schedule

The checkpoint expands `32` physical decoder layers into `56` executed steps at runtime (`4` stem layers `0..3`, `24` looped trunk layers `4..27` run twice in sequence across two iterations, and `4` tail layers `28..31`). Across the `56` steps, `42` are Gated DeltaNet linear-attention layers and `14` are grouped-query full-attention layers (`16` query heads, `4` KV heads, `head_dim=256`, `rotary_dim=64`), with distinct KV/recurrent slots per loop step and shared weights across the trunk's two passes.

Each trunk layer runs a two-stream manifold-constrained hyper-connection (`mHC`) pre/post routing block (`Sinkhorn-Knopp`, `20` iterations, `pre_scale=1.0`), fused across adjacent steps via `sn_mhc_post_and_pre_norm`. Sibling linear projections (`qkv` + `zba` on linear-attention layers and `q` + `k` + `v` on non-tail attention layers) are stacked with `OutSplits`, reducing executed dense dispatches from `281` to `226` per token.

- **Weights and formats:** BF16 safetensors (`26,249 MB` executed linear traffic/step) and AWQ W4A16 `g128` (`7,398 MB` executed linear traffic/step, lossless FP32 scale/zero metadata).
- **Context length:** Static YaRN scaling enabled at `factor=2.0` (`262,144` base to `524,288` max context), plus `--rope-yarn` override.
- **Vision inputs:** Supports the checkpoint's `model.visual.` tower through `--vision`.
- **Prompt chunking and decoding:** Uses `2048`-row linear-attention prompt chunks on Apple M3 Ultra and M5 Ultra (via single-chain prompt planning on `wide_prompt`), native Multi-Token Prediction (`MTP`) speculative drafting when `mtp.safetensors` is present beside the checkpoint or passed via `--drafter` (falling back to Copy-Draft when absent; `--no-drafts` disables drafting), and fused tail greedy argmax.

## Multi-token speculative drafting (MTP)

When `mtp.safetensors` is placed next to the checkpoint (from `sensenova/SenseNova-Loop-9B`, Apache-2.0) or passed with `--drafter <path>`, the engine attaches the native Metal MTP drafter (`TF_MTP_DRAFTS`, default `3` steps). Intermediate prompt chunks advance the drafter state via `keysOnly` attention absorb (skipping full-chunk query projections, MLP, and norm until the final kept token), and active streams share a single Metal command buffer per round via `draft_round` and `sn_mtp_batch`.

- **`TF_MTP_DRAFTS`:** Draft depth per round (default `3`; clamped to `1` through the drafter's window less one).
- **`TF_MTP_PLAIN_GUARD`:** `0` or `1` (default `1`); runs a plain-round fallback alongside MTP when verification rejects the drafted chain.
- **`TF_MTP_BITS`:** `4` (default) or `16`; quantizes loaded `mtp.safetensors` projections (and a draft-only copy of `lm_head` when the target head is BF16) to AWQ 4-bit `g128` at boot, or keeps BF16 (`16`). Target verification always runs in full checkpoint precision, producing bit-identical `token_sha` outputs to `--no-drafts` at both `T=0` and seeded `T=1.0`.

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
- **MTP head (`tf-sensenova-oracle --mtp-oracle`, including split `prompt_only` prefill and 2-stream batched `draftRound`):**
  - `TF_MTP_BITS=16` on BF16: `absorb_out last cos=0.9999904 max_abs=0.06250`, `chain[0..2]` `cos=0.9999904`, `0.9999332`, `0.9999087`, greedy `20/20`, `22/22`, `22/22`.
  - `TF_MTP_BITS=4` on BF16: `absorb_out last cos=0.9844645 max_abs=3.04688`, `chain[0..2]` `cos=0.9844645`, `0.9146141`, `0.8913979`, greedy `20/20`, `22/22`, `22/22`.
  - `TF_MTP_BITS=4` on AWQ g128: `absorb_out last cos=0.9831613 max_abs=1.71875`, `chain[0..2]` `cos=0.9831613`, `0.9113104`, `0.8807949`, greedy `20/20`, `22/22`, `22/22`.

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

### MTP vs `--no-drafts` on Apple M5 Ultra (`--parallel 4 --context 36864 --prompt-cache-gib 0`)

With `mtp.safetensors` (`TF_MTP_BITS=4`, `TF_MTP_DRAFTS=3`, `TF_MTP_PLAIN_GUARD=1`), active memory once loaded is `29.6 GiB` in BF16 (vs `25.5 GiB` with `--no-drafts`) and `18.4 GiB` in AWQ g128 (vs `14.9 GiB` with `--no-drafts`).

| Check | Format | Prompt / Mode | MTP (`tok/s`) | `--no-drafts` (`tok/s`) | Gain / Exactness |
| --- | --- | --- | --- | --- | --- |
| `tools/bench_openai.py` (`256` tok, `3` reps) | BF16 | `fibonacci-raw` `T=1.0` / `T=0` | `67.61` / `73.80` (`2.639` / `2.753` tok/chunk) | `29.02` / `29.15` | `2.33x` / `2.53x`, identical `token_sha` |
| `tools/bench_openai.py` (`256` tok, `3` reps) | BF16 | `gpu-chat-no-think` `T=1.0` / `T=0` | `49.56` / `58.34` (`1.855` / `2.169` tok/chunk) | `29.05` / `29.10` | `1.71x` / `2.00x`, identical `token_sha` |
| `tools/bench_openai.py` (`256` tok, `3` reps) | AWQ g128 | `fibonacci-raw` `T=1.0` / `T=0` | `114.23` / `117.73` (`2.667` / `2.783` tok/chunk) | `55.45` / `54.28` | `2.06x` / `2.17x`, identical `token_sha` |
| `tools/bench_openai.py` (`256` tok, `3` reps) | AWQ g128 | `gpu-chat-no-think` `T=1.0` / `T=0` | `79.19` / `83.00` (`1.842` / `1.954` tok/chunk) | `55.44` / `54.28` | `1.43x` / `1.53x`, identical `token_sha` |
| `tools/bench_concurrent.py` (`c=4`, `T=1.0` / `T=0`) | BF16 | `code` (`agg`, `per-stream`) | `176.6` (`48.9`) / `205.9` (`52.3`) | `—` / `109.6` (`27.6`) | `1.88x` (`T=0`), `alone 12/12`, `serial 4/4` & `1/1`, `unequal: 0` |
| `tools/bench_concurrent.py` (`c=4`, `T=1.0` / `T=0`) | BF16 | `chat` (`agg`, `per-stream`) | `144.2` (`40.9`) / `171.6` (`44.0`) | `—` / `108.8` (`27.6`) | `1.58x` (`T=0`), `alone 12/12`, `serial 4/4` & `1/1`, `unequal: 0` |
| `tools/bench_concurrent.py` (`c=4`, `T=1.0` / `T=0`) | AWQ g128 | `code` (`agg`, `per-stream`) | `321.5` (`88.2`) / `356.4` (`91.3`) | `—` / `200.6` (`51.0`) | `1.78x` (`T=0`), `alone 12/12`, `serial 4/4` & `1/1`, `unequal: 0` |
| `tools/bench_concurrent.py` (`c=4`, `T=1.0` / `T=0`) | AWQ g128 | `chat` (`agg`, `per-stream`) | `251.7` (`67.0`) / `257.5` (`66.3`) | `—` / `198.8` (`51.0`) | `1.30x` (`T=0`), `alone 12/12`, `serial 4/4` & `1/1`, `unequal: 0` |
| `tools/bench_cache.py --require-cold` (`2,048` tok) | BF16 / AWQ | First `ttft_s` (4-run median) | `0.627s` (`0.609s`) / `0.830s` (`0.769s`) | `0.605s` (`0.597s`) / `0.788s` (`0.755s`) | `qualified: true`, `cached_tokens: 0` |
| `tools/bench_cache.py --require-cold` (`8,192` tok) | BF16 / AWQ | First `ttft_s` (4-run median) | `2.393s` (`2.377s`) / `2.976s` (`2.946s`) | `2.318s` (`2.337s`) / `2.957s` (`2.897s`) | `qualified: true`, `cached_tokens: 0` |
| `tools/bench_cache.py --require-cold` (`32,768` tok) | BF16 / AWQ | First `ttft_s` (4-run median) | `10.615s` (`10.587s`) / `12.959s` (`12.934s`) | `10.246s` (`10.209s`) / `12.561s` (`12.553s`) | `qualified: true`, `cached_tokens: 0` |

### Metal kernel contracts

- **Apple M5 Ultra (`applegpu_g17`, `tensor_units=true`):**
  - `--only sensenova --offline`: `24 kernels: 24 pass gates 1-4, 0 fail, 0 known findings; 45 controls: 45 red as planted, 0 missed; 0 limit problems`.
  - `--only qwen27`: `40 kernels: 40 pass gates 1-4, 0 fail, 0 known findings; 6 controls: 6 red as planted, 0 missed; 0 limit problems`.
- **Apple M3 Ultra (`applegpu_g15`, `tensor_units=false`, macOS `26.6.2`):**
  - `--only sensenova --offline`: `17 kernels: 17 pass gates 1-4, 0 fail, 0 known findings; 37 controls: 37 red as planted, 0 missed; 0 limit problems`.
  - `--only qwen27`: `30 kernels: 30 pass gates 1-4, 0 fail, 0 known findings; 3 controls: 3 red as planted, 0 missed; 0 limit problems`.
