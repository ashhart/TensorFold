# Experimental DeepSeek-V4.1 CUDA asset preparation

The native engine remains experimental. CPU build and host-test results do not qualify a checkpoint or GPU.
Serving still applies the native checkpoint representation and architecture refusals.

Build the CPU generators before preparing the CUDA images:

```sh
zig build dsv41-assets
zig build test-dsv41-assets-host test-dsv41-token-map
zig-out/bin/tf-dsv41-token-map --tokenizer MODEL/tokenizer.json --config MODEL/config.json --output ASSETS/engram-token-map.bin
zig-out/bin/tf-dsv41-assets aot-needs ASSETS/sigs.json ASSETS/needs.json
```

The token-map generator validates release geometry and records the original tokenizer and config SHA256 values.
Its metadata reader accepts the official unquantized configuration for asset generation; this does not admit that
matrix representation to the serving loader. The file uses `DSENGR01`, including prime buckets, multipliers and
lookup ids, with a JSON provenance sidecar. The legacy `engram-host.bin` fixture format is not a serving asset.

`aot-needs` executes the family's actual symbolic planner on the CPU. Supply the intended rank, world, context,
slot, draft, sampling, prefill and image settings through the same environment used for serving. It never opens a
CUDA driver. Enumerating an image or two-rank plan does not execute a GPU or distributed model.

The external Triton reference tools produce `sigs.json`, compile every specialization in `needs.json` for the
actual target SM, and write `aot/`. The reference RoPE generator writes `rope.bin` using fixed CPU table blocks;
its rows must cover context plus 2048. Keep the reference revision, compiler versions, options, manifests and
hashes with these generated assets. Do not substitute captured images from another checkpoint or profile.
The recipe's reference tools are versioned separately from this native runtime.

Build installable ARM64 binaries on the native ARM64 host with real runtime and family CUDA images:

```sh
zig build dsv41-serve -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Ddsv41-nvcc=/usr/local/cuda/bin/nvcc -Dsm=121
```

This installs rank zero's `tensorfold-dsv41` and the follower tool `tf-dsv41`. The runtime reads
`TF_DSV41_ASSETS/{rope.bin,engram-token-map.bin,aot/}`. Compile-only targets use inert synthetic images and
cannot produce a serving qualification receipt. Keep generated images and compiler caches outside source control.


## Repeat-divergence diagnostics

Set `TF_DSV41_LAYER_HASH=PREFIX` on every rank to retain the existing per-layer trace hashes.
Use that setting when generating AOT needs too; the CPU planner emits the traced variants without opening a hash file.
Each process creates a new `PREFIX.rankN.jsonl` file and refuses an existing file. Use a fresh prefix per run.
Unset or empty attaches no writer and adds no trace launches, device copies or synchronizations.

JSONL version 1 starts with a header. Request records carry a sequence, slot and SHA-256 of the full prompt's
little-endian u32 token IDs. Layer records add the logical window, phase (`prefill`, `replay`, `decode`), start,
rows, row offset, columns, layer, window-token SHA-256 and SHA-256 of the post-only BF16 streams.
Rank zero shares request metadata over the plan link before executing it. Sequence zero marks boot/calibration.
Mixed windows are recorded per slot; padding is excluded. Compare the same rank, request and window shape across runs.
A resumed prompt omits cached-prefix layers, so missing records alone do not show an arithmetic mismatch.

Hashing synchronizes and reads back the producing stream at every layer. Traced programs run eagerly, including
batched windows, whatever `TF_DSV41_GRAPHS` asks for. This can hide a timing race
and has unmeasured overhead. Check traced/untraced tokens separately; equal traced runs do not qualify graph replay,
cache resume, RoCE ordering or a checkpoint. Keep both ranks' logs and exact request/output IDs with the receipts.

Compare hashes on and off with `TF_DSV41_GRAPHS=0` in both arms, and keep a separate graphs-on arm without hashes.
The eager hash path cannot reproduce a failure confined to graph replay.
