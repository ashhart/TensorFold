# Zig engine preview

This branch is an early look at TensorFold's native engine. It's written in Zig and drives Metal (and CUDA) directly, with no Python or MLX in the decode path. It isn't a release. Expect rough edges, and please report what you find or send a pull request.

The rule the whole engine is built around: drafted output is byte-for-byte identical to decoding one token at a time. Guessed tokens, trees of guesses and copies from the context are all checked together in one forward ("lanes"). Only tokens the model would have produced itself are kept.

## Models tested so far

| Model | Backend | Status |
|---|---|---|
| Nemotron 3.5 Lightning 30B-A3B, MLX 4-bit + MTP head ([TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit](https://huggingface.co/TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit)) | Metal, Apple M5 | Served through the OpenAI-compatible server. Exact against the Python engine on the same chip. Keeps each conversation's prompt state between requests, so a later turn reads only its new tokens. Measured below. |
| Qwen 3.8 Flash Next, mlx-q6g32 | Metal | Served through the same server, one reply at a time, with prompt reuse between requests. Kernels and packs come from TF_FLASHNEXT_DUMP, a folder made by tools/zig/flashnext_dump.py and its helpers. Also served from two Macs at once in [speed-up mode](docs/speed-up-mode.md). |
| Nemotron 3.5 Lightning 30B-A3B | CUDA, NVIDIA GB10 (DGX Spark) | `tensorfold run` from the command line. Token-identical to the Python engine on 16 of 16 runs. Speed about level with the Python engine. Not wired into the server yet. |
| Qwen3.5-2B, pinned MLX affine 4-bit/group-64 checkpoint | Metal, Apple M5 Max | Text generation through the native server and lane core, with shared tied embeddings. [Recipe, scope and verification](docs/recipes/qwen3.5-2b-native.md). M1–M4 untested for this model. |
| Kimi K3 | Metal, several Macs over Thunderbolt | Research code for multi-Mac tensor parallelism (`zig/src/families/kimi_k3`, `zig/src/cluster`). Not ready for testing. |

## Speed so far

On an M5 Max, Nemotron 3.5 Lightning 4-bit, one greedy stream, output identical to plain decoding:

| Engine | Essay | Story | Code | Edit a file | Rename across a file |
|---|---|---|---|---|---|
| Zig engine (this branch) | 277 tok/s | 244 | 357 | 459 | 581 |
| TensorFold Python engine | ~220 | | | | |
| mlx_lm server (greedy prose) | ~175 | | | | |

- Against TensorFold's own Python engine, one stream is about 1.2-1.3x faster. That's 277 against 220 on an essay, and 217 against 184 averaged over 64 story and essay prompts.
- Against mlx_lm's server it's about 1.6x on prose.
- Edits and renames are fastest because copies from the context fill many lanes per round.

Concurrent sessions on the same machine:

| Sessions | Zig engine, total tok/s | Python engine, total tok/s |
|---|---|---|
| 8 | 384 | 388 |
| 16 | 449 | 458 |
| 32 | 509 | 502 |
| 64 | 502 | 607 |

The two engines are level up to 32 sessions. Past 32 the Zig engine stops scaling, because a forward holds at most 32 rows (see below).

## Build

You need Zig 0.17.0 and Xcode's Metal toolchain.

```bash
zig build native -Dcpu=apple_m1
```

That writes `zig-out/native/bin/tensorfold-native`. `zig build test` runs the host-side unit tests. `zig build` with no target also builds the command-line `tensorfold run` and the test tools.

## Run

```bash
hf download TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit --local-dir ~/models/nemotron-lightning
zig-out/native/bin/tensorfold-native serve ~/models/nemotron-lightning --name nemotron --port 8090 --temperature 0 --no-thinking
```

Then point any OpenAI-compatible client at `http://127.0.0.1:8090/v1`. Add `--parallel 32` for up to 32 sessions at once. `--no-drafts` turns the lanes off, which is the reference for any exactness check. Nemotron keeps each conversation's prompt state between requests, so a later turn reads only its new tokens; the memory for those states is sized as for Flash Next below, and `--prompt-cache-gib 0` turns it off.

Flash Next is the same `serve` command on a qwen4_exp checkpoint in mlx-q6g32. Set `TF_FLASHNEXT_DUMP` to the dump folder; [the speed-up mode guide](docs/speed-up-mode.md) shows how to make one. The host has one lane, so `--parallel` does not run two Flash Next replies at once. The memory kept for earlier prompts' states defaults to what 70% of RAM leaves past the loaded server, less 2 GiB; `--prompt-cache-gib` sets less, and `0` turns it off.

To serve Flash Next from two Macs at once, each holding the whole model, see [speed-up mode](docs/speed-up-mode.md).

## Known gaps

- `--learn` keeps shared prompt prefixes on disk for GLM only. Nemotron and Flash Next keep theirs in memory, so a restart forgets them.
- On M1 to M4, chips without tensor units, prompt kernels use the simdgroup-matrix layout and Nemotron's window attention is rewritten to it. Both are checked at load. Dense projections and routed experts are already proven row-exact there. The Mamba tree conv/scan and the norms are still open.
- A forward holds at most 32 rows.
- The native server serves Nemotron 3.5 Lightning, Qwen 3.8 Flash Next, GLM-5.3-Flash and the Qwen3.5-2B checkpoint in its recipe. Flash Next takes one reply at a time.

## Where the work goes next, and where you can help

1. **Nemotron's prompt states on disk, and a pool for them** (`zig/src/families/nemotron/snapshot.zig`). Nemotron keeps conversation states between requests now, copied on the GPU. Next: `--learn` for its shared prefixes, as GLM has, and a pool of kept buffers, as Flash Next has, so a long-running server does not allocate one per state.
2. **Exactness on M1 to M4** (`zig/kernels/metal`). Prompt kernels and Nemotron window attention already use the simdgroup-matrix layout and are checked at load. The remaining kernels still need the per-kernel sweep: one row alone against the same row inside a 2-, 3- and 8-row window, then fix the kernel whose bits move.
3. **More than 32 rows per forward** (`zig/src/native/metal.zig`, `batch_rows`). Lifting the cap lets 64+ sessions scale, and lets one stream run wider windows.
4. **Cheaper extra lanes** (`zig/kernels/metal`). Past 16 lanes the routed-expert kernel is limited by arithmetic, not memory. A round of 8 lanes costs 2.2x a round of one, and 32 lanes cost 6.2x. Flattening that curve speeds up both one stream and many sessions.
5. **New model families** (`zig/src/families`). Qwen 3.8 Flash Next is served on Metal. Each new family follows the Nemotron layout: a weight loader, kernels checked op by op against the Python engine, a full forward whose tokens match it, then the lanes. [Adding a Zig family](docs/recipes/adding-a-zig-family.md) has the steps in order, what each one gained, and which code owns it.
6. **The CUDA backend** (`zig/src/cuda`, `zig/build/cuda.zig`). Nemotron is exact on GB10 from the command line. It needs the server wiring and more families.
7. **One-pass drafting** (experimental, `--block-lanes`). The draft head can fill every lane in one pass instead of level by level. The engine side is in, but the placeholder lanes need trained rows before they land tokens.

House rules for pull requests:
- Output must stay identical to `--no-drafts` on every prompt.
- No precision trades: bf16 activations, fp32 accumulation.
- No file over about 600 lines, and one-line comments.
- `zig build test` passes.

## Reporting

Open an issue with your chip, macOS version, the exact command and the server's `done req-...` log lines. If drafted output ever differs from `--no-drafts` output on an M5, that's the most valuable report you can send.
