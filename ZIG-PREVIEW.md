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

The two engines were level up to 32 sessions, and past 32 the Zig engine stopped scaling, because a forward held at most 32 rows. A shared round now holds four rows a lane, 64 at least and 128 at most, so 32 and 64 drafting sessions keep rows for their drafts. On an M5 Ultra with `--parallel 64`, 64 sessions went from 795 to 1,006 tok/s in total on code and from 753 to 847 on chat, 32 sessions from 795 to 1,019 and 755 to 843, every reply equal to its solo run (the table above is an earlier measurement on another machine).

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

Flash Next is the same `serve` command on a qwen4_exp checkpoint in mlx-q6g32. Set `TF_FLASHNEXT_DUMP` to the dump folder; [the speed-up mode guide](docs/speed-up-mode.md) shows how to make one. The host has one lane, so `--parallel` does not run two Flash Next replies at once. The memory kept for earlier prompts' states defaults to what 70% of RAM leaves past the loaded server, less 2 GiB; `--prompt-cache-gib` sets less, and `0` turns it off. `--learn` keeps shared prefixes (a system prompt and its tools) on disk as well, under `~/.cache/tensorfold/learned` by default, keyed by the checkpoint, the kernel sources, the chip and OS build and a startup probe's bits, so a fresh server, or an upgrade that computes the same bits, resumes them without reading them again. On one Mac for now: a speed-up pair refuses it.

To serve Flash Next from two Macs at once, each holding the whole model, see [speed-up mode](docs/speed-up-mode.md).

## Known gaps

- `--learn` keeps shared prompt prefixes on disk for GLM, Nemotron and Flash Next on one Mac. A Flash Next speed-up pair keeps them in memory only.
- On M1 to M4, chips without tensor units, prompt kernels use the simdgroup-matrix layout and Nemotron's window attention is rewritten to it. Both are checked at load. Dense projections and routed experts are already proven row-exact there. The Mamba tree conv/scan and the norms are still open.
- A shared forward holds at most 128 rows (64 up to `--parallel 16`), and a lone stream's window at most 64.
- The native server serves Nemotron 3.5 Lightning, Qwen 3.8 Flash Next, GLM-5.3-Flash and the Qwen3.5-2B checkpoint in its recipe. Flash Next takes one reply at a time.

## Where the work goes next, and where you can help

1. **`--learn` for Flash Next** (`zig/src/native/flashnext_host.zig`). GLM and Nemotron keep shared prompt prefixes on disk and resume them after a restart; Flash Next keeps its states in memory only. The prompt cache's `write`, `read` and `forget` functions over its `snapshot.zig` do it, with an identity made as `glm_host.zig` and `native/metal.zig` make theirs.
2. **Exactness on M1 to M4** (`zig/kernels/metal`). Prompt kernels and Nemotron window attention already use the simdgroup-matrix layout and are checked at load. The remaining kernels still need the per-kernel sweep: one row alone against the same row inside a 2-, 3- and 8-row window, then fix the kernel whose bits move.
3. **More than 128 rows per forward** (`zig/src/native/metal.zig`, `batch_rows`; `zig/src/families/nemotron/state.zig`, `max_rows`). A shared round holds four rows a lane up to 128 now, which is what 64 sessions need. Past that, each row costs a Mamba state slot (47 MiB across the model's 23 Mamba layers) and the expert kernels' cost per row keeps climbing (item 4), so 128+ sessions, and wider windows for one stream, need both looked at together.
4. **Cheaper extra lanes** (`zig/kernels/metal/nemotron_experts.metal`). Past 16 lanes the routed-expert kernel falls below the memory line: it streams unique expert weights at about 1 TB/s up to 8 rows, 730 GB/s at 32 and 540 GB/s at 64. `tf_experts_w` (a 16-group's weight words as the outer loop, three members a pass, four simdgroups) took 6% off at 32 rows and 10% at 64, bit for bit. The rest of the gap is per-member work that the weight stream does not hide; a change that keeps each lane's summation order is allowed, and `tf-nemotron-experts` bit-checks and times a candidate at every width.
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
