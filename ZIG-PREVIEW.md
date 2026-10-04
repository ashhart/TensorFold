# Zig engine preview

This branch is an early look at TensorFold's native engine. It's written in Zig and drives Metal (and CUDA) directly, with no Python or MLX in the decode path. It isn't a release. Expect rough edges, and please report what you find or send a pull request.

The rule the whole engine is built around: drafted output is byte-for-byte identical to decoding one token at a time. Guessed tokens, trees of guesses and copies from the context are all checked together in one forward ("lanes"). Only tokens the model would have produced itself are kept.

## Models tested so far

| Model | Backend | Status |
|---|---|---|
| Nemotron 3.5 Lightning 30B-A3B, MLX 4-bit + MTP head ([TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit](https://huggingface.co/TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit)) | Metal, Apple M5 | Served through the OpenAI-compatible server. Exact against the Python engine on the same chip. Measured below. |
| Nemotron 3.5 Lightning 30B-A3B | CUDA, NVIDIA GB10 (DGX Spark) | `tensorfold run` from the command line. Token-identical to the Python engine on 16 of 16 runs. Speed about level with the Python engine. Not wired into the server yet. |
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

Then point any OpenAI-compatible client at `http://127.0.0.1:8090/v1`. Add `--parallel 32` for up to 32 sessions at once. `--no-drafts` turns the lanes off, which is the reference for any exactness check.

## Known gaps

- No prompt reuse between turns yet. Each turn of a long chat reads the whole conversation again.
- Not exact on M1 to M4 yet. On chips without tensor units, drafted output can differ from plain output, so use an M5 for now. The dense projections and routed experts are already proven row-exact there. The open suspects are the attention kernels sized per window, the Mamba tree conv/scan, and the norms.
- A forward holds at most 32 rows.
- Nemotron 3.5 Lightning is the only model served.

## Where the work goes next, and where you can help

1. **Prompt reuse across turns** (`zig/src/server`, `zig/src/native`). Keep the conversation's state between requests, so a new turn only reads its new tokens. This is the biggest win for agent and chat clients.
2. **Exactness on M1 to M4** (`zig/kernels/metal`). Each kernel must give a row the same bits however many rows share the forward. The method is a per-kernel sweep: one row alone against the same row inside a 2-, 3- and 8-row window, then fix the kernel whose bits move.
3. **More than 32 rows per forward** (`zig/src/native/metal.zig`, `batch_rows`). Lifting the cap lets 64+ sessions scale, and lets one stream run wider windows.
4. **Cheaper extra lanes** (`zig/kernels/metal`). Past 16 lanes the routed-expert kernel is limited by arithmetic, not memory. A round of 8 lanes costs 2.2x a round of one, and 32 lanes cost 6.2x. Flattening that curve speeds up both one stream and many sessions.
5. **New model families** (`zig/src/families`). Qwen 3.8 Flash Next is being ported now. Each family follows the Nemotron layout: a weight loader, kernels checked op by op against the Python engine, a full forward whose tokens match it, then the lanes.
6. **The CUDA backend** (`zig/src/cuda`, `zig/build/cuda.zig`). Nemotron is exact on GB10 from the command line. It needs the server wiring and more families.
7. **One-pass drafting** (experimental, `--block-lanes`). The draft head can fill every lane in one pass instead of level by level. The engine side is in, but the placeholder lanes need trained rows before they land tokens.

House rules for pull requests:
- Output must stay identical to `--no-drafts` on every prompt.
- No precision trades: bf16 activations, fp32 accumulation.
- No file over about 600 lines, and one-line comments.
- `zig build test` passes.

## Reporting

Open an issue with your chip, macOS version, the exact command and the server's `done req-...` log lines. If drafted output ever differs from `--no-drafts` output on an M5, that's the most valuable report you can send.
