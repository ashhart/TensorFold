# What's new in TensorFold

`tensorfold update` prints the sections below that are newer than the version you had. Each release's page on
GitHub has the full notes and the measurements behind them.

## 0.3.6 (in progress)

- **GLM-5.3-Flash on Macs.** GLM runs on Apple Silicon with drafted replies equal to serial ones, prompts prefill
  through the absorbed attention path, and tool calls parse in both servers. Thanks to @chadhurley25075-png (#9, #39)
  and @jeidbugs404 (#35).
- **Gemma 4 26B-A4B on the lanes,** exact at every width. Thanks to @cshintov (#10).
- **Memory back after long prompts.** The server hands MLX's freed buffers back when it goes idle. Thanks to
  @kingjamez (#44).
- **Flash Next on 128 GB Macs.** `--ple-on-ssd` reads its n-gram tables from disk instead of memory (#16).
- **Faster prompt processing.** Flash Next sizes its prompt chunks to the memory it has, and Nemotron takes prompts in
  chunks of up to 8,192 tokens on M5 GPUs. The weights stay wired in memory while a server runs.
- **MLX 0.32.2 or newer** is required on Macs. Earlier builds processed prompts more slowly.
- **What's new after an update.** `tensorfold update` now prints these notes when it finishes, and the first run of a
  new version prints a link to them.

## 0.3.5.1 (28 Sep 2026)

- Qwen3.8-27B loads on M1 and M2 Macs again. Kernels there fit Metal's per-kernel thread limit, with the same sums in
  the same order, so drafted output still equals serial output.
- M3, M4 and M5 run 0.3.5's machine code unchanged.
- Thanks to @hichaiuse, @simonmd, @gcarusso, @tonydehnke, @Cyb3r-Monk and @tinyapps for the reports and the repro.

## 0.3.5 (27 Sep 2026)

- **Concurrent requests share each verification round.** `--parallel auto` is on by default, and every stream's reply
  equals the same request served alone, on Metal and on CUDA.
- **Follow-up turns resume at the start of their newest messages,** with output identical to a fresh prompt. A 12-turn
  agent session with the 27B spent 14.9 s on first tokens instead of 31.9 s.
- **Memory that fits.** The whole process stays inside 70% of RAM, an omitted `--context` defaults to the window the
  machine can hold, and a prompt past it gets a clear 400.
- **Flash Next prefill** with sparse prompt attention, 1.1-1.4x faster than 0.3.4.1 on an M3 Ultra (@quigles1977, #29).
- **2- to 8-bit weights** on the lanes, so mixed-precision 27B checkpoints decode fully (@jasontitus, #34).
- **CUDA:** FP8 prefill and shared expert kernels, concurrent streams for the 27B and Flash Next, and admission from
  available memory before loading.
- **API:** raw `/v1/completions` prompts, `ignore_eos` and `stop`, request `reasoning_effort` and typed tool arguments
  (@chris247474, #28), `developer` messages, `parallel_tool_calls: false`, and cancellation when a client disconnects.

## 0.3.4.1 (27 Sep 2026)

- Prompt processing is back to MLX's speed on every model. Prompts prefill through MLX's own forward on a fixed
  2,048-token grid, so a resumed conversation still equals a fresh one byte for byte.
- Flash Next's peak memory stays within 20 GB of its weights up to a 196k-token prompt.

## 0.3.4 (26 Sep 2026, pre-release)

- Every model runs on the lane engine, and the serial engine is gone. Nemotron drafts on M1 to M4 with row-exact
  kernels.
- Qwen3.8-27B on M1 to M4 decodes at 1.9 to 4x serial, through a new 4-bit matmul on the simdgroup matrix units.
- CUDA: every default path verifies at least two rows a round.

## 0.3.3 (26 Sep 2026)

- Qwen3.8-27B verifies drafted tokens together on every M1 to M5 GPU, with output byte-identical to serial decoding.
- Streamed `/v1/completions` send plain text.

## 0.3.2 (26 Sep 2026)

- `tensorfold update` installs the newest release.
- GLM-5.3-Flash reads Mia-AiLab's EXL3 weights on two DGX Sparks (experimental).

## 0.3.1 (26 Sep 2026)

- `tensorfold info` shows how a checkpoint stores its weights and which backends read them. `serve` and `pull` refuse
  checkpoints no engine reads yet, before anything downloads.

## 0.3.0 (26 Sep 2026)

- NVIDIA GPUs: `tensorfold serve` picks CUDA on Linux and runs on one GPU or two (one rank per DGX Spark).
- A new family, GLM-5.3-Flash, on two Sparks.

## 0.2.0 (25 Sep 2026)

- A rewrite: `tensorfold serve`, `pull`, `models` and `info` for Nemotron 3.5 Lightning, Qwen3.8-27B and Qwen3.8
  Flash Next on Apple Silicon, with drafts that never change the output.
