# Recipe book

Each family page describes its supported checkpoint, kernels and operating limits.

| Family | Recipe |
| --- | --- |
| Nemotron 3.5 Lightning | [MLX](nemotron-3.5.md), [Intel Arc, experimental](xpu-arc.md) |
| Qwen3.8-27B | [MLX, quantization and CUDA](qwen3.8-27b.md) |
| Qwen3.5-2B | [Native Metal preview, tied affine 4-bit](qwen3.5-2b-native.md) |
| Qwen3.8 Flash Next | [MLX prefill and CUDA](qwen3.8-flash-next.md), [CUDA images](flash-next-vision.md) |
| Ternary Bonsai 2 27B | [MLX](ternary-bonsai-2.md) |
| GLM-5.3-Flash | [MLX on a 256 GB Mac, two-rank CUDA](glm-5.3-flash.md) |
| Gemma 4 26B-A4B | [MLX, fused one-row decode](gemma-4.md) |
| DeepSeek-V4-Flash | [MLX on a 256 GB Mac, DSpark and MTP drafts](deepseek-v4-flash.md) |
| Qwen3.6-35B-A3B | [MLX with MTP drafts, one-GPU CUDA](qwen3.6-moe.md) |

## Capability floor

Quoted from the family pages, not measured per card.

| Engine | Floor |
| --- | --- |
| CUDA, other families | Compute capability 8.9 or newer: Ada RTX 40, Hopper, and Blackwell cards (RTX 50, RTX PRO 6000, DGX Spark GB10). RTX 30 (8.6) is not supported yet. |
| Flash Next, CUDA | sm_120 and sm_121 only: DGX Spark GB10, RTX 50, RTX PRO 6000. A card below sm_120 refuses Flash Next at startup. |
| Intel GPU (XPU), Nemotron | Experimental. Weights need 17.3 GiB of device memory (18 to 19.4 GB in use from 4k to 128k context); measured on an Arc Pro B70 (32 GB). Cards below 24 GB are untested and a 16 GB card cannot hold the weights. |
| MLX, Apple Silicon | GLM-5.3-Flash is written for a 256 GB Mac, about 151 GiB resident. DeepSeek-V4-Flash is the same, about 151 GiB resident. Flash Next's default command sizes to a 128 GiB M4 Max. Qwen3.8-27B on a 32 GB Mac needs more than the default 22.4 GiB. Machine classes, not measured minimums. |

New families go on the Zig engine: [adding a Zig family](adding-a-zig-family.md) has the steps in order, what each
one gained, and which code owns it. For the Python engine, frozen at 0.6.5, the contributor guides cover
[adding an MLX family](adding-a-family.md), [adding a CUDA family](adding-a-cuda-family.md) and
[CUDA implementation rules](cuda.md).
[EXL3 weights](exl3.md) and [universal EXL3 experts](exl3-universal-experts.md) describe the shared EXL3
module every CUDA family can read: any codebook, any width per tensor, one grouped launch per MoE projection.

## The contract

Drafted output must equal the same engine's serial output. A resumed prompt must equal that prompt served
fresh, and every concurrent stream must equal its solo run. This contract applies within the same weights,
runtime and settings; another backend or quantization can have different arithmetic.

The sampler keys each draw by seed, absolute position and token ID. A draft is accepted only when it
matches that draw from the target. Each verify row must use the serial row's arithmetic, including
attention boundaries, routing ties and recurrence updates. Rollback must restore all state after the
accepted path, including draft-head state.

Prompt reuse is equally strict. The MLX planner computes chunks from rendered tokens and caches only
compatible boundaries. Its resume points are assistant-message starts and the second message start
when the tokenizer exposes those markers. Chunks skip resume points less than 256 tokens from the
previous start and otherwise end at the first eligible point or after 2,048 tokens. Without recognized
markers, chunks use the 2,048-token grid. Rewriting earlier template text can invalidate reuse.
The chunk scheme and kernel/runtime identity must match before a stored prefix can be reused.

## Measurements

`tools/bench_openai.py` contains public fixtures: a raw Fibonacci-function prompt and a chat prompt asking
how matrix multiplication uses a GPU. The chat fixture disables thinking. After starting a server with
`--name bench`, run from the repository root:

```bash
python3 tools/bench_openai.py http://127.0.0.1:8080 bench \
  --tokens 64 --reps 5 --temperatures 1.0,0 --output bench.json
```

The client warms each cell, uses seeds 1234 through 1238, and reports the median decode rate after the
first token. Sampled cells use temperature 1, top-k 20 and top-p 0.95. The client sends `ignore_eos: true`;
MLX, GLM CUDA and the Qwen3.8-27B and Qwen3.6 CUDA engines honor it, while the Flash Next and Nemotron CUDA engines can stop
at EOS before the requested limit.
It measures throughput; it does not itself prove token equality. Record checkpoint and tokenizer revisions,
runtime versions, backend, rank count, launch command and output hashes with a result. Compare serial and drafted output separately.

For concurrent MLX workloads, use `tools/bench_concurrent.py` with `--alone --serial` to compare each
concurrent reply with a solo run and check those solo runs against `draft: false`. Long-context and memory
measurements also need a public prompt fixture and its exact rendered-token count. Separate cold, resumed
and post-restart requests.

All 0.3.5 decode, prefill, concurrency and peak-memory results are TBD [release-0.3.5]. Older public-fixture
results in the CUDA family pages are historical measurements, not release qualification.

## Kernel checks

Use exact equality for serial versus multi-row kernels and committed caches. Use a separate trusted
forward for model-quality checks; agreement with one's own serial implementation does not establish
model fidelity. Include real head dimensions, sparse-attention boundaries, partial keeps and concurrent
streams at different lengths. After changing a kernel or runtime, regenerate serial references.

Measure dependent operations and complete requests. Independent microbenchmarks can hide launch cost,
weight-cache effects and lost overlap with other work.
