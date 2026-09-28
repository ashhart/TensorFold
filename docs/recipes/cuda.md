# CUDA implementation

CUDA families read supported checkpoints through PyTorch loaders and execute family-specific Triton and
CUDA kernels. Use the [runbook](../../RUNBOOK.md#nvidia-gpus) for the container and two-rank setup.

| Family | CUDA execution |
| --- | --- |
| [Qwen3.8-27B](qwen3.8-27b.md#cuda) | One or two ranks, DFlash2 trees and context copies |
| [Flash Next](qwen3.8-flash-next.md#cuda) | One or two ranks, MTP chains and CUDA graphs |
| [Nemotron 3.5 Lightning](nemotron-3.5.md#cuda) | One or two ranks, MTP chains and CUDA graphs |
| [GLM-5.3-Flash](glm-5.3-flash.md#cuda) | Two ranks, MTP and optional DFlash2 |
| [Qwen3.6-35B-A3B](qwen3.6-moe.md#cuda-execution) | One rank, MTP chains and context copies, CUDA graphs |

An EXL3 checkpoint's trellis is read by one shared module for every family, any codebook (3inst, mcg, mul1)
and any width 1 to 8, mixed across a checkpoint and inside one MoE layer: `src/tensorfold/cuda/exl3/`. A family
whose CUDA engine reads it declares `EXL3_VARIANT = "any"`, and TensorFold checks the checkpoint before
downloading. The dense linear layer, its plan and its measured throughput are in [EXL3 weights](exl3.md);
`python -m tensorfold.cuda.exl3.inspect MODEL_DIR` prints what a checkpoint holds.

## Arithmetic and state

Each engine defines its own serial reference. A verify row uses the same group order, K split and
rounding as that row alone. Attention partitions depend on absolute key position; router ties use a
stable ID order. Recurrent commits replay the accepted path with the same update routine.

The default two-rank decode paths gather fp32 partials and add them in rank order. The dense Qwen
prefill path gathers bf16 partials and adds them in fp32. Rank 0 chooses the window and both ranks
execute the same forwards. Prefix token IDs must describe the state actually cached on each rank.
CUDA graphs replay the same kernels using stable buffers; changing capture shapes must preserve these rules.

## Shared kernels and prefill

`tensorfold/cuda/kernels/qmm.py` packs 4-bit weights for the shared CUDA matmul. Its decode kernel fixes
the K split by weight shape, while the prefill kernels use separate arithmetic. Dense Qwen prefill uses
FP8 activations in chunks of up to 4,096 tokens. It retains prompt-end states and prefills replies again
on a follow-up, because prefill and decode use different arithmetic.

`tensorfold/cuda/kernels/gdn.py` and `attention.py` support several streams in one call. Each stream
supplies its own tree, cache offsets and accepted path. `tensorfold/cuda/experts.py` groups routed
row/expert pairs so the MLX 4-bit formats of Flash Next, GLM and Nemotron share expert kernels, with
separate prefill and decode forms. A shared call must preserve each row's arithmetic and each stream's cache.

## Requests and memory

CUDA `--parallel auto` serves one request at a time. Set an explicit `--parallel N` above one for shared
Qwen3.8-27B rounds on one or two ranks, or Flash Next on one rank. Flash Next rejects this setting with
`--tp 2`; Nemotron, GLM and Qwen3.6 remain serialized. The shared scheduler admits requests between decode
rounds, then verifies each active stream's drafts together and commits each stream independently. On the 27B,
a new prompt prefills 1,024 tokens a round while the other streams keep decoding, and its state is kept at
message starts (the second message and the last assistant turn), so prompts that share a system prompt or
extend a conversation resume there with a fresh prefill's bits.

Cache capacity is fixed at startup and bounds prompt plus reply.
A positive context that exceeds the startup budget is refused; automatic capacity is an estimate.
Unified-memory GPUs share physical RAM with host buffers and file-backed model data. Admission uses
available host memory, including reclaimable page cache, and considers mapped-table residency when sizing
an automatic window. It accounts for stream count and retained caches where concurrency is enabled.

Two-rank Flash Next, Nemotron and GLM requests finish on both ranks after a client disconnects, keeping the
collective sequence aligned. MLX disk snapshots and cache-budget flags do not configure these CUDA
caches. The CUDA CLI also does not apply `--alias`, `--thinking-budget` or `--reasoning-effort`; use `--name` for the served model ID and
`--thinking` or request `chat_template_kwargs.enable_thinking` for the chat template.

## Measuring

Use the [public benchmark command](README.md#measurements), the same client and fixtures for each engine,
and record runtime and checkpoint revisions. Decode, prefill, memory and comparative speed for 0.3.5 are
TBD [release-0.3.5].

Compare exact output separately from throughput. Test long prompts as well as short fixtures and compare
resumed requests with fresh ones. When profiling NCCL, exclude annotation events from summed GPU time to
avoid counting a collective twice. Record system memory pressure alongside GPU timing on unified-memory
systems rather than inferring a kernel regression from one slow run.

The [CUDA family guide](adding-a-cuda-family.md) specifies the interface and required checks.
