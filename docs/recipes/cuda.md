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

## Checkpoints

On CUDA, TensorFold serves NVFP4 and EXL3 checkpoints, usually the ones Mia-AiLab's DGX Spark recipes run or your
own exports, and MLX 4-bit checkpoints as the portable option: the same files a Mac serves. `tensorfold serve` loads
the checkpoint you name; it picks none by itself.

| Family | NVFP4 | EXL3 | MLX 4-bit |
| --- | --- | --- | --- |
| Qwen3.8-27B | `nvidia/Qwen3.8-27B-NVFP4`, one rank | `turboderp/Qwen3.8-27B-exl3`, one rank | one or two ranks |
| Flash Next | `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` (a mirror of local-inference-lab's), `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, `RadixArk/Qwen3.8-Flash-Next-NVFP4`, one rank | `turboderp/Qwen3.8-Flash-Next-exl3`, one rank | one or two ranks |
| GLM-5.3-Flash | not read | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, two ranks (experimental) | two ranks |
| Qwen3.6-35B-A3B | not read yet | not read yet | one rank |
| Nemotron 3.5 Lightning | not read yet | not read yet | one or two ranks |

Mia-AiLab's checkpoints on Hugging Face (30 Sep 2026):
- Loaded and served here: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (two Sparks) and
  `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` (found by its `model_type`, `qwen3_8_flash_next`).
- Not tried yet: `Mia-AiLab/Qwen3.8-27B-EXL3`, `Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw`,
  `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`, `Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` (a drafter),
  `Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw` and `-3.0bpw` (DeepSeek-V4 has no CUDA engine yet).
- Not readable: the GGUF repositories (`Qwable-3.6-27b`, `Qwable-3.6-27b-MTP`, `Qwable-3.6-35b`,
  `Gemmable-4-12B-MTP-GGUF`, `Gemmable-4-31B-MTP-GGUF`); TensorFold reads no GGUF.

Prompts take bf16 activations by default. What that costs against the FP8 prompt path (`--prefill-fp8`), by
format ([prompt precision](#prompt-precision)):
- EXL3, every family: nothing; EXL3 prompts never took FP8 activations.
- NVFP4: about level on Flash Next (0.94-1.03x from 2k to 64k on local-inference-lab's export; RadixArk's has no
  FP8 prompt kernel, so nothing changes); the 27B's NVFP4 export is not measured yet.
- MLX 4-bit: 0.73-0.82x on the 27B and 0.90-0.96x on Qwen3.6 from 2k to 128k; `--prefill-fp8` gives that speed
  back at FP8's precision. Flash Next's, GLM's and Nemotron's MLX 4-bit prompts were already bf16, so nothing
  changes for them.

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
the K split by weight shape, while the prompt kernel (`qmm_prefill.cu`) rounds each weight once to bf16 and adds
every product over K in one fp32 chain, the arithmetic of MLX's prompt matmul. NVFP4, FP8 and MXFP8 weights take
`nvfp4/prompt.cu`, where each weight is exact in bf16. Prompts run in chunks of up to 4,096 tokens with bf16
activations, as decode does. A row's bits never depend on its chunk or the kernel's tile, so a resumed prompt
equals a fresh one; they differ from decode's, so the engines retain prompt-end states and prefill replies
again on a follow-up.

`tensorfold/cuda/kernels/gdn.py` and `attention.py` support several streams in one call. Each stream
supplies its own tree, cache offsets and accepted path. `tensorfold/cuda/experts.py` groups routed
row/expert pairs so the MLX 4-bit formats of Flash Next, GLM and Nemotron share expert kernels, with
separate prefill and decode forms. A shared call must preserve each row's arithmetic and each stream's cache.

<a id="prompt-precision"></a>

### Prompt precision

Prompt matmuls take bf16 activations by default. `--prefill-fp8` switches the ones that have an FP8 kernel (the
27B's and Qwen3.6's MLX 4-bit projections, and the FP8, NVFP4 and MXFP8 layers of NVFP4 checkpoints) to e4m3
activations with one scale a row, the arithmetic of TensorFold 0.5.0's prompts; `--no-prefill-fp8` asks for bf16 by
name. e4m3 keeps 3 mantissa bits to bf16's 7, and one scale a row loses a row's small values when one of its channels
is large. Either way a drafted reply equals the same server's serial one and a resumed prompt equals a fresh one;
only the prompt's own arithmetic changes.

Quality over 8 sequences of 4,096 tokens (wikitext-2, CPython source, chats), every position scored. The reference
is an fp32 forward from the checkpoint (every activation in fp32, the 4-bit weights dequantized exactly); Flash Next
has no fp32 forward, so its reference is the engine's own decode path. KL is KL(reference || prompt path) over the
vocabulary, top-1 the share of positions whose likeliest token matches the reference's.

| Model | Prompt rows: KL mean, top-1 | The reply after a 3,072-token prompt: KL mean, top-1 | PPL on wikitext / code |
| --- | --- | --- | --- |
| Qwen3.8-27B MLX 4-bit, bf16 | 0.0031, 99.2% | 0.0037, 99.4% | +0.20% / +0.03% |
| Qwen3.8-27B MLX 4-bit, FP8 | 0.0624, 93.7% | 0.0156, 98.0% | +1.30% / +4.10% |
| Qwen3.6-35B-A3B MLX 4-bit, bf16 | 0.0043, 98.2% | 0.0035, 98.1% | +0.04% / -0.31% |
| Qwen3.6-35B-A3B MLX 4-bit, FP8 | 0.0289, 94.8% | 0.0071, 97.4% | +0.68% / +1.82% |
| Flash Next NVFP4 (against decode), bf16 | 0.0081, 97.9% | | +0.17% overall |
| Flash Next NVFP4 (against decode), FP8 | 0.0167, 96.9% | | +0.39% overall |

The engine's decode path lands at 0.0023 (27B) and 0.0045 (Qwen3.6) on the same reference, so bf16 prompts sit at
decode's level. Rounding each 4-bit weight to bf16 in the prompt matmul, as MLX does, measures the same as exact
weights.

Cold prefill on one DGX Spark (GB10), served, prompts of Python standard-library code with a unique first line,
median of two, tok/s. vLLM's numbers come from one DGX Spark too, on NVIDIA's NVFP4 checkpoints of these models,
whose matmuls take FP4 activations (FP8 on their FP8 layers):

| Model | Prompt | 2k | 8k | 16k | 32k | 64k | 128k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B MLX 4-bit | bf16 | 1,350 | 1,418 | 1,379 | 1,298 | 1,164 | 965 |
| | FP8 | 1,842 | 1,950 | 1,878 | 1,728 | 1,498 | 1,181 |
| | vLLM, NVFP4 checkpoint | 2,253 | 1,969 | 1,683 | 1,300-1,428 | 897-1,244 | 999 |
| Qwen3.6-35B-A3B MLX 4-bit | bf16 | 7,260 | 7,658 | 6,910 | 6,311 | 5,102 | 3,675 |
| | FP8 | 7,766 | 8,208 | 7,715 | 6,735 | 5,376 | 3,819 |
| | vLLM, NVFP4 checkpoint | 5,907 | 5,881 | 5,090 | 3,951 | 2,693 | |

On Flash Next's NVFP4 checkpoint (MXFP8 attention and DeltaNet layers), bf16 prompts run 0.94-1.03x the FP8 ones
from 2k to 64k. Most of the 27B's prompt time is matmuls, four fifths of a chunk at short context. At bf16 they run at
about 88 TFLOPS, 84% of the GB10's practical bf16 rate, against about 130 for the FP8 kernel.

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
caches. The CUDA CLI also does not apply `--alias`; use `--name` for the served model ID. `--thinking`,
`--reasoning-effort` and `--thinking-budget` set the defaults a request's `chat_template_kwargs.enable_thinking`,
`reasoning_effort` and `thinking_budget` override, as on the Mac.

## Measuring

Use the [public benchmark command](README.md#measurements), the same client and fixtures for each engine,
and record runtime and checkpoint revisions. Decode, prefill, memory and comparative speed for 0.3.5 are
TBD [release-0.3.5].

Compare exact output separately from throughput. Test long prompts as well as short fixtures and compare
resumed requests with fresh ones. When profiling NCCL, exclude annotation events from summed GPU time to
avoid counting a collective twice. Record system memory pressure alongside GPU timing on unified-memory
systems rather than inferring a kernel regression from one slow run.

The [CUDA family guide](adding-a-cuda-family.md) specifies the interface and required checks.
