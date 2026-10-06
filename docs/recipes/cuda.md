# CUDA implementation

CUDA families read supported checkpoints through PyTorch loaders and execute family-specific Triton and
CUDA kernels. Use the [runbook](../../RUNBOOK.md#nvidia-gpus) for the container and two-rank setup.

Native Windows is experimental, one GPU a process and not yet run on Windows hardware: see the
[runbook](../../RUNBOOK.md#windows-with-an-nvidia-card).

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
| GLM-5.3-Flash | not read | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (Brandon M. Music's; re-hosted as `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`), two ranks (experimental) | two ranks |
| Qwen3.6-35B-A3B | not read yet | not read yet | one rank |
| Nemotron 3.5 Lightning | not read yet | not read yet | one or two ranks |

Mia-AiLab's checkpoints on Hugging Face (30 Sep 2026):
- Loaded and served here: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (two Sparks; a byte-identical re-host of Brandon M.
  Music's `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, under his ShapleyMCG License 1.0) and
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
  FP8 prompt kernel, so nothing changes). The 27B's NVFP4 export runs its prompts in its own math by default
  ([NVFP4 precision](#nvfp4-precision)): 6,296-7,484 tok/s from 2k to 32k on an RTX PRO 6000, 0.95-0.97x vLLM's;
  at `--precision full` its bf16 prompts run 3,005-3,267 tok/s.
- MLX 4-bit: 0.73-0.82x on the 27B and 0.90-0.96x on Qwen3.6 from 2k to 128k; `--prefill-fp8` gives that speed
  back at FP8's precision. Flash Next's, GLM's and Nemotron's MLX 4-bit prompts were already bf16, so nothing
  changes for them.

## Native prompt controls

The native `tensorfold run` command accepts `--device N` to choose a CUDA ordinal. If
the flag is absent, it uses `TF_CUDA_DEVICE`, then device 0. `--segments N` enables
staggered prompt work for 1 through 4 whole 2,048-row chunks; without the flag it
uses `TF_CUDA_SEGMENTS`, then 1. The `tensorfold segments` command compares the
available segment counts and can profile the serial chunk parts.

CUDA capture and packing use the repository's Triton manifest tool. It records
Triton and extension launches, maps them to cached kernels and metadata, and
checks the packed manifest on CPU before a CUDA run.

## Arithmetic and state

Each engine defines its own serial reference. A verify row uses the same group order, K split and
rounding as that row alone. Attention partitions depend on absolute key position; router ties use a
stable ID order. Recurrent commits replay the accepted path with the same update routine.

A two-rank engine opens its communicator with `tensorfold.cuda.comm.open_comm`: NCCL, wrapped by a registered
transport (a `Transport` subclass) when `TF_COMM_BACKEND` names one; NCCL stays the control channel, and the ranks
refuse to start with different transports. A model exchange calls `fast_gather`, which takes the transport's
`all_gather_fast` where there is one; `exchange` trades tensors with a peer (an NCCL send / receive group), and
`check` raises a transport's recorded failure after a synchronizing exchange. A transport moves bytes, so a reply
never depends on it.

The default two-rank decode paths gather fp32 partials and add them in rank order. The dense Qwen
prefill path gathers bf16 partials and adds them in fp32. Rank 0 chooses the window and both ranks
execute the same forwards. Prefix token IDs must describe the state actually cached on each rank.
CUDA graphs replay the same kernels using stable buffers; changing capture shapes must preserve these rules.

## Shared kernels and prefill

`src/tensorfold/cuda/kernels/qmm.py` packs 4-bit weights for the shared CUDA matmul. Its decode kernel fixes
the K split by weight shape, while the prompt kernel (`qmm_prefill.cu`) rounds each weight once to bf16 and adds
every product over K in one fp32 chain, the arithmetic of MLX's prompt matmul. NVFP4, FP8 and MXFP8 weights take
`src/tensorfold/cuda/nvfp4/prompt.cu`, where each weight is exact in bf16. Prompts run in chunks of up to 4,096 tokens with bf16
activations, as decode does. A row's bits never depend on its chunk or the kernel's tile, so a resumed prompt
equals a fresh one; they differ from decode's, so the engines retain prompt-end states and prefill replies
again on a follow-up.

`src/tensorfold/cuda/kernels/gdn.py` and `attention.py` support several streams in one call. Each stream
supplies its own tree, cache offsets and accepted path. `src/tensorfold/cuda/experts.py` groups routed
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

<a id="nvfp4-precision"></a>

### NVFP4 precision

An NVFP4 checkpoint names the formats of its activations as well as its weights. `--precision checkpoint`, the
default, runs that math, as the checkpoint's own runtimes do: each row is quantized under the checkpoint's static
input scale, to NVFP4 (per-16 e4m3 block scales) in NVFP4 layers and to e4m3 in FP8 layers. `--precision full` runs
bf16 activations against the stored weights exactly. The weights are the same in both modes; only the math changes.
A drafted reply equals the same server's serial one, and a resumed prompt a fresh one, in either mode. Which math a
GPU runs under `checkpoint` (its startup line says which; no GPU from 8.9 is refused):

| GPU | NVFP4 layers | FP8 layers |
| --- | --- | --- |
| SM 12.x (RTX 50, RTX PRO 6000 Blackwell, DGX Spark) | FP4 x FP4, block-scaled mma | FP8 x FP8 |
| SM 8.9-10.x (RTX 40, H100, H200, B200) | W4A16 (bf16 activations) | FP8 x FP8 |

`nvidia/Qwen3.8-27B-NVFP4` ran on an RTX PRO 6000 Blackwell Max-Q and a DGX Spark (GB10). The SM 8.9, 9.0 and 10.0
builds are compiled, and their math is checked on Blackwell: the SM 8.9-10.x choice served with drafted == serial,
and the split-K reduction those GPUs take without clusters bit for bit against the clusters' one. They have not run
on those cards yet.

Quality over 8 sequences of 4,095 positions (wikitext-2 test x 4, CPython source x 4), against an fp32 forward of
the stored weights; vLLM's prompt log-probs give top-1 and perplexity only. Seven sequences leave out one wikitext
slice where the fp32 reference itself is off at a few positions.

| Math | KL mean, 8 / 7 sequences | Top-1, 8 / 7 | Perplexity against fp32, 8 / 7 |
| --- | --- | --- | --- |
| `--precision checkpoint` | 0.0703 / 0.0662 | 92.9% / 93.1% | +3.89% / +4.24% |
| `--precision full` | 0.0026 / 0.0006 | 98.9% / 99.0% | -0.22% / -0.04% |
| vLLM MTP=3 (checkpoint math) | | 93.0% / 93.2% | +3.36% / +3.97% |

Speed on an RTX PRO 6000 Blackwell Max-Q at a 250 W power limit (the card allows up to 325 W),
tok/s, the median of two passes, against vLLM MTP=3 on the same weights and window (34,816 tokens), DFlash2 drafts,
256-token replies; cold prompts are the final build's (three of each length; its MLPs quantize the SwiGLU rows
straight to down's NVFP4 input). Streams are concurrent requests; 4 and 8 are aggregates. Under load both engines sat
at 247-250 W (a 0.3 s 2k prompt barely reaches it), the SM clock between 1,522 and 2,032 MHz, so compute-bound cells
(prompts, 8 streams) are the ones a higher limit would move most. vLLM ran at `--gpu-memory-utilization 0.29`, and at
0.32 for 8 streams (at 0.29 its cache admits fewer than 8 requests at once).

| Cell | checkpoint | full | vLLM | checkpoint / vLLM |
| --- | ---: | ---: | ---: | ---: |
| 1 stream, code / chat, greedy | 272.8 / 182.9 | 222.2 / 155.6 | 151.4 / 130.6 | 1.80x / 1.40x |
| 1 stream, code / chat, sampled | 274.1 / 162.2 | 238.8 / 151.4 | 134.2 / 115.7 | 2.04x / 1.40x |
| 4 streams, code / chat, greedy | 675.8 / 466.5 | 482.0 / 361.5 | 557.3 / 451.7 | 1.21x / 1.03x |
| 4 streams, code / chat, sampled | 654.9 / 420.5 | 466.3 / 348.1 | 535.3 / 422.6 | 1.22x / 0.99x |
| 8 streams, code / chat, greedy | 858.0 / 656.8 | 597.5 / 457.4 | 1,009.5 / 907.5 | 0.85x / 0.72x |
| 8 streams, code / chat, sampled | 834.5 / 584.1 | 578.9 / 428.4 | 896.8 / 730.6 | 0.93x / 0.80x |
| Cold prompt, 2k / 8k | 6,990 / 7,484 | 3,005 / 3,267 | 7,390 / 7,804 | 0.95x / 0.96x |
| Cold prompt, 16k / 32k | 7,100 / 6,296 | 3,213 / 3,039 | 7,350 / 6,491 | 0.97x / 0.97x |

On one DGX Spark (an earlier build, whose prompts ran on the decode kernel), one stream measured 59.47 tok/s against
vLLM's 25.32 on code and 40.69 against 24.27 on chat (greedy, checkpoint math), 8 streams 0.85-1.07x vLLM, and the
first token at 2k came after 1.23 s against vLLM's 0.84 (ahead from 32k).

`--prefill-fp8` belongs to `--precision full`: under the checkpoint's math its prompts already take FP4 and FP8
activations, so the server refuses the pair. The bf16 GDN gates of NVFP4 and EXL3 checkpoints take a bf16 prompt kernel
whose rows never depend on their chunk; an EXL3 27B's prompt bits change with it, still equal to its fresh prefill.

## Requests and memory

CUDA `--parallel auto` serves one request at a time. Set an explicit `--parallel N` above one for shared
Qwen3.8-27B or Flash Next rounds on one or two ranks; Nemotron, GLM and Qwen3.6 remain serialized. The shared
scheduler admits requests between decode rounds, then verifies each active stream's drafts together and commits
each stream independently. On the 27B, a new prompt prefills 1,024 tokens a round while the other streams keep
decoding, and its state is kept at message starts (the second message and the last assistant turn), so prompts that
share a system prompt or extend a conversation resume there with a fresh prefill's bits.

Cache capacity is fixed at startup and bounds prompt plus reply.
A positive context that exceeds the startup budget is refused; automatic capacity is an estimate.
The budget grants a discrete card its free memory less a floor of a tenth of the card, at least 4 GiB, for the
CUDA context and workspace memory the estimate does not count. `TENSORFOLD_MEMORY_RESERVE_GIB` moves that floor
(at least 2 GiB), and `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` caps the grant from above in GiB, an absolute budget like
the MLX one. A floor close to the smallest can end requests with CUDA errors mid-reply
(`PYTORCH_CUDA_ALLOC_CONF` `=` `expandable_segments:True` reduces fragmentation near the cap).
Unified-memory GPUs share physical RAM with host buffers and file-backed model data. Admission uses the
host's available memory, reclaimable page cache included, less a floor of a tenth of RAM (at least 4 GiB) that
`TENSORFOLD_MEMORY_RESERVE_GIB` can move, and considers mapped-table residency when sizing an automatic
window. It accounts for stream count and retained caches where concurrency is enabled.

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
