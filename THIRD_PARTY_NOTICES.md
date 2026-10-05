# Third-party notices

TensorFold uses [MLX](https://github.com/ml-explore/mlx) and
[mlx-lm](https://github.com/ml-explore/mlx-lm), MIT License, Copyright © 2023 Apple Inc.
They are installed as dependencies.

## MLX and mlx-lm adaptations

The DeltaNet implementations in `src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py` and
`lane_tree.py` adapt mlx-lm's `qwen3_5` and `gated_delta` model math and kernels under its MIT License.

## Qwen Flash Next

The n-gram ID helpers in `src/tensorfold/families/qwen4_exp/model.py` and `cuda/ngram.py` translate
Hugging Face transformers' `models/qwen4_exp/modeling_qwen4_exp.py` into MLX and NumPy with renamed
identifiers. Copyright 2026 The Qwen Team and The HuggingFace Inc. team, Apache License 2.0.
See [the license text](LICENSES/Apache-2.0.txt). The same helpers appear in mlx-vlm's
`models/qwen4_exp/language.py`, MIT License, Copyright © 2025 Prince Canuma.

## GLM-5.3-Flash on Apple Silicon

The MLX engine of `glm5_next` (`src/tensorfold/families/glm5_next/`: the forward pass in `model.py`, `kda.py`,
`mla.py` and `mlp.py`, the draft head in `mtp.py`, `runtime.py`) and its Metal kernels
(`src/tensorfold/kernels/glm/flash/v1/`) are written for TensorFold. What they follow or port:

- The forward pass follows, op for op on its prefill path, the GLM-5.3-Flash (`glm5_next`)
  implementation added to [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) by PR #2030 (by Lazarus-931; MIT License,
  Copyright (c) 2025 Prince Canuma), as vendored by [oMLX](https://github.com/jundot/omlx) (Apache-2.0). Nothing is
  imported from either at runtime.
- `kernels/glm/flash/v1/kda.py` is ported from mlx-vlm PR #2105 ("glm5_next: fuse the KDA decode chain into one
  Metal kernel", by avlp12; `mlx_vlm/models/glm5_next/fused_kda.py`; closed without merging, MIT License,
  Copyright (c) 2025 Prince Canuma): the whole KDA decode step in one Metal kernel. TensorFold runs a window of
  rows in order inside the launch, folds the 4-bit `f_b` / `g_b` projections in with MLX's one-row `qmv_quad`
  arithmetic, and keeps its own rounding points. Its precision rules (precise exp, uncontracted sums of squares)
  are also used in `fused.py`, `moe.py` and `hc.py`.
- `kernels/glm/flash/v1/sparse_attention.py` is mlx-vlm's `indexed_sparse_attention` kernel
  (`mlx_vlm/models/sparse_attention.py`) as extended by mlx-vlm PR #2245 ("Fix GLM-5.3 cached decode batch
  invariance", by raullenchai; closed without merging, MIT License, Copyright (c) 2025 Prince Canuma), adapted to
  TensorFold's single latent cache.
- mlx-vlm PR #2107 (the sparse indexer's incremental decode and a stale-pool fix, by avlp12) needed no code:
  TensorFold's cache already pools once per completed block. Its stale-pool case is pinned by
  `tests/test_glm5_ported_kernels.py`.
- The hyper-connection kernel `_HC_SPLIT` in `kernels/glm/flash/v1/kernels.py`, and the sinkhorn and collapse in
  `hc.py`, repeat the `hc_sinkhorn_collapse` kernel of mlx-vlm's `mlx_vlm/models/deepseek_v4/hyper_connection.py`
  (MIT License, Copyright (c) 2026 Apple Inc.), with its output type set to the input's.
- The 4-bit matvec `_QMV_ROWS` in `kernels.py` is Flash Next's `qmv_rows` with MLX's group-64 scale indexing, and
  the expert kernels (`_EXPERT_GROUP`, `_EXPERT_QMV`) follow Flash Next's `expert_group` / `grouped_gateup`. The
  row kernels in `kernels.py`, `moe.py` and `hc.py` repeat the arithmetic and partitions of MLX 0.32's own kernels (MIT
  License, Copyright © 2023 Apple Inc.): `qmv_fast`, `qmv_quad` and `gather_qmv_fast` (`quantized.h`), `GEMVKernel`
  and `GEMVTKernel` (`gemv.h`) and the `rms_norm` kernels, one row per grid slice with the tiling MLX picks for one
  row, so each row keeps MLX's one-row bits.

## CUDA

CUDA backends use [PyTorch](https://github.com/pytorch/pytorch), BSD-3-Clause, and
[Triton](https://github.com/triton-lang/triton), MIT, supplied by NVIDIA's container rather than bundled.
Dense Qwen implements mlx-lm's model math. CUDA DFlash2 implementations port z-lab's architecture under
the MIT License, Copyright © 2026 Z Lab.

Flash Next implements transformers' model math under the attribution above. Its new DeltaNet kernel
follows flash-linear-attention's numerics, MIT; its NCCL wrapper follows vLLM's stream convention,
Apache-2.0, without copying either implementation.

GLM's CUDA engine implements transformers' `models/glm5_next/modular_glm5_next.py` math, Apache-2.0,
without including that source. Its draft inputs and thinking-off rendering follow the public GLM recipe
from MiaAI-Lab without including recipe code. Other parts of that recipe are adapted for the DeepSeek-V4.1 path
and the shared CUDA server; see "GLM-5.3-Flash TensorFold recipe (MiaAI-Lab)" below.

GLM EXL3 (`families/glm5_next/cuda/exl3.py`, `exl3.cu`, `exl3_mm.py`), the shared EXL3 module
(`src/tensorfold/cuda/exl3/`), the EXL3 loaders of Qwen3.8-27B and Qwen3.8 Flash Next
(`families/qwen3_5/cuda/exl3_load.py`, `families/qwen4_exp/cuda/exl3.py`) and DeepSeek-V4.1's EXL3 reader, rank
split and prompt expert kernel (`families/deepseek_v41/cuda/reader.py`, `weights.py`, `families/deepseek_v41/split.py`,
`families/deepseek_v41/cuda/experts_prompt.cu`, whose 128-point Hadamard follows the butterfly order of ExLlamaV3's
`rot_in`) read
[ExLlamaV3](https://github.com/turboderp-org/exllamav3)'s EXL3 format: its trellis layout and bitstream, its
"3inst", "mcg" and "mul1" codebooks, its half-integer bit widths and its tensor-core fragment order. Flash
Next's packs also carry ExLlamaV3's n-gram row codec, read as its `ngram_dequant` reads it. ExLlamaV3 uses the
MIT License, Copyright © 2025 Turboderp. TensorFold's decoders and kernels are separate implementations,
checked bit for bit against ExLlamaV3's dequantization. `tools/dsv41_exl3_vs_exllamav3.py` imports ExLlamaV3's
`LinearEXL3` at run time as a reference; ExLlamaV3 is not bundled.

Flash Next's optional int8 and int4 KV caches (`families/qwen4_exp/cuda/kvcache.py`) follow the cache quantization scheme of [ExLlamaV3](https://github.com/turboderp-org/exllamav3) `-cq 8` and `-cq 4` (MIT License, Copyright (c) 2025 Turboderp, text below): groups of 32, one fp16 absmax scale per group, the group rotated by a 32-point Hadamard, midpoint-grid codes, `compand_a == 0`. 8-bit stores each code as a signed int8 (`q - 128`). 4-bit stores two unsigned codes per byte, low nibble first (the same bits as ExLlamaV3's little-endian packing, a uint8 tensor rather than their uint32 words). Their dequantizer folds another `1/sqrt(32)` into the scale and applies the unnormalized butterfly on the way out; this cache applies the normalized H32 to the query and to the merged output instead, and leaves the stored codes rotated. Scales match their quantizer bit for bit. Reconstructed values agree within fp16/bf16 rounding (under 0.01 on random groups), not bit for bit. The quantizer and the attention dequant are written for TensorFold and checked against an independent reference of that arithmetic.

### RoCE all-gather

The two-rank RoCE all-gather (`src/tensorfold/cuda/rdma/`: `__init__.py`, `gather.cu`, `gather.cpp`, `rdma_proxy.c`;
`tools/rdma_bench.py`; the DeepSeek-V4.1 TP2 default, `TF_COMM=nccl` opts out) re-implements the "RoCEnante"
one-shot protocol of [b12x](https://github.com/local-inference-lab/b12x) (`b12x/comm/roce/` at commit
8a99d639410e39d5f39cb4037675331beceea1d4; Apache License 2.0, Luke Alonso and the b12x contributors), after its CUDA
C++ port in MiaAI-Lab's GLM-5.3-Flash TensorFold recipe (`patches/0006-cuda-roce-allgather.patch`, Apache License
2.0, Copyright 2026 MiaAI-Lab). It keeps a pinned host region of control word, flags and double-buffered slots; a
proxy thread that RDMA-writes the payload, then a sequence flag, on one reliable QP; the GPU polling its own flag; and
a device epoch for CUDA-graph replay. The kernel's PTX memory-order helpers, flag wait and doorbell, and the wrapper's
`NCCL_IB_HCA` parse, NCCL fall-through and error text, follow that port. TensorFold's version is limited to two ranks
and one HCA. The proxy is a rewrite, and each file states what was changed. The parked multi-HCA port itself is not
included. b12x ships no NOTICE file. See [the license text](LICENSES/Apache-2.0.txt).

### DeepSeek-V4.1-Flash model

The `deepseek_v41` family (`src/tensorfold/families/deepseek_v41/`: `config.py`, `reference.py`, `split.py`,
`engram.py` and `cuda/`) implements the architecture of
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (MIT License, Copyright (c)
2023 DeepSeek): hyper-connections, compressed and windowed MQA with sinks, the sparse indexer, the Engram tables, the
fp8 KV layout and the DSpark draft head. Its math was worked out from [vLLM](https://github.com/vllm-project/vllm)'s
DeepSeek-V4.1 port (`vllm/model_executor/models/deepseek_v4_1/`, `deepseek_v4/`) and DSpark speculator
(`vllm/v1/worker/gpu/spec_decode/dspark/`), Apache License 2.0, Copyright contributors to the vLLM project.
`notes/dsv41/ARCH.md` is a spec written from that source; the source is not included. The exception is `engram.py`,
which ports vLLM's `common/engram.py` token-compression normalizer, `find_next_prime` and hash multipliers to NumPy,
with the same constants, so its hashes match bit for bit; the file states this. Other parts follow vLLM's conventions:
the DSpark taps follow its `eagle3_utils` layout and fixed-k verify policy, `split.py` follows its TP=2 layout, and
`/tokenize` (`src/tensorfold/cuda/http.py`) follows its request and response shape (MiaAI-Lab's GLM patch 0037 adds
the same route). The DSML reply markup read by `cuda/dsml.py`, `server/tools.py`, `server/text.py` and
`cuda/reply_text.py` follows DeepSeek's V4.1 chat encoding. No DeepSeek code is included: the chat template is read
from the checkpoint at run time. `tools/dsv41_vllm_dump_patch.py` patches a local vLLM install by matching two lines of
its `model.py`. `notes/dsv41/golden*.json` and `vllm_*.json` are outputs recorded from a vLLM server.
See [the license text](LICENSES/Apache-2.0.txt).

`src/tensorfold/families/deepseek_v41/cuda/mqa_fp4.cu` (the CUDA chunk pass of decode attention over the NVFP4
compressed KV cache) adapts FlashInfer's "Cake" DeepSeek-V4.1 mixed-cache decode kernel,
[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
`csrc/cake_dsv4/sm_120a/cake_sparse_mla_dsv41_mixed_h32.cu` (`decode_dual`), commit
`2c1c0525067452c411944fb1c40d8112040ea229` (PR #5983), Apache License 2.0, Copyright 2025-2026 NVIDIA and Copyright
2023-2026 FlashInfer community: one gathered candidate set shared by every local head, bf16 `mma.sync` with the FP4
rows widened exactly (its two-`prmt` E2M1 table, or `cvt.rn.bf16x2.e2m1x2` on CUDA 13.2+), a lane owning whole scale
groups, V^T through `ldmatrix.trans`. It is rewritten by hand for TensorFold's `Fp4Rows` planes, bf16 window ring,
per-stage partials folded in a fixed tree and its own CUDA merge; the file lists what was changed. Its NOTICE line is in our `NOTICE`. See
[the license text](LICENSES/Apache-2.0.txt).

The DeepSeek-V4.1 tool-call constraint in `src/tensorfold/engine/grammar.py` (`TOOL_TAG`) uses the built-in
`deepseek_v4_1` structural tag of [xgrammar](https://github.com/mlc-ai/xgrammar) (Apache License 2.0), the optional
`tensorfold[grammar]` dependency; nothing from it is copied. `deploy/dsv41-tp2/Dockerfile` builds locally on NVIDIA's
`nvcr.io/nvidia/pytorch:26.07-py3` (NVIDIA Deep Learning Container License) and pip-installs xgrammar, transformers,
tokenizers and safetensors (Apache-2.0); none is bundled here. `rdma_proxy.c` is compiled at run time against the
system's libibverbs (rdma-core, GPL-2.0 or BSD-2-Clause).

## Vendored code and weights

`src/tensorfold/drafters/vendor/z_lab_dflash/model_mlx.py` is the unmodified `dflash/model_mlx.py` from
[z-lab/dflash](https://github.com/z-lab/dflash), MIT License, Copyright © 2026 Z Lab.

`src/tensorfold/families/deepseek_v4/vendor/encoding_dsv4.py` is the unmodified `encoding/encoding_dsv4.py` of
[deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (revision 60d8d70), and
`tests/fixtures/deepseek_v4/` holds two of its test cases, MIT License, Copyright (c) 2023 DeepSeek.
The MTP layer TensorFold drafts with comes from that checkpoint's last shard (MIT), converted by
`families/deepseek_v4/convert.py`.

`tests/fixtures/deepseek_v41/config.json` is the `config.json` of
[Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) (EXL3 1.4.2,
`mul1`), an ExLlamaV3 quantization of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash);
both model cards state the MIT License (Copyright (c) 2023 DeepSeek). `notes/dsv41/inspect.txt` lists that
checkpoint's tensors as read by TensorFold's inspection tool; no weights are included.

TensorFold ships no model weights. The `z-lab/Qwen3.8-27B-DFlash2` model card states Apache-2.0.
The optional `incoai/GLM-5.3-Flash-DFlash2` model card states CC BY-NC-ND 4.0, for non-commercial use
without derivatives. The DeepSeek-V4.1 path loads `Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw` and reads the
Engram tables in place from shards 47-48 of `deepseek-ai/DeepSeek-V4.1-Flash`, both downloaded by the user; both
model cards state the MIT License (Copyright (c) 2023 DeepSeek). Each checkpoint keeps its own license.

## MIT License text

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Flash Next CUDA image integration

The multimodal rotary and image-feature integration is adapted from MiaAI-Lab's
[Flash Next vision patch 0008](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/a3aa89835022c55ca8e55008c37785954834e04f/patches/0008-flash-next-vision.patch),
MIT License, Copyright (c) 2026 MiaAI-Lab. The license is included in `LICENSES/MiaAI-Lab-MIT.txt`.
The port preserves the v0.5 CUDA execution APIs and adds an offline EXL3 vision adapter.

## DeepSeek-V4.1-Flash serving (deepseek-v41-tensorfold-spark)

Parts of the DeepSeek-V4.1 serving path are adapted from Jay Leaton's
[deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark):

- `src/tensorfold/families/deepseek_v41/cuda/dsml.py` and `tests/test_dsv41_dsml.py` adapt its DeepSeek-V4.1 family's
  DSML reply parser (`engine/serving/dsml.py`) and its tests, MIT License, Copyright (c) 2026 Jay Leaton. The
  license is included in `LICENSES/JayLeaton-MIT.txt`. Each file keeps its SPDX line and states what was changed.
- The DSML tool-call grammar in `src/tensorfold/engine/grammar.py` (`tool_spec`, `_vocab`, the `tools` compile)
  ports `_tool_spec` and `_vocab` of its GLM Spark grammar module (`glm5_next/spark/grammar.py` in
  `patches/0001-spark-stack-060.patch`; MIT License, Copyright (c) 2026 TensorFold contributors and Copyright (c) 2026
  Jay Leaton, glm53-tensorfold-spark) and the tool compile of its `engine/serving/structured.py` (MIT License,
  Copyright (c) 2026 Jay Leaton).
- The NVMe tier of kept prompts (`src/tensorfold/families/deepseek_v41/cuda/kvdisk.py`) adapts its session store's
  disk tier (`engine/serving/sessdisk.py`: the O_DIRECT file helper, the entry format, reconcile / trim / retain /
  delete, the compat ident and `from_env`), and the startup intersection of both ranks' entries follows its `_serve`,
  MIT License, Copyright (c) 2026 Jay Leaton. The license is included in `LICENSES/JayLeaton-MIT.txt`; the file
  keeps its SPDX line and states what was changed.
- The routed experts' load path (`src/tensorfold/cuda/exl3/x3ld.cu`, `x3ld.cpp`, `x3ld.py`, used by
  `src/tensorfold/cuda/exl3/experts.py` under `TF_EXPERT_LOADS`) comes from its DeepSeek-V4.1 family's
  `src/tensorfold/families/deepseek_v41/cuda/x3ld.cu`, `x3ld.cpp` and `expert_loads.py`
  (`patches/0002-deepseek-v41-family.patch`), MIT License, Copyright (c) 2026 Jay Leaton. It applies the load path of
  his glm53-tensorfold-spark patch 0580 (16-byte, several-deep weight loads) to TensorFold's grouped EXL3 expert
  kernel (`experts_grouped.cuh`). The kernel and bindings are moved here unchanged. `x3ld.py` drops the PDL launch and
  probe options and renames the switch to `TF_EXPERT_LOADS`. The license is included in `LICENSES/JayLeaton-MIT.txt`;
  each file keeps its SPDX line and states what was changed.
- The prepared per-rank weight folders (`src/tensorfold/families/deepseek_v41/cuda/fastboot.py`, used by `make prepare`
  and `TF_DSV41_PREPARED` in `deploy/dsv41-tp2/`) adapt its DeepSeek-V4.1 family's `fastboot.py` (`patches/0002`): the
  key, the manifest and `data.bin` layout, the parallel O_DIRECT reader with per-chunk SHA-256, and the `prepare` /
  `check` commands. MIT License, Copyright (c) 2026 Jay Leaton. Its design comes from his GLM fast-restart patch 0140
  (`glm5_next/spark/fastboot.py` in `patches/0001-spark-stack-060.patch`; MIT License, Copyright (c) 2026 TensorFold
  contributors and Copyright (c) 2026 Jay Leaton). The license is included in `LICENSES/JayLeaton-MIT.txt`; the file
  keeps its SPDX line and states what was changed. `tests/test_dsv41_fastboot.py` is written for TensorFold, after the
  cases of its test.
- The bulk L2 prefetch (`_l2_bulk` / `bulk_table` / `l2_bulk` in `src/tensorfold/families/deepseek_v41/cuda/kernels.py`,
  a Triton re-implementation used by the `TF_L2_PREFETCH=bulk` sites of `serial.py`) and the `bulk` / `lines` kernels of
  `tools/l2probe.py` adapt the `cp.async.bulk.prefetch.L2` and `prefetch.global.L2` kernels of its GLM Spark engine's
  `glm5_next/spark/l2pf.cu`, with the 16-byte-rounded (address, bytes) table and 32 KB pieces of `l2pf.py`
  (`patches/0001-spark-stack-060.patch`, from glm53-tensorfold-spark patch 0460). MIT License, Copyright (c) 2026
  TensorFold contributors and Copyright (c) 2026 Jay Leaton, as its NOTICE records for the GLM Spark engine; MiaAI-Lab's
  GLM patch 0046 labels the same code Apache-2.0. The license texts are in `LICENSES/MIT.txt` and
  `LICENSES/JayLeaton-MIT.txt`. The probe's go / no-go measurement follows the idea of its
  `families/deepseek_v41/cuda/l2probe.py` without its code.
- `deploy/dsv41-tp2/watchdog.sh` and `deploy/dsv41-tp2/systemd/` adapt the watchdog of its `scripts/serve.sh` and
  `scripts/systemd/`, and `tools/dsv41_soak.py`, `tools/dsv41_stress.py` and `tools/dsv41_structured.py` adapt its
  `bench/soak.py`, `bench/stress.py` and `bench/structured.py`:
  Apache License 2.0, Copyright 2026 Jay Leaton (https://x.com/jayleaton); its NOTICE line is in our `NOTICE`, and
  each file states what was changed. See [the license text](LICENSES/Apache-2.0.txt).
- The fatal / forward-progress fields of `/health` (`src/tensorfold/cuda/health.py`) follow the idea of its GLM Spark
  engine's `health.py`; no code is copied.
- Ideas from the same recipe, re-implemented here (no code copied):
  - The decode indexer top-k (`src/tensorfold/families/deepseek_v41/cuda/topk.py`) follows the design of its
    `csa2/dtopk.py` (`patches/0002`; MIT License, Copyright (c) 2026 Jay Leaton): digit histograms over an
    order-preserving key, a masked radix walk and an ordered compaction. It is built like TensorFold's own
    `families/glm5_next/cuda/sparse.py` `_select_rows`.
  - The exact bounded prefill tail (`BOUNDED_TAIL`, `tail_min`, `deep_from` in
    `src/tensorfold/families/deepseek_v41/cuda/serial.py`; the kept prompts' `vd` in `multi.py`) follows its CED
    bounded-replay prefill (`replay.py`, `TF_DSV41_PREFILL=replay`, an approximate replay over a prompt's last 128 rows
    that it attributes to DeepSeek's V4.1-Flash technical report). Ours is exact.
  - The THP guards in `deploy/dsv41-tp2/docker-compose.yaml` (`NUMPY_MADVISE_HUGEPAGE=0`, `MIMALLOC_ALLOW_THP=0`) follow
    the prefill stall reported in its G12 notes.

## GLM-5.3-Flash TensorFold recipe (MiaAI-Lab)

Parts of the DeepSeek-V4.1 path and the shared CUDA server are adapted from MiaAI-Lab's (Mia's AI Lab)
[GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
recipe, Apache License 2.0, Copyright 2026 MiaAI-Lab. Its NOTICE line is in our `NOTICE`. See
[the license text](LICENSES/Apache-2.0.txt).

- `src/tensorfold/families/deepseek_v41/cuda/pool.py` adapts its `glm5_next/cuda/pool.py`
  (`patches/0030-glm-multi-stream-engine.patch`): `ALIGN`/`align_up`, `Extent`, `Pool` (lowest-base first fit over
  2048-aligned extents, `gaps`, `room_after`, `resize`, `move`, `remove`) and `_move` (here `move_rows`).
- The extent lifecycle in `src/tensorfold/families/deepseek_v41/cuda/multi.py` (`_make_room`, `_grow`, `_room`,
  `_settle`, `yield_for`) follows its `glm5_next/cuda/multi.py` (patches 0030, 0040-0042): grow in place, else move,
  else evict kept prompts, and the newest replayable stream gives its rows back. So does rank 0's ADMIT / EVICT / MOVE /
  GROW protocol. `_settle` is adapted from its `_settle`; the rest is rewritten. The re-queue in
  `src/tensorfold/cuda/scheduler.py` (`_give_way`) follows its `GlmScheduler._requeue`.
- `src/tensorfold/cuda/copy_drafts.py` (copy / prompt-lookup drafts) and its use in
  `families/deepseek_v41/cuda/engine.py` adapt its `glm5_next/cuda/copy_drafts.py` and decode wiring (patches
  `0007-glm-copy-drafts` and `0032-glm-code-copy-drafts`). TensorFold made it model-agnostic, on by default under
  `TF_COPY_*`, windowed and truncatable.
- Design only, no code copied: the indexer reads only each row's visible keys (`_index_scores` in
  `families/deepseek_v41/cuda/kernels.py`, and `cuda/topk.py`), after its visible-pools bound (patch
  `0043-glm-visible-pools`, whose idea its credits trace to TensorFold PR #140 by mikolaj92). The CUDA `/tokenize`
  route matches its patch 0037.
- The RoCE all-gather credit (its patch 0006, after b12x) is under "RoCE all-gather" above.

Each adapted file states what was changed. `tests/test_dsv41_pool.py`, `tests/test_dsv41_kept.py` and
`tests/test_copy_drafts.py` are written for TensorFold.

## Ideas from AGPL-licensed recipes (no code included)

- The GB10 display carveout (`src/tensorfold/cuda/carveout.py`, `TF_CARVEOUT=1`, counted by `capacity.available_bytes`)
  uses the technique of `release/runtime/sources/display_kv.c` in Emi Huang's (coolbho3k)
  [DeepSeek-v4.1-Flash-2x-DGX-Spark](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark) (revision 91b19f6,
  AGPL-3.0-only): a DRM dumb framebuffer from the scanout reservation, registered with CUDA as cache backing.
- DeepSeek-V4.1's candidate-block filter (`kernels.candidate_blocks`, `kernels.mask_to_blocks`, `topk.candidate_flags`)
  follows the semantics of that repository's `release/runtime/ds41/dcp_candidates.py`: the block max, the newest block
  pinned, the top 2048 kept and -1 padding.
- `notes/dsv41/ARCH.md` cites one indexing fact from MiaAI-Lab's DeepSeek-V4.1 vLLM recipe overlay
  ([DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks),
  AGPL-3.0), and `tools/dsv41_vllm_dump_patch.py` inserts TensorFold code into a private copy of that recipe.

These TensorFold files were written independently from public interfaces and descriptions and contain no code from
those repositories, so the AGPL does not apply to them.

## Ideas from the literature

- Copy drafts are prompt lookup decoding: Apoorv Saxena, "Prompt Lookup Decoding" (2023,
  https://github.com/apoorvumang/prompt-lookup-decoding), the same idea as vLLM's n-gram speculation.
- Row-invariant decode and verify (`families/deepseek_v41/cuda/multi.py`, `serial.py`; checked by `tools/dsv41_soak.py`)
  follow the batch-invariance approach of Horace He and Thinking Machines Lab, "Defeating Nondeterminism in LLM
  Inference" (2025, https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
