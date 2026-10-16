# Third-party notices

TensorFold uses [MLX](https://github.com/ml-explore/mlx) and
[mlx-lm](https://github.com/ml-explore/mlx-lm), MIT License, Copyright © 2023 Apple Inc.
The native engine installs and links neither. It reads checkpoints in their formats, and its Metal kernels under
`zig/kernels/metal/` repeat the arithmetic of MLX's kernels and mlx-lm's model math, many of them bit for bit.
The Python engine, frozen at 0.6.6, installs MLX and mlx-lm as dependencies.

## MLX and mlx-lm adaptations

In the Python engine, frozen at 0.6.6, the DeltaNet implementations in
`src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py` and
`lane_tree.py` adapt mlx-lm's `qwen3_5` and `gated_delta` model math and kernels under its MIT License.
The native engine carries that adaptation. `zig/kernels/metal/qwen3_5/gdn_pre.metal`, `gdn_chain.metal` and
`gdn_post.metal` are generated from the Python engine's dense Qwen kernels, whose recurrence step is `lane_tree.py`'s.
`zig/kernels/metal/core/deltanet.metal` and Flash Next's `zig/kernels/metal/flashnext/q4_gdn_*.metal` run the same
model math.

## Qwen Flash Next

The n-gram ID helpers translate Hugging Face transformers' `models/qwen4_exp/modeling_qwen4_exp.py` with renamed
identifiers. The native engine's are in Zig and Metal, in `zig/src/families/flashnext/config.zig` and the
`fz_ple_ids` kernel of `zig/src/families/flashnext/replay.zig`. The Python engine's, frozen at 0.6.6, are in MLX and
NumPy, in `src/tensorfold/families/qwen4_exp/model.py` and `cuda/ngram.py`.
Copyright 2026 The Qwen Team and The HuggingFace Inc. team, Apache License 2.0.
See [the license text](LICENSES/Apache-2.0.txt). The same helpers appear in mlx-vlm's
`models/qwen4_exp/language.py`, MIT License, Copyright © 2025 Prince Canuma.

## GLM-5.3-Flash on Apple Silicon

The native GLM-5.3-Flash engine (`zig/src/families/glm/`) runs the Python `glm5_next` family's decode kernels.
`tools/zig/gen_glm_kernels.py` generates `zig/kernels/metal/glm/` from that family's Metal kernels
(`src/tensorfold/kernels/glm/flash/v1/` at the `v0.6.6` tag), which @chadhurley25075-png wrote in
[#9](https://github.com/ashhart/TensorFold/pull/9) ([credits](https://github.com/ashhart/TensorFold/blob/main/docs/credits.md)).
`zig/kernels/metal/glm_kda_prompt.metal` extends the generated KDA kernel to prompt chunks.
In the Python engine, frozen at 0.6.6, the MLX engine of `glm5_next`
(`src/tensorfold/families/glm5_next/`: the forward pass in `model.py`, `kda.py`,
`mla.py` and `mlp.py`, the draft head in `mtp.py`, `runtime.py`) and its Metal kernels
(`src/tensorfold/kernels/glm/flash/v1/`) are written for TensorFold. What they follow or port:

- The forward pass follows, op for op on its prefill path, the GLM-5.3-Flash (`glm5_next`)
  implementation added to [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) by PR #2030 (by Lazarus-931; MIT License,
  Copyright (c) 2025 Prince Canuma), as vendored by [oMLX](https://github.com/jundot/omlx) (Apache-2.0). Nothing is
  imported from either at runtime. The native forward in `zig/src/families/glm/` follows the Python one.
- `kernels/glm/flash/v1/kda.py` is ported from mlx-vlm PR #2105 ("glm5_next: fuse the KDA decode chain into one
  Metal kernel", by avlp12; `mlx_vlm/models/glm5_next/fused_kda.py`; closed without merging, MIT License,
  Copyright (c) 2025 Prince Canuma): the whole KDA decode step in one Metal kernel. TensorFold runs a window of
  rows in order inside the launch, folds the 4-bit `f_b` / `g_b` projections in with MLX's one-row `qmv_quad`
  arithmetic, and keeps its own rounding points. Its precision rules (precise exp, uncontracted sums of squares)
  are also used in `fused.py`, `moe.py` and `hc.py`. The native `zig/kernels/metal/glm/kda_rows*.metal` are
  generated from it.
- `kernels/glm/flash/v1/sparse_attention.py` is mlx-vlm's `indexed_sparse_attention` kernel
  (`mlx_vlm/models/sparse_attention.py`) as extended by mlx-vlm PR #2245 ("Fix GLM-5.3 cached decode batch
  invariance", by raullenchai; closed without merging, MIT License, Copyright (c) 2025 Prince Canuma), adapted to
  TensorFold's single latent cache. The native `zig/kernels/metal/glm/sparse_attention*.metal` are generated from it.
- mlx-vlm PR #2107 (the sparse indexer's incremental decode and a stale-pool fix, by avlp12) needed no code:
  TensorFold's cache already pools once per completed block. Its stale-pool case is pinned by the Python engine's
  `tests/test_glm5_ported_kernels.py`.
- The hyper-connection kernel `_HC_SPLIT` in `kernels/glm/flash/v1/kernels.py`, and the sinkhorn and collapse in
  `hc.py`, repeat the `hc_sinkhorn_collapse` kernel of mlx-vlm's `mlx_vlm/models/deepseek_v4/hyper_connection.py`
  (MIT License, Copyright (c) 2026 Apple Inc.), with its output type set to the input's. The native
  `zig/kernels/metal/glm/hc_expand_10.metal` is generated from `hc.py`, and `zig/kernels/metal/core/hc_boundary.metal`
  runs the same split, Sinkhorn and collapse.
- The 4-bit matvec `_QMV_ROWS` in `kernels.py` is Flash Next's `qmv_rows` with MLX's group-64 scale indexing, and
  the expert kernels (`_EXPERT_GROUP`, `_EXPERT_QMV`) follow Flash Next's `expert_group` / `grouped_gateup`. The
  row kernels in `kernels.py`, `moe.py` and `hc.py` repeat the arithmetic and partitions of MLX 0.32's own kernels (MIT
  License, Copyright © 2023 Apple Inc.): `qmv_fast`, `qmv_quad` and `gather_qmv_fast` (`quantized.h`), `GEMVKernel`
  and `GEMVTKernel` (`gemv.h`) and the `rms_norm` kernels, one row per grid slice with the tiling MLX picks for one
  row, so each row keeps MLX's one-row bits. The native `zig/kernels/metal/glm/qmv*.metal`, `gemv*.metal`,
  `moe_*.metal` and `router.metal` are generated from these.

## Gemma 4 on Apple Silicon

The native Gemma 4 engine (`zig/src/families/gemma4/`) ports the Python engine's dense Gemma 4 decode, which @gilby
wrote in [#360](https://github.com/ashhart/TensorFold/pull/360) on @cshintov's Gemma kernels from
[#10](https://github.com/ashhart/TensorFold/pull/10) ([credits](https://github.com/ashhart/TensorFold/blob/main/docs/credits.md)).
Its reference is gilby's `gemma4-dense` branch at `a6ff795e`: `RowDecode` in `src/tensorfold/kernels/gemma/v1/decode.py`
and the `_EMIT`, `qkv_prep`, `attn_tail` and `dense_tail` kernels and the attention partials and merge in `glue.py` and
`attention.py`. `zig/kernels/metal/gemma4/glue.metal` repeats that arithmetic, and each kernel names its source.

Where that reference calls MLX or mlx-lm, `glue.metal` repeats their kernels' arithmetic (MLX and mlx-lm, MIT License,
Copyright © 2023 Apple Inc.; see [the license text](LICENSES/MIT.txt)): `gemma_rms_looped` is MLX's `rms_looped` from
`mlx/backend/metal/kernels/rms_norm.metal`, the kernel `mx.fast.rms_norm` runs for rows past 4,096, with its thread
counts, reads and rounding points; `gemma_embed` is MLX's affine dequantize followed by a bf16 multiply;
`gemma_geglu` and `gemma_softcap` are `geglu` and `logit_softcap` from mlx-lm's `models/gemma4_text.py` as MLX's
bf16 ops; and `gemma_rope_freqs` is mlx-lm's `ProportionalRoPE` frequencies, computed as MLX's `pow` computes them.

The image tower (`zig/src/families/gemma4/vision.zig`, `zig/kernels/metal/gemma4/vision.metal`) follows
`Gemma4VisionModel` and `Gemma4MultimodalEmbedder` in transformers' `models/gemma4/modeling_gemma4.py`, and
`zig/src/families/gemma4/image.zig` follows `Gemma4ImageProcessorPil` in `image_processing_pil_gemma4.py`
(`get_aspect_ratio_preserving_size`, the patch layout and the rescale). Copyright 2026 the HuggingFace Team, Apache
License 2.0; see [the license text](LICENSES/Apache-2.0.txt). The engine runs none of their code.

The drafter (`zig/src/families/gemma4/dflash.zig`) implements z-lab's DFlash for Gemma 4 31B from the MLX model listed
under [vendored code](#vendored-code-and-weights) (z-lab/dflash, MIT License, Copyright © 2026 Z Lab); the native
engine vendors no copy.

## CUDA

The Python engine's CUDA backends, frozen at 0.6.6, use [PyTorch](https://github.com/pytorch/pytorch), BSD-3-Clause, and
[Triton](https://github.com/triton-lang/triton), MIT, supplied by NVIDIA's container rather than bundled.
Dense Qwen implements mlx-lm's model math. CUDA DFlash2 implementations port z-lab's architecture under
the MIT License, Copyright © 2026 Z Lab.
The native CUDA build links neither PyTorch nor Triton. It runs its own kernels (`zig/kernels/cuda/`), and it loads
Triton-compiled cubins (`zig/src/cuda/triton.zig`) only from a captured set named by `TENSORFOLD_CUDA_KERNELS` or
installed beside the executable.

Flash Next implements transformers' model math under the attribution above. In the Python engine, its new CUDA
DeltaNet kernel follows flash-linear-attention's numerics, MIT; its NCCL wrapper follows vLLM's stream convention,
Apache-2.0, without copying either implementation.

The Python engine's GLM CUDA engine implements transformers' `models/glm5_next/modular_glm5_next.py` math,
Apache-2.0, without including that source. Its draft inputs and thinking-off rendering follow the public GLM recipe
from Mia-AiLab without including recipe code.
No native CUDA family serves GLM.

In the Python engine, frozen at 0.6.6, GLM EXL3 (`families/glm5_next/cuda/exl3.py`, `exl3.cu`, `exl3_mm.py`), the
shared EXL3 module (`src/tensorfold/cuda/exl3/`) and the EXL3 loaders of Qwen3.8-27B and Qwen3.8 Flash Next
(`families/qwen3_5/cuda/exl3_load.py`, `families/qwen4_exp/cuda/exl3.py`; the module and the loaders are @vcruz305's,
from [#42](https://github.com/ashhart/TensorFold/pull/42)) read
[ExLlamaV3](https://github.com/turboderp-org/exllamav3)'s EXL3 format: its trellis layout and bitstream, its
"3inst", "mcg" and "mul1" codebooks, its half-integer bit widths and its tensor-core fragment order. Flash
Next's packs also carry ExLlamaV3's n-gram row codec, read as its `ngram_dequant` reads it. ExLlamaV3 uses the
MIT License, Copyright © 2025 Turboderp. TensorFold's decoders and kernels are separate implementations,
checked bit for bit against ExLlamaV3's dequantization.

The native `zig/src/core/exl3_format.zig` reference decoder, the `zig/src/core/exl3_projection*.zig` decoder and
`zig/kernels/metal/core/exl3_projection.metal` separately implement the same format and codebooks
under the Turboderp attribution above. No captured kernel binaries or model weights are bundled.

The EXLR8 reader, decoder, pull and kernels (`zig/src/core/exlr8*.zig`, `zig/src/cli/pull_exlr8*.zig`,
`zig/src/cuda/exlr8.zig`, `zig/kernels/metal/core/exlr8*.metal`, `zig/kernels/cuda/exlr8*.cu`) read kindlingai's exlr8
v2 format as the Loader section of the [glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4)
model card lays it out. The format's packing, manifests and error tables are kindlingai's, offered under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); its codebook, Hadamard rotation and tile order are
ExLlamaV3's, under the Turboderp attribution above. TensorFold's implementations are its own, checked bit for bit
against the format's reference loader (MIT License, Copyright (c) 2025 Turboderp), which is not included.
`zig/tests/fixtures/exlr8/oracle.json` holds that loader's digests and a few sampled values of the release; no
weights are bundled.

In the Python engine, frozen at 0.6.6, Flash Next's optional int8 and int4 KV caches
(`families/qwen4_exp/cuda/kvcache.py`, by @vcruz305 in [#47](https://github.com/ashhart/TensorFold/pull/47)) follow
the cache quantization scheme of [ExLlamaV3](https://github.com/turboderp-org/exllamav3) `-cq 8` and `-cq 4`
(MIT License, Copyright (c) 2025 Turboderp, text below): groups of 32, one fp16 absmax scale per group, the group
rotated by a 32-point Hadamard, midpoint-grid codes, `compand_a == 0`. 8-bit stores each code as a signed int8
(`q - 128`). 4-bit stores two unsigned codes per byte, low nibble first (the same bits as ExLlamaV3's little-endian
packing, a uint8 tensor rather than their uint32 words). Their dequantizer folds another `1/sqrt(32)` into the scale
and applies the unnormalized butterfly on the way out; this cache applies the normalized H32 to the query and to the
merged output instead, and leaves the stored codes rotated. Scales match their quantizer bit for bit. Reconstructed
values agree within fp16/bf16 rounding (under 0.01 on random groups), not bit for bit. The quantizer and the attention
dequant are written for TensorFold and checked against an independent reference of that arithmetic. The native engine
has no quantized KV cache and refuses `--kv-dtype`.

## Vendored code and weights

In the Python engine, frozen at 0.6.6, `src/tensorfold/drafters/vendor/z_lab_dflash/model_mlx.py` is the unmodified
`dflash/model_mlx.py` from
[z-lab/dflash](https://github.com/z-lab/dflash), MIT License, Copyright © 2026 Z Lab.
The native engine vendors no copy. Its DFlash2 drafter, `zig/src/families/qwen27/dflash/`, implements z-lab's
architecture under that attribution.

In the Python engine, frozen at 0.6.6, `src/tensorfold/families/deepseek_v4/vendor/encoding_dsv4.py` is the
unmodified `encoding/encoding_dsv4.py` of
[deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (revision 60d8d70), and
`tests/fixtures/deepseek_v4/` holds two of its test cases, MIT License, Copyright (c) 2023 DeepSeek.
The MTP layer that engine drafts with comes from that checkpoint's last shard (MIT), converted by
`families/deepseek_v4/convert.py`. The native engine has no DeepSeek-V4 family and ships none of these files.

TensorFold ships no model weights. The `z-lab/Qwen3.8-27B-DFlash2` model card states Apache-2.0.
The optional `incoai/GLM-5.3-Flash-DFlash2` model card states CC BY-NC-ND 4.0, for non-commercial use
without derivatives; only the Python engine reads that drafter. Each checkpoint keeps its own license.

## Structured output: libtfgrammar

`lib/libtfgrammar` in the release archives is `zig/src/grammar/tf_grammar.cc` compiled with the C++ sources of
[xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8, Apache License 2.0, Copyright (c) 2024 by XGrammar Contributors
(its `NOTICE`: "XGrammar Copyright (c) 2024 by XGrammar Contributors"), and the headers it bundles:
[dlpack](https://github.com/dmlc/dlpack), Apache License 2.0, and [picojson](https://github.com/kazuho/picojson),
BSD 2-Clause License, Copyright 2009-2010 Cybozu Labs, Inc. and 2011-2014 Kazuho Oku. Nothing of theirs is in this
repository; the build reads a checkout given with `-Dxgrammar=DIR`, and each archive carries xgrammar's and dlpack's
license and xgrammar's notice in `LICENSES/`. picojson's license:

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
   disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Cortheon

`--cortheon` (`zig/src/server/cortheon/`) adapts [Cortheon](https://github.com/ashhart/Cortheon)'s task-classification word lists,
its stage sequences and their purposes, its rules for when a task needs outside sources, and its instruction-filter patterns,
rewritten in Zig. MIT License, Copyright (c) 2026 Cortheon contributors; the license text is below.

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

This notice covers the Flash Next image input on CUDA that @shantanugoel added to the Python engine in
[#146](https://github.com/ashhart/TensorFold/pull/146), under `src/tensorfold/vision/` and
`src/tensorfold/families/qwen4_exp/cuda/`; that engine is frozen at 0.6.6. The native engine reads no vision tower
and ships none of this code.
The multimodal rotary and image-feature integration is adapted from MiaAI-Lab's
[Flash Next vision patch 0008](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/a3aa89835022c55ca8e55008c37785954834e04f/patches/0008-flash-next-vision.patch),
MIT License, Copyright (c) 2026 MiaAI-Lab. The license is included in `LICENSES/MiaAI-Lab-MIT.txt`.
The port preserves the v0.5 CUDA execution APIs and adds an offline EXL3 vision adapter.
