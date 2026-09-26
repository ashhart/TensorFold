# Third-party notices

TensorFold depends on [MLX](https://github.com/ml-explore/mlx) and
[mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT License, Copyright © 2023
Apple Inc.), installed as packages.

## Code adapted from mlx-lm

These files reproduce parts of mlx-lm 0.31.3 so their results match mlx-lm's
bit for bit, under mlx-lm's MIT License (Copyright © 2023 Apple Inc.):

- `src/tensorfold/kernels/gdn_capture.py` follows
  `mlx_lm/models/qwen3_5.py` (`GatedDeltaNet.__call__`) op for op.
- `src/tensorfold/kernels/lane_tree.py` repeats the arithmetic of mlx-lm's
  `gated_delta_step` Metal kernel (`mlx_lm/models/gated_delta.py`) inside its
  own kernels.

## Code adapted from transformers and mlx-vlm

The n-gram embedding's id helpers of Qwen3.8 Flash Next (`_splitmix64`, `_is_prime`,
`_find_nth_prime_after`, the per-layer multipliers and the shift-and-xor id mixing) in
`src/tensorfold/families/qwen4_exp/model.py` and `src/tensorfold/families/qwen4_exp/cuda/ngram.py` are
translated, with renamed identifiers, into MLX and NumPy from Hugging Face transformers'
`models/qwen4_exp/modeling_qwen4_exp.py` (Copyright 2026 The Qwen Team and The HuggingFace Inc. team),
licensed under the Apache License, Version 2.0; the license text is in
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt). The same helpers are in mlx-vlm's
`models/qwen4_exp/language.py` (MIT License, Copyright (c) 2025 Prince Canuma).

## GLM-5.3-Flash on Apple Silicon

The MLX engine of `glm5_next` (`src/tensorfold/families/glm5_next/model.py`, `mtp.py`, `runtime.py`) and its Metal
kernels (`src/tensorfold/kernels/glm/flash/v1/`) are written for TensorFold. What they follow or port:

- `families/glm5_next/model.py` follows, op for op on its prefill path, the GLM-5.3-Flash (`glm5_next`)
  implementation added to [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) by PR #2030 (by Lazarus-931; MIT License,
  Copyright (c) 2025 Prince Canuma), as vendored by [oMLX](https://github.com/jundot/omlx) (Apache-2.0). Nothing is
  imported from either at runtime.
- `kernels/glm/flash/v1/kda.py` is ported from mlx-vlm PR #2105 ("glm5_next: fuse the KDA decode chain into one
  Metal kernel", by avlp12; `mlx_vlm/models/glm5_next/fused_kda.py`; closed without merging, MIT License,
  Copyright (c) 2025 Prince Canuma): the whole KDA decode step in one Metal kernel. TensorFold runs a window of
  rows in order inside the launch, folds the 4-bit `f_b` / `g_b` projections in with MLX's one-row `qmv_quad`
  arithmetic, and keeps its own rounding points. Its precision rules (precise exp, uncontracted sums of squares)
  are also used in `fused.py`.
- `kernels/glm/flash/v1/sparse_attention.py` is mlx-vlm's `indexed_sparse_attention` kernel
  (`mlx_vlm/models/sparse_attention.py`) as extended by mlx-vlm PR #2245 ("Fix GLM-5.3 cached decode batch
  invariance", by raullenchai; closed without merging, MIT License, Copyright (c) 2025 Prince Canuma), adapted to
  TensorFold's single latent cache.
- mlx-vlm PR #2107 (the sparse indexer's incremental decode and a stale-pool fix, by avlp12) needed no code:
  TensorFold's cache already pools once per completed block. Its stale-pool case is pinned by
  `tests/test_glm5_ported_kernels.py`.
- The hyper-connection kernel `_HC_SPLIT` in `kernels/glm/flash/v1/kernels.py`, and the sinkhorn and collapse in
  `fused.py`, repeat the `hc_sinkhorn_collapse` kernel of mlx-vlm's `mlx_vlm/models/deepseek_v4/hyper_connection.py`
  (MIT License, Copyright (c) 2026 Apple Inc.), with its output type set to the input's.
- The 4-bit matvec `_QMV_ROWS` in `kernels.py` is Flash Next's `qmv_rows` with MLX's group-64 scale indexing, and
  the expert kernels (`_EXPERT_GROUP`, `_EXPERT_QMV`) follow Flash Next's `expert_group` / `grouped_gateup`. The
  row kernels in `kernels.py` and `fused.py` repeat the arithmetic and partitions of MLX 0.32's own kernels (MIT
  License, Copyright © 2023 Apple Inc.): `qmv_fast`, `qmv_quad` and `gather_qmv_fast` (`quantized.h`), `GEMVKernel`
  and `GEMVTKernel` (`gemv.h`) and the `rms_norm` kernels, one row per grid slice with the tiling MLX picks for one
  row, so each row keeps MLX's one-row bits.
- The GLM tool-call argument conversion in `src/tensorfold/server/http.py` (`coerce_glm_value`) is adapted from
  oMLX's `_coerce_param_value` (`omlx/api/tool_calling.py`, Apache License, Version 2.0, text in
  [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)), without its repair of near-valid JSON.

## CUDA engines

On Linux the CUDA engines use [PyTorch](https://github.com/pytorch/pytorch) (BSD-3-Clause) and
[Triton](https://github.com/triton-lang/triton) (MIT), taken from NVIDIA's PyTorch container, not bundled. Their
kernels are written for TensorFold. What they follow:

- `src/tensorfold/families/qwen3_5/cuda/` implements the model math of mlx-lm's `qwen3_5` model (MIT, above).
- The CUDA DFlash2 drafters (`families/qwen3_5/cuda/dflash2.py`, `families/glm5_next/cuda/dflash2.py`) port the
  DFlash2 architecture of z-lab's `dflash/model_mlx.py` to PyTorch and Triton, under its MIT License
  (Copyright (c) 2026 Z Lab, text below).
- `src/tensorfold/families/qwen4_exp/cuda/` implements the model math of transformers'
  `modeling_qwen4_exp.py` (Apache-2.0, above) in its own kernels; `gdn.cu` reproduces the gated delta rule
  with flash-linear-attention's numerics (MIT) in a new kernel, and `comm.py` calls NCCL on the caller's stream
  the way vLLM's `pynccl` does (Apache-2.0), without code from either.
- `src/tensorfold/families/glm5_next/cuda/` implements the math of the GLM-5 definition in Hugging Face
  [transformers](https://github.com/huggingface/transformers) (`models/glm5_next/modular_glm5_next.py`,
  Apache-2.0), with its own kernels; no code from it is included. The hidden states GLM's DFlash2 reads and
  its thinking-off chat rendering were checked against Mia-AiLab's GLM-5.3-Flash DGX Spark recipe; no code from
  that recipe is included either.

## Vendored code

- `src/tensorfold/drafters/vendor/z_lab_dflash/model_mlx.py` is
  `dflash/model_mlx.py` from [z-lab/dflash](https://github.com/z-lab/dflash),
  MIT License, Copyright (c) 2026 Z Lab, unmodified.

## Model weights

TensorFold ships no weights. The DFlash2 draft model it can use for
Qwen3.8-27B (`z-lab/Qwen3.8-27B-DFlash2`) is published under Apache-2.0 per its
model card. GLM-5.3-Flash's optional DFlash2 draft model
(`incoai/GLM-5.3-Flash-DFlash2`) is published under CC BY-NC-ND 4.0 (non-commercial
use only) per its model card; without it GLM drafts with its own MTP head. Each
model you serve keeps its own license.

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
