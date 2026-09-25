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

## Vendored code

- `src/tensorfold/drafters/vendor/z_lab_dflash/model_mlx.py` is
  `dflash/model_mlx.py` from [z-lab/dflash](https://github.com/z-lab/dflash),
  MIT License, Copyright (c) 2026 Z Lab, unmodified.

## Model weights

TensorFold ships no weights. The DFlash2 draft model it can use for
Qwen3.8-27B (`z-lab/Qwen3.8-27B-DFlash2`) is published under Apache-2.0 per its
model card; each model you serve keeps its own license.

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
