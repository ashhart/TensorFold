# DeepSeek reference precision

The reference harness compares Metal arithmetic with two CPU paths on the same
inputs: the FP32 source contract and an FP64 ideal. Existing tolerances remain
part of the tests; moving their explanations here does not change them.

CSA2 isolation uses synthetic tensors and emits `CSA64` receipts. Cache codes
are generated directly so Metal and the references decode identical BF16 keys.
Window rows use a power-of-two FP8 scale and FP8 codes per 32-column block.
Compressed rows use an FP8 scale and FP4 codes per 16-column block. Encoding
uses nearest-integer, ties-to-even semantics where Metal uses `rint`.

mHC isolation emits `HC64` receipts and accepts retained rows and checkpoint
scale/base tensors. Statistic, dense, split, collapse and post-expansion stages
run on identical inputs. The rounding gate bounds Metal's distance from the
ideal relative to the FP32 contract's own distance. The amplification path
retains pre-rounding FP64 outputs for analog-gain measurements and probes
one-ULP input changes. Neither isolation harness injects source coefficients to
make an operator pass.

## Attention boundaries

The FP32 sparse-head reference uses a wide FP64 dot followed by one FP32
narrowing per score, FP32 online softmax, BF16-rounded probabilities before
the value product, and FP32 per-tile accumulator narrowing. The FP64 path
removes those FP32 narrowings while retaining the algorithm's quantized inputs,
specified probability rounding and final output rounding.

Optional capture of `acc / denominator` before output rounding exposes
arithmetic ULP differences that the final BF16 output can hide. Indexer probes
report the selection boundary margin in both precisions and count changed
top-k picks on the same inputs.

## Other reference boundaries

BF16 cells decode exactly at either reference precision. RMS uses the source
contract's FP32 mean projection for FP32, and a wide mean, division and
reciprocal square root for FP64. Direct FP64-to-BF16 conversion rounds once,
nearest-even, without an intermediate FP32 rounding. Explicit decoded BF16 or
native FP32 weight cells let operator tests run without a checkpoint tensor.

The dense projection reference keeps the quantized path's input rotation and
output stages. It replaces tile decoding with an FP64 dense dot over the
f16-rounded rotated input row.

The margin-capture runner prefills a long prompt, then copies one diagnostic
token's per-layer score tiles, top-k heap and positions for offline analysis.
It does not reload the model or run a second reference chain. This capture is
distinct from the per-window comparison of FP32 and FP64 selection margins.

## Selection and rotary isolation

The indexer compares the top-k boundary margin with one stored-score quantum.
A margin near that quantum can make the selected set sensitive to arithmetic
rounding. Both selectors resolve equal scores by lowest position. The position
view starts after score, heap and candidate storage; reading the backing buffer
without its offset would make the position check compare the wrong words.

The rotary contract computes frequency, angle and trigonometric functions in
FP32. The ideal retains those values and the products in FP64. Metal's factor
table supplies its own unary results. Cases use 64 rotary columns, apply YaRN
only to the compressed spectrum and include positions up to resident capacity.

## Retained mHC checks

The split-stage isolation supplies identical projected mixes and inverse to
each path, leaving unary and reduction boundaries to compare. The test keeps
its documented one-ULP Metal exp allowance and NRMS bound of `1e-7` for
near-cancelled cells, where absolute ULP counts can grow.

Retained cancellation fixtures require the original FP32 accumulation order
exactly. Their FP64 results document the intended half-ULP difference. The post
fixture combines a column `{1, 2^-24, 2^-24, 0}` with stream `{1,1,1,1}`,
branch `{1}` and post `{-1,0,0,0}`. The two half-ULP terms follow `1.0`.

The retained-row checks expand token 671 into four streams and use checkpoint
FFN scale/base tensors. A separate bitwise comparison keeps the verbatim
pre-parameterization FP32 algorithm as a control. The amplification check
batches the first teacher row with 16 stream-cell perturbations through Metal
and compares their response with the FP64 analog gain.
