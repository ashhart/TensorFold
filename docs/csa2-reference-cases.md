# CSA2 sink and output-mix reference cases

`zig/tests/deepseek_v41_reference/operator_csa2_fp64_sink.zig` compares the
Metal sparse-attention output with the FP64 equation. The cases below preserve
the test's fixed seeds, shapes and tolerances.

## Sink normalization

One live column fixes the running maximum at one score. Changing the sink
therefore isolates the `sink - max` normalization term. The sweep includes
underflow, balanced mass and overflow boundaries of FP32 exponentiation.
Near overflow, a one-ULP change in the maximum can determine whether the
denominator remains finite; the corresponding output can be near `1e-38`.
The exact boundary depends on the generated score, not just the sink value.

The per-head case assigns sink values `-36`, `-12`, `12` and `36`. These
different masses check both head indexing and normalization arithmetic.

## Online tile denominator

The equal-score case spans ten tiles and 639 terms. Equal exponentials make
the unsunk denominator equal to the number of terms. Ramped scores exercise
the online correction as earlier mass falls toward FP32 underflow. Descending
scores keep the running maximum fixed and remove that rescaling step. The
single-tile case applies no inter-tile correction.

## Output mix and cancellation

Window and compressed rows share an accumulator but use different scale grids.
The dominance cases reverse their relative magnitudes across six orders of
magnitude; a third case balances their scales.

The mirror cases use 128 window rows with a large shape whose sign alternates
across columns and rows. A uniform query has zero dot product with that shape.
The large contributions also cancel in the weighted sum, leaving a small shared
perturbation. This exposes error that would be hidden by a large final result.
Shape exponents are 8, 16 and 23; the shared perturbation is scaled by `2^-16`.
Smaller perturbations can disappear in the FP8 representation itself.

These are operator comparisons. Passing them does not establish whole-model
logit parity, distributed state correctness or throughput.
