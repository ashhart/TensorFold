# DeepSeek V4.1 prompt experts on M5 tensor units: the exactness evidence (#3330)

On generation 17 GPUs (M5) the grouped prompt experts send a 32-wide group of products to the tensor unit only when
the group's act_quant values make every fp32 partial sum exact: 12 * sum < 2^24 * lowest set bit (`xu_exact` in
`zig/kernels/metal/deepseek_v41/experts_units.metal`). Then any summation order gives decode's bits. Other groups run
decode's 32 fmas in k order. The argument rests on a hardware fact: that the tensor unit keeps all 24 bits of such sums.
These runs check that fact and the rule.

## What runs

`zig/tests/dsv41_metal/units_test.zig`, part of `zig build test-dsv41-metal`:

- **Host, the rule against exact integer sums.** 200 groups of 16 rows built at and just past the bound, 32 weight
  columns each. Every safe row's sums come out exact in six orders (the given one and five shuffles). Past the
  bound, some order rounds.
- **M5, near the bound.** Groups of 16 rows and 32 fp4 columns. Each row's magnitude sum is set just under the bound,
  some rows within a few units of it, and a quarter of the groups just past it. Rows share positions: tops at the
  largest code, lows at the lowest bit with odd weights, and zeros. Columns make the sums swing and cancel in eight
  patterns. Up rows go through the production act_quant (`tf_dsv41_xu_quant`), down rows through `tf_dsv41_xu_dprep`.
  A probe then runs each group on the tensor unit, as `xu_chain` takes a safe group, and decode's fmas
  (`xu_exact_sum`) as the fallback does. Both results are compared bit for bit with the exact integer sum. The
  kernels' safe flags must equal the rule. `TF_DSV41_UNITS_BATCHES` sets the batches a layout (default 2, 8192 groups
  each).

The test fails unless every safe sum is exact on both paths, the flags agree, and past the bound the two paths split.

## Results

| Machine | Layout | Safe sums | Rows within 0.1% of the bound | Tensor unit vs exact | Chain vs exact | Sums past the bound | Tensor unit vs chain there |
|---|---|---|---|---|---|---|---|
| M5 Max | up (act_quant codes) | 61,481,856 | 72,935 | 0 | 0 | 5,627,008 | 86,750 |
| M5 Max | down (dprep halves) | 57,755,648 | 894,086 | 0 | 0 | 9,353,216 | 103,364 |
| M5 Ultra | up (act_quant codes) | 61,481,856 | 72,935 | 0 | 0 | 5,627,008 | 86,750 |
| M5 Ultra | down (dprep halves) | 57,755,648 | 894,086 | 0 | 0 | 9,353,216 | 103,364 |

The generator is seeded, so both machines saw the same groups, and their split counts past the bound are identical
too. The raw output is in `m5-max.txt` and `m5-ultra.txt`.

The contract carries a planted control for the rule: `TF_CTL 6` sends every group to the tensor unit. The down gates'
data holds one group that decode's fmas round onto a bf16 tie and an exact sum does not. The control goes red at the
shapes gate, while the kernels match the oracle through the fallback.

## Not covered

Later generations: the kernels run on generation 17 only, until a later chip runs this test.
