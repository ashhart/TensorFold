# HIP row projection qualification

`zig build hip-affine-test -Dhip-arch=gfx1151`

This is one four-bit BF16 affine row projection with FP32 sums, group sizes
32/64, and explicit round-to-nearest-even BF16 output. Each output uses the
same 32-lane reduction regardless of batch width. It is not a model runner,
fast-prefill implementation, or a cross-backend bit-equality guarantee.

The model-free tests check:

- Row-alone versus batches 1/2/4/8/16/17/32, three repeats, across K
  64/192/512/576/1088 and both group sizes.
- A separate-stream bounded load, with partial progress observed during the
  foreground sequence and full completion afterward. This is not an
  instruction-overlap or occupancy measurement.
- A cancellation-sensitive fixed case, expected BF16 `0x3d80`.
- Pinned BF16 outputs for all ten matrix shapes (7680 outputs), checked
  before row-alone/batch comparisons, and a separate fresh-launch float64
  dequantized dot-product check with relative L2 below 0.0021 per shape.
  The accuracy-only inputs use biases near -8*scale to exercise cancellation;
  these differ from the exact matrix-reference inputs.
- A fixed-input fixture returning `0x3fc4`, plus a cancellation-sensitive
  K=576/1088 fixtures that distinguish contraction and shuffle/pass order,
  with N=2 and groups 32/64, repeated through batch widths
  1/2/3/4/8/16/17/32 under the bounded separate-stream load.
  Constructive cases pin x-sum order, subnormal output 0x0040, and exact
  BF16 halfway values of both parities.

`hip_affine_g64.hex` is a 206-byte little-endian fixture from the Python
Triton affine oracle in TensorFold PR #144. Its decoded SHA-256 is
`85eaf08cdad842c50068788e5801dc4ab3b78296e18a6f16a790229cbc719f99`.
The header is five u64 values (M=1,N=1,K=64,bits=4,group=64), followed by
BF16 X, packed u32 W, BF16 scales/biases, and BF16 output. This fixed case
also passes the HIP row reduction; it does not establish generic parity
with the Python implementation.

`hip_affine_matrix.hex` stores little-endian BF16 results in group-size,
K, row, column order, matching the deterministic inputs in the test.
`hip_affine_sensitive.hex` uses the same header and payload format as the
single-output fixture, one record per line, with dimensions in each header.
Searched records use M=2, N=2 and groups 32/64; constructive records use N=1.
The test repeats fixture rows modulo M to fill larger batches. These are engine
regression references, not a cross-backend equality requirement.
The independent NumPy FP32 emulator emits the two references with
`python3 zig/tests/hip_affine_fixtures.py matrix` and
`python3 zig/tests/hip_affine_fixtures.py sensitive`; compare stdout with
the corresponding hex file. The sensitive generator also asserts that
all 119 alternative shuffle orders, both FMA forms, BF16 sums, pairwise
product/x-sum grouping, reversed K passes, a split scale/bias epilogue and
an emulated flush-to-zero epilogue move at least one output.
`python3 zig/tests/hip_affine_fixtures.py search` reproduces the greedy
order-coverage search over seeds 1–1000. Fixture generation uses NumPy
2.4.3; use that version when regenerating Generator-based inputs.
The `accuracy` option reproduces the cancellation-input float64 comparison
and asserts separation from the BF16-per-add-sum emulator. It is not a
device measurement.

There are four device-test blocks. The runner also includes host/mock
tests imported by the device-test module; its total is not a count of
independent GPU checks. Device receipts and arithmetic negative-control
results belong in the pull request.

NaN payload/conversion behavior is not covered by these fixtures. Neither
finite fixtures nor a subnormal case establish all-bit-pattern semantics.
