# HIP row projection qualification

`zig build hip-affine-test -Dhip-arch=gfx1151 -Dhipcc=/usr/bin/hipcc`

This is one four-bit BF16 affine row projection with FP32 sums, group sizes
32/64, and explicit round-to-nearest-even BF16 output. Each output uses the
same 32-lane reduction regardless of batch width. It is not a model runner,
fast-prefill implementation, or a cross-backend bit-equality guarantee.

The model-free tests check:

- Row-alone versus batches 1/2/4/8/16/17/32, three repeats, across K
  64/192/512/576 and both group sizes.
- A separate-stream bounded load, with partial progress observed during the
  foreground sequence and full completion afterward. This is not an
  instruction-overlap or occupancy measurement.
- A cancellation-sensitive fixed case, expected BF16 `0x3d80`.
- A SHA-256-pinned fixed-input fixture and all 16 single-bit output mutations.
  The row kernel must return `0x3fc4`; each mutation must be rejected.

`hip_affine_g64.hex` is the 206-byte little-endian fixture exported from the
Python affine reference at TensorFold PR #144 commit
`88417d3da743c9e132a11495712c67316f20f48b`. Its SHA-256 is
`85eaf08cdad842c50068788e5801dc4ab3b78296e18a6f16a790229cbc719f99`.
The header is five u64 values (M=1,N=1,K=64,bits=4,group=64), followed by
BF16 X, packed u32 W, BF16 scales/biases, and BF16 output. This fixed case
also passes the HIP row reduction; it does not establish generic parity
with the Python implementation.

Measured on gfx1151 with HIP 7.1 and Zig 0.17.0, 7 October 2026:
28/28 host tests and 22/22 device tests passed. A separate 24-case,
4032-output comparison against a direct float64 dequantized dot product
gave per-case relative L2 errors 0.0014415–0.0019564, maximum absolute error
0.23693, and maximum RMSE 0.033885. These are fixture-specific measurements,
not a universal error bound. Against Metal `core/row_projection Sum.f32` at
`7ae6df7c3aa979d60819b210893802f8f6fca059`, two BF16 outputs differed;
cross-backend bit equality is reported separately from row/batch invariance.
