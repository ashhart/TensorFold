# Retired kernel oracles

These development-only Python references preserve TensorFold kernels from
`2fbd46c` (the native branch before its upstream 0.3.6.1 rebase). They retain the
original project's MIT license, in the repository root. `row_forward.py` and
`row_qmv.py` keep the kernel helpers, without the old model integration.

Upstream removed or changed interfaces for row-QMV/folded-norm kernels, the
single-stream tree tail, grouped Flash experts, Nemotron shared-slot routing,
row experts and the combined Mamba step. The native catalog still embeds these
interfaces, so their independent Python references must remain available. Flash's
snapshot also preserves the original row-projection recipe. The exporter uses the
current split module for Flash attention and current modules for dense/SIMD kernels,
sampling and the remaining Nemotron helpers.

These references are never imported by production TensorFold or the native
executable. Kernel replay computes expected outputs through Python MLX, with no
native source loaded into that oracle. The current upstream tests remain part of
the fixture suite; explicit legacy cases retain coverage of removed interfaces.
