# HIP model-free kernels

`zig build hip-kernel-test -Dhip-arch=gfx1100`

The kernels a model engine composes, compiled by hipcc for one `-Dhip-arch` with the caps table's instruction
switches, embedded, and launched from Zig on #463's runtime. No model or checkpoint is involved.

The GPU tests check:

- Every kernel the launchers use resolves in the embedded code objects for the device's architecture.
- The planned decode walk over a permuted page table writes the flat walk's bits for 8, 6, 4 and 1 query heads
  a KV head, and sits within 1e-5 of float64 at positions 0, 37, 300 and 1023.
- The 64-row prompt tile over pages writes the flat tile's bits, within the cache type's rounding of float64.
- The DeltaNet recurrence, token-serial and chunked, within 5e-3 of a float64 recurrence at 64 to 4170 tokens.
- The merged decode launches (router, residual-and-norm tails) no further from float64 than twice the launches
  they replace; the merged conv split, MoE pick and gated norm writing the replaced chains' bytes.
- Casts, sums, column copies, cache writes, argmax, top-k and the MoE route giving torch's bits; silu products,
  gates, the MoE activation and RoPE within one rounding of float64.
- Every MLX affine entry the GPU has (dot2 tiles, the gfx11 matrix-core GEMM tile, the K-parallel and stream tiles),
  on the served models' shapes and a ragged sweep of every width and group, within the error an fp32 sum may have
  against float64; entries the registry picks between by rows write equal bytes.
- A row's product writes the same bytes alone and in a round of 2 to 32 rows, dense and routed, in every prompt tile.
- Products sharing x, and the routed gate and up with its activation, in one launch no further from float64 than
  twice the launches they replace.

Build notes:

- `-Dhipcc` and `-Dhip-include` as for `hip-gpu-test`.
- No rocWMMA: the gfx11 matrix-core tile uses the compiler's WMMA builtins, so ROCm packages without
  `rocwmma/rocwmma.hpp` (Ubuntu's ROCm 7.1) build it.
- `-Dhip-device-lib=<dir>` names ROCm's device bitcode directory. The default is `<rocm>/lib/llvm/amdgcn/bitcode`,
  TheRock's layout; Arch's ROCm keeps it in `<rocm>/amdgcn/bitcode`.
