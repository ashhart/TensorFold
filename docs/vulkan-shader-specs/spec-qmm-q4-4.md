# Clean-Room Vulkan Compute Shader Behaviour Specification — Q4 Affine Projection Family (4 Files)

**Clean-room rule**: Implement every shader below solely from this behaviour specification and TensorFold's own Zig host code (`zig/src/vulkan/ops_q4.zig`, `zig/src/vulkan/tile.zig`, `zig/tests/vulkan/stage1_fixture.zig`) and Metal/CUDA/HIP kernels. Do not open, read, or copy from `omarchy-mlx` or any other external inference engine.

## 0. Shared Contract, ABI & Numeric Rules

- **Target**: GLSL `#version 450` or `#version 460`, compiled by `glslang -V --target-env vulkan1.2 -S comp -I. <defines> -o <out.spv> <src.comp>`.
- **Workgroup size**: Literal `LocalSize` only (never `LocalSizeId`).
- **Extensions**: Declare `#extension GL_EXT_shader_16bit_storage : require` and `#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require` for all BF16 (`uint16_t`) storage buffer accesses.
- **Bindings (5, writes mask `1 << 4`)** for all 4 files:
  - `binding = 0` readonly: `x` — activations `[M, K]`, row-major (`uint16_t` BF16 by default, `float` under `-DACT_F32`).
  - `binding = 1` readonly: `w` — packed 4-bit unsigned weights (`uint` or `uvec4`): 8 unsigned 4-bit codes per 32-bit word (`code j = (word >> (4 * j)) & 0xfu` for `j = 0..7`).
  - `binding = 2` readonly: `scales` — per-group BF16 (`uint16_t`) scale factors, group size `G = 64` (`K % 64 == 0`).
  - `binding = 3` readonly: `biases` — per-group BF16 (`uint16_t`) additive biases, group size `G = 64`.
  - `binding = 4` writeonly: `out` — output `[M, N]`, row-major (`uint16_t` BF16 by default, `float` under `-DACT_F32`).
- **Push constants (120 bytes, `Params` std430)** (`ops_q4.zig:59-71`):
  - `count = M * N`
  - `operation = 4` (4-bit affine)
  - `reduce_size = 64` (group size)
  - `output_size = M * N`
  - `matrix_m = M`, `matrix_n = N`, `matrix_k = K` (`K >= 64`, `K % 64 == 0`)
  - `lhs_offset`, `rhs_offset`, `aux_offset`, `output_offset` (element offsets; `0` when bound via sub-buffer views, still honor if nonzero).
  - `shape[0] = 1` (single batch).
- **Exact mathematical definition** (verified against `stage1_fixture.zig` and `contract_qmm.zig` `.dequantized` f64 reference `depth = K + 2`, `accum = f32`):
  - For output element `(m, n)` with `m in [0, M)`, `n in [0, N)`:
    $$\text{out}[m, n] = \sum_{k=0}^{K-1} x[m, k] \cdot \left(\text{float}(q[n, k]) \cdot \text{scale}[n, \lfloor k/64 \rfloor] + \text{bias}[n, \lfloor k/64 \rfloor]\right)$$
    accumulated in IEEE-754 `float32` using the direct dequantized weight `w = float(q) * scale + bias; acc += x * w;`.
  - **Nonfinite requirement (Gate 1)**: Do NOT factor `scale * sum(x * q) + bias * sum(x)` across a group. Under `contract_qmm.zig` (`q.form = .dequantized`), when `x = +inf`, `q = 0`, and `bias != 0`, the dequantized weight `w = bias` is nonzero so `x * w` yields `±inf`, whereas `x * 0.0` in a factored `sum_xq` produces `NaN`.
  - Under BF16 output: single RNE `f32_to_bf16` store (`lsb = (u >> 16u) & 1u; bias = 0x7fffu + lsb; uint16_t((u + bias) >> 16u)`, NaN -> `(u >> 16u) | 0x0040u`).
  - Under `-DACT_F32`: store raw `float32` output directly.
  - Must never read or write outside `[0, M*N)` (guard tail poison checks in `stage1_fixture.zig`).

---

## 1. `qmm.comp` (PR `#1544`) — Scalar Row-Major Q4 Projection
- **Variants (`zig/build/vulkan.zig:58,71`)**:
  - `qmm_bf16`: `-DUSE_BF16` (BF16 `x` & `out`)
  - `qmm_f32`: `-DUSE_BF16 -DACT_F32` (F32 `x` & `out`)
- **Weight layout (row-major)**:
  - `w`: `[N, K / 8]` `uint` (`word = w[n * (K / 8) + k / 8]`)
  - `scales`, `biases`: `[N, K / 64]` `uint16_t` (`g_idx = n * (K / 64) + g`)
- **Alignment**: Must work at any 4-byte-aligned view offset (including non-16-byte-aligned `x` or `w` offsets, e.g. offset 68 or 72 bytes — `ops_q4.zig:302-305`).
- **Dispatch (`ops_q4.zig:142`)**: `groups = (ceil((M * N) / local_size_x), 1, 1)`.
- **Behaviour**: Thread `tid = gl_GlobalInvocationID.x < M * N` computes `m = tid / N`, `n = tid % N`, loops over groups `g = 0 .. K/64 - 1` (8 `uint` words per group, 8 nibbles per word), accumulates in `float32`, stores `out[m * N + n]`.

---

## 2. `qmm_q4_vec4.comp` (PR `#1990`) — 16-Byte Vectorized Row-Major Q4 Projection
- **Variants (`zig/build/vulkan.zig:59`)**:
  - `qmm_q4_vec4`: no defines (always BF16 `x` & `out`, BF16 `scales`/`biases`).
- **Weight layout**: Same row-major layout as `qmm.comp`, with host-guaranteed 16-byte alignment on `x` and `w` (`ops_q4.zig:77-79`).
- **Dispatch (`ops_q4.zig:142`)**: `groups = (ceil((M * N) / local_size_x), 1, 1)`.
- **Behaviour**:
  - Same arithmetic and output as `qmm_bf16`, using 16-byte (`uvec4`) loads from `w` (4 `uint` words = 32 4-bit codes per `uvec4` load, 2 `uvec4` loads per 64-element group) and 16-byte (`uvec4`) loads from `x` (8 BF16 elements per `uvec4` load, 8 `uvec4` loads per 64-element group).
  - Unpack each pair of BF16 values from a `uint` word `p` via `uintBitsToFloat((p & 0xffffu) << 16u)` and `uintBitsToFloat(p & 0xffff0000u)`.

---

## 3. `qmm_q4_tile.comp` (PR `#2015`) — 32-Column Tiled Q4 Projection
- **Variants (`zig/build/vulkan.zig:60,62`)**:
  - `qmm_q4_tile`: no defines (BF16 `x` & `out`)
  - `qmm_q4_tile_f32`: `-DACT_F32` (F32 `x` & `out`)
- **Weight layout (`zig/src/vulkan/tile.zig:19-45`, host-repacked into 32-column tiles, `N` padded to `padded(N) = ceil(N / 32) * 32`)**:
  - Let `vectors = K / 32` (number of 16-byte `uvec4` weight vectors per column; each `uvec4` holds 32 4-bit codes, 2 vectors per 64-element group).
  - Let `groups = K / 64`.
  - For column `c in [0, N)` (`tile = c / 32`, `lane = c % 32`):
    - Weight `uvec4` vector `j in [0, vectors)` is at `uvec4` index: `((tile * vectors + j) * 32u + lane)`.
    - Scale / bias BF16 for group `g in [0, groups)` is at `uint16_t` index: `((tile * groups + g) * 32u + lane)`.
  - Adjacent lanes `lane = 0..31` in a 32-lane subgroup load contiguous 16-byte vectors (512-byte coalesced transaction).
- **Dispatch (`ops_q4.zig:126-127`)**: `groups = (ceil((M * N) / local_size_x), 1, 1)`.
- **Behaviour**:
  - Thread `tid = gl_GlobalInvocationID.x < M * N` computes `m = tid / N`, `n = tid % N` (or maps consecutive threads across columns `n` within a row `m` so `n % 32` matches subgroup lane!).
  - Accumulate group `g = 0 .. groups - 1` (vectors `2*g` and `2*g + 1`) in `float32`, loading `x` via 16-byte `uvec4` (`bf16`) or `vec4` (`-DACT_F32`), and store `out[m * N + n]`.

---

## 4. `qmm_q4_tile_rows.comp` (PR `#2017`) — 4-Row x 2-Column Tiled Prompt Q4 Projection
- **Variants (`zig/build/vulkan.zig:61`)**:
  - `qmm_q4_tile_rows`: no defines (BF16 `x` & `out` only; selected when `M >= 4`).
- **Weight layout**: Identical 32-column tiled layout to `qmm_q4_tile.comp` (`tile.zig`).
- **Exact thread grid (`ops_q4.zig:126`)**:
  - `row_blocks = (M + 3) / 4` (each thread computes up to 4 consecutive rows `m0 .. min(m0 + 3, M - 1)`).
  - `col_tiles = (N + 31) / 32`; **16 threads per 32-column tile** (each thread computes **2 columns** within its 32-column tile: `c0 = tile * 32 + sub`, `c1 = tile * 32 + sub + 16` where `sub in [0, 16)` — note that `sub = 0..15` coalesces the first half of the tile and `sub + 16 = 16..31` coalesces the second half, OR consecutive `(2*sub, 2*sub+1)` — either matches the exact tiled address formula `((tile * vectors + j) * 32 + col % 32)`!).
  - Total threads launched: `threads = row_blocks * col_tiles * 16`.
  - Thread `tid = gl_GlobalInvocationID.x < threads`:
    - `tile_threads = col_tiles * 16`
    - `rb = tid / tile_threads`, `rem = tid % tile_threads`
    - `tile = rem / 16`, `sub = rem % 16`
    - `m0 = rb * 4`
    - `c0 = tile * 32 + sub`, `c1 = tile * 32 + sub + 16`
- **Behaviour**:
  - Reuses the loaded weight vectors for columns `c0` and `c1` across all active rows `m = m0 + r` (`r in 0..3`, `m < M`), and reuses loaded `x[m, ...]` across both `c0` and `c1`.
  - Guard column bounds: if `c0 < N`, store `out[m * N + c0]`; if `c1 < N`, store `out[m * N + c1]` for each active `m < M`. Never write out-of-bounds columns in padded tail tiles (`N .. padded(N) - 1`) or rows `>= M`!
  - Exact same `float32` accumulation and single RNE `f32_to_bf16` store as `qmm_q4_tile.comp`.
