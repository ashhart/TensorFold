# Clean-Room Vulkan Compute Shader Behaviour Specification — Core, Pointwise, Attention, GDN, Greedy (17 Files)

**Clean-room rule**: Implement every shader below solely from this behaviour specification and TensorFold's own Zig host code / Metal / CUDA / HIP kernels. Do not open, read, or copy from `omarchy-mlx` or any other external inference engine.

## 0. Shared ABI, Compilation & Numeric Rules

- **Target**: GLSL `#version 450` or `#version 460`, compiled by `glslang -V --target-env vulkan1.2 -S comp -I. <defines> -o <out.spv> <src.comp>`.
- **Workgroup size**: Must emit literal `LocalSize` (`layout(local_size_x = ..., local_size_y = ..., local_size_z = ...) in;`). Never use specialization-constant `LocalSizeId` (`dispatch.localSize` refuses `LocalSizeId`).
- **Storage buffer extensions**: Any shader variant declaring `uint16_t` storage buffers MUST declare BOTH:
  - `#extension GL_EXT_shader_16bit_storage : require`
  - `#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require`
  *(In particular, `fast_rope.comp` under `-DORACLE_QWEN35 -DACT_OUT_BF16` writes `uint16_t` BF16 output and MUST enable these extensions.)*
- **Subgroup requirement**: Only shaders in the GDN group may use subgroup operations; `Gdn.init` guarantees `VK_SHADER_STAGE_COMPUTE_BIT`, `VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT`, and `subgroupSize % 4 == 0`.
- **Shared 120-byte push-constant block (`Params`, std430)** (all fields `uint` except `alpha`/`beta` `float`, exact byte offsets verified by `zig/src/vulkan/params.zig`):
  - `0..32`: `count`, `operation`, `lhs_size`, `rhs_size`, `reduce_size`, `output_size`, `lhs_offset`, `rhs_offset`, `output_offset` (9 words = 36 bytes, used as prefix by `cast.comp`, `binary_vec.comp`, `conv_dw1d.comp`)
  - `36..56`: `aux_size`, `aux_offset`, `matrix_m`, `matrix_n`, `matrix_k`, `flags` (15 words = 60 bytes, used as prefix by `argreduce_*`, `gdn_conv_decode.comp`)
  - `60..64`: `float alpha`, `float beta` (17 words = 68 bytes, used as prefix by `fast_norm.comp`, `fast_norm_gated.comp`)
  - `68`: `dims`
  - `72..84`: `uint shape[4]`
  - `88..100`: `uint in_strides[4]`
  - `104..116`: `uint out_strides[4]`
- **Exact BF16 conversion rules** (verified by `stage1_fixture.zig` `bf16_vectors` and `stage2_fixture.zig`):
  - **BF16 -> F32 (`bf16_to_f32(uint h)`)**: exact zero-extension widening `uintBitsToFloat((h & 0xffffu) << 16u)`. Preserves `-0.0` (`0x8000`), all subnormals (`0x0001..0x007f`), `±inf`, and NaNs.
  - **F32 -> BF16 (`f32_to_bf16(float x)`)**: IEEE-754 round-to-nearest-even (RNE) to `uint16_t`:
    - Let `u = floatBitsToUint(x)`.
    - If `(u & 0x7fffffffu) > 0x7f800000u` (NaN): return `uint16_t((u >> 16u) | 0x0040u)` (quiet NaN preserving sign/class).
    - Otherwise (including `±0`, subnormals, normals, `±inf`): `uint lsb = (u >> 16u) & 1u; uint bias = 0x7fffu + lsb; return uint16_t((u + bias) >> 16u);`
    - Note: preserves `-0.0` (`0x8000`), rounds exact half-way ties to even LSB, handles subnormals (`0x1p-133 -> 0x0001`, `0.75*0x1p-133 -> 0x0001`, `0x1p-134 -> 0x0000`, `3*0x1p-134 -> 0x0002`), and leaves `±inf` unchanged (`0x7f80` / `0xff80`).

---

## 1. Stage 1 — Data Movement & Dequant (4 files)

### 1.1 `cast.comp`
- **Variants**:
  - `cast_bf16_f32`: `-DSOURCE_BF16`
  - `cast_f32_bf16`: `-DDEST_BF16`
- **Bindings (3, writes mask `1 << 2`)**:
  - `binding = 0` readonly: `uint16_t` (under `SOURCE_BF16`) or `float` (under `DEST_BF16`)
  - `binding = 1` readonly: unused dummy buffer
  - `binding = 2` writeonly: `float` (under `SOURCE_BF16`) or `uint16_t` (under `DEST_BF16`)
- **Push constants**: 36 bytes (`count..output_offset`).
- **Dispatch**: `groups = (ceil(count / local_size_x), 1, 1)` (e.g. `local_size_x = 256`).
- **Behaviour**: For `i = gl_GlobalInvocationID.x < params.count`:
  - `SOURCE_BF16`: `out[params.output_offset + i] = bf16_to_f32(in[params.lhs_offset + i])`.
  - `DEST_BF16`: `out[params.output_offset + i] = f32_to_bf16(in[params.lhs_offset + i])`.
  - Must not touch elements `>= count` (guard tail poison checks).

### 1.2 `copy_general.comp`
- **Variants**:
  - `copy_u32`: `-DUSE_U32` (`uint` -> `uint`, bit-exact)
  - `copy_u16`: `-DUSE_U16` (`uint16_t` -> `uint16_t`, bit-exact)
  - `copy_f32_bf16`: `-DUSE_F32_TO_BF16` (`float` -> `uint16_t`, RNE `f32_to_bf16`)
- **Bindings (4, writes mask `1 << 2`)**:
  - `binding = 0` readonly: source elements
  - `binding = 1` readonly: dummy
  - `binding = 2` writeonly: destination elements
  - `binding = 3` readonly: dummy
- **Push constants**: 120 bytes (`Params`).
- **Dispatch**: 1D grid covering `params.count` elements (grid-stride or `gl_GlobalInvocationID.x < params.count`).
- **Behaviour**:
  - Supports `params.dims` in `1..4`.
  - Given linear element index `idx` in `[0, params.count)`, decompose `idx` into coordinates `c[0..dims-1]` in row-major order over `params.shape[0..dims-1]`:
    - Check exact convention used by TensorFold callers:
      - `ops.zig:210`: `dims=1, shape={count,0,0,0}, in_strides={1,0,0,0}, out_strides={1,0,0,0}`
      - `ops_q4.zig:167`: `dims=2, shape={m, chunk, 0, 0}, in_strides={chunk, 1, 0, 0}, out_strides={n, 1, 0, 0}, output_offset=first` (row `c[0]` in `[0, m)`, col `c[1]` in `[0, chunk)`)
      - `attention.zig:163`: `dims=3, shape={head_dim, kv_heads, rows, 0}, in_strides={1, head_dim, kv_heads*head_dim, 0}, out_strides={1, capacity*head_dim, head_dim, 0}, output_offset=base*head_dim`
    - Note that in `ops_q4.zig` (`dims=2`) `shape[0]` is outermost and `shape[1]` is innermost (`in_strides={chunk, 1}`), while in `attention.zig` (`dims=3`) `shape[0]` has stride `1` and `shape[2]` is outermost (`in_strides={1, 256, 512}`) — **Wait: verify coordinate decomposition order**:
      - If `idx` is decomposed with `shape[dims-1]` innermost (standard C row-major: `for (d = dims - 1; d >= 0; --d) { coord = rem % shape[d]; rem /= shape[d]; in_idx += coord * in_strides[d]; out_idx += coord * out_strides[d]; }`):
        - For `dims=2` (`shape={m, chunk}`): `c[1] = idx % chunk`, `c[0] = idx / chunk` -> `in = c[0]*chunk + c[1]*1`, `out = output_offset + c[0]*n + c[1]*1`. EXACT MATCH!
        - For `dims=3` (`shape={256, 2, rows}`): `c[2] = idx % rows`, `c[1] = (idx/rows) % 2`, `c[0] = idx/(2*rows)` -> visits EVERY `(c[0], c[1], c[2])` bijection over `[0..256) x [0..2) x [0..rows)`! Because `c[d]` is matched with `in_strides[d]` and `out_strides[d]`, ANY bijection over `shape[0..dims-1]` maps every source coordinate to its exact corresponding destination coordinate! Using standard descending-dimension decomposition `for (int d = int(params.dims) - 1; d >= 0; --d)` works identically for BOTH `ops_q4.zig` AND `attention.zig`!
  - `src_i = params.lhs_offset + in_idx`, `dst_i = params.output_offset + out_idx`.
  - Copy bit-exact (`USE_U32`/`USE_U16`) or RNE convert (`USE_F32_TO_BF16`).

### 1.3 `gather_take.comp`
- **Variants**: `gather_u32` (`-DUSE_U32`), `gather_u16` (`-DUSE_U16`).
- **Bindings (3, writes mask `1 << 2`)**:
  - `binding = 0` readonly: `table` (`uint` or `uint16_t`)
  - `binding = 1` readonly: `ids` (`uint` when `params.operation == 1u`, or `int` when `0u`)
  - `binding = 2` writeonly: `out` (`uint` or `uint16_t`)
- **Push constants**: 120 bytes (`Params`). Caller (`ops.zig:229`) sets:
  - `count = ids_count * window`
  - `operation = 1` (raw `uint` indices)
  - `reduce_size = rows` (number of rows in table)
  - `matrix_m = stride` (`window`)
  - `dims = 1`, `shape[0] = window`, `in_strides[0] = 1`, `out_strides[0] = ids_count`, `flags = 1`
- **Dispatch**: `groups = (ceil(count / local_size_x), 1, 1)`.
- **Behaviour**: For `gid = gl_GlobalInvocationID.x < params.count`:
  - `uint id_idx = gid / params.matrix_m;`
  - `uint elem_in_row = gid % params.matrix_m;`
  - `uint row = ids[params.rhs_offset + id_idx];`
  - If `row < params.reduce_size`: `out[params.output_offset + gid] = table[params.lhs_offset + row * params.matrix_m + elem_in_row];`
  - Else (out-of-range id): write zero (`0u` or `uint16_t(0u)`).

### 1.4 `dequant.comp`
- **Variants**: `dequant_bf16` (`-DUSE_BF16`).
- **Bindings (4, writes mask `1 << 3`)**:
  - `binding = 0` readonly: `w` (`uint`, packed 4-bit codes: 8 codes per `uint`, code `j` in bits `4*j .. 4*j+3`)
  - `binding = 1` readonly: `scales` (`uint16_t` BF16)
  - `binding = 2` readonly: `biases` (`uint16_t` BF16)
  - `binding = 3` writeonly: `out` (`uint16_t` BF16)
- **Push constants**: 120 bytes (`Params`). Caller (`ops.zig:248`) sets:
  - `count = rows * (k / 8)` (number of packed `uint` words across all rows)
  - `operation = 4` (4-bit codes)
  - `reduce_size = group` (group size, multiple of 8, e.g. 64)
  - `matrix_n = words_per_row` (`k / 8`)
  - `matrix_k = groups_per_row` (`k / group`)
- **Dispatch**: `groups = (ceil(count / local_size_x), 1, 1)` (1 thread per packed `uint` word -> writes 8 BF16 elements).
- **Behaviour**: For `word_idx = gl_GlobalInvocationID.x < params.count`:
  - `row = word_idx / params.matrix_n`, `w_col = word_idx % params.matrix_n`.
  - `elem_col = w_col * 8u`, `g_col = elem_col / params.reduce_size`.
  - `scale = bf16_to_f32(scales[params.rhs_offset + row * params.matrix_k + g_col])`.
  - `bias = bf16_to_f32(biases[params.aux_offset + row * params.matrix_k + g_col])`.
  - `packed = w[params.lhs_offset + word_idx]`.
  - For `j = 0..7`: `float q = float((packed >> (4u * j)) & 0xfu); out[params.output_offset + word_idx * 8u + j] = f32_to_bf16(q * scale + bias);`.

---

## 2. Stage 2 — Norm & MLP (3 files)

### 2.1 `fast_norm.comp`
- **Variants**:
  - `norm_bf16`: `-DUSE_BF16 -DORACLE_QWEN35` (BF16 in, BF16 out)
  - `norm_f32`: `-DUSE_BF16 -DORACLE_QWEN35 -DACT_IN_F32 -DACT_OUT_F32` (F32 in, F32 out)
  - `norm_f32_in`: `-DUSE_BF16 -DORACLE_QWEN35 -DACT_IN_F32 -DACT_OUT_BF16` (F32 in, BF16 out)
  - `norm_bf16_in`: `-DUSE_BF16 -DORACLE_QWEN35 -DACT_OUT_F32` (BF16 in, F32 out)
  - Gamma (`binding = 1`) is ALWAYS BF16 (`uint16_t`).
- **Bindings (4, writes mask `1 << 3`)**: `0`: `x`, `1`: `gamma` (BF16), `2`: unused bias dummy, `3`: `out`.
- **Push constants**: 68 bytes (`count..beta`). Caller (`norm_mlp.zig:173`) sets:
  - `lhs_size = width`, `reduce_size = width`, `output_size = rows`, `alpha = eps` (finite `> 0`, e.g. `1e-6`).
- **Dispatch**: 1D workgroup grid `min(rows, maxComputeWorkGroupCount.x)` with grid-stride loop over rows: `for (uint row = gl_WorkGroupID.x; row < params.output_size; row += gl_NumWorkGroups.x)`. `local_size_x = 256`.
- **Behaviour**:
  - For each row of length `W = params.reduce_size`:
    1. Accumulate `sum_sq = sum_{i=0..W-1} (x_f32[i] * x_f32[i])` in `float` across the 256 threads via shared memory tree reduction.
    2. Compute `inv = inversesqrt(sum_sq / float(W) + params.alpha)` (or `1.0 / sqrt(...)`).
    3. For each column `i` in `[0, W)`: `float y = bf16_to_f32(gamma[i]) * (x_f32[i] * inv);`
       - Note exact association `gamma * (x * inv)` and zero-row guarantee: when `x == 0`, `y == 0.0` (exact `±0`, never nonzero).
    4. Store `y` as F32 (`ACT_OUT_F32`) or `f32_to_bf16(y)` (BF16 out).

### 2.2 `binary_vec.comp`
- **Variants**:
  - `add_bf16` (no defines): BF16 `a`, BF16 `b` -> BF16 `out`
  - `add_preserve_bf16` (`-DPRESERVE_F32`): same, with `layout(...)` / SPIR-V `DenormPreserve` float32 mode (via `#extension GL_EXT_control_flow_attributes` or `#pragma` / GLSL `GL_KHR_memory_scope_semantics`? Note: in GLSL `glslang`, float32 denorm preserve is enabled via `#extension GL_EXT_shader_float_controls` / `GL_KHR_shader_subgroup`? Check `glslang` support or `layout(denorm_preserve)` — specifically GLSL does NOT have a `denorm_preserve` layout qualifier unless using `SPV_KHR_float_controls`; verify with `glslang -V --target-env vulkan1.2` and `spirv-val`, or inspect how `stage2_fixture.zig` tests `add_preserve_bf16`: it requires that `0x0001 + 0x0001 == 0x0002` and `0x0001 + 0x8000 == 0x0001` without flushing BF16 subnormals!).
  - `add_f32` (`-DACT_F32`): F32 `a`, F32 `b` -> F32 `out`
  - `add_wide_bf16` (`-DADD_WIDE`): F32 `a`, BF16 `b` -> BF16 `out`
  - `add_bf16_f32` (`-DADD_WIDE_F32`): BF16 `a`, F32 `b` -> F32 `out`
- **Bindings (3, writes mask `1 << 2`)**: `0`: `a`, `1`: `b`, `2`: `out`.
- **Push constants**: 36 bytes (`count..output_offset`). Caller (`norm_mlp.zig:198`) sets `count = elems`, `operation = 0` (add), `lhs_size = elems`, `rhs_size = elems`.
- **Dispatch**: `groups = (min(ceil(count / local_size_x), max_x), 1, 1)` with grid-stride loop over `count` (or 4-element vectors since `count % 4 == 0` and 16B-aligned once bug #8 lands).
- **Behaviour**: For every `i` in `[0, count)`:
  - Widen BF16 inputs to `float` (`bf16_to_f32`), compute exact IEEE-754 `float sum = a_f32 + b_f32`.
  - Store `sum` directly (F32 out) or via `f32_to_bf16(sum)` (BF16 out).
  - Must match host `toBf16(fromBf16(a) + fromBf16(b))` **bit for bit** on all finite inputs (including `-0 + -0 -> -0`, `+0 + -0 -> +0`, `0x0001 + 0x0001 -> 0x0002`, `0x7f7f + 0x7f7f -> +inf`) and preserve nonfinite classes (`+inf`, `-inf`, `nan`).

### 2.3 `swiglu.comp`
- **Variants**:
  - `swiglu_bf16`: `-DUSE_BF16 -DORACLE_QWEN35` (BF16 `gate`, `up`, `out`)
  - `swiglu_f32`: `-DUSE_BF16 -DORACLE_QWEN35 -DACT_F32` (F32 `gate`, `up`, `out`)
- **Bindings (5, writes mask `1 << 2`)**: `0`: `gate`, `1`: `up`, `2`: `out`, `3`: unused dummy, `4`: unused dummy.
- **Push constants**: 16 bytes (`uint count, gate_offset, up_offset, materialize`).
- **Dispatch**: 1D grid-stride loop over `count` (`count % 4 == 0`).
- **Behaviour**: For every `i` in `[0, count)`:
  - Load `g = gate[gate_offset + i]` and `u = up[up_offset + i]` as `float`.
  - Compute `float y = (g / (1.0 + exp(-g))) * u;` in float32 (single BF16 rounding on store under `swiglu_bf16`, no intermediate BF16 round of `silu(g)`).
  - Exact zero/nonfinite controls (`stage2_fixture.zig`):
    - `g = +0.0` -> `0.0`; `g = -0.0` -> `-0.0` (preserves zero sign!)
    - `g = +inf, u = 1.0` -> `+inf` (`exp(-inf) == 0`, `+inf / 1 == +inf`)
    - `g = -inf, u = 1.0` -> `NaN` (`-inf / +inf`)
    - `g = NaN` -> `NaN`.

---

## 3. Stage 3 — Attention & Trig Helper (3 files)

### 3.1 `vulkan_trig.h` *(clean-room replacement for `omarchy_trig.h`)*
- Provides float32 `sin`/`cos` helper functions with clean range reduction (`x` reduced modulo `2*pi` / Cody-Waite or Payne-Hanek / quadrant reduction to `[-pi/4, pi/4]` + minimax polynomial) so large RoPE angles (`pos * inv_freq` up to `pos ~ 8192`) remain accurate within a few ULPs across all Vulkan drivers (Apple Honeykrisp, NVIDIA, AMD, Lavapipe).
- Clean-room: standard mathematical range reduction + minimax polynomial, zero code copied from any external engine.

### 3.2 `fast_rope.comp`
- **Variants**:
  - `rope_bf16`: `-DUSE_BF16 -DORACLE_QWEN35` (BF16 in, BF16 out; addresses packed 32-bit pairs when `count = total / 2` OR elements — note `attention.zig:133,156` passes `count = plan.words = total / 2` for `rope_bf16` and `count = plan.elements = total` for `rope_f32` and `rope_f32_kcache`!)
  - `rope_f32`: `-DORACLE_QWEN35` (F32 in, F32 out)
  - `rope_f32_kcache`: `-DORACLE_QWEN35 -DACT_OUT_BF16` (F32 in, BF16 out — **MUST declare `GL_EXT_shader_16bit_storage` and `GL_EXT_shader_explicit_arithmetic_types_int16`** [Synthena bug #5]!)
- **Bindings (3, writes mask `1 << 1`)**:
  - `binding = 0` readonly: input (`uint16_t` under `USE_BF16`, `float` otherwise)
  - `binding = 1` writeonly: output (`uint16_t` under `USE_BF16` or `ACT_OUT_BF16`, `float` otherwise)
  - `binding = 2` readonly: `int offsets[]` (`offsets[0]` is the `i32` base position).
- **Push constants**: 120 bytes (`Params`). Caller (`attention.zig:214`) sets:
  - `count`: `total / 2` (for `rope_bf16`) or `total` (for `rope_f32` / `rope_f32_kcache`)
  - `flags = 1u | 8u`: **MUST test `(params.flags & 1u)` (forward rotation) and `(params.flags & 8u)` (passthrough tail `d >= 2 * half_dims`)** — `attention.zig:397-398` pins those literal flag-test expressions!
  - `alpha = 1.0` (position scale)
  - `beta = log(theta) / float(half_dims)` (where `theta = 1e7`, `half_dims = 32`)
  - `dims = half_dims` (`32`)
  - `shape = { heads, rows, head_dim, 0 }` (e.g. `{8, rows, 256, 0}` or `{2, rows, 256, 0}`)
  - `in_strides = { in_head, in_time, 1, 0 }`
  - `out_strides = { out_head, out_time, 1, 0 }`
  - `output_offset = out_base` (in elements, e.g. `base * head_dim`)
- **Behaviour**:
  - Half-split RoPE (`rotary_dim = 2 * half_dims = 64`, `head_dim = 256`):
  - **CRITICAL (Synthena bug #6)**: Route EVERY frequency calculation through ONE dedicated helper matching Metal's `pow` semantics: `inv_freq(uint i) = pow(theta, -float(i) / float(half_dims))` (note `theta` can be recovered from `exp(params.beta * float(params.dims))` OR passed/computed consistently via `pow`, NEVER bypassing with inline `exp(-float(i) * params.beta)` for some paths and `pow` for others!). Confirm directed bit comparison against Metal/host reference.
  - For each `(t, h)` in `[0, rows) x [0, heads)`:
    - `pos = params.alpha * float(int(t) + offsets[0])`.
    - For `i` in `[0, half_dims)`:
      - `angle = pos * inv_freq(i)`; `(c, s) = trig_cos_sin(angle)`.
      - `x0 = load_f32(in_base + i)`, `x1 = load_f32(in_base + i + half_dims)`.
      - Forward (`(params.flags & 1u) != 0u`): `y0 = x0 * c - x1 * s`, `y1 = x0 * s + x1 * c`. (If bit 0 clear: inverse sign on `s`.)
      - Store `y0` at `out_base + i` and `y1` at `out_base + i + half_dims` (single RNE BF16 store when output is BF16; no intermediate BF16 rounding).
    - For `d` in `[2 * half_dims, head_dim)` when `(params.flags & 8u) != 0u`:
      - Passthrough tail: copy `in[in_base + d]` -> `out[out_base + d]` bit-exact when BF16->BF16 or F32->F32, and RNE `f32_to_bf16` when F32->BF16 (`rope_f32_kcache`).

### 3.3 `sdpa_decode_native.comp`
- **Variants**:
  - `sdpa_rows_bf16`: `-DBF16_IO -DSDPA_DIM=256 -DSDPA_ROWS` (Q/K/V/Out all BF16)
  - `sdpa_rows_f32q`: `-DBF16_IO -DF32_Q -DF32_OUT -DSDPA_DIM=256 -DSDPA_ROWS` (Q/Out F32, K/V cache BF16)
- **Bindings (4, writes mask `1 << 3`)**: `0`: `Q`, `1`: `K_cache` (BF16), `2`: `V_cache` (BF16), `3`: `Y` (output).
- **Push constants**: 120 bytes (`Params`). Caller (`attention.zig:179`) sets:
  - `matrix_m = query_heads` (`8`), `matrix_n = kv_heads` (`2`), `matrix_k = total_keys` (`prefix + rows <= 7168`)
  - `alpha = scale` (`1.0 / 16.0`)
  - `dims = rows` (`q_len`)
  - `shape = { head_dim (256), capacity * head_dim, head_dim (256), 1 }`
  - `in_strides = { 1, capacity * head_dim, head_dim (256), 1 }`
  - `out_strides = { query_heads * head_dim (2048), 0, 0, 0 }`
- **Dispatch**: `groups = (query_heads, rows, 1)`. One workgroup per `(head = gl_WorkGroupID.x, row = gl_WorkGroupID.y)`.
- **Limits**: Shared memory `<= 30720` bytes (`attention.zig:32`), `local_size_x = 256` (fits `maxComputeWorkGroupInvocations >= 256`).
- **Behaviour**:
  - Causal visibility: `uint visible_keys = params.matrix_k - params.dims + row + 1u;` (strictly reads keys `0 .. visible_keys - 1`; keys `>= visible_keys` MUST NEVER be read because tests poison future keys with `0xffff` NaN!).
  - `uint kv_head = head / (params.matrix_m / params.matrix_n);`
  - Query row address: `Q` has shape `[rows, query_heads, 256]`, so element `d` of `(row, head)` is at `(row * params.matrix_m + head) * 256u + d`.
  - Output row address: `Y` has shape `[rows, query_heads, 256]`, element `d` at `row * params.out_strides[0] + head * 256u + d`.
  - K/V cache address: `[kv_heads, capacity, 256]`, key `k` element `d` at `kv_head * params.shape[1] + k * params.shape[2] + d`.
  - **CRITICAL (Synthena bug #7 — single-key exact transport)**: When `visible_keys == 1u`, check that the single dot-product score `s0 = dot(q_scaled, k0)` is finite (if nonfinite, let softmax produce NaN as usual); when `s0` is finite, **bypass zero-start accumulation** and store widened `V[0]` directly (exact bit-for-bit BF16 `-0.0` `0x8000` and subnormal `0x0001` preservation!).
  - For `visible_keys > 1u`: compute float32 scores `s[k] = dot(q * scale, k_row[k])` for `k < visible_keys`, find `m = max_k s[k]`, compute `w[k] = exp(s[k] - m)` and `denom = sum_k w[k]`, accumulate `acc[d] = sum_k (w[k] * v_row[k][d])` in float32, store `acc[d] / denom` (F32 under `F32_OUT`, RNE BF16 otherwise).

---

## 4. Stage 4 — Gated DeltaNet (5 files)

### 4.1 `conv_dw1d.comp`
- **Variants**:
  - `conv_dw1d_bf16`: `-DUSE_BF16` (BF16 in, BF16 weight, BF16 out)
  - `conv_dw1d_f32`: `-DUSE_BF16 -DACT_F32` (F32 in, BF16 weight, F32 out)
- **Bindings (3, writes mask `1 << 2`)**: `0`: `conv_input` (`[tokens + 3, C]`), `1`: `weight` (`[C, 4]`, BF16), `2`: `conv_out` (`[tokens, C]`).
- **Push constants**: 36 bytes (`count..output_offset`). Caller (`gdn.zig:306`) sets:
  - `count = tokens * C`, `operation = 4` (taps), `lhs_size = tokens + 3`, `rhs_size = tokens`, `reduce_size = C` (`6144`), `output_size = 1`.
- **Behaviour**: For `gid` in `[0, count)`: `t = gid / C`, `ch = gid % C`.
  - `float acc = 0.0; for (uint k = 0; k < taps; ++k) acc += load_act(conv_input[(t + k) * C + ch]) * bf16_to_f32(weight[ch * taps + k]);`
  - Store `acc` (F32 under `ACT_F32`, RNE BF16 otherwise).

### 4.2 `gdn_conv_decode.comp`
- **Variants**: `gdn_conv_decode_bf16` (`-DUSE_BF16`).
- **Bindings (5, writes mask `(1 << 3) | (1 << 4)`)**:
  - `0` readonly: `conv_state_in` (`[3, C]` BF16)
  - `1` readonly: `x` (`[C]` BF16)
  - `2` readonly: `weight` (`[C, 4]` BF16)
  - `3` writeonly: `activated` (`[C]` BF16)
  - `4` writeonly: `conv_state_out` (`[3, C]` BF16)
- **Push constants**: 60 bytes (`count..flags`). Caller (`gdn.zig:329`) sets `count = C (6144)`, `operation = 4`, `lhs_size = 3`, `reduce_size = C`, `flags = 0`.
- **Host flag-pin test (`gdn.zig:553-554`)**: Source MUST test `(params.flags & 1u)` (optional SiLU) and `(params.flags & 2u)` (optional norm) — both remain inactive when `flags == 0`.
- **Behaviour**: For channel `ch` in `[0, C)`:
  - Tap values `s0 = state_in[0*C + ch]`, `s1 = state_in[1*C + ch]`, `s2 = state_in[2*C + ch]`, `s3 = x[ch]`.
  - Compute `acc = s0*w[0] + s1*w[1] + s2*w[2] + s3*w[3]` in float32.
  - If `(params.flags & 1u) != 0u`, apply `acc = acc / (1.0 + exp(-acc))`. (Bit 1 reserved/optional per host pin.)
  - Store `activated[ch] = f32_to_bf16(acc)`.
  - Shift state bit-exact: `state_out[0*C + ch] = state_in[1*C + ch]`, `state_out[1*C + ch] = state_in[2*C + ch]`, `state_out[2*C + ch] = x[ch]`.

### 4.3 `gated_delta_prefill_oracle.comp`
- **Variants**:
  - `prefill_recur_bf16`: `-DUSE_BF16` (`q/k/v/beta/y` BF16; `h0/hf/g_f32/snapshot` F32)
  - `prefill_recur_f32`: `-DACT_F32` (`q/k/v/beta/y/h0/hf/g_f32/snapshot` all F32)
- **Bindings (10, writes mask `(1 << 6) | (1 << 7) | (1 << 9)`)**:
  - `0`: `q` (`[T, Hv, Dk]`), `1`: `k` (`[T, Hv, Dk]`), `2`: `v` (`[T, Hv, Dv]`), `3`: `g_bf16` (unused dummy when bit 2 set), `4`: `beta` (`[T, Hv]`), `5`: `h0` (`[Hv, Dv, Dk]` F32), `6`: `y` (`[T, Hv, Dv]`), `7`: `hf` (`[Hv, Dv, Dk]` F32), `8`: `g_f32` (`[T, Hv]` F32), `9`: `snapshot` (`[(chunks - 1), Hv, Dv, Dk]` F32).
- **Push constants**: 128 bytes (`Params` 120B + `uint chunk; uint chunks;`). Caller (`gdn.zig:373`) sets:
  - `dims = tokens`, `matrix_m = Dk (128)`, `matrix_n = Dv (128)`, `matrix_k = Hv (16)`, `flags = 4u` (scan pass) or `4u | 8u` (output pass), `chunk`, `chunks`.
  - **Host flag-pin test (`gdn.zig:549-550`)**: Source MUST test `(params.flags & 4u)` (read `g` from `binding 8` F32) and `(params.flags & 8u)` (output pass vs scan pass).
- **Dispatch**: `groups = (Hv * 4, chunks, 1)` (4 row blocks of 32 `Dv` rows per head).
- **Behaviour** (exact match to `contract_gdn_ref.zig:176-227` `walk`):
  - State layout: `S[h, row, i]` at `((h * Dv + row) * Dk + i)` where `row` in `[0, Dv)`, `i` in `[0, Dk)`.
  - Two-pass chunked execution when `chunks > 1` (single output pass when `chunks == 1`):
    - **Scan pass (`(flags & 8u) == 0u`, `gl_WorkGroupID.y == 0`)**: Starting from `h0`, step `t = 0 .. tokens - 1`. At each interior chunk boundary `(t + 1) % chunk == 0 && t + 1 < tokens`, save state `S` into `snapshot[((t + 1)/chunk - 1) * Hv * Dv * Dk + ...]`. At `t = tokens - 1`, write final state to `hf`. (Workgroups with `gl_WorkGroupID.y > 0` exit immediately in scan pass.)
    - **Output pass (`(flags & 8u) != 0u`)**: Chunk `c = gl_WorkGroupID.y` covers `t_start = c * chunk`, `t_end = min(tokens, t_start + chunk)`. Load initial chunk state from `h0` if `c == 0`, else from `snapshot[(c - 1) * Hv * Dv * Dk + ...]`. For each `t` in `[t_start, t_end)`:
      1. `kv = sum_{i=0..Dk-1} (S[h, row, i] * g[t, h]) * k[t, h, i]` (float32)
      2. `delta = (v[t, h, row] - kv) * beta[t, h]`
      3. `S[h, row, i] = S[h, row, i] * g[t, h] + delta * k[t, h, i]`
      4. `o = sum_{i=0..Dk-1} S[h, row, i] * q[t, h, i]`
      5. Store `y[(t * Hv + h) * Dv + row] = o` (RNE BF16 or F32).
      6. If `c == chunks - 1`: write final state `S` to `hf` (covers `chunks == 1` cleanly too!).

### 4.4 `gated_delta_decode.comp`
- **Variants**: `gated_delta_decode_bf16` (`-DUSE_BF16 -DORACLE_QWEN35`).
- **Bindings (10, writes mask `(1 << 6) | (1 << 7)`)**:
  - `0`: `q` (`[Hv, Dk]` BF16), `1`: `k` (`[Hv, Dk]` BF16), `2`: `v` (`[Hv, Dv]` BF16), `3`: `a` (`[Hv]` BF16), `4`: `b` (`[Hv]` BF16), `5`: `state_in` (`[Hv, Dv, Dk]` F32), `6`: `y` (`[Hv, Dv]` BF16), `7`: `state_out` (`[Hv, Dv, Dk]` F32), `8`: `a_log_u16` (`[2*Hv]` `uint16_t` packing F32 `A_log[h]` as little-endian lo/hi halfwords), `9`: `dt` (`[Hv]` BF16).
- **Push constants**: 120 bytes (`Params`). Caller (`gdn.zig:402`) sets `matrix_m = 128`, `matrix_n = 128`, `matrix_k = 16`, `flags = 8u | 512u`, `in_strides[2] = 16`.
- **Host flag-pin test (`gdn.zig:544-545`)**: Source MUST test `(params.flags & 8u)` (compute raw gates from `a, b, a_log, dt`) and `(params.flags & 512u)` (unpack F32 `A_log` from two `uint16_t` halfwords `uint(lo) | (uint(hi) << 16u)`).
- **Dispatch**: `groups = (Hv (16), Dv / 32 (4), 1)`.
- **Behaviour**:
  - Raw gates prologue (exact match to `stage4_fixture_core.zig:79-90` `gateRefOracle`):
    - `s = bf16_round(bf16_to_f32(a[h]) + bf16_to_f32(dt[h]))`
    - `sp = bf16_round(max(s, 0.0) + log(1.0 + exp(-abs(s))))`
    - `a_log = uintBitsToFloat(uint(a_log_u16[2*h]) | (uint(a_log_u16[2*h + 1]) << 16u))`
    - `g = exp(-exp(a_log) * sp)` (float32)
    - `beta = bf16_round(1.0 / (1.0 + exp(-bf16_to_f32(b[h]))))`
  - Single-step recurrence (`T = 1`) over the workgroup's 32 `row` slices of head `h`: identical math to `gated_delta_prefill_oracle.comp`, writing `y[h * Dv + row]` (RNE BF16) and `state_out[(h * Dv + row) * Dk + i]` (F32).

### 4.5 `fast_norm_gated.comp`
- **Variants**:
  - `fast_norm_gated_bf16`: `-DUSE_BF16 -DORACLE_QWEN35` (BF16 `y`, `weight`, `z`, `out`)
  - `fast_norm_gated_f32`: `-DUSE_BF16 -DORACLE_QWEN35 -DACT_F32` (F32 `y`, `z`, `out`; BF16 `weight`)
- **Bindings (4, writes mask `1 << 3`)**: `0`: `y`, `1`: `weight` (`[128]` BF16, repeated per head row), `2`: `z`, `3`: `out`.
- **Push constants**: 68 bytes (`count..beta`). Caller (`gdn.zig:422`) sets `operation = 0`, `lhs_size = 128`, `reduce_size = 128`, `output_size = tokens * 16`, `alpha = eps`.
- **Dispatch**: 1D grid-stride loop over `output_size` head-rows (`width = 128`).
- **Behaviour** (exact match to `stage4_fixture_state.zig:360-389`):
  - For each head-row `row` of width `128`:
    1. `ss = sum_{i=0..127} y_f32[i]^2`, `inv = 1.0 / sqrt(ss / 128.0 + eps)`.
    2. `scaled = y_f32[i] * inv`.
    3. Under `fast_norm_gated_bf16`: `normed = bf16_to_f32(f32_to_bf16(bf16_to_f32(weight[i]) * scaled))` (intermediate BF16 round matches Metal `gdn_post`!); under `fast_norm_gated_f32`: `normed = bf16_to_f32(weight[i]) * scaled`.
    4. `silu = z_f32[i] / (1.0 + exp(-z_f32[i]))`.
    5. Store `silu * normed` (RNE BF16 or F32).

---

## 5. Stage 5 — Greedy Argmax (2 files)

### 5.1 `argreduce_chunk.comp`
- **Variants**: `argreduce_chunk` (`-DUSE_BF16`).
- **Bindings (2, writes mask `1 << 1`)**:
  - `0` readonly: `logits` (`uint16_t` BF16, length `reduce_size`)
  - `1` writeonly: `partials` (array of `(float val, uint idx)` pairs, 8 bytes/chunk, indexed by `gl_WorkGroupID.x`)
- **Push constants**: 60 bytes (`count..flags`). Caller (`greedy.zig:139`) sets:
  - `operation = 1` (argmax), `reduce_size = count`, `matrix_m = 4096` (`chunk_len`), `matrix_n = chunks`.
- **Dispatch**: `groups = (chunks, 1, 1)`, `local_size_x = 256` (each thread scans up to 16 elements in chunk `gl_WorkGroupID.x`).
- **Behaviour**:
  - Chunk covers `[start, end)` where `start = gl_WorkGroupID.x * params.matrix_m`, `end = min(start + params.matrix_m, params.reduce_size)`.
  - Widens each BF16 logit to `float` (`bf16_to_f32`).
  - Tie-breaking rule (`stage5_fixture.zig:19-33`): **strictly greater (`val > best_val`) wins; equal values (including `+0.0 == -0.0`) ALWAYS keep the LOWEST global index (`idx < best_idx`)!**
  - Writes `(best_val, best_idx)` for this chunk to `partials[gl_WorkGroupID.x]`.

### 5.2 `argreduce_combine.comp`
- **Variants**: `argreduce_combine` (no defines).
- **Bindings (2, writes mask `1 << 1`)**:
  - `0` readonly: `partials` (`struct { float val; uint idx; }`, length `matrix_n <= 256`)
  - `1` writeonly: `results` (`uint[]`, writes winning index to `results[params.output_offset]`, where `output_offset = 2` = `Result.index`)
- **Push constants**: 60 bytes (`count..flags`). Caller (`greedy.zig:146`) sets `operation = 1`, `matrix_n = chunks`, `output_offset = 2`.
- **Dispatch**: `groups = (1, 1, 1)`, `local_size_x = 256`.
- **Behaviour**:
  - Reduces the `matrix_n` `(val, idx)` pairs in shared memory using the exact same tie-breaking rule: higher `val` wins; exact ties (`a.val == b.val`, including `+0.0 == -0.0`) keep the smaller `idx`!
  - Lane 0 writes `results[params.output_offset] = best_idx`.
