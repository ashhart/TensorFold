# XPU port: navigation guide

A map of the Intel XPU (Arc Pro B70, Xe2) backend of the Zig engine, for a developer opening it for the first time:
where each piece lives, how a request travels through it, and which contracts the tests defend. Everything here is
read from the sources; [README.md](README.md) holds the measurements, the option list and the validation
figures, and this guide does not repeat them.

The port serves one model, Qwen3.8-27B (`model_type` `qwen3_5`: 48 Gated DeltaNet (GDN) layers and 16 attention
layers, the pattern being "every fourth layer is attention"), straight over **Level Zero** (loaded with `dlopen`, no
C header, no SYCL, no PyTorch). All GPU code is OpenCL C compiled offline by `ocloc`, plus GEMMs generated as ISA by
nGEN, and is embedded in the executables.

## 1. Where things are

| Path | What it is |
|---|---|
| [`zig/src/xpu/ze.zig`](../../zig/src/xpu/ze.zig) | Level Zero binding: device (found by PCI id), memory, modules, kernels, in-order command lists, kernel timestamps. A missing symbol or a failed call stops everything. |
| [`zig/src/xpu/qwen27.zig`](../../zig/src/xpu/qwen27.zig) | The model's shape and `Runner`, the object the host drives (prompt, step, verify, commit, slots). |
| [`zig/src/xpu/dev.zig`](../../zig/src/xpu/dev.zig) | Safetensors reader (mmap), the Level Zero device wrapper (`Dev`), kernel profiling. |
| [`zig/src/xpu/load.zig`](../../zig/src/xpu/load.zig) | Weights onto the device: int4 projections (MLX and GPTQ) and the layer table. |
| [`zig/src/xpu/forward.zig`](../../zig/src/xpu/forward.zig) | Activations, prompt GEMM dispatch (fp16, W4A8, W4A4) and `forward`, one pass over the layers. |
| [`zig/src/xpu/runner_multi.zig`](../../zig/src/xpu/runner_multi.zig) | Shared verify rounds across slots (`verifyMulti`, `keepPaths`, `stepMulti`). |
| [`zig/src/xpu/slot_states.zig`](../../zig/src/xpu/slot_states.zig) | Slot state allocation and the prompt cache's snapshots (`saveSlot`, `restoreSlot`). |
| [`zig/src/xpu/drafter.zig`](../../zig/src/xpu/drafter.zig) | The DFlash2 drafter: per-slot contexts, batched lattice. |
| [`zig/src/xpu/drafter_tree.zig`](../../zig/src/xpu/drafter_tree.zig) | The draft tree policy (best-first search over the lattice) and `treeRound`. |
| [`zig/src/xpu/qwen27_main.zig`](../../zig/src/xpu/qwen27_main.zig) | The `xpu-qwen27` test and profiling executable. |
| [`zig/src/native/xpu.zig`](../../zig/src/native/xpu.zig) | The server's entry point for the XPU engine (`open`), the lane host's setup and the prompt cache's hooks. |
| [`zig/src/native/xpu_lanes.zig`](../../zig/src/native/xpu_lanes.zig) | The **lane host** backend (`--parallel N`): slots, shared rounds, lone driver, structured output. |
| [`zig/kernels/xpu/qwen27/*.cl`](../../zig/kernels/xpu/qwen27) | OpenCL C kernels (section 5); `glue.cl` includes `glue_attn.cl` and `glue_quant.cl`. |
| [`zig/tools/xpu/w4_ngen.cpp`](../../zig/tools/xpu/w4_ngen.cpp), `w4_prompt.hpp`, `w4a8_prompt.hpp` | nGEN generator of the prompt GEMMs (section 8). |
| [`zig/tools/xpu/rotate_w4.py`](../../zig/tools/xpu/rotate_w4.py) | Converter to Hadamard-rotated int4 weights (see the [recipe](RECIPE-qwen38-27b-b70.md)). |
| [`zig/build/xpu.zig`](../../zig/build/xpu.zig) | Build steps `xpu-qwen27` and `xpu-native`. |
| [`tools/xpu/`](../../tools/xpu) | Quality studies: `a8_study.py` (W4A8), `head_study.py` (the requantized output head). |
| [`tools/bench_openai.py`](../../tools/bench_openai.py), [`tools/bench_concurrent.py`](../../tools/bench_concurrent.py) | Server benches; `bench_concurrent.py --alone --serial` also checks the bit-for-bit contracts end to end. |

The port modifies no core file: it plugs into the core's lane host and its lone driver (`Lone.resumes`) as they
are; `build.zig` only adds the XPU steps.

## 2. Building

Prerequisites: Zig 0.17; the `xe` kernel driver and Intel's compute runtime (the Level Zero loader,
`libze_loader.so.1`, is opened at run time); `ocloc`, Intel's offline compiler; a C++17 compiler (`c++`); the nGEN
headers of oneDNN.

```sh
zig build xpu-native -Doptimize=ReleaseFast -Dngen-include=<path to ngen>   # zig-out/xpu/bin/tensorfold-native
zig build xpu-qwen27 -Doptimize=ReleaseFast -Dngen-include=<path to ngen>   # test executable xpu-qwen27
zig build grammar    -Dxgrammar=<xgrammar checkout>                         # libtfgrammar, structured output
```

Build options defined in [`zig/build/xpu.zig`](../../zig/build/xpu.zig) and
[`zig/build/grammar.zig`](../../zig/build/grammar.zig): `-Docloc=` (default `ocloc`), `-Dxpu-device=` (default `bmg`,
the `ocloc` device name), `-Dngen-include=` (default `ngen`), `-Dxgrammar=`.

What the two XPU steps do (they share the same embedded kernels):

1. Each `.cl` file is compiled by `ocloc` with `-cl-std=CL3.0 -cl-intel-256-GRF-per-thread`:
   `glue`, `decode`, `gemm` (with `-DGSZ=64`), `gemm128` (the same file with `-DGSZ=128`), `gdn` (from
   `gdn_chunk.cl`) and `drafter`.
2. `w4_ngen.cpp` is built for the host and **run at build time** to produce the nine prompt-GEMM binaries (zebin):
   `ngen_wide`, `ngen_narrow`, `ngen_wide128`, `ngen_narrow128`, `ngen_swiglu`, `ngen_swiglu128`, `ngen_a8`,
   `ngen_a8_swiglu`, `ngen_a4` (the options each gets are listed in `zig/build/xpu.zig`).
3. Everything is embedded with `@embedFile` into the `xpu_qwen27_kernels` module that `qwen27.zig` imports as `kb`.

`xpu-native` is `tensorfold-native` with the XPU engines linked in (`native/xpu.zig`).

`zig build grammar` needs a checkout of **xgrammar at the pinned tag `v0.2.8`** (with submodules; the build refuses any
other checkout) and builds `libtfgrammar` from it plus [`zig/src/grammar/tf_grammar.cc`](../../zig/src/grammar/tf_grammar.cc).
The server loads it lazily, at the first structured request, from `<server directory>/../lib/`, then from beside the
server, then from the dynamic loader's path (`zig/src/core/lanes/grammar.zig`). Without it, structured requests are
refused and everything else works.

## 3. The runner: `qwen27.zig`

### Load

`Runner.init(dir, context)`:

1. Reads the option variables (section 9) into module-level switches.
2. Looks for a file named `hadamard128` in the checkpoint directory: if present, the weights are Hadamard-rotated
   (`rot_on`) and every GEMM input is rotated where it is produced.
3. Memory-maps the safetensors shards listed in `model.safetensors.index.json`.
4. Opens the Level Zero device whose PCI id is `0xe223` (the B70), creates one in-order immediate command list, loads
   the embedded modules and creates every kernel with its work-group size.
5. Calls `load`, which picks the format from the tensors: a checkpoint with `in_proj_qkv.qweight` is **GPTQ**
   (`GS = 128`, `loadGptq`), otherwise **MLX affine** (`GS = 64`). Projections are repacked on the GPU into the GEMM
   layout (`repack`, `repack_gptq`); in the GPTQ case the output head is requantized to affine 4/64 with MLX's
   algorithm (`quantize_mlx`).
6. Allocates the activations (`Acts`) and slot 0's states.

A `Proj` is one fused projection in GEMM layout: packed words `[K/8, N]`, scales and biases `[K/gs, N]`, `N` padded
to the 256-column tile, and a `wide` flag (wide shapes use the GEMM family without K slices, narrow ones the sliced
family; see section 5). Projections are fused by stacking on N: gate|up (interleaved by blocks of 16 columns so the
prompt GEMM can apply SwiGLU in its epilogue), q|k|v, the GDN input projections.

### One forward

`forward(d, model, acts, ids, p0, mode)` runs at most `CHUNK` rows (`XPU_CHUNK`, 1 024 by default) at positions
`p0 …`. Per layer:

```
add_rmsnorm  ->  mixer_in GEMM  ->  mixer (attention or GDN)  ->  mixer_out GEMM
add_rmsnorm  ->  gate|up GEMM + SwiGLU  ->  down GEMM            (tap rows for the drafter on 5 layers)
```

then the final norm, the head GEMM, and `argmax_part` / `argmax`. What changes with the row count `T`:

| `T` | GEMM | GDN | Attention |
|---|---|---|---|
| 16 or fewer (decode, verification) | decode kernels (`w4k*` family) | serial `gdn_step`, as decode does | `attn_decode` + `attn_merge` |
| more than 16 (prompt) | nGEN prompt GEMMs (`w4g_*`, or `w4a8` / `w4a4` with the W4A8 options) | chunked `gdn_prep`, `gdn_state`, `gdn_out` on 64-row chunks (padded with neutral rows) | `attn_prompt` (`attn_prompt8` with the int8 cache) |

Below 17 rows the GDN runs serially so that a verified row equals the serial one bit for bit; that is the foundation
of the speculative-decoding contracts (section 10).

`Mode` selects the variants: `want` (no logits, last row, all rows), `stash` (keep the GDN inputs so a partial keep can
be replayed), `taps` (write the five tap rows for the drafter), `tree` (rows form a tree: a per-row table of depth and
ancestors, `TREE` in `decode.cl` and `glue.cl`) and `multi` (each row belongs to a different slot: a per-row table of
state and cache addresses, `MT`, plus per-row positions).

### Runner API

State per sequence ("slot", at most `MAX_SLOTS` = 16): the GDN state (`rec`, plus `rec_save` for the state after a
verified block), the convolution state (ping-pong pair), and the K/V cache of the 16 attention layers. `select(slot)`
makes a slot current (allocating on first use); `slotBytes()` is what one slot costs.

| Call | Purpose |
|---|---|
| `init`, `reset` | Load; start a fresh sequence in the current slot. |
| `prompt(ids, last)` | Feed prompt tokens in `CHUNK`-row forwards. With `last`, the final row's greedy token (`last`) and logits are kept. |
| `step(token)` | One serial decode token. |
| `verify(ids)` | Run a block of at most 16 rows (the pending token, then drafts as a chain); returns every row's greedy token. The state moves by all rows until `commit`. |
| `commit(keep)` | Keep the first `keep` rows: a whole block takes the state the forward wrote (`rec` and `rec_save` swap); a shorter one rebuilds the GDN and convolution states by replaying the kept rows from the stash. |
| `verifyTree(ids, parents)` / `commitPath(path)` | The same for a tree: `parents[t]` is the row below row `t`; `commitPath` keeps one root path, replays the states along it and moves the path's K/V rows to their final positions (`kv_move`). |
| `verifyMulti(wins)` | One forward for **several slots**: each `Win` is a slot's window (its pending token, then drafts, as a chain or as a tree via `parents`), 16 rows in all. Nothing moves until the keep. |
| `keepMulti(keep)` / `keepPaths(paths)` | Keep, per window, a prefix or a root path of the last `verifyMulti`. A whole chain window takes the state its forward wrote; otherwise its states are replayed from the stash. After tree windows every window replays. |
| `stepMulti(slots, tokens)` | One token for each of several slots: a `verifyMulti` of one-row windows, all kept. |
| `rowLogits(row)`, `logits()` | The logits of a row of the last verify, or of the kept last row, as fp32 on the host (for sampling). |

## 4. The drafter: `drafter.zig`

DFlash2 is a small model (5 layers, hidden size 5 120, 32 query heads and 8 key/value heads of 128) that proposes
tokens; it only changes how many are accepted, because the target verifies every token. Its bf16 weights are
quantized to affine 4/64 **at load** on the device (`quantize4`) and run on the target's int4 GEMM. It reads
`model.safetensors` in the drafter directory (its projections plus the `candidate_selector` codebooks).

- **Context.** The drafter attends the *target's hidden states* ("taps") of the committed rows, taken after target
  layers 5, 19, 33, 47 and 61 (`TAP_LAYERS`). `addTaps` pushes `n` rows through `fc`, its norm and each layer's K/V
  projection and appends them to the context. Each runner slot has its own `Ctx`: per layer, K and V in a **ring** of
  `CAP` = 2 080 positions (`WIN` + `MAXL` + 17, where `WIN` = 2 047 context rows), allocated on first use by `use(slot)` (~42 MB).
  `skip`, `restart` and `end` manage the window (a long prompt only feeds the last `WIN` rows).
- **Block and lattice.** A block of `L` rows (the pending token plus `L − 1` masks) attends the context and itself,
  not causally. Per depth the drafter keeps the top 16 candidates (`TOPK`) with their logits and a selector
  projection (rank-256 predecessor/successor codebooks). `lattice` runs one block; **`latticeMulti` runs one block per
  slot in a single batched pass** (up to `MAXS` = 16 blocks, `MAXB` = 128 rows), and `take(i)` makes block `i` the
  current one.
- **Choosing drafts.** `chain` follows the best child under the selector's score; `tree` is a best-first search
  (a small heap, policy constants `EDGE`, `NOISE`, `SCALE`, `BRANCH`) over the lattice that returns up to `nodes` drafts
  with their parents. For a sampled request it mixes in the target's keyed Gumbel noise at each position, so drafts
  follow the draws the serial reply would make.
- **`treeRound`.** One complete round for a lone stream: lattice, tree, `Runner.verifyTree`, walk the target's draws
  down the tree, `commitPath`, then `addTaps` for the kept rows. It returns the accepted tokens and the *terminal draw*
  (the next pending token).
- **`absorbMulti`.** Pushes the kept rows' taps of several slots in one pass (for a tree, only the kept path's rows).

## 5. Kernels, by file

Kernel code uses subgroups of 16, DPAS (Xe2's matrix unit) and 2D block loads. Every file states its bit-level
conventions in its header comment; read those before changing a reduction order.

| File | Kernels | Role |
|---|---|---|
| `glue.cl` | `embed`, `embed_dense` | Token embedding (MLX 4-bit table, or the GPTQ fp16 table). |
| | `add_rmsnorm`, `add_rmsnorm_dec` | Residual add, RMSNorm, fp16 output plus the per-group row sums the GEMM family needs; the `_dec` variant uses 512 work-items per row for the decode and verification shapes. Can fuse the W4A8 quantization and the Hadamard rotation. |
| | `dense_rows` | The GDN gates `a`, `b` of the GPTQ checkpoint, a small dense fp16 projection by DPAS. |
| | `gdn_pre`, `gdn_pre_rows` | GDN front end: causal convolution (kernel 4) with its state, SiLU, q/k norms, `β` and `g`. One row per group (`gdn_pre`, also the multi-slot and tree variant) or 8 rows by head (`gdn_pre_rows`, long prompt chunks). |
| | `gated_norm` | GDN back end: gated RMSNorm into the `mixer_out` input. |
| | `attn_prep`, `attn_gate`, `rot_heads` | Attention front end (RMS norms, RoPE, K/V cache write at the row's position or its slot's address) and back end (sigmoid gate); head rotation for the K/V cache studies. |
| | `swiglu`, `rowsums_ff` | SwiGLU and its row sums, and the row sums alone after the GEMM-epilogue SwiGLU. |
| | `quant_a8`, `quant_a4`, `fakequant_rows` | Activation quantization for the W4A8 / W4A4 prompt GEMMs; the study version that quantizes and dequantizes in place. |
| | `argmax_part`, `argmax`, `tap`, `conv_commit` | Greedy token (two kernels), tap rows for the drafter, convolution-state replay after a partial keep. |
| | `repack`, `repack_gptq`, `quantize_mlx` | Load-time repacking of MLX and GPTQ weights into the GEMM layout, and the MLX-algorithm requantization of the GPTQ head. |
| `decode.cl` | `gdn_step` | Serial GDN for a few rows, for decode, verification (also replay with `ROWS`, trees with `TREE`, several slots with `MT`). |
| | `attn_decode`, `attn_merge` | Attention over the cache by key slices (`SPLIT` = 32) and their ordered merge. Tree rows see their ancestors; `MT` selects each row's slot. |
| | `kv_move` | Moves a kept path's K/V rows to consecutive positions, in depth order. |
| | `attn_prompt`, `attn_prompt8` | Prompt attention: FlashAttention-2 on DPAS, causal, bf16 cache and int8 cache (Q·Kᵀ in int8 DPAS, P·V in fp16). |
| | `exl3_gemv` | EXL3 decode probe (speed measurement only; `XPU_EXL3_PROBE`). |
| `gdn_chunk.cl` | `gdn_prep`, `gdn_state`, `gdn_out` | Chunked GDN for prompts (64-token chunks on bf16 DPAS, state in fp32): per-chunk preparation, the serial state recurrence across chunks, and the output. |
| `gemm.cl` | `w4_*`, `w4n_*`, `w4s3_*`, `w4k*` | Weight-only int4 GEMM families, compiled for group 64 or 128. The retained decode/verification kernels are `w4k1_decode`, `w4k1_d64` (wide shapes) and `w4k_dsplit` (narrow); the others are earlier families and sweeps kept as references (`w4_*` is the g64 fp16-dequantization reference). `rowsum`, `rowsum_hl` and `w4s3_reduce` are helpers. |
| `drafter.cl` | `dconv`, `dprep`, `dattn`, `rowsums_rows`, `taps_in`, `to_res`, `topk_part`, `quantize4` | DFlash2: grouped two-tap dynamic convolution, q/k/v preparation with a ring cache, block attention, tap handling, top-16 of the logits rows, load-time int4 quantization. |

The GEMM families share one arithmetic contract: **the bits of a row do not depend on the number of rows, on its
position, or on the tiling**. Each row has its own accumulator and the summation order is fixed (K in slices of groups
each summed from 0, then `((t0 + t1) + t2) + t3`, then the bias by DPAS), so the decode kernels, the nGEN prompt
kernels and every split produce the same bits. This is what lets `verify`, `verifyMulti` and a plain `step` agree.

## 6. The server hosts

`native/xpu.zig` is what `tensorfold-native serve` calls. `open` refuses any `model_type` other than `qwen3_5`,
takes `--context` (32 768 when not given), builds the `Runner` and serves it through the core's lane host for any
`--parallel` value (the core removed its serial host in 1.0.9).

**Lane host (`--parallel N`).** `openLanes` creates `Lanes` (`xpu_lanes.zig`) as the backend of the core's lane
host (`engine_api.LaneHost`, [`core/lane_host.zig`](../../zig/src/core/lane_host.zig)). The number of slots is
`slotsThatFit`: the device's free memory read through Sysman once the model, the first stream's state and the drafter
are loaded (`Gpu.freeMemory`), less a 2 GiB margin and the prompt cache, divided by `slotBytes()`, plus the first
stream, at most `MAX_SLOTS`. When the margin and the cache do not fit, there is no slot and the engine refuses to
open with a message. At startup `measure()` times a verify plus keep of 1 to 16 rows on a
scratch context in the last slot (logged as `window ms by rows`); the core's depth rule uses these costs to weigh
expected tokens per millisecond (without them it drafts at most 3 rows).

`Lanes.backend()` exposes the vtable the core drives: `prefill`, `first` / `first_masked`, `queue`, `read`, `verify`,
`keep`, `draft`, `tree`, `release`. How each maps to the runner:

| Backend call | Runner / drafter calls |
|---|---|
| `prefill` | `select(slot)`, `reset`, `prompt(ids, true)` with taps on; the last row's logits kept for sampled or structured streams. A free slot is taken from the top, so slot 0 stays for the lone driver. |
| `queue` | `select(slot)`, `step(token)` (a one-token advance). |
| `verify(windows)` | `settle`, then **`verifyMulti`**: every stream's window (pending token plus drafts, as a chain or a tree) in one forward of at most 16 rows; each row is drawn with its stream's sampling at its position, or among its grammar's allowed tokens (`drawMasked`). |
| `keep(windows, paths)` | `keepPaths`, after absorbing the kept rows' taps into each stream's drafter context (`absorb`). |
| `draft(requests)` | `absorb` the kept rows not yet taken, then **one batched drafter pass** (`latticeMulti`) for all streams, then per stream a tree (`tree`) of its asked drafts, or a chain below `TREE_MIN` = 5 drafts. The result is held per slot and handed to the core by `tree`. |

`settle` exists because the core calls `keep` only to drop rows: a verify whose rows all stayed is kept lazily, at the
start of the next call.

**Lone driver and resume.** A single stream that asks for drafts does not use shared rounds. `Lanes.lone` takes it in
slot 0 and runs `alone`, a loop of `treeRound` (DFlash2 trees, sampled or greedy) that calls the host's `yield` hook
before each round; when another stream arrives it saves `cache_len` and the pending token and returns `true`, and the
lane core takes the stream over in shared rounds. When the others finish and one stream remains, the core hands it back
through `resumeLone` (the core's `Lone.resumes` and `Engine.detach`; `lone` sees `hooks.resumed`): the drafter context is restarted if it does not end at the
runner's position, and the loop continues.

**Structured output.** The lane host is created with `.structures = true`. With `libtfgrammar` available, a request
with `response_format` or `guided_*` fields gets its tokens masked: windows carry one grammar bit-mask per row; greedy
takes the allowed argmax (first on ties), sampled takes the keyed sampler over the allowed tokens (their top `k`).
`first_masked` does the same for the first token. Such a request does not use the lone driver's trees.

## 7. A request, end to end (lane host)

1. The HTTP server tokenizes and renders the chat template, then submits a stream to the core.
2. `prefill`: the stream gets a slot; the prompt runs in 1 024-row forwards (`Runner.prompt`). Taps of the last `WIN`
   rows feed the slot's drafter context.
3. `first` draws the first token from the kept logits (argmax, or the sampler at the stream's seed and position).
4. Rounds: the core asks `draft` for the streams that want drafts, builds one window per stream (pending token plus
   drafts), and calls `verify` (one forward, at most 16 rows across streams). It decides each window's accepted path
   from the drawn tokens and calls `keep`; the runner replays or swaps states and the drafters absorb the kept rows.
   A lone stream runs the same idea in `alone` with `treeRound`.
5. Tokens are committed to the stream and streamed back until a stop token or `max_tokens`.

## 8. The prompt GEMM generator: `w4_ngen.cpp`

`w4_ngen.cpp` (the command line) builds the kernels of `w4_prompt.hpp` (fp16 activations) and `w4a8_prompt.hpp`
(int8 or int4 activations), written against nGEN (oneDNN's header-only assembler for Intel GPUs). Given a hardware (`xe2` or `xe3`), a group size
(64 or 128), an output path and option words, it writes a zebin: `./w4_ngen xe2|xe3 64|128 OUT.zebin [options]`. The
options each build kernel gets are in `zig/build/xpu.zig`. The generated
GEMMs have the same arithmetic and tiling as the OpenCL reference `w4n_prompt`, hence the same bits:

- A subgroup is one SIMD16 thread over 32 rows × 32 columns; a work-group is 2 × 8 subgroups (64 × 256); the
  grid is `(GM, N / 256, packets)` so that all the columns of a packet of row tiles run before the next (A stays
  in L2).
- Per group of `GSZ`: a DPAS partial sum from 0, then `acc += part · (s · 2²⁴)` (the int4 values enter the DPAS as fp16
  subnormals) and the bias term.
- Families (kernel names in the zebin): `w4g_dpb` (wide shapes), `w4g_s4dpb` (narrow shapes: K in four slices of groups
  summed separately, `((t0 + t1) + t2) + t3`), `w4g_dpb_swiglu` (gate|up interleaved, SwiGLU in the epilogue), `w4a8`
  and `w4a8_swiglu` (int8 activations, int8 DPAS, GPTQ g128 only) and `w4a4` (int4 activations on rotated weights).
- Option words: `b3` (B over three buffers), `s4` (the sliced family), `asplit` (A read in two halves, overlapping the
  DPAS), `swiglu`, and `a8` / `a4` for the integer-activation kernels. The register map is documented in each header.

## 9. Environment variables

The server and the runner read the same variables. See [README.md](README.md) for the effect and the measured value
of the user-facing options; this table says where each is read.

| Variable | Read in | Meaning |
|---|---|---|
| `XPU_CHUNK` | `Runner.init` | Prompt rows per forward (1 024; a multiple of 64). |
| `XPU_TREE`, `XPU_BLOCK` | `native/xpu.zig`, `qwen27_main.zig` | Tree nodes per round (server 15; the test runner 0, i.e. chains) and drafter block rows (8). |
| `XPU_PREFILL_INT8`, `XPU_PREFILL_INT4`, `XPU_PREFILL_INT4_SITES` | `Runner.init` | W4A8 and W4A4 prompt GEMMs (local options; W4A4 needs W4A8 and rotated weights, else `Int4NeedsRotatedWeightsAndInt8`). |
| `XPU_KV_INT8` | `Runner.init` | int8 K/V cache. |
| `XPU_PROFILE` | `Runner.init` | GPU time per kernel (timestamp events), printed by `profReport` after the prompt and the decode (test runner). |
| `XPU_DUMP_H=FILE` | `Runner.init` | Append every row's final normed hidden state (fp16) to FILE, for the quality studies. |
| `XPU_DUMP=DIR` | `Runner.init` | The first forward writes the residual stream after the embedding and after layers 0-3 (comparison with the Python port's layer reference). |
| `XPU_A8`, `XPU_A8_SITES`, `XPU_A8_ROT`, `XPU_A8_BITS`, `XPU_A8_SITES4`, `XPU_KV8` | `Runner.init` | Quantization studies (simulated quantization of GEMM inputs, rotated activations, K/V cache variants). Not for serving. |
| `XPU_CTX` | `qwen27_main.zig` | Context of the test runner (4 096). |

Test-runner-only variables (`qwen27_main.zig`): `XPU_DRAFTER=DIR` (drafted decode compared with the serial reply),
`XPU_VERIFY_TEST`, `XPU_MULTI_TEST=N`, `XPU_ROWS_PROBE`, `XPU_OVERLAP_PROBE`, `XPU_EXL3_PROBE`,
`XPU_GEMM_ZEBIN`, `XPU_A8_ZEBIN` with `XPU_A8_PROJ` (`gu`, `in`, `out`, `down`). Section 10 describes the tests.

## 10. The test runner and the correctness contracts

```sh
xpu-qwen27 MODEL ids.txt 32          # prompt token ids separated by spaces, 32 greedy tokens, timings
xpu-qwen27 MODEL --bench=4096 2      # synthetic prompt of 4 096 tokens (timing only: not real tokens)
```

It loads the model, runs the prompt, decodes `STEPS` greedy tokens serially, prints prompt and decode rates and the
token list, then runs whichever checks the environment asks for. The serial reply is the **reference** the checks
compare against.

| Contract | Checked by | What it does |
|---|---|---|
| **Oracle**: greedy tokens equal the fp32 torch reference of the Python port | Compare the printed `tokens:` line with the reference's | The reference (`tools/xpu-bench/integ/oracle.py` of the Python port, named in `qwen27_main.zig`) is not in this repository. |
| **Drafted == serial** | `XPU_VERIFY_TEST=1` | Blocks of 8 rows verified and committed with *perfect* drafts (the serial reply itself) and with *wrong* drafts (token 0): both replies must equal the serial one. |
| | `XPU_DRAFTER=DIR` (+ `XPU_TREE`, `XPU_BLOCK`) | Greedy decode with real DFlash2 drafts, as chains or as trees, must equal the serial reply; prints draft and verify times. |
| **Multi-stream == serial** | `XPU_MULTI_TEST=N` (N ≤ 16) | N slots, each prompted differently. Three checks against each slot's serial reply: shared one-row rounds (`stepMulti`); shared windows of 4 rows with perfect drafts on even slots and wrong ones on odd slots (`verifyMulti` + `keepMulti`); shared tree windows of 7 rows (`verifyMulti` + `keepPaths`, parents `−1 0 0 2 1 3 3`). Then the aggregate rate of shared rounds for 1 to N slots. |
| **Same bits per projection / per row** | By construction (section 5) and through the checks above | A row's result does not depend on how many rows ran with it, on its chunk, or on which kernel computed it. |
| Cost of a window | `XPU_ROWS_PROBE=1` | Forward time for 1 to 16 rows. |
| Where the time goes | `XPU_PROFILE=1` | GPU time per kernel. |

The server-level versions of the same contracts are run with
[`tools/bench_concurrent.py`](../../tools/bench_concurrent.py) `--alone --serial`: every concurrent reply's token hash
must equal the same request alone, and the alone run must equal the same request with `"draft": false`, for greedy and
sampled requests. `zig build test` covers the core.

When changing a kernel, the quickest honest check is: `xpu-qwen27` with `XPU_VERIFY_TEST=1`,
`XPU_DRAFTER=<drafter dir> XPU_TREE=15` and `XPU_MULTI_TEST=<n>` on real token ids (not `--bench`), then
`bench_concurrent.py --alone --serial` against a running server.
