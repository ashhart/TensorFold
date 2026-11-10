# Adding a Zig family, and taking it to full speed

This is how a model goes onto the Zig engine and gets every speed-up we know. The steps are in order. Each one is a
known win with its measured result, so a new model goes through them without measuring each one again. It comes from
three bring-ups:

- Flash Next 6-bit on an M5 Ultra, 5 Oct 2026. Served decode went from 117-120 tok/s on the Python engine to 134-160,
  and prefill from 2,000-2,500 to 4,000-4,200 tok/s.
- Nemotron 3.5 Lightning on an M5 Max, 3-4 Oct. Prose went to 272-277 tok/s, against about 220 on Python.
- Kimi K3 on four M3 Ultras, 3-4 Oct. One stream went from 2.7 to 11.6 tok/s plain, and 20.3 drafted on prose.

The Python engine's guides, [adding an MLX family](adding-a-family.md) and
[adding a CUDA family](adding-a-cuda-family.md), describe the engine frozen at 0.6.6. New families go here.

## How to use it

1. Do the steps in order, and build each one rather than probing it first. Build it once, in the core, and plug the
   family in (next section).
2. After each step, run that step's exactness check.
3. Before any speed claim, run a served side-by-side against the current release on the same machine. Cover
   generation, prefill and time to first token, 0.5k to 256k tokens, code, prose and edits, all greedy. Check the
   run's tokens.
4. When a new step proves out on a model, add it here with its result. When one loses, add it to "Don't bother".

## Build each step once, in the core

A family composes the core blocks below and writes only what is its own: its config, weight map and layer graph, and
kernels for its own mixer (KDA, GDN, MLA) or a sparse-attention indexer. Kernels stay per backend. On day one it's
exact against the reference (stage 0); after the lane kernels go in, check against the engine's own plain output
(step 4).

**The host blocks on main.** Block paths are under `zig/src/core`, other paths under `zig/src`. "Families" names the
ones that call the block now.

| Block | The call a family makes | Families |
| --- | --- | --- |
| `lanes/engine.zig`, `lanes/backend.zig` | implements `lanes.backend.Backend` (forwards and draws); the host steps `lanes.Engine.init(gpa, cfg, backend, clock)` | deepseek_v41, dory, dots_mocr, flashnext, gemma4, glm, glm_dsa, glm_ocr, kimi_k3, kolibri1, lfm2_moe, lfm3, lighton_ocr, llama, minimax_m3, nemotron, nemotron_parse, qwen27 (`native/qwen27_lanes.zig`), qwen3, qwen3_5, qwen4, spark25, step5 |
| `lanes/drafter.zig`, `lanes/drafted.zig` | returns a `lanes.drafter.Drafter` (taps, open, absorb, hold, held, release); the host wraps it with `lanes.drafted.Drafted.init(gpa, target, drafter)` (`native/cuda_drafted.zig` on CUDA) | qwen3_5 on CUDA (DFlash2, MTP), gemma4 on CUDA and Metal (DFlash, assistant), minimax_m3 (DSpark). qwen27's DFlash on Metal has its own glue |
| `lanes/solo_chunk.zig` | `lanes.solo_chunk.rows(env_name, limits)` for its solo chunk setting, then `passRows(lanes, own, fill, solo, busy)` when a stream's prompt pass starts: wider chunks while no other stream decodes (`othersDecode` alone for a family that only needs the test) | gemma4 on CUDA (`families/gemma4/cuda_solo.zig`, `TF_GEMMA4_CHUNK_SOLO`), flashnext on CUDA (`families/flashnext/cuda_chunk.zig`), glm on CUDA (`othersDecode` in `families/glm/cuda_prompt_solo.zig`) |
| `lane_host.zig` | its host calls `LaneHost.init(gpa, io, &engine, info)`, then `start` | every CUDA family through `native/cuda.zig`; on Metal, HIP and XPU through their hosts (`native/metal.zig`, `hip.zig`, `xpu.zig`, `qwen35.zig`, `qwen27_served.zig`, `glm_host.zig`, `llama.zig`, `lfm3.zig`, ...). Flash Next on Metal has its own host, one reply at a time, unless `--parallel` above one selects `native/flashnext_batch_host.zig` (`shared` in `families/flashnext/batch_meta.zig`) |
| `engine_api.zig` | its host opens it from `Open` and returns an `Engine` with its `Info`; the HTTP server drives only this | every host |
| `kv_pool.zig` | declares a `kv_pool.Layout`, sizes it with `kv_pool.plan`, holds pages in a `kv_pool.Pool` | on CUDA: dory, flashnext, gemma4, glm, kolibri1, nemotron, qwen3_5, spark25; on Metal: qwen3_5, nemotron (`TF_NEMOTRON_METAL_KV_PAGES=1`), and every Metal family's stream admission (`kv_pool.plan` in `native/metal_admit.zig`) |
| `kv_ledger.zig` | none of its own: a family that exposes `kvPool` or `ledgerPool` gets its pool's pages counted by the lane host's ledger (`native/cuda_kv_pool.zig`); `per_request` declares per-request costs (`native/cuda_ledger.zig`) | the CUDA families on `kv_pool`; no family declares `per_request` yet |
| `prompt_cache.zig` | gives a `prompt_cache.Snapshots` (`bytes`, `save`, `restore`, `drop`; optional spare, disk and rows hooks) to `prompt_cache.Store.init(gpa, snapshots, rules, budget)`, and sets the lane host's `cache` to that store; the lane host (`lane_host.zig:5`) then does the exact prompt reuse | on CUDA, every family that declares `snap_save` (`native/cuda.zig`, hooks from `native/cuda_snaps.zig`): dory, flashnext, gemma4, glm, kolibri1, minimax_m3, nemotron, qwen3_5, spark25; qwen4 through `families/qwen4/lane_snapshots.zig` (`native_host.zig:26`); on Metal, XPU and HIP the hosts' own snapshot hooks (`native/metal.zig`, `qwen35.zig`, `qwen27_served.zig`, `glm_host.zig`, `flashnext_host.zig`, `xpu.zig`, `families/dots_mocr`, `glm_ocr`, `lighton_ocr`, `nemotron_parse`, ...) |
| `prompt_radix.zig` | gives a `Store` its snapshots and pages (`Store.init(gpa, snapshots, pages, rules)`), then `begin`, `resumed`, `keep` | qwen3_5 on CUDA (`TF_QWEN_KV_SHARE`, `families/qwen3_5/cuda_prefix.zig`) and HIP; nemotron on HIP; spark25, kolibri1 and glm on CUDA under `TF_KV_SHARE` (`native/cuda_prompt_pages.zig`) |
| `prompt_tail.zig` | passes its `Facts` to `prompt_tail.start` to keep one row's logits past the last layer's keys and values | qwen3_5 on CUDA |
| `recurrent_workspace.zig` | `Workspace(DeviceBuffer)` for recurrent scratch leases, disposed after their completion watermark | qwen3_5 on CUDA |
| `checkpoint_direct.zig` | `checkpoint.direct.Layers.init(gpa, ck, readers)`, then `enter(prefix, next)` per layer: tensors read past the page cache, the next layer ahead | qwen3_5 and gemma4 on CUDA |
| `direct_gather.zig` | `direct_gather.Pair` and `Source` directly, for a loader that is not layer by layer | flashnext on CUDA |
| `rope_yarn.zig` | `rope_yarn.yarn(theta, factor, trained)`: transformers' static YaRN frequencies and attention factor | qwen3_5 on CUDA (`--rope-yarn`); the CUDA host refuses `--rope-yarn` for other families, as Metal, HIP and XPU do. flashnext keeps its own `yarn` in `families/flashnext/rotary_config.zig` |
| `affine4_host.zig` | `dequantize`, `quantize`, `f32of`, `bf16of` for MLX affine 4-bit at group 64 on the host | nemotron, qwen3_5 (`cuda_mtp_bf16.zig`) |
| `modelopt.zig` | `modelopt.admitRecipes` and `Recipe` to check an NVFP4 or MIXED_PRECISION config before loading; `zig/src/cuda/modelopt.zig` repacks its tensors on the GPU | config checks: qwen3_5 and glm on CUDA. GPU repack: dory, flashnext, minimax_m3, nemotron, qwen3_5 |
| `gguf.zig` | `gguf.Document` over a checked GGUF v3 file | qwen27 on Metal |

**What a family PR must justify.** For each block above it does not use, the pull request names the block and says why
its own code is needed, or refuses that feature by name until the block covers it.

**The rules a family PR follows.**
- Settings are read through `zig/src/platform/environment.zig`; policy switches live in `zig/src/core/profile.zig`.
- Source files stay at or below 600 lines ([CONTRIBUTING.md](../../CONTRIBUTING.md#lean-code-and-review)).
- Each new or changed kernel registers with the [kernel contract](../kernel-contract.md#adding-a-kernel) and its
  records ([the PR skill, section 4](../pr/SKILL.md#4-kernels-arithmetic-and-platforms)).
- The served gate is the family's conformance kit profile (`tools/conformance.py`, #2467) once that lands.

**The GPU side.** Shared kernels so far:
- `zig/kernels/metal/ops/qmv.metal`: affine 4-, 6- and 8-bit row matmuls with MLX's arithmetic, and
  `zig/src/core/frags.zig`, the prompt kernels' fragment layout on M1-M4.
- On CUDA, `zig/kernels/cuda/experts_rows.cu` runs the decode expert kernel over a window's rows, and
  `route_rows.cu` the router logits and softmax top-k (flashnext, glm). A row's bits never depend on its window.
- Staggered prompt segments: `zig/src/core/segments.zig` (Flash Next on Metal) and `zig/src/cuda/segments.zig`
  (Nemotron on CUDA).
- Several Macs: the MCDMA fabric (`zig/src/fabric`). Flash Next splits rows over it in
  `zig/src/families/flashnext/tp.zig`; `zig/src/families/kimi_k3/parallel.zig` is a split plan.

Family-owned GPU rounds remain: Flash Next on Metal replays its recorded kernels
(`zig/src/families/flashnext/replay.zig`), and Nemotron on Metal runs a lone greedy stream's rounds with no host wait
(`zig/src/families/nemotron/gpu_round.zig`).

A model is done when the exactness checks pass and the served side-by-side beats the release on the same machine.

## Tokenizer byte API

`Tokenizer.tokenBytes(allocator, id)` returns an allocator-owned byte slice, or `null` when the decoder
cannot safely decode a token independently of its neighbors. Free a returned slice with the supplied allocator.
Unknown IDs return an empty slice for a supported decoder; special tokens retain their literal spelling.
The bytes may contain incomplete UTF-8, including individual ByteLevel or `<0xNN>` byte-fallback pieces.
Use `Tokenizer.decode` for human-readable text; its existing replacement-character behavior is unchanged.

Supported chains contain ByteLevel or ByteFallback, Fuse, and literal Replace steps before any joining step.
An empty chain or a token-local literal Replace chain is also supported.
Strip, Metaspace, WordPiece, regex replacement, missing decoders, and replacement after joining return `null`.
This API does not enable constrained generation or add a server capability.

## The exactness checks

- **Decode kernels.** Drafted output equals the engine's own one-token-at-a-time output, token for token, at every
  depth and window width. A kernel swap on the target must hold at every width. The MTP head may differ, because its
  drafts are only proposals.
- **Prompt path.** The first token is the same, last-row logits are within bf16 rounding, and the next 48 decoded
  tokens are equal, at 8k and 32k. Run it from a fresh process: an A/B inside one process can hide uninitialized
  state.
- **Precision.** No int8 or fp8 activations. No lossy weight or cache formats, even opt-in, and no mode that changes
  output tokens, even opt-in. A new summation order at equal precision is allowed.

## Stage 0: exact on day one

1. **Record and replay.** Run the reference engine once with a hook on its kernel launches.
   - Record every kernel's source and its launch sites, keyed by role and row count. That covers one-row steps,
     verify windows of 2-16 rows, MTP absorbs and chains, and prompt chunks.
   - Write a pack of prepared weights, reference tokens and fixtures. The Zig engine loads the checkpoint plus the pack
     and replays the kernels natively.
   - Flash Next matched Python bit for bit on its first run: 64/64 tokens, 110/110 window rows, 14/14 drafts.
   - The recorder is `tools/zig/flashnext_dump.py`. Python is only a dev-time oracle here. A model isn't v1 on Zig
     until it runs without a recorded dump.

## Stage 1: the decode round

2. **One command buffer a round.** Drafts stay on the GPU. Flash Next went from 115-118 to 123.2 tok/s at depth 3.
3. **Lane matmuls sum their own groups**, and run on a serial encoder. This removed 99 launches a step: 128.9 tok/s.
4. **Full-width expert kernels at the checkpoint's bit width**, gate and up together, then down: 141.7 tok/s. This
   changes the summation order, so check drafted against the engine's own plain output, not Python's.
5. **A dense matvec for the MTP head only.** The target keeps its lane matmul, because the matvec wins one row and
   loses wide windows: 142.9 tok/s.
6. **GPU-side rounds.**
   - An accept kernel checks drafts against the target's picks on the GPU.
   - It writes the verdict and the next round's positions into a small arena.
   - Two rounds stay in flight, chained by events.
   - The host only reads accepted tokens from a ring (`(round & 511) * 20`), and picks the next draft width from
     rounds two back.
   - 144.8 tok/s, GPU busy 97%.

## Stage 2: fill the lanes

7. **The depth rule.**
   - Three draft levels: 3, 6 and 8 (3 or 7 on a paired Mac).
   - A moving average of drafts landed over drafts offered moves the level: from 3 up to 6 above 0.8, from 6 up to 8
     above 0.74, from 8 back to 6 below 0.62, and from 6 to 3 below 0.65. On a pair, 3 goes to 7 above 0.95.
   - The widest level holds while the moving average stays at 0.62 or above (0.8 on a pair). Metal's MTP chains run
     up to 8 drafts; on CUDA the recipe defaults to 6 (8 for greedy streams), and an explicit depth stops at 15.
   - Flash Next code at depth 6: 6.07 tokens a round, 242.7 tok/s.
8. **Copy drafts on the GPU.** The GPU keeps the token history and finds the longest suffix match, up to 8 tokens.
   - Lenient gate: a match of 3 that reaches 6, or that agrees with the head's first draft.
   - Strict gate: a match of 4 that reaches 9, or that agrees with the first two.
   - Stay lenient while copies land within 0.05 of the head's drafts.
   - Flash Next edits went from 228.6 to 314.1 tok/s, with code and prose unchanged.
9. **Windows of up to 16 rows**, the widest only while copies land.
   - **Routed experts past one row: a 16-group's weight words as the outer loop** (`tf_experts_w`, Nemotron on Metal,
     8 Oct 2026). The lane kernel took its members two at a time with all 16 activations of a chunk live; taking the
     four weight words as the outer loop keeps four activations a member live, so three members a pass at four
     simdgroups fit. Same bits at every width (`tf-nemotron-experts`); the up and down projections of 23 layers went
     from 2.35 to 2.27 ms at 4 rows, 6.68 to 6.48 at 16, 11.93 to 11.24 at 32 and 22.57 to 20.58 at 64, equal at 2.
     Served, cache off, alternated twice: one stream 503 to 513 tok/s on code and 348 to 351 on prose; 8 streams 947
     to 985 and 713 to 731 in all; 32 streams 792 to 802 and 752 to 761. Every reply equal to its solo and plain run.

## Stage 3: the prompt path

On M5 tensor units. Chips before M5 need their own path.

10. **8192-row prompt chunks, and attention on the tensor units.** Each row's query heads go in as one operand over its
    selected keys, so each key is read once for every head. Flash Next went from 2,000-2,500 to 3,300-3,600 tok/s.
11. **Tensor-op matmuls at the checkpoint's bit width** for dense projections and expert gathers, with gate and up
    fused. The router and the split-K hyper-connection projection go on the tensor units too.
12. **Sparse-attention indexer scores on the tensor units**, and block selection past the key budget everywhere:
    decode, prompt chunks and GPU-side rounds.
13. **The MTP head's prompt keys on the tensor units.** With step 11's fused gate-up: 4,000-4,200 tok/s.
14. **The linear-recurrence scan over 16 lanes a row** (DeltaNet). A small gain, kept.
15. **Staggered segments on two queues** (`zig/src/core/segments.zig`, any family).
    - A long prompt's call runs as two segments, each on its own Metal queue with its own prompt buffers.
    - The second segment runs each layer's mixer after the first's, so one segment's scan and glue run beside the
      other's matrix work.
    - The core owns the queues, an MTLFence per segment and an MTLEvent between them. Buffers are untracked, so
      without the fence the encoders overlap and the output breaks.
    - A family supplies hooks: begin, pre, mixer, post and finish. It also supplies `wait` and `handoff`. `wait` says
      which layers wait ahead of their pre-mixer work, such as a convolution tail the segment before writes.
    - Recurrent state slots alternate as serial chunks would.
    - Split only when every segment gets 4,096 rows or more, in even calls (`segments.next`). Each segment re-reads
      nearly every expert's weights, so small segments lose: 2k rows -7%, 3k level. Three queues gave +2%, and four
      timed out.
    - Flash Next served: TTFT -5% to -8% from 8k (32k 7.52 to 6.94 s, 128k 31.5 to 29.1 s), with the same replies.
      The second segment's buffers take about 9 GB. Pushed in c2bede8fd.

## Stage 4: serve

16. **Serve on `tensorfold-native`**, warmed at load, foreground before background, with stop strings and cancel.
17. **A Metal keepalive** for any model that holds a large residency set. That's a one-thread command buffer every
    250-500 ms, on its own queue, inside the engine process. Without it macOS unwires the set after 1-2 s idle, and
    the next request pays a full re-wire: 2.7 s on K3.

Native CUDA families may export `healthy(ctx: *anyopaque) bool`. The native host forwards this callback to
lane health and admission. It must read thread-safe host state without binding a GPU context, waiting on device
work or taking the inference thread's lock. Families without it retain the lane host's normal closing/stale checks.
Unavailable or busy admission returns HTTP 503 before a streaming response opens. Context-window refusals retain
their existing streamed error policy; failures after admission remain stream errors.

## Across several Macs

From Kimi K3 on four Macs:
- Split by tensor parallel and expert parallel over Thunderbolt RDMA, which is two-sided only.
- Send one message per exchange, with a spinning lock instead of a sleeping one. Posting fell from 14-19 us to 2-7 us.
- Put one MTLFence between consecutive round command buffers, or a later buffer's spin wait holds an earlier one's
  work.
- Give each simdgroup one weight tile in one-row dispatches. The rank-to-rank spread came from the threadgroup tail.
  +1.4%.
- Run drafters on other ranks, asked through the round's draft command.
- Send bulk copies between nodes over the Thunderbolt link addresses, not the LAN.

## On CUDA: Nemotron served on a GB10

These wins are on a branch we haven't pushed yet. The Zig CUDA engine replays the Python engine's captured Triton
kernels next to our own .cu kernels, so stage 0 holds from the start: `tensorfold run` gives Python's tokens. These
were the wins, served through `tensorfold-native`, in order. Each kept every check exact.

1. **Graphs on the engine's own sequence.** The first stream takes the sequence the window graphs were captured on
   and replays them. Other streams run eager. Served drafted +0.7-3.5%, serial +4.5%.
2. **lane_gemv and a forked shared expert.**
   - Every projection with K slices runs a column tile's slices in one CTA, summed in order, instead of a cluster per
     tile.
   - The shared expert runs on a forked graph branch while the router and plan run.
   - Byte-equal against the kernels they replace. Serial +6%, drafted +1-4%.
3. **A lone-stream driver.**
   - A lone drafted stream decodes the way `tensorfold run` does, with the depth rule that reads each draft level's
     confidence. The lane core's rule reads acceptance by depth and keeps fewer drafts a round.
   - The driver hands the stream to the lane core when another request arrives.
   - Served drafted +3-7%, 1.05-1.12x the Python engine's one-stream numbers, every rep within 1%.
4. **A graph set per draw mode.** Greedy and sampled each get their own window and head graphs, and the sampled ones
   read the rule from the sequence's buffer. Sampled streams replay graphs and run lone too. Sampled story +8-17%.
5. **Shared rounds.** Several streams' windows run in one forward, 16 rows in all.
   - The dense matmuls, router and experts run once over every row.
   - Each stream's embeds, conv, scan, KV write, attention and draw run on its own row slice, with its own state and
     rule.
   - It stays exact because every 16-row kernel is row-invariant.
   - Aggregate over taking turns, sampled with a seed a stream: 1.17-1.28x at 2 streams, 1.39-1.57x at 4, and
     1.53-1.91x at 8.
   - Greedy cells that send one sequence to every stream go higher, up to 2.91x, and swing by rep.
   - One stream is unchanged.

Next on CUDA: one MTP head forward a level over every stream's row. Each stream drafts on its own today, eagerly, at
0.75 ms a level, which is about 9 ms of a 4-stream round. Graphs for shared windows would save only 1-4.5% of a
forward.

CUDA traps:
- Zig 0.17's `readFileAlloc` reads up to the size `stat` reports, and procfs reports 0. Read /proc files with a
  streaming reader, or the GB10 memory budget silently falls back to MemFree.
- A synchronous copy (`cuMemcpyHtoD`) runs on the legacy stream. A kernel launched right after on a non-blocking
  stream can read the old bytes. Upload on the stream that reads.
- A shared round drafts before it keeps. Commit each window to its kept path at whichever comes first, or the round
  commits twice.
- Two GB10s ran identical code 4-6% apart. Compare engines on the same box.

## What to attack first, by model type

| The model has | Do first |
| --- | --- |
| Routed experts | Step 4 at its bit width, then step 11's fused gate-up. Don't group experts by lane. |
| DeltaNet, Mamba or KDA layers | Step 14, and keep recurrent state fp32. |
| Hyper-connections | Step 3. Split-K hyper-connection projections on the tensor units in prompts. |
| A sparse-attention indexer | Steps 10 and 12. |
| An MTP head | Steps 5 and 13, chains capped at 6. |
| Text that repeats its context, like edits and code | Step 8. |
| A model bigger than one Mac | The section on several Macs. |

## Don't bother: built, run, and lost

- 128-row prompt tiles: same bits, no faster than 64.
- 32-row expert tiles for every size: 4,031 against 4,205 tok/s at 8k.
- A staged DeltaNet prompt scan: slower.
- Coalesced hyper-connection kernels in decode: no change.
- The dense matvec on the target: it wins one row and loses wide windows.
- DeltaNet step and hyper-connection tensor-op kernels at 2-7 rows: exact, slower.
- Split command buffers: no gain, and they need events between them.
- Grouped experts, so each distinct expert is read once for all lanes: bit-identical and slower. The GPU cache already
  shares lanes' expert reads. Packed expert rows: no change.
- Two lane groups at once: the same work as one window of both.
- One launch for all experts: exact, slower.
- Input sums once a threadgroup, and weight prefetch beside short launches: slower.
- A persistent kernel with a GPU-wide barrier: it loses to plain dependent launches on the M5 Ultra.
- A pool of kept prompt-state buffers for Nemotron: a fresh 9.2k-token prompt costs 1.07 s whether the cache is off,
  keeps every state, or evicts and reallocates two 106 MiB states a request (22 evictions in 12 prompts). The GPU copy
  hides a new buffer's cost; Flash Next's pool stays, for its idle pre-touch.
- Nemotron's routed experts, three ways that kept every bit and gained nothing (`tf-nemotron-experts`, M5 Ultra):
  an expert's member blocks spread over grid z instead of looped (the GPU already overlaps a popular expert's loop
  with the other 230 threadgroups a layer); the activations' floats and bf16-rounded group sums prepared once a
  forward instead of remade a member (50 fewer ALU ops a chunk, the same time, so the loop is not ALU-bound);
  8-byte vector loads of the weight words and activations (slower than the compiler's own schedule). Six or eight
  members a pass, or four at two simdgroups, lose 5-40%: only two to four at four simdgroups win.
- Adaptive depth by landed+1: it lost to fixed levels. Full-vocabulary drafts lose too.
- Reshaping the 6-bit expert prompt kernels, all exact, Flash Next on the M5 Ultra at 8k:
  - 128-row paired gate-up tiles lost 25% to register spills.
  - These were all level: 128 rows on 8 simdgroups; mixed 128/32-row heights; 32-column tiles with half the shared
    memory; those tiles double-buffered; and a tile list built once a layer.
  - The paired kernel's gap to the dense projections isn't tiling, occupancy, barrier overlap or the tile search.
- Lanes without a drafter on prose, all measured on K3 and Nemotron:
  - Jacobi self-lanes landed 1.1-1.3 of 32.
  - Recycled lanes landed 0.59 a round, against MTP's 3.01.
  - A phrase store added 2-15%.
  - Logit-lens pruning found the true token in its top 64 at most 6% of the time.
  Lanes land when they're guessed right going in.

## Traps that cost a day each

- Compile Metal kernels at run time with safe math. An offline metallib gives wrong values for bf16 tensor-op inputs.
- Register arrays need full-unroll hints, or kernels run 5-9x slower. A `break` inside an unrolled loop kills the
  unroll.
- `var x: T = undefined` skips Zig field defaults, so set every field. It once gave a wrong first token, and garbage
  replies that only showed in a fresh process.
- The strict tensor-op mode lays out operands differently from our fragments. Use the relaxed mode.
- `half` is a Metal type name. macOS 27 rejects it as a variable name.
- Untracked buffers let command buffers overlap. Separate buffers need an event or fence between them.
- Ordering encoders with a shared event's signal-and-wait works, but it's slow: staggered prefill fell to 2,629 tok/s.
  Use an MTLFence inside a queue and an MTLEvent across queues.
- Per-call metadata buffers must be distinct inside one command buffer, because the GPU reads them when it runs.
- MLX checkpoints put tensors at odd offsets. Load each shard into one Metal buffer; don't mmap typed.
- A binary that leaves the build machine needs `-Dcpu=apple_m1`, or M1-M4 Macs crash on start with SIGILL.
- Check the build's exit status before copying a binary. A failed build piped through grep once shipped the old one.
- Send `"temperature": 0` to the Flash Next host. A missing temperature becomes the model default of 1.0, which the
  host refuses.
- Never quote a speed before checking that run's tokens. A Nemotron build once looked 25% faster because a bug gave
  wrong tokens that were easier to draft.

## Next steps, not yet proven

Try these on Flash Next first, then move each into the stages above with its result:
- Staggered segments for Nemotron and every other family. For Nemotron, the in-projection goes in pre, conv and scan
  are the mixer, and MoE layers need no wait.
- The hyper-connection write fused into its norm in the prompt path.
- Expert inputs read through a row map instead of copied, and the down projection read back through the inverse map.
- Draft trees in one pass per depth, so 16-32 lanes carry the head's best alternatives.
- A drafter trained on the model's own replies.

## Toolchain: Zig is pinned to 0.17.0

Zig 0.17.0 is the release of 1 Oct 2026. 0.18 is in development, and Zig breaks between releases.
- The repo enforces it. `.zig-version` holds 0.17.0, `build.zig.zon` sets it as the minimum, and `build.zig` refuses
  to compile with any other Zig.
- Official builds and their sha256, from ziglang.org's release index:
  - aarch64-macos: zig-aarch64-macos-0.17.0.tar.xz, b607e9b9234790a008116ae5bdb71c6243b84b9fb42a53a9e70fde41c06c536a
  - aarch64-linux: zig-aarch64-linux-0.17.0.tar.xz, 9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
  - x86_64-linux: zig-x86_64-linux-0.17.0.tar.xz, 1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026
- Moving to a newer Zig is a deliberate step: change `.zig-version`, rebuild everything, and rerun every exactness
  check.
