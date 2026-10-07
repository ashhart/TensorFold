# Adding a Zig family, and taking it to full speed

This is how a model goes onto the Zig engine and gets every speed-up we know. The steps are in order. Each one is a
known win with its measured result, so a new model goes through them without measuring each one again. It comes from
three bring-ups:

- Flash Next 6-bit on an M5 Ultra, 5 Oct 2026. Served decode went from 117-120 tok/s on the Python engine to 134-160,
  and prefill from 2,000-2,500 to 4,000-4,200 tok/s.
- Nemotron 3.5 Lightning on an M5 Max, 3-4 Oct. Prose went to 272-277 tok/s, against about 220 on Python.
- Kimi K3 on four M3 Ultras, 3-4 Oct. One stream went from 2.7 to 11.6 tok/s plain, and 20.3 drafted on prose.

The Python engine's guides, [adding an MLX family](adding-a-family.md) and
[adding a CUDA family](adding-a-cuda-family.md), describe the engine frozen at 0.6.5. New families go here.

## How to use it

1. Do the steps in order, and build each one rather than probing it first. Build it once, in the core, and plug the
   family in (next section).
2. After each step, run that step's exactness check.
3. Before any speed claim, run a served side-by-side against the current release on the same machine. Cover
   generation, prefill and time to first token, 0.5k to 256k tokens, code, prose and edits, all greedy. Check the
   run's tokens.
4. When a new step proves out on a model, add it here with its result. When one loses, add it to "Don't bother".

## Build each step once, in the core

A new model should get every step below by default, not a hand-built engine. So far each family was built as its own
engine, to be exact quickly, and the GPU-side wins stayed inside one family:

- Flash Next replays the Python engine's recorded lane kernels and has its own GPU-side rounds
  (`zig/src/families/flashnext/replay.zig`).
- Kimi K3 and Nemotron have their own rounds.
- Our GLM-5.3 port (not pushed yet) started with MLX-identical row kernels and a host-synchronous round loop.

Some of the host side already lives in core, but so far each piece serves one family, or two:
- the lane round loop, the depth rule and copy proposals (`zig/src/core/lanes`) are Nemotron's;
- prompt reuse (`zig/src/core/prompt_cache.zig`) is Flash Next's;
- staggered segments (`zig/src/core/segments.zig`, and `zig/src/cuda/segments.zig` on CUDA) serve Flash Next on Metal and Nemotron
  on CUDA;
- every host returns the HTTP contract in `zig/src/core/engine_api.zig`, and `zig/src/core/lane_host.zig` serves Nemotron on Metal
  and CUDA. Flash Next has its own host and serves one reply at a time.

The GPU side is barely shared yet, and that costs time. In the GLM-5.3 port each extra row in a window cost about
4.4 ms of GPU time, against about 2.3 ms for the row's own experts. Row kernels read every replicated weight again for
every row.

**What the core owns.** A family gets all of this by plugging in:

1. Lane kernels by format.
   - Formats: affine 4-, 6- and 8-bit at the checkpoint's group size, and bf16. They run on M5 tensor units and in the
     M1-M4 layout (`zig/src/core/frags.zig`).
   - Ops: dense projections; the expert gather (gate and up fused, then down); the router with its top-k and this Mac's
     expert lists; the head with its argmax; norms.
   - Each kernel reads a weight once per window and keeps every row's own sums, so a row's bits are the same at any
     width.
2. GPU-side rounds (steps 6-9).
   - The accept kernel, the ring, and two rounds in flight.
   - The MTP chain driver and copy drafts.
   - A depth rule set from measured tokens per ms for each model.
3. The prompt path (steps 10-15).
4. Several Macs.
   - Tensor and expert parallel over [MCDMA](https://github.com/ashhart/MCDMA), with exchanges inside the command buffer.
   - Experts split by rows, so both Macs do equal work for every pick.
5. Serving (steps 16-17), and the exactness suite.

**What a family supplies.** Its config, weight map and layer graph, plus kernels only for what is its own: a mixer such
as KDA, GDN or MLA, or a sparse-attention indexer. On day one it's exact against the reference (stage 0). After the
lane kernels go in, check against the engine's own plain output (step 4).

**Where each piece lives today:**

| Piece | Shared now | Family copies or gaps |
| --- | --- | --- |
| Lane kernels | `zig/kernels/metal/ops/qmv.metal` (affine 4-, 6- and 8-bit row matmuls, MLX's arithmetic) and `zig/src/core/frags.zig` (the prompt layout on M1-M4) | Flash Next: `zig/kernels/metal/decode/fn_lane.metal` (6-bit, groups of 32, tensor ops, up to 16 rows). Kimi K3: MXFP4 experts and bf16 projections. Nemotron: generated per-shape kernels |
| GPU-side rounds | the host loop, depth rule and copy proposer in `zig/src/core/lanes`, used by Nemotron | Flash Next: its own host loop, plus `fz_accept`, the ring and event chaining (`zig/src/families/flashnext/replay.zig`). Nemotron on Metal: `zig/src/families/nemotron/gpu_round.zig` for a lone stream. Kimi K3: none |
| Several Macs | the MCDMA fabric (`zig/src/fabric`) | Flash Next: `zig/src/families/flashnext/tp.zig` splits rows and holds the whole model on each Mac. Kimi K3: `zig/src/families/kimi_k3/parallel.zig` is a split plan with no link yet |
| Prompt path | segments; the prompt cache, used by Flash Next only | per-family prompt kernels |
| Serving | `engine_api` for every host; `lane_host` for Nemotron | Flash Next: its own host, one reply at a time |

**The order.**
1. GLM-5.3 builds the core pieces first:
   - the route (router logits, then the pick) and the hyper-connection boundary, each in as few launches as its
     arithmetic allows;
   - 4-bit lane kernels with groups of 64, generalized from `fn_lane.metal`;
   - GPU-side rounds lifted from Flash Next;
   - experts split by rows across two Macs.
2. Flash Next moves onto the core kernels once they give its own bits, and drops its copies.
3. Qwen3.6-35B-A3B, Kimi K3 and Nemotron follow.

A model is done when the exactness checks pass and the served side-by-side beats the release on the same Mac.

## The exactness checks

- **Decode kernels.** Drafted output equals the engine's own one-token-at-a-time output, token for token, at every
  depth and window width. A kernel swap on the target must hold at every width. The MTP head may differ, because its
  drafts are only proposals.
- **Prompt path.** The first token is the same, last-row logits are within bf16 rounding, and the next 48 decoded
  tokens are equal, at 8k and 32k. Run it from a fresh process: an A/B inside one process can hide uninitialized
  state.
- **Precision.** No int8 or fp8 activations, and no lossy cache. A new summation order at equal precision is allowed.

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
   - Three draft levels: 3, 6 and 15.
   - A moving average of drafts landed over drafts offered moves the level: up to 6 above 0.8, back to 3 below 0.65.
   - The widest level runs only while copy drafts land. MTP chains stop at 6.
   - Flash Next code at depth 6: 6.07 tokens a round, 242.7 tok/s.
8. **Copy drafts on the GPU.** The GPU keeps the token history and finds the longest suffix match, up to 8 tokens.
   - Lenient gate: a match of 3 that reaches 6, or that agrees with the head's first draft.
   - Strict gate: a match of 4 that reaches 9, or that agrees with the first two.
   - Stay lenient while copies land within 0.05 of the head's drafts.
   - Flash Next edits went from 228.6 to 314.1 tok/s, with code and prose unchanged.
9. **Windows of up to 16 rows**, the widest only while copies land.

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
      The second segment's buffers take about 9 GB. Pushed in zig-flashnext c2bede8fd.

## Stage 4: serve

16. **Serve on `tensorfold-native`**, warmed at load, foreground before background, with stop strings and cancel.
17. **A Metal keepalive** for any model that holds a large residency set. That's a one-thread command buffer every
    250-500 ms, on its own queue, inside the engine process. Without it macOS unwires the set after 1-2 s idle, and
    the next request pays a full re-wire: 2.7 s on K3.

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
