# Optimizing for speed

This page is about how to approach speed work in TensorFold: deciding whether a change is worth building, finding
which limit a kernel is up against, profiling without taking the box down, proving the change exact and measuring it
honestly. It does not teach CUDA; the guides under [Further reading](#further-reading) do that well.

## Bound before building

Before writing a kernel or a fusion, work out the most it can win.

- Measure the kernel's share of the step (decode) or of the pass (prompt) on the real server, with nsys.
- Work out the ceiling on its gain: the time the change removes if it works perfectly. For a bandwidth-bound kernel
  that is its time minus the time its bytes take at the device floor. For a fusion it is the removed kernel plus the
  removed writes and reads, if they were fully exposed.
- Set a bar in advance, at the server, not at the kernel: for example 3 % of cold TTFT or of a decode step. Write the
  hypothesis and its expected range down before measuring. A result outside the range means debug the measurement
  first.
- Stop rule: if the ceiling is under the bar, don't build it. If the built change lands under the bar at the server,
  record it as `failed` in the [optimization ledger](optimization-ledger.md) and stop.

Rank ideas by the size of the prize, not by the cost of the change. An idea that won elsewhere still needs bounding
here: TensorFold's kernels and launch shapes differ from the project that reported the gain. Bounding it is fine, but
what lands is written from our own code and the model's own definition, never read from or ported from another engine.

Worked example (Qwen3.8-27B NVFP4 on a GB10, main at the time, one hour of measurement, no code):

- Fusing the SwiGLU activation into the gate/up prompt GEMM, so its output is written once, already activated.
  The activation kernel it removes was 3.38 % of cold TTFT at 8k and 3.11 % at 32k, already at 246-248 GB/s.
  Adding the gate/up write it would also remove gave an upper bound of 4.4 % (8k) and 4.1 % (32k). Clears a 3 % bar
  only marginally.
- Splitting the GDN decode scan over more SMs, so one stream's scan uses more of the GPU. TensorFold's grid already
  launches 768 blocks a stream on 48 SMs, and the kernel is 1-2 % of a step. Bound under 1.5 %: not built.

## Which limit applies

Name the binding limit only after a one-knob test. Three cases cover most kernels:

- **Bandwidth.** Bytes moved ÷ time, against the device floor (273 GB/s on a GB10). Most decode projections run at
  73-87 % of it. A kernel well under the floor that launches fewer blocks than there are SMs wants more parallelism:
  split-K, split-V or split-KV, with a fixed merge order (see below). A kernel under the floor with plenty of blocks
  has a per-kernel efficiency problem instead (loads, reduction, input layout).
- **Launch and overhead.** Small kernels of a few microseconds, many launches a step. The fix is fusion with a
  neighbour, not tuning.
- **Compute.** Prompt GEMMs at large M. The levers are tiles, waves and L2 order (swizzle).

### The per-kernel audit

One nsys capture of the real server gives a table that answers "where does the time go" for a whole step:

| column | from |
|---|---|
| kernel, role | kernel name, plus the shape it serves |
| time a step, % of step | kernel intervals, summed per step |
| launches a step | count |
| grid, blocks vs SMs | launch record |
| achieved GB/s | bytes from the checkpoint shapes ÷ time |

Flag every row above 1 % of the step that runs fewer blocks than SMs, or well under the bandwidth floor. Also add up
the weight bytes a step: that gives the floor for the whole step, and how far the step is from it. In the example
above the audit found no kernel over 1 % under 48 blocks, and one projection at 179 GB/s where the same shape
elsewhere reached 216: a better lead than either idea above.

## Profiling safely

- nsys on the real server. Decode runs in CUDA graphs, so use `--cuda-graph-trace=node`. Use one capture per server
  start. GPU busy time is the union of kernel intervals, not the profiler's summary table.
- ncu only on a synthetic single-kernel test, never on a process holding a model. ncu replay backs up device memory
  into the same pool; on a GB10, with unified memory, a full-model ncu run exhausts memory and hangs the box.
- Small kernels: measure in the model. An isolated ~5 µs kernel measures launch overhead and cold caches.
- Kernel microbenchmarks: rotate distinct buffers larger than L2 so weights come from memory, and state the method.
- Which kernel ran comes from the trace or the log, not from reading the dispatch code.

## Exactness first

A speed change that changes tokens isn't a speed change. TensorFold's rules:

- Every speed change, opt-in ones included, keeps greedy token hashes and seeded sampled tokens equal to main. Show
  it: token hashes against main on the same prompts, greedy and seeded. There are no lossy weight or KV formats.
- Splits (split-K, split-V, split-KV) merge partial sums in a fixed order, independent of the grid and of timing. No
  atomics on the sum.
- For every kernel you did not mean to change, compare SASS between main and the branch, function by function, for
  every arch built. Moving a helper into a shared header can change SASS; check, don't reason about it. Reviewers
  want the comparison as something they can rerun.
- Register new and changed kernels with the [kernel contract](kernel-contract.md). Its gates each have a planted
  control that must fail; a run where a control passes proves nothing.

## Measuring honestly

- A-B-B-A order, a fresh server for each arm and round, enough rounds that the sign holds every round.
- Drop the page cache before each server start.
- Name the clock state (capped or not) and log SM clock, temperature and throttle reasons. Drop rounds with
  throttling or an SM-clock gap over 1 % between arms. Never compare capped with uncapped numbers.
- md5 every binary you serve, and put it in the result. A stale binary measures nothing.
- Token hashes in every speed table. With speculative decoding, speed follows acceptance, and acceptance moves with a
  1-ulp change: compare acceptance and tok/s only between token-identical builds, or report time a round and
  acceptance separately.
- Report cold TTFT and plain decode at 1 and 4 streams, against the exact kernel you replace at the family's shape.
  A step that makes a served cell slower is held back, even when it is bit-exact (#2620); a move between files shows
  main-against-branch speed on each backend it touches, with no loss (#2118).
- Keep the receipts: raw runs, server logs and the trace behind every posted number, somewhere other than `/tmp`.

## A null on a bandwidth-bound box

A GB10 is bandwidth-bound for most decode work. A compute-side change (fewer instructions, better tiles, more
overlap) can measure as a null there and still win on a discrete card. A null on a GB10 is not a verdict for that kind
of change: open the PR with the GB10 numbers and ask for a run on a discrete card.

## Where it goes

- Check shared code first. Cross-family parts live in `zig/src/core/` and the shared CUDA headers in
  `zig/kernels/cuda/common/`; the [optimization ledger](optimization-ledger.md) lists the helpers. A change proven on
  one family should be ported, not invented again. Say in the PR which parts are core and which backend-only.
- Update the optimization ledger row in the same PR, including `failed` for a measured null.
- Name profile switches through [profiles](profiles.md), and evaluation through [evaluating](evaluating.md).

## Further reading

- NVIDIA, [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- GPU MODE, [lectures](https://github.com/gpu-mode/lectures) and
  [resource stream](https://github.com/gpu-mode/resource-stream)
- Stas Bekman, [Machine Learning Engineering](https://github.com/stas00/ml-engineering)
- Simon Boehm, [SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA): a CUDA matmul taken step by step to near cuBLAS
