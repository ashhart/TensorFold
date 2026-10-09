# Measuring a usable memory profile

A checkpoint fitting in memory is the beginning of a serving profile. A useful
profile also names the context, reply reserve, request capacity, cache policy and
exact runtime. Measure those together before recommending a RAM class.

This guide separates the stable Python/MLX server from the
[native Zig preview](../ZIG-PREVIEW.md). The MLX details below were checked against
[v0.6.6](https://github.com/ashhart/TensorFold/tree/v0.6.6). They do not qualify the
preview's models, kernels or memory admission.

## Keep the memory quantities separate

| Quantity | What it establishes |
| --- | --- |
| Checkpoint bytes on disk | Storage and transfer size; packed weights may be transformed at load. |
| Resident model bytes | Loaded weight representation; excludes request state and other runtime allocations. |
| MLX active and cached bytes | Allocator use; excludes the rest of the process. |
| Process physical footprint and lifetime peak | Memory charged to the process, including allocations missed between samples. |
| Installed RAM | Host capacity; shared with macOS and other applications. |
| GPU recommended working set | A separate device limit used by the stable MLX server's budget ceiling. |

Record the working set from `mlx.core.device_info()` alongside installed RAM.
The stable server caps `TENSORFOLD_MEMORY_LIMIT_GB` by both values, then reserves
3 GiB for the rest of the process before setting the MLX limit. Read the applied
budget from startup and `/health`; recording only the requested environment
variable can mislabel a silently capped run. This is an allocator/admission budget,
not an OS-enforced physical-footprint cap; measure the process peak independently.

An emulated budget on a larger Mac is useful for finding refusal boundaries and
cache tradeoffs. It does not reproduce the smaller Mac's GPU, bandwidth, working
set, background memory pressure or thermal behavior. A recommendation needs
headroom for the host and a check that the target can apply the measured budget.
Do not derive a minimum RAM promise by dividing the budget by a default fraction.

## Freeze one serving profile

Retain the engine commit, Python and package versions, checkpoint repository and
revision, weight format, chip, OS, applied budget and complete launch options.
Pin the rendered prompt tokens too: template and tokenizer revisions affect the
actual context length. Count the encoded rendered string, rather than the number
of fields in a tokenizer result.

State whether thinking, an external drafter, other drafting methods, vision and
tools are enabled. On the stable MLX server, `--drafter none` disables an external
draft model; it does not disable every drafting method. `--no-drafts` and each
request's `"draft": false` have their own scopes. Use the same precision and
weight representation for serial and concurrent exactness comparisons.

For Ternary Bonsai on M1-M4, the stable loader can retain packed 2-bit projections,
widen some layers, or widen all layers to 4-bit storage as the budget permits.
Capture the startup representation for every cell. Equal checkpoint revisions
alone do not establish equal runtime representations across budgets or GPU
generations; do not turn within-profile token equality into a cross-profile claim.
See the [Bonsai recipe](recipes/ternary-bonsai-2.md).

## Separate cold, repeated and restarted requests

Give every independent cold run a fresh cache parent. In the stable Python server,
`--snapshot-dir` selects the prefix directory, while conversation snapshots use
its sibling `session-snapshots` directory. Emptying only the selected directory
can leave a restarted request warm. Preserve the old parent as evidence and use a
new one; never clear a user's shared cache to make a benchmark cold.

A compact workload matrix contains:

1. Startup, model loading and the first short completion.
2. Cold prompts at several lengths, with a fixed explicit reply reserve.
3. The same prompts repeated, then a follow-up turn; report cached-token counts.
4. Short concurrent requests compared with their solo token hashes.
5. Independent long requests submitted together from an empty cache, each compared
   with its solo hash. Verify that cached-token counts establish the intended cold
   condition.
6. An over-context request and a fresh process restart with the same profile.

Choose a reply reserve that represents the intended application. A short measured
output does not qualify every larger output merely because the context setting
allows it. Report requested capacity separately from simultaneous decode: memory
admission can queue requests within `--parallel`, and queueing changes TTFT.

Measure startup through shutdown. Sample process physical footprint during load,
prefill, decode, cache retention and restart, and retain the kernel's lifetime
high-water mark. Record MLX active/cached/peak bytes separately. Host swap and
compression provide context, but are not attributable to the model alone.

## Publish failures and cache tradeoffs

For each applied budget and context, retain passed, startup-refused and
request-failed cells. A failure after a long prefill is different from rejecting
before loading. Keep the requested context, largest completed prompt, explicit
output length and phase of failure. An estimated context ceiling is not a
completed workload measurement.

Report client-observed time to first generated text, decode throughput, cached
prompt tokens and process peak together. A larger budget may retain more prefixes
and reusable buffers, increasing observed footprint while improving later turns.
Repeating a long prompt can still require prefill at a smaller budget. Do not label
that request a warm-cache speed result just because it is the second request.

Document fan mode, other active workloads and thermal sampling definitions. Named
core averages and raw sensor maxima are different quantities. A fixed fan setting
without an alternating controlled comparison establishes a test condition, not a
cooling-policy speedup. Missing power or temperature data stays missing.

## Backend and preview boundaries

CUDA has separate memory admission and cache support; MLX-only snapshot flags do
not establish disk-spill behavior on CUDA. Check the backend's actual startup and
[API fields](api.md) instead of transferring an MLX command unchanged.

The Zig preview uses its own native Metal allocations and cache policy. In its
Flash Next server, `--prompt-cache-gib` controls retained prompt states, not a
whole-process memory budget. Flash Next serves one reply at a time even when
`--parallel` is supplied; Nemotron's prompt reuse has a different status. Consult
[the preview's current limitations](../ZIG-PREVIEW.md#known-gaps) before building
a matrix. Stable MLX measurements are not preview performance or exactness results.

A public receipt should contain synthetic or public prompts, generic hardware
specifications and pinned public revisions. Exclude hostnames, account names,
private network addresses, credentials and operator paths. Keep benchmark tables
and retained measurements in the pull request or linked report, rather than
turning a single machine's results into unqualified source defaults.
