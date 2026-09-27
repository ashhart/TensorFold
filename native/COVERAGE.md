# Native Metal coverage

This fork ports the inference hosts to Zig and embeds the original author's Metal
kernels. MLX-C supplies arrays, scheduling, safetensors, and general operations.
This matrix distinguishes exercised behavior from physical-device validation.

## Model paths

| Path | Native implementation | Executed checks on M5 Max |
| --- | --- | --- |
| Qwen3.8-27B target | Embedding, quantized projections, norms, GDN, attention, gated MLP, head | 993,280 Python logits exact; sampled and greedy completions |
| Dense tree verification | Parent/depth metadata, recurrent replay, KV gather and partial commit | 513-token prefix, branched rows, continuation and all 64 caches exact |
| Dense SIMD | simdgroup quantized matmul and row attention | Forced SIMD tree/cache check; 128-token GPU-sampled DFlash2 equals serial; physical M1–M4 untested |
| DFlash2 | Five layers, hidden taps, dynamic convolution, selector codebooks, candidate tree | 128 sampled tokens match Python serial; GPU radix top-k preserves output |
| Context copy | Suffix matching and target-verified proposals | Host tests and generation path |
| Nemotron target | Mamba convolution/SSM, grouped norm, NoPE attention, top-6 routing, ReLU² experts, shared experts | 524,288 Python logits exact; verified rows and all 52 caches exact |
| Nemotron MTP | Embedding/hidden fusion, attention, MoE, cloned draft cache | 32 sampled tokens match Python serial with both CPU and GPU samplers |
| Nemotron SIMD | Row quantized projections and expert kernels; MLX SDPA | Forced SIMD verified rows, rollback and all 52 caches exact |
| Nemotron long context | 128-wide tensor attention at 10K visible keys; SDPA otherwise | Kernel fixtures at 9,999 and 10,007 keys, serial/window exact |
| Flash Next target | Four residual streams, hyper-connections, GDN, sparse attention, top-10 experts/shared gate | 248,320 Python logits exact; verified rows and all 48 caches exact |
| Flash Next PLE | N-gram hashing/EOS reset, positional packed-row reads, gate, dilated convolution | Shipped hash constants checked; all 128 shard boundaries checked against MLX; rollback history/conv cache exact |
| Flash Next MTP | Embedding and stream fusion, HC attention/MoE, separate output mixer | 16 sampled tokens match Python serial |
| Flash sparse selection | Pool four keys, select top 512 blocks, include causal tail, merge attention | Prefixes 2,044/2,051/2,063; cold and populated pool, eight rows and rollback at every nonzero row |

## Shared Metal utilities

`test-metal` generates fixtures using the original Python wrappers and independently
loads/dispatches them through Zig/MLX-C. Tests fail on any bit mismatch.

| Utility | Cases |
| --- | --- |
| GPU sampling | BF16/f32 logits; greedy and sampled; seeds 0/1234/5678/UINT64_MAX; mapped and full token IDs; top-k 0/20/2048; top-p .8/.95/1; ties and dominant logits; positions through 262,144 |
| Radix top-k | Vocabularies 31/4,097; k 1/16/31/64; deterministic ties; BF16 conversion of f32 input |
| Tensor attention | D128/D256; 1/3/8 queries; 513/9,999/10,007 keys; cache views with unused trailing capacity; every batch row equals serial |
| Sparse attention | Original fused Python attention with deterministic synthetic weights at actual Flash dimensions; pooled keys and projected outputs match; rejected rows do not affect continuation |

The fixtures use synthetic inputs to reach expensive context branches cheaply.
The separate full-checkpoint tests establish model integration. Kernel export checks
ensure embedded sources remain verbatim; source export alone is not a numerical test.

## Reproduce

```sh
bash scripts/fetch-zig.sh
.zig-toolchain/zig build test -Doptimize=safe
.venv/bin/python tools/export_native_kernels.py --check
.zig-toolchain/zig build test-metal -Doptimize=safe -Dmetal-tensors=true
.zig-toolchain/zig build test-models -Doptimize=safe
.zig-toolchain/zig build test-cache-stress -Doptimize=safe
.zig-toolchain/zig build test-drafts -Doptimize=safe
.zig-toolchain/zig build test-checkpoint-files -Doptimize=safe
.zig-toolchain/zig build test-ple -Doptimize=safe
```

The suite needs the MLX prefix described in README and Python development dependencies
in `.venv`. Omit `-Dmetal-tensors=true` on GPUs without tensor units. Model paths default
to `build/models`; override with `-Dmodel-root=/absolute/path`. All fixtures and weights
stay in ignored `build/` directories. `test-models` deliberately serializes model loads.
The cache and draft suites also serialize their model loads; run these targets separately,
since simultaneous suites compete for unified memory and GPU execution time.
Use `-Dcache-family=0|1|2` or `-Ddraft-family=0|1|2` to select Qwen, Nemotron or Flash.

## Expanded acceptance and cache checks

The shared acceptance policy is used by both DFlash trees and MTP chains. Host tests
cover all 32 chain acceptance lengths against 33 output budgets, accepted/rejected/bonus
EOS, and 1,000 deterministic random trees. The thirteen host tests pass with safety
checks, including four supported config fixtures and 129 invalid recipe mutations,
safetensors parser allocation failures and all 128 PLE shard lookup boundaries.

`test-cache-stress` has passed on all five model/backend combinations below. Each
accepted prefix is compared with serial execution: logits, every cache array and its
metadata, and the next token's logits after restoring the original cache. It also drops
entire speculative passes, tests invalid commits, inserts EOS IDs into the PLE history,
and repeatedly restores snapshots and resets the model.

| Model/backend | Accepted-prefix checks | Prefix lengths | Post-warmup reset cycles | Active MLX memory, baseline = maximum |
| --- | ---: | --- | ---: | ---: |
| Qwen tensor | 384 | 0, 1, 15, 16, 17, 63, 64, 127, 128, 511, 512, 513 | 128 | 15,133,588,480 bytes |
| Qwen forced SIMD | 384 | Same | 128 | 16,734,960,640 bytes |
| Nemotron tensor | 96 | 0, 1, 15, 16, 17, 33 | 128 | 18,816,686,464 bytes |
| Nemotron forced SIMD | 96 | Same | 128 | 17,778,989,440 bytes |
| Flash, positional PLE reads | 96 | Same | 128 | 79,022,784,536 bytes |

Qwen additionally checks every path in randomized trees with 2, 7, 15, 16, 17, 31 and
32 rows, on each backend. Its chain windows reach the 128-row prefill limit; family
windows reach 16 rows. Memory figures measure MLX active allocations after synchronization,
not process RSS or peak unified-memory use.

These tests exposed a borrowed recurrent-state handle in Qwen's saved verification pass:
restoring the live cache could invalidate the handle needed for a later partial commit.
The pass now retains its replay base. Flash's saved pooled-key handle received the same
lifetime fix. Dense commit also validates position and ancestor paths before changing
the cache, and releases temporary replay arrays at the end of each commit.

`test-drafts` passes all 44 serial/drafted comparisons: 17-token greedy, 32-token CPU
sampling, 32-token Metal sampling and a two-token CPU output budget. MTP budgets are
1, 3 and 15. Use `-Ddraft-scenario=0|1|2|3` to select a scenario. The comparison
requires actual decode rounds so an immediate EOS cannot masquerade as draft coverage.
Completion evidence for this larger matrix is tracked in [WORK_PLAN.md](WORK_PLAN.md).

## Checkpoint and PLE row reads

All safetensors loads validate dtype names, bounded dimensions, checked byte-size
arithmetic and contiguous non-overlapping offsets against the actual file size before
calling MLX. Shard names from indexes must be simple safetensors filenames. Duplicate
tensor names across loaded files are rejected. Native readers additionally validate
the dtype/shape of all 384 PLE packed weight/scale/bias tensors and their total row count.

`test-checkpoint-files` verifies missing files, short headers, oversized declarations,
bad payload lengths, wrong row buffer sizes, out-of-range rows, and files truncated
after opening. Parser and file-opening checks inject failure at every Zig allocation.
The dense loader now cleans up array/linear ownership if map insertion fails.
These checks do not yet validate every model-specific tensor shape or inject failures
inside MLX itself; those remain tracked gaps.

`test-ple` independently loads each shard with MLX and compares its first, adjacent,
middle, penultimate and final rows against native positional reads: 640 exact rows.
The production lookup bounds scratch storage to 256 packed rows (25,600 bytes), followed
by GPU dequantization. Full-model active MLX memory dropped by 32,002,375,680 bytes in
the repeated cache check. The previously intermittent maximum-budget Flash timeout
did not recur in the complete Flash draft matrix after this change.

## Limits

Physical validation is on one M5 Max with 128 GiB, with both normal and forced SIMD
dispatch. Older Apple GPUs still require execution on those devices before claiming
hardware qualification. Flash Next's short tests fit through positional PLE reads; long
full-model context memory and throughput are not established on this machine.

This is functional coverage of the three upstream Metal inference recipes. Native
dispatch selects one implementation of each operation; historical or optional fused
optimization variants are not all separate native execution modes. CUDA, HTTP serving,
chat templates, vision, and disk prefix-cache persistence are outside this Metal port.
