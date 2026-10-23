# Reference only: GLM FP4-Local on two GB10s

**Not a merge proposal. No review or merge is requested.** This archive preserves
local experiments from 22–23 October 2026, including unsuccessful optimizations.
The accompanying PR is intended to be closed by its author immediately after
submission. It does not change TensorFold's runtime, defaults or support table.
Maintainers have no obligation to review, reproduce, maintain or adopt it.

## Attribution and review status

The local implementation, test orchestration, analysis and this report were
performed by OpenAI ChatGPT/Codex at the hardware owner's direction. The human
operator tested the service but did **not** perform a line-by-line code review.
This disclosure is not a claim of independent human validation, formal proof,
upstream qualification, or endorsement by OpenAI or the TensorFold maintainers.

TensorFold and its existing kernels remain the work of their original authors.
In particular, the FP4-Local loader is borrowed from
[Bizuayeu's PR #1050](https://github.com/ashhart/TensorFold/pull/1050), pinned at
`b960da830a2722e39513f354211efe2f4531cd02` for these experiments; it is **not**
claimed as new work by this archive. That PR has continued evolving independently.
This archive is not a competing implementation or a request to close that PR.

The dependency patch is separated from the local increments. It records the
loader integration on the tested main base, not the entire original contributor's
commit history. Do not reattribute it when reusing the experiment. Consult the
original PR for its complete receipts and authorship. TensorFold's Apache-2.0
license and required notices apply; model weights retain their own licenses.
No weights, model packs, native binaries or third-party shared libraries are
redistributed here. DFlash2's CC BY-NC-ND 4.0 terms must be evaluated separately.

## Scope and tested configuration

| Item | Recorded configuration |
|---|---|
| Source base | `f8d254c5e0f9ab91eb325a2e7c28d2e5cb07c94d` |
| Engine | Native Zig/CUDA; not the frozen Python inference engine |
| Hardware | Two NVIDIA GB10 hosts, 128 GB unified memory each, TP=2, RoCE |
| Toolchain | Zig 0.17.0, CUDA compiler 13.0.88, architecture-specific SM121a |
| Main checkpoint | `nvidia/GLM-5.3-Flash-FP4-Local` revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2` revision `bf582e4eacc1810f76656d1811693ff6c6737d2a` |
| Main precision | Stored mixed FP4/MXFP8/BF16/FP32; routed A4 prompt path, A16 verification/decode |
| Other precision | Existing native FP8 latent KV; no extra Dense FP8 conversion |
| Serving budget | 524288 context; four admitted streams; 2 GiB requested prefix cache; 6 GiB OS reserve |
| Draft policy | Native DFlash2 Q4, adaptive up to seven drafts, cumulative confidence 0.7 |
| Default sampling | Temperature 1, top-p 0.95, thinking enabled/high; test overrides explicitly recorded |
| Storage | One read-only checkpoint copy, second host reads it over NFS; no weight-byte changes |

Detailed OS/driver inventory is not supplied in this public bundle; compiler,
binary, image and checkpoint identities are retained. Clocks were variable, not
locked for benchmark qualification. These omissions limit reproducibility.

The selected local baseline is **v5**, binary SHA-256
`4c75b2b6182560bdbee9eef8267218bfaa975541464761a7ad0476c8fec2e0ad`.
V6/V8/V9 were not promoted. Packaging this archive did not restart or change the
running service. This is not a comparison claiming superiority to vLLM.

## What was retained, and what was rejected

1. **DFlash-only loading:** omit the unused MTP transformer while retaining the
   same draft vocabulary projection. Recorded model allocations fell from 94.83
   to 92.78 GiB per head-node receipt. This is not total process or system RSS.
2. **Native A4 prompt path:** reuse the existing compact A16 weight pack, apply
   each projection's own scales and GLM clipping, without a second weight copy.
   A generic SM121 build initially trapped; architecture-specific SM121a and an
   early native-MMA marker check corrected this. The failed launch is documented,
   not counted as successful inference.
3. **A4 N64 to N128 tiles:** retained after same-output comparisons. On the fixed
   2K–66K prompt fixtures, prefill took about 14% less time (16–17% more tokens/s).
   Decode did not improve. This comparison is within the existing A4 path; adding
   A4 in the first place is not bit-equivalent to an A16-only prompt implementation.
4. **Decode warp count and DFlash confidence tuning:** cache-resident microbench
   gains or more accepted drafts did not translate into better served results.
   Four warps and confidence 0.7 were retained; the cost override remains unset.
5. **V6 copy/tail/gather changes:** KDA direct output, shared-expert tail fusion
   and a window-wide sampled-logit gather passed listed local checks but did not
   produce a repeatable served gain. They remain independent, default-off options.
6. **V7 profiling:** target experts and KDA projections dominate the sampled
   windows. GPU activity was about 95%; that is not proof of DRAM saturation.
7. **V8 low-eight-row experts:** fewer registers did not yield a robust isolated
   chain improvement; no complete model was loaded with this candidate enabled.
8. **V9 KDA projections:** six pipeline/warp variants were screened. An existing
   ordered non-cluster O reduction, keeping SK=4, was integrated experimentally.
   Output hashes matched but served performance did not reliably beat frozen v5.

## Representative results

The public synthetic prose/HTML fixtures are not the operator's private tasks.
The reported token rate includes reasoning tokens. Prefill rows below are
cache misses after separate kernel warmup, not a fresh-process, cache-disabled
full qualification sweep. All compared prompt rows report zero reused tokens.

| Prompt tokens | V4 prefill seconds | V5 prefill seconds |
|---:|---:|---:|
| 2027 | 2.022 | 1.735 |
| 8108 | 8.503 | 7.334 |
| 32988 | 33.742 | 28.941 |
| 66429 | 67.562 | 57.888 |

Decode medians for the same short controlled fixtures were essentially flat:
40.09→39.63, 40.51→40.18, 35.66→35.59 and 47.05→46.82 tokens/s.
Do not describe v5 as a demonstrated decode acceleration.

V9 fixed-token comparisons: thinking enabled/low, temperature 1, top-p .95,
seed 119, 256 output tokens, two repetitions per cell. Identical prompt/output
hashes and speculative counters were required before interpreting timings.

| Workload | Frozen v5 | V9 off | V9 on | Reverse v5 |
|---|---:|---:|---:|---:|
| Prose 8K | 33.948 | 33.002 | 33.622 | 33.938 |
| Code 8K | 44.372 | 42.516 | 44.144 | 44.388 |
| Prose 32K | 30.182 | 29.581 | 29.718 | 30.315 |
| Code 32K | 47.660 | 47.650 | 47.238 | 47.779 |
| Prose 128K | 30.306 | 29.442 | 30.193 | not rerun |
| Code 128K | 29.280 | 28.426 | 28.645 | not rerun |

Reverse v5 has three repetitions per 8K/32K cell. All 36 new requests matched
the prior v5 correctness/speculation evidence. Sequential run variation and
recompilation effects limit causal attribution. The 128K code fixture produced
only reasoning within the cap, not a finished HTML page. Cached-prefix lengths
differ, so these rows cannot establish cold-prefill or total-wall-time gains.

## Evidence and known gaps

- V5 CUDA-host build: 1057 passed, 4 skipped. This subset is not the complete
  upstream host/golden/fuzz/Metal/device qualification.
- Listed local tests include draft/plain, fixed seed, four short solo/concurrent
  replies, >32K prefix reuse, an interrupted SSE request followed by retry,
  tools, structured output, direct images and tool-returned images.
- One 521113-token checksum/capacity request succeeded on v3. Four concurrent
  approximately 41K requests were separately exercised. Neither proves four
  simultaneous full 512K requests, nor broad long-context reasoning quality.
- Focused operator tests checked immutable-baseline bytes, five repeats, guard
  bytes, unchanged inputs and sampled independent FP64 references. They do not
  replace the complete unified kernel runner and its planted failure controls.
- No official-backend teacher-forced fidelity qualification, full all-platform
  kernel `SUMMARY`, or complete new-finding-free lean baseline is claimed.
  The historical whole-tree lean check reported 439 findings; its full pinned
  baseline comparison was not completed.
- Small intermittent paging was recorded; idle zero swap-in/out is not a claim
  of zero swap residency or an entirely paging-free run.
- V9's wider vision/concurrency/cancel/512K matrix was not rerun after it failed
  the performance promotion gate. V5 validation does not transfer automatically.

The detailed reports are historical experiment notes. Some refer to additional
private local artifacts not included here; their file names are not promises of
public downloads. Only files listed in `SHA256SUMS` belong to this bundle.

## Using the reference artifacts

`patches/00-...` is the separately attributed loader dependency integration.
`01-...` reaches frozen v5. `02`–`05` advance through the non-promoted v6–v9
experiments. Apply only in a **new disposable checkout** at the pinned base,
not directly to current main or to a running deployment:

```sh
git checkout --detach f8d254c5e0f9ab91eb325a2e7c28d2e5cb07c94d
git apply /path/to/reference/patches/00-upstream-pr1050-dependency.patch
git apply /path/to/reference/patches/01-v5-increment.patch
# Stop here for the selected v5 source; later patches are experiments.
zig build native test-cuda-host -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dstrip=true -j4 --summary all
```

Use Zig 0.17.0 and qualify the resulting binary independently. No deployment
controller, automatic image replacement or credential-bearing configuration is
included. `local-reference/...` names in provenance are anonymized labels for
local images, not published registries. Original image/archive hashes identify
the measured private artifacts; they are not hashes of this sanitized bundle.

Python scripts are development-only clients/analysis, not inference dependencies.
For clients importing `upstream/tools`, arrange a sibling `upstream` checkout of
the pinned engine, or add that directory to the import path. GPU harnesses need
the indicated built fatbins and their original runtime dependencies; these are
not redistributed. To recompute the included served comparison without a GPU:

```sh
python3 tests/compare_kda_served_v9.py --self-test
python3 tests/compare_kda_served_v9.py --baseline v5 \
  v5=evidence/decode-v7-baseline-observed.jsonl \
  v9-off=evidence/decode-v9-off.jsonl v9-on=evidence/decode-v9-on.jsonl \
  v5-return=evidence/decode-v5-after-v9.jsonl --output comparison.json
```

Public synthetic request bodies and exact prompt token IDs are retained. Generated
reply/reasoning bodies and transport chunks are omitted; runtime token hashes,
usage, speculative counters and timings remain. Hash equality is checked against
the recorded hashes, not independently recomputed from unbundled output IDs.
Private addresses, host names, user paths and image names are replaced with
documentation examples. Credentials, model files and captured profiler binaries
are excluded. The publication scan supplements, rather than replaces, inspection.

This archive intentionally records unfinished and rejected work. Anyone adopting
a patch must establish the current upstream gate, preserve attribution, review
the implementation and requalify it on their own checkpoint and hardware.
