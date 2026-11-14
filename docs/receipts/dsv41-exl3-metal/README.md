# EXL3 Metal follow-up qualification

This is a follow-up to [#2373](https://github.com/ashhart/TensorFold/pull/2373), not release qualification.
The parent is `bec140d545aa50b765f26cf295b43179c6b2834f`; the native binary's production sources are
`e070123319b69e73c969b91409dd7195fbbef93e`. The qualification tree is
`128cfe743a76f59b350d0c750a12669708029ff3`, which additionally corrects the shared-down test's scratch capacity.
The subsequent formal projection fixtures do not change those GPU dot bodies; the batched-head fix below is separate.

[environment.json](environment.json) records the checkpoint metadata fingerprints, source revision, build settings
and hardware: **M5 Ultra, 64 GPU cores, 256 GiB**, not the parent's historical 80-core machine.
The pack is pinned to `45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b`.
[model-fingerprints.json](model-fingerprints.json) binds all 39 weight shards, both streamed source shards,
the pack/source metadata and tokenizers, the chat template and the generated token map.

## Checks already exercised

- `zig build test-lane-api-host -j4 --summary all`: 545 passed.
- `zig build test-exl3 test-environment-host test-native -Dcpu=apple_m1 -j4 --summary all` on the host workstation:
  EXL3 host checks passed and environment checks passed 12/12; native compilation required the M5's Metal compiler.
- On the M5, `zig build native test-exl3 test-environment-host test-native test-deepseek-v41
  -Dcpu=apple_m1 -Doptimize=ReleaseSafe -j8 --summary all` at `59535dad914819ea4d281f87c81a138c4f2c4e1b`:
  the family GPU target passed 73/73, EXL3 host checks passed, and environment checks passed 12/12.
  The combined command failed because the native cone diagnostic imported an excluded transport module.
- After removing that dependency and making cone mismatches fail closed, `zig build native test-native
  -Dcpu=apple_m1 -Doptimize=ReleaseSafe -j8 --summary all` at the serving revision passed:
  69/69 build steps, 118 tests passed, 3 skipped.
- An actual `GLM_LOAD_LIMIT_GB=1` candidate startup refused both workspace sizes with `LoadLimitExceeded`, exit 1,
  without an allocator abort. The wide plan requested 217556406008 bytes and the narrow plan 214737522424 bytes.
- At the common four-slot, 32768-context geometry with `GLM_LOAD_LIMIT_GB=240`, the parent default workspace
  refused admission: 235727027864 resident bytes plus 8589934592 margin exceeded the 230553780224-byte allowance.
  The comparison therefore pins **4096 workspace rows in both arms** instead of raising the machine's limit.

The 73-test family record predates the final core fallback fix. The final standalone projection target,
`zig build test-exl3-gpu -Dcpu=apple_m1 -Doptimize=ReleaseSafe -j8 --summary all`, ran on the qualification tree:
**36 passed, 3 failed, 39 total; exit 1**. The parent target returned **21 passed, 6 failed, 27 total; exit 1**.
All three remaining failing test cases also fail on the parent: mixed-width routed/dense bit parity,
packed-sign filtered/dense BF16 parity, and grouped/scattered versus separate projection bit parity.
They remain merge gates; neither complete projection suite is reported as passing.
The indexed odd/wide scalar-reference check, tiled/table bit checks and new chain/MoE fusion checks pass.
The final native rebuild passed 66/66 build steps. Raw build/server logs are not published because they contain
local filesystem paths; [operator-checks.json](operator-checks.json) records the scoped outcomes.
Both revisions also reported a 16-byte allocator leak on shutdown; this receipt does not claim a clean exit.

## Reproduce the served comparison

Use separate checkouts at the parent and serving revisions, the same pack and original Engram tables, Zig 0.17.0,
an installed Metal compiler, and Python 3.10 or later. Build each with:

```sh
zig build native -Dcpu=apple_m1 -Doptimize=ReleaseSafe -j8
python3 docs/receipts/dsv41-exl3-metal/prompts.py /tmp/exl3-prompts
```

Run the prompt generator from the candidate checkout; the parent does not contain it.
Run only one server at a time. For each checkout, use this common command; on the candidate add
`TF_DSV41_CONE_CHECK=4096,16384` to run full/cone logits, cursor and rollback checks before readiness.
The checker restores the configured cone mode afterwards; any mismatch fails startup.

```sh
GLM_LOAD_LIMIT_GB=240 TF_DSV41_PREFILL_STEP=4096 \
./zig-out/native/bin/tensorfold-native serve <pack> \
  --engram-stream-dir <orig> --chat-template <pack>/chat_template.jinja \
  --context 32768 --parallel 4 --prompt-cache-gib 0 \
  --host 127.0.0.1 --port 8941 --name exl3-qualification --keep-warm 0
```

Once `/v1/models` is ready, run the existing client from a checkout containing it; set `ARM=base` or `ARM=candidate`
for the currently running server:

```sh
python3 tools/bench_concurrent.py http://127.0.0.1:8941 exl3-qualification \
  --tokens 64 --reps 1 --temperatures 0,1.0 --levels 1,4 --alone --serial \
  --prompts /tmp/exl3-prompts/short-prompts.json --label "$ARM-short" --output "$ARM-short.json"
python3 tools/bench_concurrent.py http://127.0.0.1:8941 exl3-qualification \
  --tokens 64 --reps 1 --temperatures 0 --levels 1,4 --alone --serial \
  --prompts /tmp/exl3-prompts/long-prompts.json --label "$ARM-long" --output "$ARM-long.json"
```

The client can finish without rejecting unequal tokens. Require complete full-token hashes and compare both arms:

```sh
python3 tools/bench_exl3_compare.py base-short.json candidate-short.json
python3 tools/bench_exl3_compare.py base-long.json candidate-long.json
```

## Formal projection and batched-state checks

On the same 64-core M5, the projection contract passed 219 width/codebook registrations over 13 entrypoints:
all gates 1–4 passed, with 876/876 real GPU fault controls caught and no known findings.
The fixtures compare every output column to independently scalar-decoded FP64 rows and require the replaced
sixteen-warp reduction's exact bytes. KU2 prompt tiles additionally equal the KU1 template's bytes.
Structured full-sized matrices vary three K bands independently; precision plants repeat a random decoded tile.
The reported FP64 worst column and squared error are retained separately for each shape and comparator.

```sh
zig build tf-kernel-contract -Dcpu=apple_m1 -Doptimize=ReleaseSafe -j8
./zig-out/bin/tf-kernel-contract --only exl3-metal --no-limits
./zig-out/bin/tf-kernel-contract --only exl3-metal --no-gates --offline /tmp/exl3-offline
```

The separate resource run passed all 13 live launches, with zero limit problems and both resource controls caught.
These projection pipelines are selected only with M5 tensor units; M1/M2 projections are explicitly unserved,
not claimed to have passed offline or runtime checks there. The offline resource controls did run.
`zig build test-exl3-contract-host test-kernel-contract -Dcpu=apple_m1 -j4 --summary all`
passed 232/232 host tests in 33/33 build steps.

Finite packed-key fixtures exposed a batched split-merge bug at 16 heads: the merger retained the 64-head defaults.
It now receives the actual head count and width. The same test checks four independent slots at 16, 32 and 64 heads,
including wrapped/long positions, both RoPE directions, KV glue and publication against the per-slot path.
`zig build test-deepseek-v41-rows -Dcpu=apple_m1 -Doptimize=ReleaseSafe -j8 --summary all`
failed before the repair and then passed 6/6 tests in 41/41 build steps.
The fixture excludes E4 NaN codes from its finite-data leg and explicitly requires finite attention outputs.

## Limits

These are single-run synthetic correctness/timing probes, not representative throughput or quality gates.
The long fixture repeats public prose; the short fixture mixes completion and chat requests.
The client warms the server; disabling prompt retention does not make disk/Engram pages cold.
`aggregate_tps` sums per-stream decode rates; use `end_to_end_tps` for completed tokens per concurrent request window.
This does not replace cold `bench_cache.py --require-cold`, `bench_openai.py`, all contexts, cancellation/resume,
explicit MTP/DSpark, or other-chip checks. The automatic 8192/4096 policy is not measured by the pinned comparison.
Formal registration and planted controls for every added kernel, independent whole-model fidelity and the remaining
parent model gates are still required before this contribution can leave draft status.
The new family controls use shared strict validation; they do not add a global profile or claim profile qualification.
