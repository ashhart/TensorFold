# GLM target-row sampling checks

Use the version in `.zig-version` and the final PR head. Run one model process under the existing GPU lock
and admission guard. Keep the host's current load, context and cache budgets; do not increase them for this check.
This recipe qualifies sampling, separately from the cache lifetime repair in #835. It carries the shared-sampler
repair from #524; that PR's previous model receipts do not qualify this combined tree.

## Host and synthetic Metal checks

```sh
zig build native -Dcpu=apple_m1
zig build test test-golden -Dcpu=apple_m1
MTL_DEBUG_LAYER=1 zig build test-glm-sampling -Dcpu=apple_m1
python3 -m unittest discover -s tools/zig -p test_check_glm_sampling.py
zig fmt --check build.zig zig
python3 tools/zig/lean_check.py
```

The lean baseline is main `c7137bb3`: 527 findings. Compare identities with `tools/zig/lean_new.py`;
a nonzero checker result is not a clean run. Every new finding must be fixed.
The synthetic gate uses the actual shared `tf_sample_full` and production dispatch, including mixed and solo rows.

## Served model checks

Start the newly built native server using the existing admitted configuration and a public model name.
Replace `local-model` and the URL below with the values exposed by that server.

```sh
python3 tools/zig/check_glm_sampling.py http://127.0.0.1:8080 local-model --output glm-sampling-receipt.json
python3 tools/bench_concurrent.py http://127.0.0.1:8080 local-model --tokens 64 --reps 2 --temperatures 0,1 --levels 1,4 --alone --serial --output glm-concurrent-receipt.json
```

Require positive restored-token counts, native token hashes, drafted telemetry greater than zero, equality between
plain and drafted replies, and solo/concurrent equality with both draft modes. Report any cold/restored mismatch;
do not waive one because prompt chunk kernels can differ. Repeat with the MTP head omitted, using
`--skip-drafted-qualification` for the first client, and label that run as unqualified for drafting.
Cancel a streaming sampled reply, then repeat a known fresh request and compare its native token hash.

## Standalone depth-zero pipeline

Build `zig build tf-glm-run -Dcpu=apple_m1`. Supply public token arrays using the existing `PROMPTS_JSON` format
shown by `tf-glm-run`: `{"prompts":[{"name":"public-case","ids":[1,2,3]}]}`.
Use meaningful public prose/code token arrays from the checkpoint tokenizer for qualification.

```sh
GLM_SAMPLING=12,0.7,40,0.9,0.05 GLM_DEPTHS=0,3 GLM_MAX=64 GLM_CANCEL_TEST=1 zig-out/bin/tf-glm-run MODEL_DIR PROMPTS_JSON
GLM_SAMPLING=12,0,0,1,0 GLM_DEPTHS=0,3 GLM_MAX=64 zig-out/bin/tf-glm-run MODEL_DIR PROMPTS_JSON
```

Depth zero must exercise the pipelined plain rounds with sampled rules at each input position plus one.
Require identical token arrays at both depths, repeated-seed equality and cancellation followed by a valid fresh run.
Run with and without MTP where supported. Two-Mac sampled requests must still be refused before state changes;
check greedy peer behavior on two Macs when available.

Record the exact final commit, GPU/OS, checkpoint revision, runtime, public inputs, request settings and token hashes.
Include pass/fail/skip counts, before/after decode and cold prompt timings, and physical/device memory peaks.
Host checks and semantic compilation do not qualify device execution, model parity or performance.
