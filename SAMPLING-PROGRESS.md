# GLM sampling validation

The sampler integrates with the GLM Metal family. Sampled TP2/EP requests remain unsupported.
Memory admission changes belong to PR #521 and are kept separate.

## Regression gates

The bounded block now contains the actual highest candidates, independent of atomic arrival order.
The full top-k boundary and nucleus normalizer are independent of the 1,024-token storage capacity.
Min-p is applied to the final race, after nucleus normalization. Production and device tests share payload/bindings.

Use the exact version in `.zig-version`:

```sh
zig test -lc -lm --dep lanes -Mroot=zig/src/families/glm/draw_test.zig -Mlanes=zig/src/core/lanes/lanes.zig
zig build native -Dcpu=apple_m1 -Dversion=1.0.0
zig build test test-golden -Dcpu=apple_m1
MTL_DEBUG_LAYER=1 zig build test-glm-sampling -Dcpu=apple_m1
python3 -m unittest discover -s tools/zig -p test_check_glm_sampling.py
python3 tools/zig/lean_check.py
```

The synthetic device gate needs no checkpoint. It checks the real shader, production payload and bindings,
output offsets/sentinels, filtered counterexamples, vocabulary boundaries, mixed modes, repeated/reordered and solo rows.
A host compile is not a substitute for running that gate on Metal.

## Real-model gate

Use one model process and the existing GPU lock/admission guard. Enable native prompt-prefix retention.
Use context and cache budgets admitted by the host; do not raise shared limits without authorization.

```sh
python3 tools/zig/check_glm_sampling.py http://127.0.0.1:8080 local-model --output glm-sampling-receipt.json
python3 tools/bench_concurrent.py http://127.0.0.1:8080 local-model --tokens 64 --reps 2 --temperatures 0,1 --levels 1,4 --alone --serial --output glm-decode-receipt.json
```

The first tool requires native token-ID hashes, a cold baseline, and positive cached-token counts after it.
It compares fresh/restored, plain/drafted, and mixed solo/concurrent requests with active filters.
An absent cache hit fails the gate. `GLM_CUTS` checks chunk seams, not snapshot restoration.

Earlier revisions reported full-model unfiltered sampling/cancellation; those do not qualify the repaired shader.
Rerun sampled `tf-glm-run` at depths 0 and 3, cancellation/reset and greedy regression on the final commit.
TP2 device validation requires a second Mac and remains unrun when that hardware is unavailable.

## Receipts and publication

Record final commit, GPU/OS, checkpoint revision, runtime libraries, exact public prompts/settings and token hashes.
Include pass/fail/skip counts, cache hits, before/after timings and peak physical/device memory where reported.
For combined validation with #521, record both commits and any additional local patches.
Keep internal progress notes and local user paths out of source, receipts, PR text and outgoing commit messages.
Link the maintainer-agreed Metal sampling issue/gate; the CUDA checkpoint issue alone is not that agreement.

## Host repair checks

Using Zig 0.17.0 on Linux: 15/15 draw-rule tests, 5/5 served-gate client tests and 89/89 server/cache tests passed.
The production head, generation and backend paths and synthetic gate passed Apple Silicon semantic compilation.
The CPU golden harness reports 312/319 equal, seven documented differences and zero unexpected differences.
It also emits allocator-leak diagnostics; that harness and the server code were unchanged by this repair.
Linux `zig build test` reports 75/76 passed: the direct-I/O read test fails on this filesystem.
The identical direct-I/O test also fails from the untouched reviewed head, so this is an inherited failure.
The macOS-only named golden step is unavailable on Linux; the same checker was run directly with `fake_serve`.
The base reports 512 lean findings; the previous PR head and repaired tree report 511, with zero added findings.
Synthetic Metal and full-model validation of this repair remain pending until run on supported hardware.
