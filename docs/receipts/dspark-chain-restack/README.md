# DSpark chain restack

The chain is rebuilt on main d33dcfc767ee093393c6bfa00b3949bd7b4d8e88. Rejected shared/lone guards are absent. Safety fixes are carried in the first drafter PR, and the matrix-at-a-time loader retains BF16 by default with explicit 8-bit opt-in.

The attached historical load receipt records pre-allocation-refactor BF16 startup peak 16,036,097,352 bytes versus 15,527,930,776 bytes after the refactor, a 508,166,576-byte (484.6 MiB) reduction. Both are one stream, context 4096, prompt cache zero. It retains its original source/base identifiers; it is historical evidence, not a new measurement of these rebased commits. No M5 was used for this restack.

Local host/build results and any available M4 device replay will be recorded separately. The existing M5 report in PR1381 shows a remaining downstream 1–2% loss at four/eight streams despite 16-stream gains; restacking does not waive Ash’s performance requirement or count that tradeoff as approved.

## New M4 load comparison

The rebased sources were measured before any request at one stream, context 4096, prompt cache zero, BF16 default. #1392 source `51cbe02b1` peaked at 16,037,424,528 bytes; #1393 source `c1b9d325a` at 15,511,169,992 bytes: a reduction of 526,254,536 bytes (501.88 MiB). Both reported `context_fitted: false` and shut down gracefully. One run each; this is startup peak, not serving footprint or a minimum-RAM guarantee. [Full /memory results and binary identities](m4-load-peaks.json).

## New M4 correctness replay

Final code source `f9de08d58` passes the selected LFM2 runner: 11 kernels pass gates 1–4, all 17 controls red as planted, zero limit problems. Independent drafter reference: 9/9 picks match; worst confidence difference 0.00106. Twenty steady rounds report zero host allocations/resizes. This is not the full all-family kernel runner.

Serving scope is context 4096 / parallel capacity 16 / 64 generated tokens on M4 Pro 48 GiB, macOS 26.5.1 (25F80), Metal 32023.883, Zig 0.17.0. BF16 cache/resume and fresh replies match with 3517 reused tokens. Metal API validation was enabled for the functional server. Stress: six slots / 60 seconds, 120 complete and 79 cancelled; 16 slots / 90 seconds, 250 complete and 132 cancelled; zero wrong replies and zero errors. All 29 sampled cases match plain, solo drafted and concurrent execution. Eight 8-bit cells at 1/4/8/16 streams across code/prose match their plain and solo oracles. Five servers shut down gracefully; no memory-pressure admission warning occurred.

[Full hashes, cache/sample/stress/8-bit results and source scope](m4-validation.json), [host-check summaries](host-checks.json), [M4 reproducer](m4_replay.py). Use this checkout at `tf-dspark-1394` beside `tf-dspark-1392` and `tf-dspark-1393`, or adapt those binary locations; set TF_RESTACK_ROOT/TF_RESTACK_OUT as needed and the qualification helper’s checkpoint environment variables. The helper reserves the existing GPU lock nonblockingly and refuses a recognized active model. No M5 was accessed.

Default build: 239/239 steps. All five stage trees pass global formatting, whitespace and lean comparison (425 inherited findings; zero new). The new replay does not replace the original receipts’ separate long-context scope, and does not claim the remaining 4/8-stream performance loss is fixed.
