# DSpark chain restack

The chain is rebuilt on main d33dcfc767ee093393c6bfa00b3949bd7b4d8e88. Rejected shared/lone guards are absent. Safety fixes are carried in the first drafter PR, and the matrix-at-a-time loader retains BF16 by default with explicit 8-bit opt-in.

The attached historical load receipt records pre-allocation-refactor BF16 startup peak 16,036,097,352 bytes versus 15,527,930,776 bytes after the refactor, a 508,166,576-byte (484.6 MiB) reduction. Both are one stream, context 4096, prompt cache zero. It retains its original source/base identifiers; it is historical evidence, not a new measurement of these rebased commits. No M5 was used for this restack.

Local host/build results and any available M4 device replay will be recorded separately. The existing M5 report in PR1381 shows a remaining downstream 1–2% loss at four/eight streams despite 16-stream gains; restacking does not waive Ash’s performance requirement or count that tradeoff as approved.

## New M4 load comparison

The rebased sources were measured before any request at one stream, context 4096, prompt cache zero, BF16 default. #1392 source `51cbe02b1` peaked at 16,037,424,528 bytes; #1393 source `c1b9d325a` at 15,511,169,992 bytes: a reduction of 526,254,536 bytes (501.88 MiB). Both reported `context_fitted: false` and shut down gracefully. One run each; this is startup peak, not serving footprint or a minimum-RAM guarantee. [Full /memory results and binary identities](m4-load-peaks.json).
