# Conformance ledger: one output and cache oracle

Every speed experiment (the research queue's rank 0 prerequisite) must leave these invariants as they are: the same token IDs, termination, accepted path and visible cache state as the unchanged serial decode on the same engine, sampling recipe and seed. Each row names the host tests that pin it. They run without a GPU over the lane core's fake target (`zig/src/core/lanes/fake.zig`), whose every token is a function of the whole fed history, so a rollback, position or cache mistake changes the reply. A PR that changes decoding cites the rows it relies on and keeps their tests green.

**Run them:** `zig build test-shared-lane-regressions-host` (the whole lane core, `zig/src/core/lanes/lanes.zig`). The `core/lane_host_*` rows also run under `zig build test` (the engine API's module).

The new tests are in `zig/src/core/lanes/conformance_test.zig`, named "conformance: …" with the rows they pin.

## C1. Fresh equals resumed

A reply resumed from a kept prompt state emits the fresh run's tokens, and holds the fresh run's cache.

| Test | What it pins |
|---|---|
| conformance_test "fresh and resumed serial decode emit the same tokens and hold the same cache every round" | Plain and drafted, greedy and keyed. The tokens and stop reason equal the fresh run's, and both equal `fake.next`'s decode. `cached` is 5 against 0. After every step the cache is the fresh decode's (prompt and committed tokens). For one-token rounds the cache digest equals the fresh run's at every step. |
| drafted_test "a restored prompt, part or whole: the drafter starts past it and the output is the one-token decode" | Tokens and `cached`, greedy and sampled, restores at 12 and at the whole prompt. |
| fill_test "grafted replies: resumed and fresh solo and neighbors equal plain" | Tokens and reason, with 1, 2 or 4 streams and tree windows; `cached == 5`. |
| lone_costs_test "lone price: changing widths keeps drafted, plain, resumed, fresh and concurrent outputs exact" | Tokens. |
| lane_host_reuse_test, lane_host_flush_test, lane_host_rows_test | Through the lane host: resumed from RAM or disk, the tokens equal the fresh decode. |

## C2. Tree equals chain equals plain

Tree windows commit the one-token decode.

| Test | What it pins |
|---|---|
| conformance_test "tree windows commit the plain decode's tokens, and their rejected rows never stay in the cache" | Chain and tree windows (suffix-match grafts beside the head's chain), fresh and resumed, emit the plain run's tokens and reason. Tree rows really are verified (`branch_rows > 0`). The cache is checked every step. |
| fill_test "a shared round that trims a window with suffix-match grafts replies as the one-token decode" | Two streams sharing a round; branch rows kept. |
| recall_engine_test "Recall's continuations graft beside the head's chain and keep the reply's tokens" | Tokens, with Recall's grafts. |
| grammar_budget_test "… masks each row as its plain step does, chain or tree" | Per-row grammar masks only. |

## C3. Reproducible, greedy and keyed

| Test | What it pins |
|---|---|
| conformance_test "the same request twice, greedy and keyed, emits the same tokens in the same rounds" | Plain, chain and tree, greedy and seed 23: the same tokens, steps, drafted rows and cache digest every step. A keyed reply differs from the greedy one (so the key matters). |
| engine_test "drafted rounds commit the one-token decode, greedy and sampled"; "shared rounds commit what each stream commits alone" | Drafted equals plain, shared equals alone. |
| drafted_rows_test "drafted rounds give the rows of one-token rounds, greedy and sampled" | Logprob rows as well. |
| rollback_test "rollback: heads landing no draft, some or all commit the one-token decode, alone and shared, greedy and sampled" | With heads landing 0, 1, 2 or all of their drafts. |
| sampling_identity_test "host sampling identity: keyed bits equal main's formula …"; sampling_race_test | The keyed draw itself. |

## C4. Stops

EOS and `max_new` end a drafted reply where the plain decode ends, with its reason.

| Test | What it pins |
|---|---|
| conformance_test "eos and max_new end a drafted reply where the plain decode ends, with its reason" | Plain and drafted, greedy and keyed. At an EOS reached partway: reason `stop`, and the tokens up to and including it. At `max_new`: reason `length`, and exactly that many tokens. |
| stream.zig "commit stops at eos and length" | `Stream.commit` at EOS (its length path isn't exercised). |
| handover_test, open_first_test "max-one and EOS terminate successfully without preparing any drafts", engine_test "structured output: the grammar's stop token ends a reply that ignores eos" | Handed-over streams, one-token stops, grammar stops. |

## C5. Ties and non-finite logits

| Test | What it pins |
|---|---|
| sampling.zig "temperature 0 draws greedily, an exact tie going to the lower id" | Greedy ties. |
| sampling_row.zig "candidate selection preserves full-vocabulary keyed draws and tie order" | Keyed and greedy draws over bf16 rows; a NaN word is `NonfiniteLogits`. |
| sampling_row_test "the blocked row scan equals the scalar walk over sizes, ties, zeros, masks, tails and non-finite words"; "… keeps the first of two equal or adjacent words …"; "… refuses every non-finite encoding in a block nothing is allowed from" | The row scan exhaustively. |
| sampling_selection_test "host sampling: NaNs and infinities retain the original full-sort behavior", sampling_identity_test "… a uniform that rounds to one (+inf noise) …" | Non-finite host draws. |
| core/tree_round.zig "finite ties choose first token and nonfinite kinds invalidate a row" | Tree rounds. |

**Not pinned on the host through the engine:** the fake target's logits are always finite. Non-finite rows are pinned at the sampler, which every backend's host draw uses; device argmax kernels are pinned by the kernel contract.

## C6. Malformed tree parents refused

| Test | What it pins |
|---|---|
| conformance_test "a parent at or after its child is refused at accept.depths and windows.build, before any backend" | A row that is its own parent, or a child of a later row, is `ParentAfterChild` at `accept.depths`. A plan whose draft hangs under a later draft is refused by `windows.build`, and the backend has run no round. A valid tree builds with its row parents and depth-keyed positions. |
| accept.zig "sanitize drops orphans" | A forward parent and its orphaned child are dropped when a window is cut. |
| trim.zig "select keeps listed drafts with their parents re-indexed" | Selecting a node without its parent is `ParentNotKept`. |

Backend-side refusals:
- `families/qwen27/state.zig` "invalid parents and context overflow refuse before state mutation";
- `families/qwen27/attention_plan.zig` (`AttentionParents`);
- `core/tree_slots.zig` "forests reset to committed state and bad parents never make a schedule".

## C7. Off-path state guarded

Rejected rows never become visible state.

| Test | What it pins |
|---|---|
| rollback_test (every step) | One cache per live stream, its digest the fresh run's. Heads land 0, 1, 2 or all drafts, alone or shared. A cancel between any two rounds, or at any prompt ask, leaves peers and the next request fresh. |
| conformance_test "tree windows …" and "fresh and resumed …" | The same cache check every step for tree windows and resumed streams, which rollback_test doesn't cover (it runs chains, unrestored). |
| drafted_test "the strict target holds only what Features promises: a keep drops the rejected rows" | A drafter can't read rejected rows. |
| commit_storage_test | Host stream state (context, rows, force queue, grammar) at the accepted prefix on every failure. |

## Not pinned on the host

- **Device arithmetic:** row-invariance of verify widths (the kernel contract's gates) and exact KV contents on a GPU. The fake's cache is its token history, which pins positions and rollback, not bytes.
- **Non-finite logits through a backend's own argmax:** the fake's logits are finite (C5).
- **Multi-rank (TP) agreement.**

These need the device checks each family's PR names.
