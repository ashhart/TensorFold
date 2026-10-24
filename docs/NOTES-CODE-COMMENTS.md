# Notes implementation commentary

Detailed invariants retained from inline comments during the current-main rebase. Source comments reference the numbered sections below; executable behavior and tests are unchanged by this relocation.

## 1. zig/src/core/lane_host_forget_test.zig

Engine.forget_prompts on a lane host: the engine thread drops the kept states holding a span, between rounds. The planted state is first shown to resume (the check can fail), then shown gone while an unrelated one resumes.

## 2. zig/src/core/prompt_cache_forget.zig

Every kept state whose tokens hold `span` dropped (but `held`), every last prompt holding it replaced, and every learned state on disk whose tokens hold it removed: an erased note's tokens, out of the cache. An empty span forgets nothing.

## 3. zig/src/server/notes/alias.zig

What populate() encoded, kept by the store between versions: each name fact is encoded once, when written, and indexed by its alias keys and its place, so that a layered version recomputes only the keys a write touched.

## 4. zig/src/server/notes/alias.zig

Build only from current structured name attributes. Ambiguous aliases are disabled in the legacy unique map rather than selecting the last entity. Retain all distinct candidates separately for the weighted resolver.

## 5. zig/src/server/notes/alias.zig

populate() with the encodes of earlier calls for the same model kept in `memo` (the writer's, under its lock). A layered version (idx.layer.core set) gets only the keys its delta touched: the name facts written or retired since its core, and the first-person words when "me" came after the core; the rest it reads from its core.

## 6. zig/src/server/notes/bounded.zig

Bounded writes (docs/NOTES-WRITE-ERASE-DESIGN.md, Gate 1): a version is a core (a full suffix-array index, shared by reference count) plus a delta of the notes written since it (layer.zig). A delta past the soft limit (Store.side_max) starts a merge on its own thread from the version just published (merge.zig, core_build.zig: the suffix arrays merged linearly, sa.merge; nothing copied under the writer's lock); the writes go on against the old core meanwhile. Past the hard limit a write waits for that merge (back-pressure): the only place a write waits for merge work. The journal is untouched: everything here is derived state.

## 7. zig/src/server/notes/bounded.zig

Frees a core whose last version went, on a thread of its own: a core is the whole store's columns and suffix arrays, and the store's allocator zeroes what it frees, so freeing one in a write would write out the store.

## 8. zig/src/server/notes/bounded.zig

A full build: every note in the index's own columns (sa: a new core, its suffix arrays merged from the current core's; places move only in an erase, whose own build replaces the current core).

## 9. zig/src/server/notes/core_build.zig

A merge's new core, made from a published version alone (its accessors, its core's suffix arrays and the side notes' token ids it keeps): no copy of the writer's model, so a merge takes nothing under the writer's lock. The result is what Index.buildWith makes of the same notes (notes_delta_test.zig and notes_layer_test.zig check it, lookups and accessors, against that full build).

## 10. zig/src/server/notes/index.zig

rankAt() by the note itself: a place below 2^24 and a told tick in 0..told_span (2^40 ticks: past the year 3000); ids ascend with places (model.zig's placeOf relies on it too), so a tie by place is Python's tie by id.

## 11. zig/src/server/notes/index.zig

A core place's override in this version, if a write since the core retired it. The core's `retired` bits (set by the writer, never cleared) skip the table for the places no version on it has retired; a bit set by a newer version only sends an older one to its own table, which then has no entry: the core's value stands.

## 12. zig/src/server/notes/index.zig

Python's tie order after the score, (-told, id), as a key that sorts ascending. By the note itself when its told and place fit (rank_by_note: `(told_span - told) << 24 | place`; ids ascend with places), so no write sorts the other notes; else the full build's rank of every note, `trank << 32 | place`.

## 13. zig/src/server/notes/layer.zig

value.zig's value entities of the side's written notes. A core written note that a name added since the core may name is not revisited here (a common name can be in most notes): valueEntityAt() resolves it when read.

## 14. zig/src/server/notes/merge.zig

A merge off the writer's path: a new suffix-array core built on its own thread from one published version. The version is immutable and its index held (a reference, as a layered version holds its core), so the merge reads a consistent snapshot while writes go on; nothing of the writer's model is copied or read (core_build.zig). The version itself is not held: the lines it replaced may go with it as usual, since the new core keeps only their slices for places retired since, which the versions on it read from their own overrides.

## 15. zig/src/server/notes/overlay.zig

Bounded immutable-version metadata delta over a borrowed, retained base. The caller owns base lifetime. Fork before changing a published version.

## 16. zig/src/server/notes/sa.zig

One note's folded keys and token ids for the builder; `cur`: its suffixes go to the current view's array. In a merge (sa.merge) an old note may come without ids: its keys' wordiness is the old base's.

## 17. zig/src/server/notes/sa.zig

build() of `notes` whose first `old.notes` are the notes `old` was built over (the same keys, by place; their current or not may differ now): the same base, bit for bit, made by a linear merge of `old`'s sorted arrays with the newer notes' suffixes sorted alone, instead of sorting every suffix again. A suffix ends at its note's separator, so suffixes of two sets of notes order without reading across them; the dictionary's new codes keep the old keys' order, so the old arrays stay sorted under them.

## 18. zig/src/server/notes/sa.zig

Every note's keys as `b` holds them (its text decoded by the dictionary), flat, with each note's start: what a merge reads the old notes from, so that it needs no copy of the writer's notes.

## 19. zig/src/server/notes/sa.zig

build()'s order of two suffixes: their first three codes as its radix sort reads them (across a separator, as the packed prefix does), then within an equal prefix SortCtx.less (positions, when a separator ends both).

## 20. zig/src/server/notes/sa.zig

One view of a merge: the old base's suffixes of notes in that view now (from either old array, each already in suffix order) and the newer notes' (sorted), merged three ways into `out`.

## 21. zig/src/server/notes_decode.zig

The time one reply's checks may take in all, the name scans and the lookups, in ms ("notes_decode_budget_ms"). A lookup under way is never cut: the budget is read before each one, so a reply can go over it by one lookup.

## 22. zig/src/server/notes_decode.zig

After a reply token: whether a lookup runs now. A new name not covered asks for one; the cap, the budget and the gap decide. `found`: the entity whose name the token ended (null: none).

## 23. zig/src/server/notes_decode.zig

The entity whose name the reply's last tokens end, when it has a current note outside the block and no earlier name of it asked for a lookup in this reply; null: none.

## 24. zig/src/server/notes_decode_fields.zig

A request's notes_decode (default on), notes_decode_place, notes_decode_names (false: no check point at all), notes_decode_dry (checks only), and the limits: notes_decode_checks, notes_decode_budget_ms and notes_decode_gap.

## 25. zig/src/server/notes_delta_test.zig

Differential: after every step of random writes, take-backs, erases, merges and restarts, every fixture question answers on the store's version (a core plus a delta, merged on a thread or in the write) exactly as on a full build of the same notes by the old path (every note copied, the tie rank by a global sort, aliases from nothing). Versions held across a step answer as the oracle did before it.

## 26. zig/src/server/notes_delta_test.zig

The alias tables of `got` (read through its accessors: a layered version's overrides, then its core's) against `want`'s: every key either names, in both, the same entity and candidates, or names nothing in both.

## 27. zig/src/server/notes_erase_cache_test.zig

An erase reaches the engine's prompt cache and its --learn files: a chat with the note in its block keeps a state (and learns it to disk); the erase forgets both, an unrelated state survives, and a restart brings nothing back. The negative control is the same run with an engine that has no forget hook: the state stays, and the reply says so.

## 28. zig/src/server/notes_erase_cache_test.zig

An engine over a real prompt-cache store: each request keeps its state before the last token, as a shared cut (so --learn writes it to disk), and answers "o".

## 29. zig/src/server/notes_erase_cache_test.zig

A note as the client sends it, as the block quotes it (written out by hand, apart from line.quote), and how the test ages it before the erase.

## 30. zig/src/server/notes_erase_merge_test.zig

An erase while a background merge (notes/merge.zig) is under way, or done and not yet layered on: the erase stops it and drops the core it made (that core holds the erased note, at the old places). After it every accessor of the version lookups read equals a full build of the store's notes, no lookup finds the note, and it stays gone after the next merges and a restart.

## 31. zig/src/server/notes_erase_merge_test.zig

`running`: the merge is held when the erase begins and let go while the erase waits for it; else it is done (ready) and no write has taken its core yet.

## 32. zig/src/server/notes_erase_places.zig

How a place stands for this erase: it may hold the note, or the server can show it does not (off, empty, or forgotten: the engine searched it for the note's tokens and removed every match). Anything else stays may_hold.

## 33. zig/src/server/notes_erase_places.zig

The tokens a prompt that carried `note` in a notes block holds, whatever line wraps it (null: none can be shown). The block holds the note as `[id] "<quoted>"` and maybe a label (line.render), so the tokens searched for are those of the quoted text as the block renders it (line.quote: JSON escaping), cut at a space on each side: the first and last word can merge with the quote marks, the id or the label. That part is checked against the tokenizer itself: it must lie inside every wrapper the block can give the line (any id, every label, a line before and after or none, the line first or last), else one more word goes from each side; a note without such a part with at least four stable tokens gives null; then the erase cannot show its prompts gone.

## 34. zig/src/server/notes_erase_places.zig

Every erased note forgotten by the engine, by the tokens its quoted text has in a block: kept prompt states and --learn files that hold them.

## 35. zig/src/server/notes_erase_places_test.zig

The erase reply's "complete": true when every place is shown clear (off, empty or forgotten). Disk writers that were off for this-process notes, or off in every recorded start covering an older note, report off. Promotion is off for a this-process note when this process made no pocket; unknown origin stays may_hold. Remaining_places names only may_hold.

## 36. zig/src/server/notes_erase_span_test.zig

The erase's search span on a real model tokenizer (NVIDIA Nemotron 3.5 Lightning's tokenizer.json): it must lie inside the tokens of a real chat prompt whose notes block holds the note, with any id and any label, while the raw text's tokens (the first fix's search) do not for an escaped note. Skipped when the tokenizer is not on disk: TF_NOTES_TOKENIZER_DIR names its folder, else the Hugging Face cache path on petrus's Macs is tried.

## 37. zig/src/server/notes_ops.zig

The text of every note the removed writes made, copied out before the old model was freed: what the engine's prompt cache is told to forget, as the notes block quoted it.
