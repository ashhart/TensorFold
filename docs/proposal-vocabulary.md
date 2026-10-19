# Proposal vocabulary artifacts

The shared `core.proposal_vocab` parser is preparation code with no serving caller.
It changes no default, target projection, target sampling, draft list or model admission.
Backend scoring, coverage-based routing, fallback, CLI serving options and device qualification are later work.

Version 1 is strict JSON, bounded to 16 MiB. Unknown, missing and duplicate fields refuse.
Integer fields require JSON integer syntax, never strings, floats or exponent notation.
`zig/src/core/testdata/proposal-vocab.json` is a complete synthetic example, not a calibrated release artifact.

- `identity` binds target, drafter, head, tokenizer and generation recipe by name, revision and lowercase SHA256.
  It also binds full vocabulary, head hidden width, stored format, group and scale recipe.
  The parser compares the complete identity with the caller's actual identity, including at equal vocabulary size.
- `ids` contains a positive number of strictly ascending unique global IDs below the full vocabulary.
  `candidate_size` equals its exact logical length. `ids_sha256` hashes ordered little-endian u32 values.
  Backend padding belongs only to a physical stride; it never inserts fake IDs.
- `mandatory_ids` is the complete validated generation/control/byte-fallback recipe set.
  It must equal the caller's expected set and be contained in `ids`.
- `provenance` records corpus, prompt, response, split and settings digests, the collector identity,
  mandatory recipe source and policy version. Version 1 accepts committed response-ID provenance only.
- `coverage` records integer token and covered counts for calibration, size-selection and held-out splits.
  Each split has one `*` total and one or more nonoverlapping bucket rows whose counts exactly sum to that total.
  These counts describe measured support; they neither prove next-token recall nor authorize activation.

The artifact owns its parsed data. Its exposed IDs and strings are const views valid until `deinit`.
The raw artifact SHA256 is its generation fingerprint; serialization changes also change that fingerprint.
Use `global` only for logical columns, `column` for global lookup and `physicalCount` for checked backend stride.
A null mapping is a refusal to interpret padding or an omitted token as a compact candidate.
No score selector, grammar rule, mutable stream state or target fallback is implemented in this parser.

Run `zig build test-proposal-vocab-host` for parser, ownership, identity, padding and coverage checks.
The planted `proposal-vocab-bad.json` has the same vocabulary and valid ID checksum but a wrong tokenizer digest.
It must refuse with `IdentityMismatch`.

## Offline response-ID mode

```
python3 tools/draft_vocab.py responses MANIFEST.json RECEIPTS.jsonl OUT.json --size 32768
```

This mode uses only Python's standard library. Existing `TOKENIZER OUT patterns...` text mode and its flags remain
unchanged and still require the existing `tokenizers` package. That mode's text coverage is not response calibration.
`responses` is a reserved subcommand; a tokenizer file with that basename can still be given as `./responses`.

The synthetic input examples are `tools/tests/fixtures/proposal-vocab-manifest.json` and
`tools/tests/fixtures/proposal-vocab-receipts.jsonl`. The manifest seals identity, complete mandatory IDs, collector,
settings, corpus revision and the predeclared record/split/bucket plan. Each record belongs to one bucket.
`prompt_group`, `template_group`, `source_group` and prompt digests cannot cross splits. A template group identifies
a prompt-generation template, not the shared chat serialization recipe. A source group identifies a corpus unit.
Every declared bucket must have calibration, selection and held-out records. Split the plan before collection.

Receipts must identify the same collector, full-target/head/tokenizer recipe and settings. They declare unrestricted
full-target generation and committed response IDs only. Each planned record appears exactly once with its prompt
and response-ID digests. Unknown fields, prompt/echo arrays, missing records, duplicate keys, invalid IDs, checksum
errors and identity changes refuse. The builder checks these declarations; collector trust and public/synthetic
corpus review remain the producer's responsibility. It does not collect data or authenticate a model run.

Only calibration response frequencies rank candidates: descending count, then ascending global ID. The complete
mandatory recipe occupies slots first; the highest-ranked remaining observed IDs fill the requested size.
The final ID map is ascending global ID. There is no low-ID fill in this mode. Too few observed IDs or too many
mandatory IDs refuse. Selection and held-out IDs never affect ranking. Output reports exact overall and per-bucket
counts separately for each split; no threshold automatically selects a size or enables the artifact.

`corpus_sha256` binds the canonical input manifest, including corpus revisions and the entire split plan.
Prompt, response and split hashes additionally bind the sorted receipt IDs and their token digests. Object keys
are canonicalized; manifest record order is part of the sealed plan. Receipt arrival order does not affect output.
Digests are content hashes, never local paths or timestamps. Output replacement occurs only after full validation.
The manifest and each receipt are bounded to 16 MiB, receipts total 64 MiB/100,000 records, and each response to
1,000,000 IDs. The artifact remains within the parser's 16 MiB limit.

Run `zig build test-proposal-vocab-tool-host` for the offline and legacy-CLI fixtures. These tests use a synthetic
stand-in tokenizer for the unchanged text-mode contract; they do not qualify a real tokenizer or model.
Shared scoring, forced/grammar/OOV fallback, streams, allocations on devices, collective placement and all family
adapters remain steps 3-8. Head-fraction measurements and native exactness/performance gates still apply.
