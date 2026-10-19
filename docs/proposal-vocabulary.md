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
