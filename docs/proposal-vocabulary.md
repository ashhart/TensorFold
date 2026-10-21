# Proposal vocabulary artifacts

The shared `core.proposal_vocab` parser admits immutable artifacts for optional shared proposal services.
It changes no default, target projection, target sampling, draft list or model admission.
The optional router is available to backend producers; CLI activation, family adapters and device qualification are later work.

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
Optional shared routing and backend row views are available as preparation APIs. Collective placement and all family
adapters remain later steps. Head-fraction measurements and native exactness/performance gates still apply.

## Optional shared routing

`lanes.proposal_head.Service` is immutable model-owned configuration, absent by default. Construct it against the
actual artifact identity before publication. Native Backend and external Drafter expose the same optional service.
Explicit calibrated mode refuses missing subset support, wrong identity and unqualified multi-rank placement.
Full diagnostic mode requires an available full head. No family has an adapter or enables this service yet.

A calibrated request uses the subset only with sixteen covered committed context tokens and covered pending,
forced and stop IDs. Unknown first-token handles, short histories and structured requests take full fallback;
without admitted full proposal resources, they carry depth zero through absorption. The stream keeps drafts enabled.
The router clears held validity before each planned call and after allocation, producer or cancellation failure.
External Drafted always absorbs kept rows before a positive-depth planned hold; paused requests never reuse fillers.
Coverage is reconstructed from committed context, so an omitted token delays subset resumption until it leaves the
last sixteen tokens. Target verification and correction remain full-width, including omitted tokens and EOS.

Backend producers must implement planned drafting and held invalidation explicitly; unsupported services refuse
before work. Plan references and artifact maps remain alive through the model's queued work. There is no hot swap.
With no service, or legacy mode, the original callback receives the original requests without routing or allocation.
Run `zig build test-proposal-router-host test-shared-lane-regressions-host` for recording, lifecycle and parity fixtures.

## Shared scoring and stored rows

A service binds one scorer before publication. Its descriptor records the actual head geometry, stored format,
group/scale recipe, tied ownership, backend device and queue. The scorer accepts normalized BF16 rows with original
stream, position and numerical-role metadata. Its result carries logical width, physical pitch, immutable global-ID
mapping and the existing argmax, keyed, top-k or coupled selection recipe. Short buffers and mismatched generations,
queues or dimensions refuse. Mixed views partition by generation, binding, shape, role, queue and rank, then scatter
by original row index. No target random draws or family-specific vocabulary policy belong in this interface.

`core.proposal_rows.Source` copies original row bytes into separately owned, zero-padded planes after identity,
shape, placement and memory-price checks. BF16 preserves signed zeros and NaN payloads. Affine rows require explicit
BF16 scale/bias storage; codes, scales and biases are copied without rounding. Other scale dtypes and opaque bases
refuse this subset capability. Logical IDs are never synthesized for padding. The full head remains available.

`cuda.qlinear.proposal` uses the existing BF16 packer or affine-4 packer and projection owners. Decode/prompt role
stays explicit. A shape that would change the original split-K basis refuses the compact view. The shared affine
Metal encoder likewise retains its existing bits, group and output role; its real encoder is compile-checked only.
There is no new precision option. Row selection uses the common tree-round argmax and BF16 top-k rules; keyed CUDA
draws reuse sample's global-ID kernel, with each padded row's logical bound and actual pitch.

Prepared CPU planes, packed words, mapping buffers and device allocations have explicit prices. Admission must
include the old admitted head, source planes and the prepared/upload peak before publication. `uploadBytes` bounds
the prepared generation, new device buffers, temporary int64 IDs and synchronous copy custody. Failed upload keeps
the prepared generation intact. Captured pointers retain the generation; deinit refuses live captures or an
undrained queue. Model owners drain before releasing artifacts, views or graph bindings. There is no live reload.

Run `zig build test-proposal-rows-host test-proposal-head-cuda-host compile-proposal-metal-host` for exact byte,
padding, mapping, recording-driver and compile-only encoder checks. These are host receipts, not GPU arithmetic,
model equality or speed measurements. No native family exposes a proposal-vocabulary serving option yet.

## Existing Flash Next table

Flash Next's default table now enters through the common legacy artifact and row views.
It remains an existing recipe, with no response-calibration claims and no new serving option.
A legacy artifact has no measured coverage and cannot enter calibrated admission.
The target vocabulary, masks, sampling and weights are unchanged.

CUDA retains its 79,591 sorted IDs, logical logit pitch and original BF16 packer and lane launches.
The group-32 affine option keeps its existing precision choice and projection callback.
The Metal pack retains its existing 79,616 real IDs, including the extra IDs its old tiling recipe added.
Those IDs are legacy support, separate from the zero padding of a calibrated view.
Pack source identities, cache keys, raw table bytes and runtime readers keep their current layout.
HIP shares the built-in table reader and retains its original row spans and projection path.
These bridges do not enable arbitrary calibrated lists on unqualified backends.

Host comparisons cover both previous and common paths on the same table and synthetic stored rows.
They compare IDs, selected codes and side planes, packed bytes, cache fingerprints and CUDA launch records.
Flash Next CUDA and Metal checkpoint token hashes and drafted/plain equality still require their device gates.
