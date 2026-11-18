# Exact drafted/plain tokens near BF16 ties

`draft_divergence.py` selects a small set of token prefixes from recorded greedy
traces, then compares one serial completion with requested draft caps 1, 4 and 8.
Every token ID must match. A `near-tie` label explains a mismatch; it never waives
one. Selection and labels reuse `zig/qwen27_margins.py`'s BF16 spacing rule.

Run from the repository root. Python's standard library is sufficient. The
`--replay` mode opens no endpoint and needs no model or GPU:

```sh
python3 tools/draft_divergence.py \
  --recording tools/tests/fixtures/draft_divergence/recording.json \
  --profiles tools/tests/fixtures/draft_divergence/profiles.json \
  --replay tools/tests/fixtures/draft_divergence/equal.json \
  --lookback 2 --tokens 8 --output equal-result.json
```

The example prints exactly:

```text
DRAFT_DIVERGENCE status=pass cases=1 comparisons=3 divergences=0
```

Replacing `equal.json` with `divergent.json` exits 1 and identifies the first
mismatch at output index 2 for requested depth 4. These files are **synthetic unit
fixtures**, not measurements of a checkpoint, backend, GPU or numerical error.

## Recordings and selection

A recording has `schema: 1`, an `identity` object of nonempty strings, and `traces`.
Use pinned checkpoint, tokenizer, backend, precision and configuration identifiers
in the identity; every endpoint profile must repeat the same identity exactly.
This is the operator's declaration, not server attestation.

Each trace has a unique `name`, a nonempty `prompt` of token IDs, generated
`tokens`, and one ordered raw `top2` row per generated token:

```json
{
  "schema": 1,
  "identity": {"checkpoint": "pinned-revision", "tokenizer": "pinned-tokenizer", "backend": "precision-and-config"},
  "traces": [{"name": "example", "prompt": [1, 2], "tokens": [12],
              "top2": [{"ids": [12, 99], "logits": [16.0, 15.9375], "position": 2}]}]
}
```

`position` is the absolute zero-based generated token position, including the
prompt. Its first entry is `len(prompt)`. The first top-two ID must equal the
recorded greedy token. All logits must be finite raw scores, ordered largest
first. Do not substitute log-softmax values: BF16 spacing depends on absolute
magnitude and is not invariant under that transformation.

Select positions whose margin is at most one BF16 spacing of the highest raw
logit. Defaults: at most 8 distinct prefixes, up to 8 generated tokens before
the tie, and up to 32 generated tokens per run. Cases are ranked by margin divided
by spacing, then name. Their prompt is the original token prefix, with no text
round trip. Increase `--tokens` above `--lookback`; windows stop at the end of the
recording. No selected positions is an error, not a passing empty test.

## Served endpoints and draft depths

The current server accepts `draft: false/true`; it has no per-request depth cap.
Configure one server endpoint per supported cap using that family's actual
configuration. Do not add a fictitious `draft_depth` request field. Each profile
has `depth`, `url`, `model` and the recording's exact `identity`. The fixture
`profiles.json` illustrates this shape but contains placeholder URLs only.

For a real family run, prepare `recording.json` from that same checkpoint,
tokenizer, backend and configuration, and `profiles.json` from its configured
servers. Run on the hardware selected for the family:

```sh
python3 tools/draft_divergence.py --recording recording.json \
  --profiles profiles.json --depths 1,4,8 --cases 8 --lookback 8 --tokens 32 \
  --timeout 120 --output divergence-result.json
```

For each requested cap, the nearest supplied supported cap is selected; ties
choose the smaller cap. JSON records both numbers. Different configured caps
require distinct URLs. The operator must ensure their configurations differ
only in draft cap and all resolve to the same model. Caps are configuration
metadata; the response does not attest the actual draft length at every round.

The lowest-cap endpoint also provides the serial reference with `draft: false`.
Requests use `/v1/completions`, token-ID-array prompts, greedy sampling,
`ignore_eos: true`, `return_token_ids: true` and nonstreaming responses. A base URL
may end with `/v1` or `/v1/completions`. Redirects are refused. Authentication is
not implemented; URLs containing credentials or query strings are refused.

The tool requires `tensorfold.token_ids`, a matching `tensorfold.drafts` boolean,
and consistent completion-token counts. It also requires positive
`speculative.drafted` work for each drafted run. Equality without observed draft
work, an early stop, or failure of the serial run to reproduce the selected
recording window yields `incomplete`, never `pass`.

## Divergence JSON and logits

Every case contains serial and drafted IDs and the first mismatch for each cap.
`output_index` is zero-based within the generated window; `absolute_position`
includes the complete prompt. A length mismatch reports a null missing token.
Any mismatch returns `divergent`, even if coverage is also incomplete.

The current API does not expose raw logits. Its logprobs are not raw logits and
are never used here. JSON retains the source recording's two raw scores and
reports each `recorded_logits_for_tokens` entry only when the first-divergence
history matches that recording and its token is present in the recorded top two.
These are historical scores, not new plain/drafted measurements. Missing scores
are null.
Live results leave `plain_top2` and `drafted_top2` null.

Offline observations may additionally carry per-run raw captures. Their shape is
`{"response": <complete API JSON>, "top2": <aligned raw rows>}`. A replay file maps
`CASE:plain` and `CASE:REQUESTED_DEPTH` to these observations; `CASE` is
`TRACE_NAME:SOURCE_GENERATED_INDEX`. The tool's output retains the observations
and can itself be replayed by saving its `observations` object as a replay file.
Each optional capture must align with that run's IDs and absolute positions.
When both captures are supplied, `plain_top2` and `drafted_top2` report both
measured score pairs and the shared helper labels `near-tie` or `bug`. Without
them, `recorded-near-tie` explicitly describes the historical source alone.
Never claim fresh per-run logits from an ordinary live API result.

Exit codes: 0 strict equality with complete coverage, 1 any divergence, 2 an
incomplete run or malformed input/response. A JSON file and one summary line are
written for completed evaluations and handled errors. Endpoint errors stop the
run and produce `status: error`; rerun that input after resolving the endpoint.

## Fixture checks

```sh
python3 -m pytest tools/tests/test_draft_divergence.py
python3 -m pytest tools/tests
```

The new fixture tests also run without pytest:

```sh
python3 -m unittest discover -s tools/tests -p test_draft_divergence.py -v
```

Fixture checks exercise request construction with an injected client. They do
not run a device, open a socket, or replace the family's actual hardware check.
