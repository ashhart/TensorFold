# DeepSeek-V4.1-Flash on one Mac

TensorFold serves the whole DeepSeek-V4.1-Flash from the resident EXL3 packs on one Apple
Silicon Mac. The engine runs on the lane core — one stream per slot, windows of rows — so
concurrent requests each keep their own caches and Engram cursor, and replies are greedy
(`temperature 0`); drafts, grammar masks and tree requests are refused until their pieces
land. The served checks hold: a drafted request reports `"draft": false` and equals the
no-drafts reply, a resumed conversation equals the fresh one, and concurrent lanes equal
their solo runs.

## What you need

- One Mac with 512 GB of unified memory. The 2.9bpw pack stages about 413 GB of weights
  and the 2.8bpw pack about 190 GB; the server stays at 70% of RAM unless
  `GLM_LOAD_LIMIT_GB` opts past it, as GLM's load does.
- The original release's inference directory — `tokenizer.json`, `config.json`,
  `chat_template.jinja` and the two Engram tables — beside the pack or passed with
  `--engram-table-dir`. The weights come under DeepSeek's own terms; the inference code
  is MIT and the native reference follows it (see THIRD_PARTY_NOTICES).
- An EXL3 pack. The qualification ran jayleaton's 2.9bpw and 2.8bpw repacks; the 2.8bpw
  pack's duplicated tensors are owned by the index, so a shard contributes only the
  tensors the index maps to it.

## The token map

The Engram tables are keyed by a compressed vocabulary derived from the release's own
tokenizer and inference config. The native engine builds the map itself — decode each
id, fall back to the raw token when the decoder emits U+FFFD, normalize through the
release's chain (NFKD, strip marks, lowercase, the whitespace replaces), dedupe — with
the multipliers drawn from NumPy's PCG64 exactly as the source does. The result is
byte-identical to the map shipped beside the 2.9bpw pack. When `engram-token-map.bin`
is missing beside the checkpoint and the original's tokenizer sits beside the tables,
the engine builds it at open; no Python step. To build it by hand:

```sh
zig build tf-dsv41-token-map && ./zig-out/bin/tf-dsv41-token-map \
  --tokenizer <orig>/tokenizer.json --config <orig>/config.json --output <pack>/engram-token-map.bin
```

## Serve

```sh
GLM_LOAD_LIMIT_GB=430 tensorfold-native serve <pack> \
  --engram-table-dir <orig> --chat-template <orig>/chat_template.jinja \
  --context 65536 --parallel 2 --no-drafts --prompt-cache-gib 0 \
  --host 127.0.0.1 --port 8099 --name DeepSeek-V4.1-Flash --keep-warm 0
```

`--parallel` is the lane count: each lane is one stream with its own caches and Engram
cursor. `--no-drafts` is the honest flag for now — the engine decodes greedily and a
drafted request answers with `"draft": false` and the same tokens. The resident load
reads no weights from disk during execution; a run that does fails the receipt's
residency gate.

## The numbers

The per-operator reference tests run every stage against an FP64 ideal at three sides —
kernel, FP32 source contract, FP64 ideal — and assert rounding level. The whole-model
picture is sensitivity-dominated: the FP32 oracle against the FP64 oracle on the same
tokens is 2.90% NRMS, Metal against FP32 is 4.10% at 2.9bpw and 5.16% at 2.8bpw (the
error grows as the quantization coarsens, the signature of the model amplifying
FP32-level differences), and the indexer's 512-pick selection boundary lands inside an
exact BF16 tie at every indexed layer past the window fill — so pick order there is
decided by sub-quantum noise, folded into the sensitivity accounting. The 1% absolute
gate stays a trip-wire next to argmax agreement, which held at every width.

## The dual-width oracle

`oracle-precision` runs the FP32 and FP64 oracles on the same checkpoint bytes and the
same tokens with the GPU out of the loop, and keeps both logit vectors:

```sh
zig build tf-dsv41-run && ./zig-out/bin/tf-dsv41-run oracle-precision \
  <pack> <orig> ids.json <out>
```

It reports the NRMS between the two widths, the FP32 and FP64 argmax per token, and
both wall times; the window is capped at 16 tokens, one teacher step. The same run
prints the residency receipt, so a page touched from disk during execution is visible
in the artifact rather than inferred from wall time.

## Receipts

The qualification receipts — the served parity legs, the lane bit-equality, the oracle
and margin captures — live beside the PRs on the fork; the tree carries the tests that
regenerate them.
