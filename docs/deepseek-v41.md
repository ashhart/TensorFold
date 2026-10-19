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
byte-identical to the map shipped beside the original qualified 2.9bpw pack.
Full normalization parity for other tokenizers, including Hangul, remains
[unqualified](engram-token-map.md#normalization-qualification). When `engram-token-map.bin`
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

## 256 GB: the Engram tables from disk

The two Engram tables are about 203 GB of the release, and a token touches only 48 of
their rows (24 per table, 256 bytes each). `--engram-stream` leaves them in their
shards: the load checks both table headers as a resident load does but stages neither,
and it stages only the tensors the backbone binds (no `mtp.*`, `vision.*`, `aligner.*`
or image markers). Each chunk then hashes its row IDs in a command buffer of its own,
reads each table's distinct rows with uncached `pread`s on a few threads, and runs the
rest of the forward with the same gather kernel over the staged rows, so a streamed
reply is the resident reply bit for bit (the gather check stages a synthetic shard and
compares the two gathers byte for byte). The scale columns, 8 bytes a row, are read
once at open and held in host memory (about 6.1 GB); they count against the load
limit, not Metal admission.

```sh
tensorfold-native serve <pack> \
  --engram-table-dir <orig> --engram-stream --chat-template <template> \
  --context 65536 --parallel 1 --no-drafts --prompt-cache-gib 0 \
  --host 127.0.0.1 --port 8099 --name DeepSeek-V4.1-Flash --keep-warm 0
```

- Set `--context` explicitly: `--context 0` means the config's 1,048,576 positions.
- The bound backbone tensors must fit the load limit without the tables. The
  measured LibertAI 1017 2.9bpw checkpoint stages about 202.37 GB and runs with
  `GLM_LOAD_LIMIT_GB=220` on 256 GB. Fit depends on the pack and which tensors the
  backbone binds, not only its advertised bpw. See the
  [pinned checkpoint and receipts](receipts/ds1017-tp2-2026-10-19/README.md).
- Every row read costs a random read: put the tables on the internal SSD.
- Releases without `chat_template.jinja` (DeepSeek-V4.1-Flash-1017 ships only
  `encoding/`): use V4.1's template after checking it against the release's goldens,
  ```sh
  zig build tf-dsv41-template-check && ./zig-out/bin/tf-dsv41-template-check \
    <template> <orig>/encoding/tests
  ```
  which renders each `test_input_N.json` with the engine's Jinja and compares it with
  `test_output_N.txt`. Replies' `<｜DSML｜ calls>` blocks (V4.1's spelling, a space
  after the bar) parse into `tool_calls` like V4's `<｜DSML｜tool_calls>`.
- The token map depends only on the tokenizer and config; a release with V4.1's
  tokenizer and config has V4.1's map, but its tables are its own.

The residency receipt changes with it, and this is a gate change: `tf-dsv41-run` with
`DSV41_ENGRAM_STREAM=1` passes when the execution's disk reads stay within the row
bytes the reader counted itself and no page-ins occur; the resident gate (no disk
reads at all) is unchanged.

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

## Receipts

The qualification receipts — the served parity legs, the lane bit-equality, the oracle
and margin captures — live beside the PRs on the fork; the tree carries the tests that
regenerate them.

## Experimental two-Mac MoE

The development branch adds `--speed-up SETTINGS` for two Metal ranks. On two
M3 Ultra Macs with 256 GB each, the measured logits and tokens match the single-rank
engine. TCP over TB5 is slower in this implementation: 2.78 decode forwards/s
against 4.06 on one Mac. The [development receipts](receipts/ds1017-tp2-2026-10-19/README.md)
include model and HTTP state checks, failure/restart controls and open release gates.
These measurements do not qualify MCDMA, RDMA, drafts or vision.

Each rank holds the full backbone and reads its Engram tables from its own SSD.
The ranks split the output columns of the MoE gate, up and down projections and
exchange their output halves. Each dot product still covers its complete input
width. Attention and routing run on both ranks. This partition does not halve the
memory required on either Mac. The recorded comparisons cover only the pinned checkpoint and inputs.

For MCDMA, use the existing [speed-up link settings](n-link-speed-up.md), with rank 0
on one Mac and rank 1 on the other, one peer per rank, and the same library build.
Alternatively, select TCP explicitly:

```json
{"transport":"tcp","rank":0,"host":"192.0.2.10","port":19750,"session":"my-ds41-test"}
```

Rank 0 binds the stated numeric address. Rank 1 uses the same address, port and
session, with `"rank":1`. Scoped IPv6 is supported, with the scope naming the local
interface on each Mac. The port and the next port carry ordered window writes and
completion acknowledgements respectively. Wildcard addresses are refused. Use a
trusted private link: the session check prevents accidental pairing, but supplies
neither authentication nor encryption. TCP uses ordinary sockets and does not open
an RDMA device. Its latency and bandwidth must be measured separately from MCDMA.
If checkpoint loading finishes at different times, `"connect_timeout_ms":600000`
allows ten minutes for startup pairing. The default is 30 seconds, the maximum is
15 minutes, and message and completion waits remain bounded by 30 seconds.

The CPU-only `zig build tf-tcp-probe -Dcpu=apple_m3` target builds a link check. Run
`tf-tcp-probe RANK HOST PORT SESSION ROUNDS` on both Macs before loading a model.
It checks peer payloads and reports exchange times for 64 bytes, 4 KiB, one routed
MoE row and 256 rows. Those times are transport measurements, not inference speed.

On both Macs, pass the respective settings file alongside the streamed-Engram options:

```sh
TF_DSV41_SPLIT_RESIDENCY=1 TF_DSV41_RESIDENCY_REFRESH=1 \
GLM_LOAD_LIMIT_GB=220 tensorfold-native serve <pack> \
  --engram-stream --engram-table-dir <original-1017> \
  --chat-template <qualified-template> --speed-up <rank-settings.json> \
  --context 8192 --parallel 2 --no-drafts --prompt-cache-gib 0 \
  --host 127.0.0.1 --port 8099 --keep-warm 0
```

Send generation requests to rank 0. Rank 1 follows its reset, forward, commit and
rollback commands and refuses independent generation. Each command checks the
starting cursor; the acknowledgement checks the resulting cursor. A failed exchange
invalidates the pair. Restart both ranks after a failure.

Both ranks must run the same executable, checkpoint metadata, token map, context,
chunk size, lane count and arithmetic settings. Startup compares these identities.
It does not hash all weight and Engram bytes: the launch recipe must verify their
SHA256 manifests separately, including the original 1017 table revision. A successful
startup is not a substitute for those checks.
