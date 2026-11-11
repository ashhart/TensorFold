# DeepSeek-V4.1 prefill hand-off (CUDA to Metal)

A GB10 pair prefills a prompt; a Mac decodes it. The hand-off is the state a prefill leaves behind, carried by MKVH
protocol 2 (recurrent layers of named parts) and converted to the Metal engine's own cache layout
on arrival. The library is `zig/src/families/deepseek_v41/handoff/`; it is host-only and independent of the transport.

A hand-off prefix is **outside the exactness contract**: it is opt-in (`--prefill-from`), labelled remote, never kept as a
native state, and the Metal engine re-quantises some of it (below). Decode after an import is not bit-identical to a
Mac-only prefill, and is not claimed to be.

## What travels

Everything is bf16 or f32 in a neutral form; no CUDA or Metal byte format crosses the link.

| part | layers | dtype, shape | content |
|---|---|---|---|
| `swa` | every backbone layer (0..39) | bf16 `[min(upto,128), 512]` | the last window's keys, oldest first (positions `upto-rows .. upto-1`), normed and roped, fp8 NoPE dequantised exactly |
| `carry` | the ratio-2 KV sources (2, 8, 14) | f32 `[2, 512]` | the compressor's open-group row of position `upto-1`: kv then gate |
| `ring` | the DSpark blocks (40..42) | bf16 `[rows, 512]` | the drafter's last keys, `rows = min(upto, 128)`; optional |
| `comp@P` | the KV sources (2, 8, 14, 20) | bf16 `[n, 512]` | compressed rows `P*page_rows ..` of the paged pool, exact |
| `ixk@P` | the KV sources | bf16 `[n, 128]` | index keys of the same rows, exact |

`comp`/`ixk` windows are whole pages cut to at most 2 MiB (one MKVH frame each); `P` is the first logical page. The
mailbox frame limit is the reason for the cut: a part is one frame. The manifest is one `recurrent` layer per model
layer, `parts` in the order above, dims `{tap, channel}`; `prompt_tokens` is `upto`, `tp_rank` / `tp_size` name the
shard. Nothing is added to the wire: no new dtype, dim name or protocol.

Not shipped: token ids (the decoder holds the prompt, and the manifest's `token_sha256` is verified; Engram's lookback ids
are the last 3 prompt tokens), the position (it is `upto`), the CED stash (the decoder recomputes it), fp8 bytes.

## Why bf16

CUDA stores comp rows as 584 bytes (448 e4m3 NoPE codes with a UE8M0 scale per 64 dims, 64 bf16 RoPE dims, a pad),
index keys as 132 bytes (128 e4m3 and a power-of-two fp32 scale). The Metal engines store other formats:

| engine | window | pool row | index key |
|---|---|---|---|
| release lane (`metal/kv.zig`) | bf16, mirrored over `window + 16` slots | bf16 | 68 B (E2M1 + E8M0, 32-blocks) |
| served lane (`metal_kv_state.zig`) | 528 B (fp8 e4m3 + UE8M0, 32-blocks), slot `pos % 128` | 288 B (E2M1 + E4M3, 16-blocks) | 68 B |

So the conversion is not a memcpy. CUDA fp8 decodes to bf16 exactly (a product of an e4m3 value and a power of two), the
neutral form loses nothing, and the importer packs it with scalar mirrors of each Metal packer (`fp4.zig`). The loss is
exactly what the Metal engine itself applies to a bf16 row: the fp4 pool (served lane), RoPE dims of a window row to fp8
(served lane), index keys to fp4 (both).

## Shards

* **Replicated KV (default)**: one producer, `tp_size 1`, the rank-0 view. Works with `prefill_source.Remote` as is.
* **KV_SPLIT**: logical page `pg` lives on rank `pg % world`. Each rank exports a shard (`tp_rank`, `tp_size = world`):
  shard 0 carries `swa`, `carry`, `ring` and its pages, others only their `comp`/`ixk` pages. The decoder pulls every
  shard into one import. `prefill_source.Remote` pulls one producer (`tp_rank 0`, `tp_size 1`), so a split pair needs
  either a multi-link pull on the Mac or a gather on the producer into one `tp_size 1` export (the exporter takes a
  split view: `Paged.ranks`).
* The CUDA snapshot must hold the rings of every layer; a state without one is an error (`MissingRing`), never zeros.
  Replay prefill (`TF_DSV41_PREFILL=replay`) is exported: the cut stands after the decoder replay, so the live slot holds the
  decoder layers' rings (the last 127 positions the replay wrote). The window's oldest position of those layers was never
  written by the replay (the next token does not read it) and is zero in the export.

## What a reader does

* **Protocol 1 reader**: refuses the manifest and every header as foreign (`BadManifest` / `Protocol`); it never lands a
  byte. Nothing falls back to a wrong layout.
* **Protocol 2 reader** (`fabric.decoder.pull` with `protocol = 2`): receives each part as a recurrent-layer part
  (`rows = 1`, `row_start` = part index). `p2.ShardSink` maps it to `Import.put`, which accepts exactly the parts of the
  plan its shape implies (name, dtype, shape, byte length; duplicates and unknown parts refused; every row finite).
  `Import.complete()` fails (`Incomplete`) until every part has landed.
* The decoder builds the shape itself: `upto` from the prompt it pulls, `shards` from its link set, the drafter rows
  `min(upto, 128)` or 0. `p2.check` compares a manifest with that plan exactly.

## Producing on the GB10

`dsv41-serve` exports a prompt's hand-off when a request names one: `kv_transfer_params.mcdma_handoff` with a 32-hex `id`
and `export_from` 0 on `/v1/chat/completions` or `/v1/completions`. The prompt prefills to its last-but-one token, the
reply is one token, and the state at the cut goes to the producer (`cuda_export_serve.zig`).

* `--mcdma-produce LINK` attaches to mcdma-rpcd's listen daemon and holds each export until the decoder closes it
  (`TF_PRODUCE_HELD_GIB`, as for Qwen).
* `--mcdma-produce http` holds them the same way and serves them on the server's own port, for a Mac with no link
  ([Over plain HTTP](#over-plain-http)).
* `TF_DSV41_HANDOFF_DUMP=DIR` also (or instead, with no link) writes each export as `DIR/<id>.mkvh2`, with
  `<id>.tokens.json` (the prompt ids it covers) beside it, or `<id>.error` with the reason. `<id>.mkvh2` is what the
  producer's replies are, concatenated: the protocol-2 MANIFEST, then every DATA frame (a 128-byte header and the
  payload). `handoff_file.pull` replays it through the decoder's checks into any sink, the importer's `p2.ShardSink`
  included, so a hand-off moves between machines without the link. A file is written whole and renamed, so one that
  is there is complete.
* The export reads rank 0's active slot: its snapshot (`Sessions.snapshot`) and its pool pages, through the same
  `PageStore` the NVMe park uses. A replicated pool holds the same pages and rings on every rank, so no other rank takes
  part. The snapshot's raw bytes are read at the cut, on the engine thread, and converted to the library's bf16 parts on a thread of
  their own (at most two at a time, a third converts inline), so the decode lanes keep running; a decoder that asks before the
  parts are done waits for them (`.wait` on a link, a held request over HTTP). The raw bytes are freed when the parts stand.
* The model name is `LaneHost.Remote.checkpointId` of the checkpoint folder (config and weight index hashed): the
  decoder must serve identical files, or give the manifest's name.

The engine refuses, at start-up and by name, a configuration it cannot export: no session store (`TF_DSV41_POOL_TOKENS`),
`TF_DSV41_KV_SPLIT=1` (half the pages are on the other rank), a non-default prompt shape, and fp4 comp rows. Replay prefill is
exported, with `TF_DSV41_PF_4K` and the other knobs that need it. So is a prompt run in pieces between decode rounds
(`TF_DSV41_PREFILL_PIECES=1`): the cut stands in the prompt's last step, after its last piece and the decoder replay and before
the last token's window, and the other streams keep decoding while the pieces run. The pair's serving profile (pieces, replay,
drafts) therefore exports in the same process, with a replicated pool.

## Assumptions that the library cannot prove

* The Metal engines compute the same normed, roped, compressed row as CUDA for the same tokens, to bf16 noise. The
  library converts layouts; it does not make two engines agree. A first comparison run measures it.
* The drafter ring of a position is `p % 128`-indexed on both sides. A wrong drafter ring only lowers draft acceptance.
* A ratio-2 carry is the projection of position `upto-1`; only the odd-`upto` case occupies a slot on Metal (the other is
  reset: kv 0, score `-inf`).

## The Mac side

`serve --prefill-from URL --mcdma-link NAME` on the DeepSeek V4.1 Metal lane (`native/deepseek_v41_remote.zig`, modelled
on the Qwen3.8-27B pull) gives the lane host a `lane_remote.Source`: `wants` is the shared prefix policy
(`TF_PREFILL_MIN`, `TF_PREFILL_TAIL`, `TF_PREFILL_CHUNK` below), `fill` asks the GB10 over HTTP, pulls protocol 2 into a **blank Snapshot**
(`Model.snapshotBlank`: zeroed `storage`, `live` and taps) through `landing.Landing` (the Packed layout over the Snapshot's own
host slices, so nothing is staged), seals it (`Model.snapshotSeal`: position, the Engram record of the last three tokens'
compressed ids, taps with no rows) and returns `{ saved, at }`. `Backend.prefill` restores it with `snapshotRestore` and
prefills the tail, exactly as for a kept prompt state. `drop` frees the Snapshot; `TF_PREFILL_KEEP` decides whether states
built on it are kept (marked remote). The drafter's rings are not in a Snapshot (the lane holds them in a state of its own), so
`ring` parts are read and dropped; the drafter has no conditioning rows until local forwards fill the taps, which can lower
draft acceptance and nothing else.

### How much of the prompt is handed over

The prefix is the prompt less its **tail**, `TF_PREFILL_TAIL` tokens (default 1): the Mac asks the producer for the whole prompt
(the trigger sends `prefix + 1` tokens, so the producer's export stands at `prefix`, the last prompt token being the one its reply
token comes from), lands `prefix = len - tail` tokens, and the pass on the Mac is the one last token: a one-row, decode-class
forward whose logits are the first token's. That removes the prefill-class pass the 512-token boundary used to leave on the Mac
(96 to 461 tokens: 0.55 s of weight streaming plus 0.9 ms a row, measured). `TF_PREFILL_TAIL=0` goes back to the boundary
(`TF_PREFILL_CHUNK`, default 512, the prefix `(len - 1) / chunk * chunk`). A tail of 9 to 63 tokens runs the slow shared-body
kernels (a 52-row chunk measured 0.85 s against 0.59 s for 100 rows) and is warned about; 1 to 8, or 64 and up, are the sensible values.

* The rows move, they do not disappear: the producer prefills the tail's tokens too (about 0.9 ms a row at 1.0 to 1.1k tokens
  a second, the Mac's own pace), so the saving is the Mac's fixed pass (about 0.5 s a request), growing as the producer gets
  faster than the Mac.
* **Nothing is shipped for the first token.** The producer's reply token is not used: the first token is drawn on the Mac, from the Mac's logits
  of the last prompt token, with the request's own sampling (temperature, top_p, seed, penalties, grammar), exactly as after a local
  prefill. Shipping the producer's token or logits would save one decode-class step (about 30 ms), and would need the producer to capture its
  state after the last token (it stands at the last-but-one), the request's sampling in the trigger, and a drafter with no conditioning at
  all. A part for it is not defined; the wire, the manifest (`first_token` stays the page-row offset it was) and the plan are unchanged.
* **Old and new peers**: every pairing works, because only the prefix length the Mac asks for changes. A producer cuts at the last-but-one
  token of whatever prompt it is sent; an older Mac asks for a boundary prefix, a newer one for all but its tail, and the producer exports each
  as it is asked. A protocol 1 reader refuses the export as foreign, as before.
* **Exactness**: the prefix is still remote (`exact_mode: remote_prefix`, `pulled_prefix` = the prefix) and never kept as native; with
  the default tail a larger share of the prompt is outside the contract (all but the last token, not all but 96 to 461).
* **The drafter** starts with one conditioning row instead of the up to 128 a tail forward gives: the landed Snapshot has no taps, and the
  one-row step fills one. Lookup drafts are unaffected (the history is the prompt); the trained proposer's acceptance is lower for the
  first rounds, until the taps ring fills. A draft is always verified, so this costs speed over the first tokens and never a token.
  `TF_PREFILL_TAIL=64` to `128` keeps a prefill-class tail that conditions it, for a reply long enough to repay the 0.6 to 0.7 s.

### Without a network

`TF_DSV41_HANDOFF_CHECK=roundtrip` (the Metal lane's own prefill exported through `export_metal.zig`, written as a hand-off
file, landed through the same sink) or `TF_DSV41_HANDOFF_CHECK=<file>` (a `blob.zig` file: tokens, protocol 2 manifest, parts
in delivery order; a synthetic one from `synth.zig`, or a producer's parts dumped) runs at load. With
`TF_DSV41_HANDOFF_TOKENS=<ids>` (the full prompt; the file covers its first `upto` tokens, at least one more must follow) and
`TF_DSV41_HANDOFF_STEPS=N` (default 16) it prefills locally, restores the native snapshot as a control (must be exact),
lands the hand-off in a blank Snapshot, then compares the next N logit rows (the reference's tokens fed to both): top-1
agreement and NRMS, and the free-running token streams. Agreement is reported, not required.

A file may also be a producer's stream, `<id>.mkvh2` (`TF_DSV41_HANDOFF_DUMP` on the GB10 pair): its `<id>.tokens.json` beside it
gives the exported prefix's ids, and the stream is replayed through the decoder's checks for this checkpoint's name
(`blob.fromStream`) before it is landed; `TF_DSV41_HANDOFF_FILE` takes the same. `TF_DSV41_HANDOFF_UPTO=N` sets a round trip's
prefix (default the last whole 128 below the prompt), for an odd one, where the carries matter. `TF_DSV41_HANDOFF_BISECT=1`
also takes each part kind alone (window rings, carries, pool rows, index keys) from the hand-off and the rest from the local
prefill, compares the logits as above, and prints the landed rows' relative RMS error against the local prefill's per part
and per layer.

`TF_DSV41_HANDOFF_DUMP=<path>` keeps the round trip's file (a path ending in `.mkvh2` keeps it as a producer's stream instead,
the body `GET /v1/handoff/<id>` serves, for the checkpoint's own name: a fake producer can serve it to the HTTP decoder); `TF_DSV41_HANDOFF_FILE=<path>` serves such a file as the lane
host's remote source (instead of `--prefill-from`): a request whose prompt extends the file's tokens gets that prefix landed
from it, through the lane's own `wants` / `fill` / restore / `drop` path and keep rule, and any other prompt is prefilled here.

## Over plain HTTP

With no ConnectX link between the pair and the Mac, the hand-off travels over the LAN on the ports the servers already
have. It is the same stream a link carries, so nothing in the library changes: the manifest, the part checks and the landing
are the ones `TF_DSV41_HANDOFF_FILE` uses.

**Producer** (the GB10 pair's rank 0, an export profile as above): `dsv41-serve ... --mcdma-produce http`. `http` is a reserved
link name: no mcdma-rpcd, no listen daemon. Each export is held as for a link (`TF_PRODUCE_HELD_GIB`, 120 s) and the server
answers, on its own port and behind its own API keys (`--api-key`, `--api-key-file`):

* `GET /v1/handoff/<id>`: HTTP 200, `application/octet-stream`, `Content-Length`, and the producer's replies to a decoder's
  OPEN and every PULL as one body: the protocol-2 MANIFEST, then each DATA frame (128-byte header, payload, CRC-32). The
  frames are cut from the held bf16 parts a message at a time through one 4 MiB buffer: no second copy of an export exists.
  It may be fetched again until it is deleted or expires (a retry is a new GET; there are no ranges).
* `DELETE /v1/handoff/<id>`: the decoder is done; the export is released at once (HTTP 200, or 404 when none is held).
* 400 when the id is not exactly 32 hex digits; 404 for an id never exported, deleted, or expired (and from a server that was not
  started with `--mcdma-produce http`); 409 with the reason when the export failed (the trigger's reply was still one token).
  A key without the `inference` scope gets 403, no key 401. Only a held export is reachable: the route reads no file and
  takes no path.

The body is not encrypted: it is a state derived from the prompt. Keep the pair's port on a trusted LAN, and use a key.

**Decoder** (the Mac): `serve ... --prefill-from http://PRODUCER:PORT --mcdma-link http`. A request whose prompt is long enough
(`TF_PREFILL_MIN`, `TF_PREFILL_TAIL`) triggers the export with the same `POST /v1/completions` + `kv_transfer_params`, then
`GET /v1/handoff/<id>` is read frame by frame straight into the landing (`handoff_file.pullFrom`: the manifest must be for this
checkpoint, prompt and shard, every frame arrives in order and checksummed, nothing may follow the last; the export is never held
whole here), the copy on the producer is deleted, and the Mac prefills the tail and decodes. `TF_PREFILL_API_KEY_FILE`'s key is
sent on all three requests. `TF_PREFILL_TIMEOUT_S` (default 60 s plus 1 s per 400 tokens) bounds the trigger and, separately,
the transfer. Any failure (the producer down, a 401, a 404, a cut stream, a wrong checkpoint or prompt, a bad checksum, a
timeout) is logged and that prompt is prefilled on the Mac; a checkpoint, prompt or protocol mismatch also turns remote prefill
off for the rest of the run, as over a link. The result is labelled remote as ever and is outside the exactness contract.

The pair, then the Mac (names and paths are placeholders):

```
# each Spark: the export profile (replicated KV, whole-prompt prefill, paged pool), rank 1 as ever `tf-dsv41 follow`
export TF_EXPERIMENTAL=deepseek TF_DSV41_POOL_TOKENS=65536   # plus the pair's usual TF_TP_* and TF_DSV41_* settings
unset TF_DSV41_PREFILL_PIECES TF_DSV41_KV_SPLIT
# rank 0 only: the serve line gains one flag (and a key)
tensorfold-dsv41 serve $MODEL --host 0.0.0.0 --port 8000 --name DeepSeek-V4.1-Flash-TF --no-thinking \
    --mcdma-produce http --api-key-file $KEYFILE
# the Mac
TF_PREFILL_API_KEY_FILE=$KEYFILE tensorfold-native serve $PACK --engram-stream-dir $ENGRAM --chat-template $PACK/chat_template.jinja \
    --prefill-from http://PRODUCER:8000 --mcdma-link http --prompt-cache-gib 4
```

Both ends must load the same checkpoint files: the manifest's model name is the hash of `config.json` and
`model.safetensors.index.json`. The producer logs `handoff: exports are served over HTTP, GET /v1/handoff/<id> on this server's port` at start
and `handoff <id>: N tokens, P parts, M MiB of bf16 rows` per export; the Mac's reply says what happened in its `tensorfold` block:
`"exact_mode": "remote_prefix"`, `remote_prefix` and `pulled_prefix` in tokens, as over a link. A failed pull logs
`remote prefill failed (REASON); the prompt is prefilled here` once per request and the reply has none of those fields.

## Tests

`zig build test-dsv41-handoff` (host only; includes the export glue over a synthetic engine snapshot, the file round trip, the
producer's held-export bookkeeping with its readers, and the HTTP decoder against a loopback fake producer: a whole fetch
landed in a Snapshot's buffers, 401, 404, 409, a cut stream, a wrong checkpoint or prompt, a producer that never answers): codec exhaustively against independent tables, packers against independent
f64 references, plan partitions, export-to-import round trips for both Metal layouts (replicated, split 2 and 3 shards,
short, odd and even `upto`), protocol 2 end to end over the fake link, and layout guards that fail when an engine's
constants change. `zig build test-handoff-routes-host` runs `GET`/`DELETE /v1/handoff/<id>` over real sockets (the whole stream under
its `Content-Length`, the key gate and scopes, bad ids, 404 and 409). `compile-dsv41-handoff-aarch64-linux` / `-macos` compile the tests for the targets.
