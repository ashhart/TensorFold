# DeepSeek-V4.1 prefill hand-off (CUDA to Metal)

A GB10 pair prefills a prompt; a Mac decodes it. The hand-off is the state a prefill leaves behind, in a neutral
bf16/f32 form that is converted to the Metal engine's own cache layout on arrival.
The library is `zig/src/families/deepseek_v41/handoff/`; it is host-only and independent of the transport.

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
| served lane (the EXL3 Metal lane) | 528 B (fp8 e4m3 + UE8M0, 32-blocks), slot `pos % 128` | 288 B (E2M1 + E4M3, 16-blocks) | 68 B |

So the conversion is not a memcpy. CUDA fp8 decodes to bf16 exactly (a product of an e4m3 value and a power of two), the
neutral form loses nothing, and the importer packs it with scalar mirrors of each Metal packer (`fp4.zig`). A packer
returns `error.NotExact` instead of a row the lane's format would hold with other values, so a hand-off lands only when
it is lossless there: an fp4 pool row, an fp8 window row and an fp4 index key each take a record whose values are
already on that format's grid, and one that is not is refused rather than rounded.

## Shards

* **Replicated KV**: one producer, `tp_size 1`, the rank-0 view.
* **KV_SPLIT**: logical page `pg` lives on rank `pg % world`. Each rank exports a shard (`tp_rank`, `tp_size = world`):
  shard 0 carries `swa`, `carry`, `ring` and its pages, the others only their `comp`/`ixk` pages. The decoder pulls every
  shard into one import.
* The CUDA snapshot must hold the rings of every layer; a state without one is an error (`MissingRing`), never zeros.

## What a reader does

* **Protocol 1 reader**: refuses the manifest and every header as foreign (`BadManifest` / `Protocol`); it never lands a
  byte. Nothing falls back to a wrong layout.
* **Protocol 2 reader** (`fabric.decoder.pull` with `protocol = 2`): receives each part as a recurrent-layer part
  (`rows = 1`, `row_start` = part index). `p2.ShardSink` maps it to `Import.put`, which accepts exactly the parts of the
  plan its shape implies (name, dtype, shape, byte length; duplicates and unknown parts refused; every row finite).
  `Import.complete()` fails (`Incomplete`) until every part has landed. A release layout whose window ring or drafter
  ring holds fewer rows than the window is refused at `Import.init` (`RingTooSmall`), before any row lands.
* The decoder builds the shape itself: `upto` from the prompt it pulls, `shards` from its link set, the drafter rows
  `min(upto, 128)` or 0. `p2.check` compares a manifest with that plan exactly.

## Assumptions that the library cannot prove

* The Metal engines compute the same normed, roped, compressed row as CUDA for the same tokens, to bf16 noise. The
  library converts layouts; it does not make two engines agree. A first comparison run measures it.
* The drafter ring of a position is `p % 128`-indexed on both sides. A wrong drafter ring only lowers draft acceptance.
* A ratio-2 carry is the projection of position `upto-1`; only the odd-`upto` case occupies a slot on Metal (the other is
  reset: kv 0, score `-inf`).

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
  is there is complete. With a link or `http`, the file is written only once the export is admitted under
  `TF_PRODUCE_HELD_GIB`: an export refused at the cap leaves `<id>.error` and nothing else.

* The export reads rank 0's active slot: its snapshot (`Sessions.snapshot`) and its pool pages, through the same
  `PageStore` the NVMe park uses. A replicated pool holds the same pages and rings on every rank, so no other rank takes
  part. The raw bytes are converted to the library's bf16 parts when the cut is taken.
* The model name binds the checkpoint the server loaded (`checkpoint_identity.Bound`). Before the load reads a byte,
  config.json, the index and every shard it names are stamped (device, inode, size, modify and change times); after the
  load the name hashes the config, index, shard headers and every shard byte (cached per file under that same stamp),
  and every file is checked against its stamp before and after the hash. The decoder must serve identical bytes, or
  give the manifest's name. A file replaced or rewritten between the stamp and the name, a shard that cannot be read,
  or one that does not match its recorded sha256 refuses the start: no name is given and nothing is exported.
  Not covered: a write that keeps a file's size and both of its times (only a write in the same clock tick as the
  file's last change on a coarse-clock filesystem), a change after the name, and files outside the checkpoint folder
  (the assets folder, `TF_DSV41_PREPARED` images, which are keyed by the pack's full digest read in the same load).

The engine refuses, at start-up and by name, a configuration it cannot export: no session store (`TF_DSV41_POOL_TOKENS`),
replay prefill, a prompt run in pieces between decode rounds, `TF_DSV41_KV_SPLIT=1` (half the pages are on the other rank),
a non-default prompt shape, and fp4 comp rows.

## Over plain HTTP

With no ConnectX link between the pair and the decoder, the hand-off travels over the LAN on the ports the servers already
have. It is the same stream a link carries, so nothing in the library changes: the manifest and the part checks are the
ones the `.mkvh2` file replay uses.

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

The producer logs `handoff: exports are served over HTTP, GET /v1/handoff/<id> on this server's port` at start
and `handoff <id>: N tokens, P parts, M MiB of bf16 rows` per export.

## Tests

`zig build test-dsv41-handoff` (host only; includes the export glue over a synthetic engine snapshot, the file round trip and
the producer's held-export bookkeeping with its readers): codec exhaustively against independent tables, packers against independent
f64 references, plan partitions, export-to-import round trips for both Metal layouts (replicated, split 2 and 3 shards,
short, odd and even `upto`), protocol 2 end to end over the fake link, and layout guards that fail when an engine's
constants change. `zig build test-handoff-routes-host` runs `GET`/`DELETE /v1/handoff/<id>` over real sockets (the whole stream under
its `Content-Length`, the key gate and scopes, bad ids, 404 and 409). `compile-dsv41-handoff-aarch64-linux` / `-macos` compile the tests for the targets.
