# Stage protocol fixtures

Golden bytes of every stage protocol message (RFC #2005 section 6.4, boundary format v0), good and bad. A CUDA or
Metal stage decodes each file and must take the good ones and refuse each bad one with the error named below.
`fixtures_test.zig` re-encodes every file and compares bytes, and runs `fixtures.judge` on each. To rewrite them after
a deliberate format change: `TF_STAGE_FIXTURES_OUT=$PWD/zig/src/fabric/stage/fixtures zig build test-stage`.

Everything is little-endian. Every protocol payload starts with an 8-byte prefix: magic `0x50534654` as a u32
(bytes `54 46 53 50`, "TFSP"), u16 version 1, u8 message (1 describe, 2 ready, 3 plan), u8 reserved 0. Rows payloads have
no prefix: they are raw rows after the 32-byte frame header of `frame.zig`.

## The fixtures' model and placement

- engine version `tensorfold stage-v0 test`, KV bf16, 8 layers, drafter taps at layers 1 and 5;
- boundary: hidden 4, rows bf16, 2 planes, draft width 2;
- shapes: prompt chunk 4, max rows 8, parallel 4, context 64;
- rank 0: CUDA, layers 0..4; rank 1: Metal, layers 4..8 (ranges are half-open: `first` up to, not including, `end`).

The round in every plan and rows fixture: stream 3 open at 0 with 4 rows; stream 7 continue at 10 with 1 row,
2 accepted, generation 1; stream 9 resume from 32 at 32 with 2 rows, generation 2; stream 12 close at 20, generation 1;
one chunk cut at row 4; 7 rows.

## Layouts

Description (message 1), 72 bytes, then `n_taps` u32 tap ids (strictly ascending, inside the range), then the
version text (`n_text` bytes, printable ASCII, 1..64):

| bytes | field | bytes | field |
|---|---|---|---|
| 0..8 | prefix | 32..36 | hidden width |
| 8..12 | rank | 36..40 | draft width (0: no taps) |
| 12..16 | world | 40..44 | prompt chunk |
| 16 | backend (1 cuda, 2 metal) | 44..48 | max rows a round |
| 17 | KV dtype | 48..52 | parallel |
| 18 | row dtype | 52..56 | context |
| 19 | residual planes | 56..64 | KV rows it can hold (u64) |
| 20 | n_taps (at most 16) | 64..68 | recurrent states it can hold |
| 21 | n_text | 68..72 | reserved 0 |
| 22..24 | reserved 0 | | |
| 24..28 | first layer | | |
| 28..32 | end layer (exclusive) | | |

Dtypes: 1 bf16, 2 f16, 3 f32, 4 f8e4m3, 5 nvfp4 (KV only; not a row dtype).

Ready (message 2), 16 bytes: prefix, rank u32, flags u32 (bit 0 loaded; bit 1 warm, executable for the declared
shapes, with no graph capture implied; others 0).

Plan (message 3), 24 bytes, then 28 bytes a stream, then u32 cuts: prefix, streams u32, cuts u32, total rows u32
(the streams' sum), reserved u32. A stream: id u32 (strictly ascending), op u8 (1 open, 2 continue, 3 close,
4 cancel, 5 resume), 3 reserved bytes, first position u32, rows u32, accepted u32, resume point u32 (0 unless resume),
state generation u32 (0 for open; a stage refuses a stream whose generation is not its slot's, `OtherGeneration`).
Cuts are row offsets into the round's rows, strictly ascending, each in 1..rows-1. No streams: the keep-alive, sent
whenever rank 0 has sent no plan for 2 s, mid-round too; it never changes a stream's state.

Rows (frame kind 3): `planes` x `rows` x `hidden` values of the row dtype, plane-major, rows in plan order; then, when
the draft width is not 0, `rows` x `draft_width` fp32 tap partial sums. The frame's rows field is the plan's total.

## Files

| file | bytes | what | result |
|---|---|---|---|
| `describe_rank0.bin` | 100 | rank 0, CUDA, layers 0..4, tap 1, 256 KV rows, 4 states | taken |
| `describe_rank1.bin` | 100 | rank 1, Metal, layers 4..8, tap 5, 128 KV rows, 4 states | taken |
| `describe_bad_magic.bin` | 100 | rank 1's, byte 0 = 0x00 | `NotStagePayload` |
| `describe_bad_payload_version.bin` | 100 | rank 1's, byte 4 = 2 | `BadPayloadVersion` |
| `describe_bad_length.bin` | 99 | rank 1's without its last byte | `BadLength` |
| `describe_bad_reserved.bin` | 100 | rank 1's, byte 22 = 1 | `BadReserved` |
| `describe_bad_backend.bin` | 100 | rank 1's, backend byte 16 = 9 | `BadBackend` |
| `describe_bad_row_dtype.bin` | 100 | rank 1's, row dtype byte 18 = 5 (nvfp4) | `BadRowDtype` |
| `describe_bad_version_text.bin` | 100 | rank 1's, version byte 3 = 0x07 | `BadVersionText` |
| `describe_bad_taps.bin` | 100 | rank 1's, tap 2 (outside its range 4..8) | `BadTaps` |
| `describe_other_rank.bin` | 100 | rank 1's, world 3 | `OtherRank` |
| `describe_other_version.bin` | 100 | rank 1's, version ends in `X` | `OtherVersion` |
| `describe_other_backend.bin` | 100 | rank 1's, CUDA where Metal is planned | `OtherBackend` |
| `describe_other_kv_dtype.bin` | 100 | rank 1's, KV f8e4m3 | `OtherKvDtype` |
| `describe_other_range.bin` | 100 | rank 1's, layers 3..8 | `OtherRange` |
| `describe_other_width.bin` | 100 | rank 1's, hidden 8 | `OtherWidth` |
| `describe_other_dtype.bin` | 100 | rank 1's, rows f16 | `OtherDtype` |
| `describe_other_planes.bin` | 100 | rank 1's, 1 plane | `OtherPlanes` |
| `describe_other_draft_width.bin` | 100 | rank 1's, draft width 0 | `OtherDraftWidth` |
| `describe_other_shapes.bin` | 100 | rank 1's, context 128 | `OtherShapes` |
| `describe_other_taps.bin` | 100 | rank 1's, tap 6 (the model's is 5) | `OtherTaps` |
| `describe_short_capacity.bin` | 100 | rank 1's, 32 KV rows (context is 64) | `ShortCapacity` |
| `ready_rank1.bin` | 16 | rank 1, loaded and warm | taken |
| `ready_cold.bin` | 16 | rank 1, loaded, not warm (well-formed; rank 0 keeps waiting) | `NotReady` (judge) |
| `ready_bad_flags.bin` | 16 | flags 7 | `BadFlags` |
| `ready_wrong_message.bin` | 16 | message byte 6 = 3 (plan) | `WrongMessage` |
| `plan_keepalive.bin` | 24 | no streams, no cuts | taken |
| `plan_round.bin` | 140 | the round below | taken |
| `plan_bad_length.bin` | 136 | the round without its last 4 bytes | `BadLength` |
| `plan_bad_reserved.bin` | 24 | keep-alive, byte 20 = 1 | `BadReserved` |
| `plan_too_many_streams.bin` | 164 | 5 streams (parallel is 4) | `TooManyStreams` |
| `plan_unordered.bin` | 140 | the round, stream 1's id 2 (after 3) | `UnorderedStreams` |
| `plan_bad_op.bin` | 140 | the round, stream 0's op byte = 6 | `BadOp` |
| `plan_bad_open.bin` | 140 | the round, the open at position 5 | `BadOpen` |
| `plan_open_generation.bin` | 140 | the round, the open with generation 1 | `BadOpen` |
| `plan_bad_resume.bin` | 140 | the round, resume point 40 past first 32 | `BadResume` |
| `plan_rows_on_close.bin` | 140 | the round, the close with 1 row | `RowsOnClose` |
| `plan_past_context.bin` | 140 | the round, stream 1 at 64 + 1 row (context 64) | `PastContext` |
| `plan_too_many_rows.bin` | 140 | the round with 9 rows (max rows 8) | `TooManyRows` |
| `plan_bad_total.bin` | 140 | the round, total byte 16 = 6 (streams sum to 7) | `BadTotal` |
| `plan_bad_cut.bin` | 140 | the round, cut at 7 (= its rows) | `BadCut` |
| `rows_round.bin` | 200 | frame header (rows 7, 168 bytes) + payload, byte i = (37 i + 11) mod 256 | taken |
| `rows_other_count.bin` | 200 | header rows 6 | `OtherRowCount` |
| `rows_other_bytes.bin` | 199 | header and payload 167 bytes | `OtherRowBytes` |
