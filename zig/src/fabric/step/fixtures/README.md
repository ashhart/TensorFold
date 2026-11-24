# Step header fixtures

Golden bytes of leader step messages (T2 of #2560), good and bad. Every user of the header (the ring, a tensor
group's leader and followers, on any backend) decodes each file and must take the good ones and refuse each bad one
with the error named below. `fixtures_test.zig` re-encodes every file and compares bytes, and runs `fixtures.judge`
on each. To rewrite them after a deliberate format change:
`TF_STEP_FIXTURES_OUT=$PWD/zig/src/fabric/step/fixtures zig build test-fabric-step`.

Everything is little-endian. A message is the 64-byte header, then its sections in ascending id, each padded to a
multiple of 8 bytes with zeros.

## The receiver

`judge` decodes with limits of 512 bytes after the header, 4 streams, 8 rows and a 64-token context, then expects
round 3 and group `0x0A0B0C0D`. Every good fixture carries that round and group.

## Layouts

Header, 64 bytes:

| bytes | field | bytes | field |
|---|---|---|---|
| 0..4 | magic `0x48534654` ("TFSH") | 24..28 | streams: the row map's entries |
| 4..6 | version 1 | 28..32 | rows: the row map's sum |
| 6 | kind (1 round, 2 state, 3 idle, 4 stop, 5 family) | 32..40 | bytes after the header |
| 7 | mode (0 window, 1 prompt; 0 unless kind is round) | 40..44 | family step kind |
| 8..16 | round | 44..48 | family flags |
| 16..20 | group | 48..64 | reserved 0 |
| 20..24 | sections: bit i set when section i follows | | |

Section prefix, 8 bytes: id u8, version u8 (1), reserved u16 0, count u32. Entries:

| id | section | entry |
|---|---|---|
| 1 | streams | 28 bytes: id u32, op u8 (1 open, 2 continue, 3 close, 4 cancel, 5 resume), 3 reserved, first, rows, accepted, resume point, generation (u32 each) |
| 2 | tokens | u32: the round's rows in entry order, then a tail that is the family's |
| 3 | cuts | u32 row offsets, strictly ascending, each in 1..rows-1 |
| 4 | sampling | 64 bytes: stream u32, top_k u32, seed u64, temperature f64, top_p f64, min_p f64, row0 u32, rows u32, pos0 u64, has u32 (bit 0 seed, bit 1 pos0), reserved u32; an absent field is 0 |
| 5 | pages | stream, first, n, kept (u32 each), then n u32 page ids; count is the number of patches |
| 6 | copies | 8 bytes: from u32, to u32 |
| 7 | snapshots | 24 bytes: op u8 (1 drop, 2 restore, 3 save), 3 reserved, stream u32 (0 for drop), id u64, at u32, reserved u32 |
| 8 | rows | 8 bytes: id u32, rows u32 (a compact row map; never with section 1) |
| 15 | family | raw bytes; count is their number |

Rules the shared decoder does not make, because they are each user's: ascending stream ids (the ring's), a `resume`
paired with a `restore`, a sampler's rows inside the header's rows, and anything about page or snapshot ids.

## Files

The round in `round.bin` is the ring fixtures' round: stream 3 open at 0 with 4 rows; stream 7 continue at 10 with
1 row, 2 accepted, generation 1; stream 9 resume from 32 at 32 with 2 rows, generation 2; stream 12 close at 20,
generation 1; one cut at row 4; 7 rows.

| file | bytes | what | result |
|---|---|---|---|
| `round.bin` | 200 | the ring's round: sections 1 and 3 | taken |
| `round_tokens.bin` | 248 | `round.bin` with 7 token ids and a 2-id tail | taken |
| `keepalive.bin` | 64 | a `round`, the header alone | taken |
| `begin.bin` | 216 | `state`, family 1: one open, a sampler with a seed, a patch of 3 pages with 1 kept | taken |
| `drops.bin` | 144 | `state`: three snapshot drops, one id past u32 | taken |
| `restore_chunk.bin` | 168 | `round`, mode prompt: a resume at 16 with 4 rows, 5 tokens, a restore | taken |
| `compact.bin` | 120 | `round`: section 8 with three streams of 1 row, 3 tokens | taken |
| `copies.bin` | 112 | `state`: a copy, then a one-page patch mid-table | taken |
| `family.bin` | 80 | `family` kind 7 with 5 bytes of section 15 | taken |
| `stop.bin` | 64 | `stop`, the header alone | taken |
| `sample.bin` | 200 | `family` kind 20, no row map: two samplers, one with rows 3..5 and no seed, one with a seed and a draw position | taken |
| `bad_magic.bin` | 200 | `round.bin`, byte 0 = 0 | `NotStepHeader` |
| `bad_version.bin` | 200 | `round.bin`, byte 4 = 2 | `BadVersion` |
| `bad_kind.bin` | 200 | `round.bin`, byte 6 = 9 | `BadKind` |
| `bad_mode.bin` | 216 | `begin.bin`, mode 1 on a `state` | `BadMode` |
| `bad_keepalive_mode.bin` | 64 | `keepalive.bin`, mode 1 | `BadMode` |
| `bad_reserved_header.bin` | 200 | `round.bin`, byte 50 = 1 | `BadReserved` |
| `too_large.bin` | 592 | a `family` with 520 bytes of section 15 | `TooLarge` |
| `bad_length_bytes.bin` | 200 | `round.bin`, header bytes 144 | `BadLength` |
| `bad_section_version.bin` | 200 | `round.bin`, section 1 version 2 | `BadSectionVersion` |
| `bad_reserved_prefix.bin` | 200 | `round.bin`, byte 66 = 1 | `BadReserved` |
| `unknown_section.bin` | 200 | `round.bin`, section id 9 | `UnknownSection` |
| `unordered_sections.bin` | 248 | `round_tokens.bin`, the cuts' id 2 | `UnorderedSections` |
| `bad_padding.bin` | 248 | `round_tokens.bin`, a padding byte 1 | `BadReserved` |
| `bad_sections.bin` | 200 | `round.bin`, mask bit 2 set with no section 2 | `BadSections` |
| `both_row_maps.bin` | 200 | sections 1 and 8 | `BothRowMaps` |
| `bad_streams.bin` | 200 | `round.bin`, header streams 3 | `BadStreams` |
| `bad_op.bin` | 200 | `round.bin`, stream 3's op 9 | `BadOp` |
| `bad_open.bin` | 200 | stream 3 opens at 1 | `BadOpen` |
| `bad_resume.bin` | 200 | stream 9 resumes from 33, past its first row 32 | `BadResume` |
| `rows_on_close.bin` | 200 | stream 12 closes with a row | `RowsOnClose` |
| `past_context.bin` | 200 | stream 9 at 63 with 2 rows | `PastContext` |
| `too_many_rows.bin` | 200 | stream 3 with 6 rows: 9 in all | `TooManyRows` |
| `bad_total.bin` | 200 | `round.bin`, header rows 6 | `BadTotal` |
| `rows_off_round.bin` | 104 | a `state` that opens a stream with 2 rows | `RowsOffRound` |
| `sections_off_round.bin` | 80 | a `stop` with section 15 | `SectionsOffRound` |
| `bad_tokens.bin` | 232 | `round.bin` with 6 tokens for 7 rows | `BadTokens` |
| `bad_cut.bin` | 200 | a cut at row 7 of 7 | `BadCut` |
| `bad_sampling.bin` | 216 | `begin.bin`, a seed with its `has` bit clear | `BadSampling` |
| `bad_pages.bin` | 216 | `begin.bin`, 4 kept of 3 pages | `BadPages` |
| `bad_snapshot_op.bin` | 144 | `drops.bin`, the first op 9 | `BadSnapshotOp` |
| `bad_snapshot.bin` | 96 | a drop that names stream 5 | `BadSnapshot` |
| `other_round.bin` | 64 | a keep-alive of round 4 | `OtherRound` |
| `other_group.bin` | 64 | a keep-alive of group 1 | `OtherGroup` |
