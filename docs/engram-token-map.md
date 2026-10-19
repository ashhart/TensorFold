# Engram token map

`zig/src/families/deepseek_v41/token_map.zig` builds the map from the original
release's tokenizer and inference config, without a Python runtime dependency.
The algorithm comes from DeepSeek AI's MIT-licensed `inference/engram.py`,
revision `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`. The source URL is retained
in the builder and its output manifest.

The builder decodes token IDs in order. When decoding emits U+FFFD, it uses the
raw token. Otherwise it normalizes the decoded text. Keys are deduplicated in
first-seen order. The compressed vocabulary size must match the config.
Each layer has 24 prime buckets, unique across layers, cumulative table offsets
and multipliers from PCG64 seeded with `10007 * layer`.

## Normalization qualification

The release uses the Rust chain NFKC, NFD, StripAccents, Lowercase,
`Replace([ \t\r\n]+, " ")`, `Replace(^ $, U+E000)`, Strip and
`Replace(U+E000, " ")`. The local implementation uses decomposition tables,
canonical ordering, Mn/Me removal and lowercase tables before the replacements
and whitespace trimming. An empty result falls back to the original text.

The intended simplification relies on `NFD(NFKC(x)) = NFKD(x)` and idempotent
NFKD. Reproducing the Unicode tables does not establish parity of the complete
Rust normalization chain. Hangul behavior remains a qualification caveat.
Do not describe this builder as generally byte-identical to the release without
comparing the resulting map against that release's tokenizer and implementation.

## Random sequence and format

`SeedSequence(seed).generate_state(4, uint64)` hashes the entropy word into a
four-word pool with NumPy's hashmix/mix scheme. A second hash cycles that pool
to draw eight words, paired little-endian. The existing reference check uses
NumPy 2.5.3 for the two seeds used by this map.

PCG64 initialization starts with zero state, steps once, adds `initstate`, then
steps again. SeedSequence supplies the seed words; the increment is
`(initseq << 1) | 1`, modulo 2^128. Bounded draws use Lemire's algorithm,
matching NumPy's `masked=false` integer path.

The binary is little-endian: magic, raw/compressed vocabulary sizes, pad ID,
order, head and layer counts, tokenizer/config hashes, per-layer blocks, then
the lookup array. A neighboring JSON manifest records the source, provenance
hashes, packed-map hash and layer parameters.
