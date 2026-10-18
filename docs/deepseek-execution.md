# DeepSeek execution contracts

The served EXL3 product assigns two columns and eight rows to a SIMD group.
Lanes 0 through 15 and 16 through 31 own the two columns. A lane decodes its tile state and
shuffles it to the lanes that sum that column. A one-row product uses eight
times as many groups as `tf_exl3_matmul_impl`; each row/column retains its
products and TwoSum steps in K order. The kept-dense product instead assigns
one output element per thread and accumulates BF16 weights in FP32, in
sequential K order, into the region consumed by output rotation.

## Concurrent groups and dispatch thresholds

Attention can group `wq_a`, `wkv` and cache-owner compressor `wkv`/`wgate`.
They read only the input and write separate outputs. A kept-dense compressor
repack declines this group without encoding work, allowing the caller to run
the individual products. Diagnostics also retain the individual path.

The concurrent forward uses barriers after other dispatches to preserve serial
ordering. `TF_DSV41_SERIAL=1` or an installed profiler selects the serial
encoder. Grouped projection jobs receive separate consecutive scratch slices;
a group is flushed before the next job would exceed capacity.

For up to four rows, attention indexes all eight slices in one product: row
`8*t+g` represents token `t`, group `g`, at the individual products' offsets.
Indexed rows decode tiles independently. With more rows, the per-slice path
shares each decoded tile across eight rows.

The following M5 Max timings are inherited tuning observations, not M3 Ultra
qualification or measurements from the TP2 run:

| Attention rows | Indexed slices | Separate slices |
| --- | --- | --- |
| 4 | 1.0 ms | 1.5 ms |
| 8 | 1.9 ms | 1.6 ms |
| 256 | 58 ms | 17 ms |

The shared expert's merged row also decodes alone, while its separate product
shares tiles across rows. Inherited per-layer M5 Max observations were
2.81 → 2.23 ms for one row, 4.20 → 4.03 ms for two and 7.23 → 7.29 ms for
four. The implementation merges only one or two rows. `work.moeSlot` preserves
routed row positions and places shared rows below their block, so SwiGLU and
combine use the same values as the separate shared path.

## Streamed Engram and lanes

Engram tables remain in their safetensors shards. Each chunk deduplicates row
IDs, reads misses through uncached preads on worker threads and compacts the
rows into staging. A bounded row cache can supply hits. The unchanged gather
kernel consumes the same row bytes as a resident table.

`Fetcher.fetch` receives all misses for one stage call, sorted ascending, and
writes each row's 256 weight bytes and 8 scale bytes at the corresponding
output slot. `Pread` is the current implementation. The interface can describe
another source, but does not itself implement remote table storage. Staging
rejects out-of-range IDs before fetching and returns the distinct row count
plus each original ID's local staging index.

Each stream owns a lane slot. A window executes as that slot's own sub-forward;
retaining only a prefix rolls the slot back and re-feeds those rows. Generation
uses greedy draws with drafts disabled. The native load limit defaults to 70%
of RAM unless `GLM_LOAD_LIMIT_GB` explicitly overrides it. `--engram-stream`
leaves the two large tables on storage and binds the backbone in memory.

## Template fixtures

The G8 checker renders a supplied template with the engine's Jinja against the
release's `encoding/tests/test_input_N.json` and `test_output_N.txt` fixtures.
It follows their `test_examples_encoding_golden_outputs` convention, defaulting
`thinking_mode` to `chat` unless a case supplies one. This checks template
bytes, not model inference or tokenizer-normalization parity.
