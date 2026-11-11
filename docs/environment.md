# Native environment validation

Malformed numeric settings are refused at startup with the variable, supplied value and accepted forms.
The same parser is used by direct CUDA entry points. An unset variable keeps its existing default.
A valid value keeps its existing meaning, including documented caps and clamping.
Diagnostics quote and escape the supplied text so it cannot create another terminal line.

| Variable | Accepted explicit value |
| --- | --- |
| `TENSORFOLD_VIDEO_TOKENS` | Nonnegative whole video-token budget; default 16384; stored as an unsigned 64-bit count |
| `TENSORFOLD_MAX_PDF_PAGES` | Whole count 1 through 4294967295, capped by `--vision-max-images`; default min(16, image limit) |
| `TENSORFOLD_PRESSURE_CACHE_PERCENT` | Whole percentage 0 through 100; default 50 |
| `TENSORFOLD_PREFIX_WAIT_MS` | Whole milliseconds 0 through 4294967295; zero disables waiting; default 60000 |
| `TENSORFOLD_RESTORE_PREFETCH` | Whole unsigned 64-bit count, capped at 65536; zero disables reading ahead; default 2 |
| `TENSORFOLD_RESTORE_PREFETCH_GIB` | Whole unsigned 64-bit GiB count, capped at 16777216; default 16 |
| `TENSORFOLD_RESTORE_PREFETCH_MS` | Whole unsigned 64-bit milliseconds, capped at 1099511627776; default 30000 |
| `TENSORFOLD_TOOL_ID_KEY_FILE` | The path of the file that holds the key for tool-call ids ([API](api.md#messages-and-tools)). The file must be a regular file, not a link, that only its owner can read or write (`chmod 600`), with 32 to 4096 bytes in it, and the path must not contain control characters. When this is unset, each server process makes its own random key. If the file is wrong, the server does not start; the message names this setting but never shows the path or what the file holds |
| `TF_STARTUP_RECEIPT` | Exactly `0` or `1`; default off. `1` writes the startup and first completion receipts |
| `TF_CUDA_DEVICE` | Whole ordinal 0 through 4294967295, or empty; default 0 |
| `TF_CUDA_SEGMENTS` | Whole count 1 through 4, or empty; default 1 |
| `TF_CUDA_CARVEOUT` | Exactly `0` or `1`; default off |
| `TF_CUDA_CARVEOUT_MIB` | Positive whole MiB count whose byte size fits the platform, or empty for 1792 MiB |
| `TF_Q27_COPY_ROWS` | Whole u32 count; zero disables copied rows; default 48 |
| `TF_Q27_DRAFT_SLOTS` | Whole u32 count, clamped to 1 through the lane stream limit; default 8. DFlash2 can draft for at most the smaller of this count and the admitted streams, and none without a drafter. |
| `TF_MTP_DRAFTS` | Whole u32 count, clamped to 1 through the drafter's window less one; existing depth when unset |
| `TF_DRAFT_NODES` | Whole native-size nonnegative count, capped by the drafter's window; zero disables trees |
| `TF_QMMF_HALF_BELOW` | Whole native-size nonnegative block count; zero disables half tiles; default 96 |
| `TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS` | Exactly `0` or `1`, checked when the CUDA backend starts; default off ([profiles](profiles.md)) |
| `TF_HEAD_GEOMETRY` | Whole native-size nonnegative geometry count in the head benchmark |
| `TF_ROCE_GID_INDEX` | Whole index 0 through 2147483647; unset searches for a RoCE v2 IPv4 GID |
| `TF_GEMMA4_CHUNK_SOLO` | Gemma 4 on CUDA: a multiple of 512 from 512 through 4096; unset, empty or 512 is off. Checked when the model loads, with the family's own message |
| `TF_DRAMA_PREFILL_ROWS` | Drama 3 on CUDA: whole prompt rows a chunk, 1 through 1024; default 1. Checked when the checkpoint opens, with its own message |
| `TF_DRAMA_REFERENCE_CACHE` | Drama 3 on CUDA: kept reference states, 0 (off) through 16; default 0. Checked when the checkpoint opens, with its own message |
| `TF_GLM_HOST_CACHE` | Exactly `0` or `1`; default off (kept GLM states on the device) |
| `TF_GLM_VISION_HOST` | Exactly `0` or `1`; default off (GLM vision weights on the device) |
| `TF_GLM_PROF` | Exactly `0` or `1`; default off (no GLM section timing) |
| `TF_GLM_LINK_WAIT` | Exactly `0` or `1`; default off (short GLM steps go to the other ranks unwaited) |
| `TF_GLM_PDL` | Exactly `0` or `1`; default on (GLM matmuls use dependent launches) |
| `TF_GLM_PROMPT_STEPS` | Exactly `0` or `1`; default on (GLM prompt passes in chunks); read on rank 0 |
| `TF_GLM_SHARED` | Exactly `0` or `1`; default on (GLM windows share a forward); read on rank 0 |
| `TF_GLM_DFLASH2_LOG` | Whole unsigned 64-bit count of DFlash2 rounds whose drafts are logged; default 0 |
| `TF_GLM_DRAFT_CONFIDENCE` | Finite number 0 through 1 where a DFlash2 chain's chance product stops it; default 0.7 |
| `TF_PROMPT_ROWS` | Native serving: a multiple of 16 from 128 through 2048; unset keeps the chip/family default |
| `TF_ANE_COLUMNS` | Whole u32 count; zero disables the experimental share; default zero |
| `TF_FLASHNEXT_SCALE_INDEX` | Flash Next on Metal before tensor units: exactly `0` or `1`; `1` has the 4-bit expert kernels read each group's scale and bias through a 16-bit index into its tensor's table of pairs, the same values with fewer weight bytes; default 0 |
| `TF_FLASHNEXT_GROUPED_ROWS` | Flash Next on Metal without tensor units: `0`, or the rows (`1` through `32`) from which a decode window runs its 4-bit experts grouped by expert, with the same output bits; default 5; ignored with `--ssd-experts` and in speed-up mode |
| `GLM_EHPROJ_NAX` | GLM-5.3-Flash on Metal: exactly `0` or `1`; `0` serves the MTP head's prompt eh_proj on the scalar kernel; default 1, the tile kernel where it builds |
| `TF_PREFILL_MIN` | With `--prefill-from`: whole prompt-token count a pull must reach past the Mac's own prefix; default 4096 |
| `TF_PREFILL_CHUNK` | With `--prefill-from`: positive multiple of 16; a pulled prefix ends on this boundary; default 512 |
| `TF_PREFILL_KEEP` | With `--prefill-from`: exactly `0` or `1`; keeps states built on a pulled prefix, marked remote, so a chat's next turn resumes on the Mac; default 1 |
| `TF_PREFILL_AHEAD` | With `--prefill-from`: whole count from 1 of queued requests with a fill under way or landed; default 4 |
| `TF_PREFILL_API_KEY_FILE` | With `--prefill-from`: the path of a file only its owner can read, holding the GB10 server's API key, which the trigger sends as a Bearer token; unset sends none, and a GB10 started with `--api-key` then answers 401 |
| `TF_PRODUCE_HELD_GIB` | With `--mcdma-produce`: whole GiB 1 through 1024 of hand-off exports a GB10 holds at once; an export past it is refused by name and the Mac prefills that prompt itself; default 32 |
| `TF_DSV41_HANDOFF_DUMP` | A directory path without control characters, used as given; a DeepSeek-V4.1 producer also writes each hand-off export there as a file, with or without `--mcdma-produce`. Unset or blank writes none |
| `TF_PREFILL_TIMEOUT_S` | With `--prefill-from`: whole seconds 1 through 86400 for the trigger and, with `--mcdma-link http`, separately for the transfer of the export; default 60 plus 1 per 400 tokens sent |
| `XPU_A8_BITS` | Exactly `4` or `8`, for the XPU quantization study; unset keeps the study's existing setting |
| `LFM3_FUSE` | Whole u32 fusion mask; default 7 |
| `TF_SLIDE_POCKET_KEY` | Exactly `auto`, `reply`, `subject` or `both`; default `auto` |
| `TF_SLIDE_FIT_GIB` | Whole native-size nonnegative GiB count; default 4 |
| `TF_SLIDE_POCKET_DUMP` | A directory path without control characters, used as given; each lesson's read rows are written there for offline study. Unset or blank writes none |
| `TF_SLIDE_FORGET_FAIL` | Whole u32 fact index, zero off; only checked in a slide-test-hooks build |

The video-token reader has no fixed model-independent token ceiling. Its unsigned storage width is not a useful token budget.
The request geometry and model context decide whether a parsed budget fits; zero remains accepted as before.

u32 means 0 through 4294967295. Native-size counts use the platform's unsigned integer size.
Existing numeric spellings accepted by each reader, such as leading zeroes, remain valid.
Pressure, prefix-wait and restore-prefetch values retain trimming of spaces and tabs.
Prefix-wait and restore-prefetch still require the first remaining character to be a digit.
CUDA device/segments and the slide test hook retain their space trimming.
Other numeric readers do not gain whitespace aliases. Empty or whitespace-only text means unset for every variable, as it did before.
`TF_GEMMA4_CHUNK_SOLO` is the exception to "as before": its old reader refused blank text, and now reads it as off.
The table checks only that it is a whole number; the lane core's `solo_chunk.parse` judges it against the family's grid.
The two `TF_DRAMA_*` settings are another: their old reader refused text of only tabs or line breaks, and now reads it as unset.

An explicit `--device` or `--segments` wins over its environment value, even if the ignored value is malformed.
`--carveout` likewise enables the carveout regardless of its enable variable.
The carveout size is checked only when the carveout is enabled, and still rounds to framebuffer rows.
`TF_NO_TREES` retains precedence over `TF_DRAFT_NODES`.
Backend-specific settings are checked when that backend starts; slide settings require learning to be enabled.
The `TF_GLM_*` rows are checked when GLM's CUDA family opens, on every rank, so they never stop another family.
Common server settings are checked before checkpoint resolution, including when no request would use them yet.
Direct family readers also refuse malformed values when called without the server.

## Existing lenient switches

These settings retain their existing boolean policies rather than gaining new aliases or refusals:

- `TF_GRAPH_CACHE`: exact `0` or case-insensitive `off` disables graphs; every other value leaves the family's settings unchanged.
- `TF_QWEN27_SLOT_RESUME`: exact `0` disables resume; every other value keeps it enabled.
- Kolibri pages: exact `TENSORFOLD_KOLIBRI_KV_PAGES=0` disables requested pages; other text keeps the option unchanged.
- Dory pages: unset or exact `TENSORFOLD_DORY_KV_PAGES=0` uses windows; every other present value selects pages.
- `TF_SLIDE_EDIT`, `TF_SLIDE_PROBE` and `TF_SLIDE_ROUTED` select modes by presence, even empty or `0`; probe wins over edit.
- `TF_COOP_SPLIT` retains its first-character `0` disabling rule, `TF_NEMOTRON_TRACE` its first-character `1` trace rule,
  and `TF_SLIDE_FIT_REDUCE` its exact `tree` reduction selection. These developer switches keep their existing semantics.
- GLM: `GLM_EHPROJ_NAX` retains its first-character `0` disabling rule, `TF_GLM_DFLASH2_PROF` enables by presence,
  even empty or `0`, and `TF_KV_POISON` (also read by Nemotron) is on only for exact `1`.
- The existing tool-grammar, rows-sync, log-time, memory-pressure and XPU-MTP boolean policies are unchanged.

- The DeepSeek-V4.1 lane's existing experimental and diagnostic switches keep their own readers
  (presence, exact `0`, or lenient numeric parsing), so they are listed here rather than gaining refusals:
  `TF_DSV41_ATTN_COUNT`, `TF_DSV41_ATTN_COUNT_LAYERS`, `TF_DSV41_ATTN_DENSE_VALUES`, `TF_DSV41_ATTN_PROF`, `TF_DSV41_ATTN_PROF_LAYERS`,
  `TF_DSV41_ATTN_SCAN_BARRIER`, `TF_DSV41_ATTN_SPLITS`, `TF_DSV41_BATCH`, `TF_DSV41_BATCH_DEBUG`, `TF_DSV41_BLOCK_FILTER`,
  `TF_DSV41_BLOCK_PROF`, `TF_DSV41_CLEAR_PARTIALS`, `TF_DSV41_CLEAR_STALE`, `TF_DSV41_CONCURRENT`, `TF_DSV41_COND_NOALIGN`,
  `TF_DSV41_COND_NOROPE`, `TF_DSV41_COND_ONE`, `TF_DSV41_COND_PROF`, `TF_DSV41_COND_ROWS`, `TF_DSV41_CPU_HASH`,
  `TF_DSV41_DENSE_COL`, `TF_DSV41_DENSE_COL_SPEC`, `TF_DSV41_DOUBLE_ATTN_COMPRESSOR`, `TF_DSV41_DOUBLE_ATTN_DENSE`, `TF_DSV41_DOUBLE_ATTN_INDEXER`,
  `TF_DSV41_DOUBLE_ATTN_MERGE`, `TF_DSV41_DOUBLE_ATTN_NORM`, `TF_DSV41_DOUBLE_ATTN_PACK`, `TF_DSV41_DOUBLE_ATTN_POOL`, `TF_DSV41_DOUBLE_ATTN_PUBLISH`,
  `TF_DSV41_DOUBLE_ATTN_QKV`, `TF_DSV41_DOUBLE_ATTN_SCAN`, `TF_DSV41_DOUBLE_ATTN_SCANX`, `TF_DSV41_DOUBLE_ATTN_WOA`, `TF_DSV41_DOUBLE_CLEAR`,
  `TF_DSV41_DOUBLE_COMBINE`, `TF_DSV41_DOUBLE_COND_PROWS`, `TF_DSV41_DOUBLE_COND_WKV`, `TF_DSV41_DOUBLE_DENSEKV`, `TF_DSV41_DOUBLE_ENGRAM`,
  `TF_DSV41_DOUBLE_EXPERTS`, `TF_DSV41_DOUBLE_FINISH`, `TF_DSV41_DOUBLE_HC_EXPAND`, `TF_DSV41_DOUBLE_HC_SPLIT`, `TF_DSV41_DOUBLE_HEAD`,
  `TF_DSV41_DOUBLE_INDEX`, `TF_DSV41_DOUBLE_INDEX_W`, `TF_DSV41_DOUBLE_KEYS`, `TF_DSV41_DOUBLE_MERGE`, `TF_DSV41_DOUBLE_MLP`,
  `TF_DSV41_DOUBLE_ROPE`, `TF_DSV41_DOUBLE_SCORE`, `TF_DSV41_DOUBLE_SPARSE`, `TF_DSV41_DRAFT_COPY`, `TF_DSV41_DRAFT_OFF`,
  `TF_DSV41_DRAFT_OFF_AFTER`, `TF_DSV41_DRAFT_OFF_BELOW`, `TF_DSV41_DRAFT_OFF_GATE_CTX`, `TF_DSV41_DRAFT_OFF_SAMPLE_LOW`, `TF_DSV41_DRAFT_PUBLISH`,
  `TF_DSV41_DSPARK_CAPTURE_ALL`, `TF_DSV41_DSPARK_CHUNK_ALL`, `TF_DSV41_DSPARK_COND_ONE`, `TF_DSV41_DSPARK_NO_CHUNK_MASK`, `TF_DSV41_DSPARK_NO_COND`,
  `TF_DSV41_DSPARK_NO_RING_ROPE`, `TF_DSV41_DSPARK_NO_ROPE`, `TF_DSV41_DSPARK_POS1`, `TF_DSV41_DSPARK_SCALE1`, `TF_DSV41_DSPARK_TRUE_POS`,
  `TF_DSV41_DUMP_SELECT`, `TF_DSV41_DUMP_SELECT_REF`, `TF_DSV41_END_MARK`, `TF_DSV41_ENGRAM_ADAPT`, `TF_DSV41_ENGRAM_CACHE`,
  `TF_DSV41_ENGRAM_LARGE`, `TF_DSV41_ENGRAM_LATCH`, `TF_DSV41_ENGRAM_MMAP`, `TF_DSV41_ENGRAM_NOCACHE`, `TF_DSV41_ENGRAM_PEND`,
  `TF_DSV41_ENGRAM_PF`, `TF_DSV41_ENGRAM_PIPE`, `TF_DSV41_ENGRAM_POS_THREADS`, `TF_DSV41_ENGRAM_SMALL_THREADS`, `TF_DSV41_ENGRAM_STAGE_HASH`,
  `TF_DSV41_ENGRAM_STREAM_DIR`, `TF_DSV41_ENGRAM_THREADS`, `TF_DSV41_ENGRAM_TRACE`, `TF_DSV41_EXL3_FMA`, `TF_DSV41_EXL3_FMA_DENSE_MULTI`,
  `TF_DSV41_FAST_ARGMAX`, `TF_DSV41_FMA_WARPS`, `TF_DSV41_FUSE_UP_SIX`, `TF_DSV41_GATE_DUMP`, `TF_DSV41_GATE_DUMP_ROWS`,
  `TF_DSV41_GATE_TREE`, `TF_DSV41_GATE_WIDE`, `TF_DSV41_GROUP_MIN_ROWS`, `TF_DSV41_HANDOFF_BISECT`, `TF_DSV41_HANDOFF_CHECK`, `TF_DSV41_HANDOFF_DUMP`, `TF_DSV41_HANDOFF_FILE`, `TF_DSV41_HANDOFF_STEPS`, `TF_DSV41_HANDOFF_TOKENS`, `TF_DSV41_HANDOFF_UPTO`, `TF_DSV41_HC_PROBE`, `TF_DSV41_HC_PROBE_INDEP`,
  `TF_DSV41_HC_PROBE_THREADS`, `TF_DSV41_HID_FULL`, `TF_DSV41_HID_HASH`, `TF_DSV41_HID_ROWS`, `TF_DSV41_HOST_PROF`,
  `TF_DSV41_I8_DIR`, `TF_DSV41_I8_ONLY`, `TF_DSV41_I8_PREFILL`, `TF_DSV41_I8_ROT`, `TF_DSV41_INDEX_KEYS`,
  `TF_DSV41_INDEX_KEYS_SCORE`, `TF_DSV41_INDEX_PARALLEL_FINISH`, `TF_DSV41_INDEX_PARALLEL_MERGE`, `TF_DSV41_INDEX_ROW_SCORE`, `TF_DSV41_INDEX_SCORE_ONE`,
  `TF_DSV41_INDEX_SELECT_ONE`, `TF_DSV41_INDEX_SELECT_ONE_ROWS`, `TF_DSV41_INDEX_SELECT_V`, `TF_DSV41_INDEX_SERIAL_MERGE`, `TF_DSV41_INDEX_WARPS`,
  `TF_DSV41_KROW_CHECK`, `TF_DSV41_LAYER_CB`, `TF_DSV41_LOGIT_DUMP`, `TF_DSV41_LOOKUP_DRAFTS`, `TF_DSV41_MTP_ADAPT`,
  `TF_DSV41_MTP_ADAPT_CONFIRM`, `TF_DSV41_MTP_ADAPT_DEBUG`, `TF_DSV41_MTP_ADAPT_DOWN`, `TF_DSV41_MTP_ADAPT_RATCHET`, `TF_DSV41_MTP_ADAPT_SIGNAL`,
  `TF_DSV41_MTP_ADAPT_UP`, `TF_DSV41_MTP_ADAPT_WARM`, `TF_DSV41_MTP_DEBUG_LOGITS`, `TF_DSV41_MTP_DEBUG_ROWS`, `TF_DSV41_MTP_DEBUG_TOP`,
  `TF_DSV41_MTP_DEPTH_MAX`, `TF_DSV41_MTP_DRAFTS`, `TF_DSV41_MTP_DRAFT_ROWS`, `TF_DSV41_MTP_PIPE`, `TF_DSV41_MTP_ROPE_ALL`,
  `TF_DSV41_MTP_ROWS`, `TF_DSV41_MTP_STOP`, `TF_DSV41_MTP_STUB`, `TF_DSV41_MTP_STUB_BLOCKS`, `TF_DSV41_MTP_STUB_COND`,
  `TF_DSV41_MTP_STUB_HEAD`, `TF_DSV41_MTP_TRUNK_RING`, `TF_DSV41_NAX_MIN_ROWS`, `TF_DSV41_NO_COND_WKV_FUSE`, `TF_DSV41_NO_ENGRAM_FLUSH`,
  `TF_DSV41_NO_FUSE_DRAFT_WO_A`, `TF_DSV41_NO_FUSE_KVGLUE`, `TF_DSV41_NO_FUSE_MOE_UP`, `TF_DSV41_NO_FUSE_QKV`, `TF_DSV41_NO_FUSE_SHARED`,
  `TF_DSV41_NO_FUSE_SHARED_ROT`, `TF_DSV41_NO_FUSE_WO_A`, `TF_DSV41_NO_GATE_UP_FUSION`, `TF_DSV41_NO_HC_BOUNDARY`, `TF_DSV41_NO_INDEX_NAX`,
  `TF_DSV41_NO_MARKOV`, `TF_DSV41_NO_REFEED`, `TF_DSV41_NO_SWIGLU_ROT`, `TF_DSV41_NO_TAIL_SNAP`, `TF_DSV41_PARTIAL_TXN`,
  `TF_DSV41_PEN_DEBUG`, `TF_DSV41_PIPE_LAYERS`, `TF_DSV41_PREFILL_STEP`, `TF_DSV41_PROFILE_GPU`, `TF_DSV41_RESUMED_TAIL`,
  `TF_DSV41_RING_KEEP`, `TF_DSV41_ROUTE_DEBUG`, `TF_DSV41_ROUTE_DEBUG2`, `TF_DSV41_ROUTE_DEBUG3`, `TF_DSV41_ROUTE_DEBUG4`,
  `TF_DSV41_ROUTE_DEBUG5`, `TF_DSV41_ROUTE_DUMP`, `TF_DSV41_ROUTE_GEMM`, `TF_DSV41_ROUTE_GEMM_MINROWS`, `TF_DSV41_ROUTE_GEMM_PRIVATE`,
  `TF_DSV41_ROUTE_GEMM_SCRATCH`, `TF_DSV41_ROUTE_HIST`, `TF_DSV41_ROUTE_IDS`, `TF_DSV41_ROUTE_TENSOR`, `TF_DSV41_SCAN_PROBE`,
  `TF_DSV41_SCAN_TWICE`, `TF_DSV41_SELECT_TWICE`, `TF_DSV41_SHARED_DENSE`, `TF_DSV41_SKIP_ATTN`, `TF_DSV41_SKIP_COMBINE`,
  `TF_DSV41_SKIP_DENSEKV`, `TF_DSV41_SKIP_ENGRAM`, `TF_DSV41_SKIP_EXPERTS`, `TF_DSV41_SKIP_HC_EXPAND`, `TF_DSV41_SKIP_HC_SPLIT`,
  `TF_DSV41_SKIP_INDEX`, `TF_DSV41_SKIP_MERGE`, `TF_DSV41_SKIP_MHC`, `TF_DSV41_SKIP_MOE`, `TF_DSV41_SKIP_SELECT`,
  `TF_DSV41_SKIP_SHARED`, `TF_DSV41_SKIP_SPARSE`, `TF_DSV41_SKIP_SWIGLU`, `TF_DSV41_SNAP_COLD`, `TF_DSV41_SNAP_HASH`,
  `TF_DSV41_SPARSE_MERGE_SPLIT`, `TF_DSV41_STAGE_CB`, `TF_DSV41_TAPS`, `TF_DSV41_TAPS_SHARED_BASE`, `TF_DSV41_TREE_PRICE`

The profile registry reports selected arithmetic/experimental overrides; it does not change their parsing or defaults.
See [profiles](profiles.md) for classes, scopes and grouped slide-learning names.
`capabilities --json` no longer advertises `TENSORFOLD_NO_UPDATE_CHECK` or `HF_HUB_OFFLINE`: the native binary reads neither.
The command-line `--no-update-check` remains an accepted no-op.
