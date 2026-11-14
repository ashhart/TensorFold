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
| `TENSORFOLD_CUDA_KERNELS` | A path without control characters, used as given: a captured kernel set's folder, or `native` for the engine's own kernels. Unset uses the binary's bundled set for the GPU when it has one, else the engine's own kernels. A family that runs only on the engine's own kernels (Qwen3 among them) refuses any other value; Qwen3 reads a blank value as unset |
| `TF_CUDA_CARVEOUT` | Exactly `0` or `1`; default off |
| `TF_CUDA_CARVEOUT_MIB` | Positive whole MiB count whose byte size fits the platform, or empty for 1792 MiB |
| `TF_Q27_COPY_ROWS` | Whole u32 count; zero disables copied rows; default 48 |
| `TF_Q27_DRAFT_SLOTS` | Whole u32 count, clamped to 1 through the lane stream limit; default 8. DFlash2 can draft for at most the smaller of this count and the admitted streams, and none without a drafter. |
| `TF_MTP_DRAFTS` | Whole u32 count, clamped to 1 through the drafter's window less one; existing depth when unset |
| `TF_DSPINE_WINDOW` | Qwen3.8-27B on CUDA with a DSpine `--drafter`: whole row count 16 through 8192 that narrows the drafter's sliding window; it must be at least the block (16) and at most the checkpoint's own `sliding_window`, else the drafter is refused when it opens; unset or blank keeps the checkpoint's (8192 for tencent/Qwen3.8-27B-DSpine-b16) |
| `TF_DRAFT_NODES` | Whole native-size nonnegative count, capped by the drafter's window; zero disables trees |
| `TF_QMMF_HALF_BELOW` | Whole native-size nonnegative block count; zero disables half tiles; default 96 |
| `TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS` | Exactly `0` or `1`, checked when the CUDA backend starts; default off ([profiles](profiles.md)) |
| `TF_PROMPT_INTERMEDIATE` | Exactly `0` or `1`, checked when the CUDA backend starts; default off. `1` keeps Flash Next's ladder of intermediate prompt-cache checkpoints under a long prompt's history |
| `TF_HEAD_GEOMETRY` | Whole native-size nonnegative geometry count in the head benchmark |
| `TF_ROCE_GID_INDEX` | Whole index 0 through 2147483647; unset searches for a RoCE v2 IPv4 GID |
| `TF_GEMMA4_CHUNK_SOLO` | Gemma 4 on CUDA: a multiple of 512 from 512 through 4096; unset, empty or 512 is off. Checked when the model loads, with the family's own message |
| `TF_DRAMA_PREFILL_ROWS` | Drama 3 on CUDA: whole prompt rows a chunk, 1 through 1024; default 1. Checked when the checkpoint opens, with its own message |
| `TF_DRAMA_REFERENCE_CACHE` | Drama 3 on CUDA: kept reference states, 0 (off) through 16; default 0. Checked when the checkpoint opens, with its own message |
| `TF_GLM_HOST_CACHE` | Exactly `0` or `1`; default off (kept GLM states on the device) |
| `TF_GLM_VISION_HOST` | Exactly `0` or `1`; default off (GLM vision weights on the device) |
| `TF_GLM_PROF` | Exactly `0` or `1`; default off (no GLM section timing). On, a prompt step is one chunk, with no pair or solo steps, so its prompt timings are of that shape |
| `TF_GLM_LINK_WAIT` | Exactly `0` or `1`; default off (short GLM steps go to the other ranks unwaited) |
| `TF_GLM_PDL` | Exactly `0` or `1`; default on (GLM matmuls use dependent launches) |
| `TF_GLM_PROMPT_STEPS` | Exactly `0` or `1`; default on (GLM prompt passes in chunks); read on rank 0 |
| `TF_GLM_SHARED` | Exactly `0` or `1`; default on (GLM windows share a forward); read on rank 0 |
| `TF_GLM_DSEL` | Exactly `0` or `1`; default on (a GLM decode window's DSA select runs on many CTAs; 0 keeps one CTA a row) |
| `TF_GLM_DFLASH2_LOG` | Whole unsigned 64-bit count of DFlash2 rounds whose drafts are logged; default 0 |
| `TF_GLM_DRAFT_CONFIDENCE` | Finite number 0 through 1 where a DFlash2 chain's chance product stops it; default 0.7 |
| `TF_GLM_CONTEXT_CEILING` | GLM-5.3-Flash on CUDA: whole token count 1 through 2147483616, the window `--context` may ask for and `--context 0` serves, in place of config.json's 1048576; read at GLM's open. Past that length the model's quality is the operator's to check. Unset keeps config.json's window |
| `TF_PROMPT_ROWS` | Native serving: a multiple of 16 from 128 through 2048; unset keeps the chip/family default |
| `TF_ANE_COLUMNS` | Whole u32 count; zero disables the experimental share; default zero |
| `TF_METAL4` | Metal: exactly `0` or `1`; `1` sends a family's GPU work through Metal 4 (macOS 26 and later) where the family supports it: the same kernels in the same order, with Metal 4's command queue, allocators, argument tables and barriers, and bit-identical outputs; `0` keeps Metal 3. DeepSeek-V4.1 defaults to `1` where the Mac has Metal 4 (its device test rigs to `0`); Macs without it stay on Metal 3 |
| `TF_METAL4_TIMELINE` | Metal 4: a whole round count, default 0; `n` puts a GPU timestamp at every stage (between barriers) of `n` rounds of each kind a family names, after 8 warm ones, and logs their time by kernel and the longest stages. DeepSeek-V4.1 names its decode rounds by rows. Stamps cost GPU time, so stamped rounds read a little long |
| `TF_DSV41_ROUND_PROF` | DeepSeek-V4.1 on Metal: exactly `0` or `1`; `1` logs decode rounds' host milliseconds (first encode to last wait) and GPU milliseconds by rows, every 1,024 rounds; default 0 |
| `TF_FLASHNEXT_SCALE_INDEX` | Flash Next on Metal before tensor units: exactly `0` or `1`; `1` has the 4-bit expert kernels read each group's scale and bias through a 16-bit index into its tensor's table of pairs, the same values with fewer weight bytes; default 0 |
| `TF_FLASHNEXT_GROUPED_ROWS` | Flash Next on Metal without tensor units: `0`, or the rows (`1` through `32`) from which a decode window runs its 4-bit experts grouped by expert, with the same output bits; default 5; ignored with `--ssd-experts` and in speed-up mode |
| `GLM_EHPROJ_NAX` | GLM-5.3-Flash on Metal: exactly `0` or `1`; `0` serves the MTP head's prompt eh_proj on the scalar kernel; default 1, the tile kernel where it builds |
| `TF_DSV41_SMALL` | DeepSeek-V4.1 on Metal: exactly `0` or `1`; `1` reads the checkpoint's config at its own small shapes (the served smoke test's tiny model); default 0 |
| `TF_DSV41_ASSETS` | DeepSeek-V4.1 on Metal: a directory path without control characters that holds the Engram token map; unset or blank reads it beside the checkpoint |
| `TF_DSV41_ENGRAM_DIR` | DeepSeek-V4.1 on Metal: a directory path without control characters that holds the packed Engram tables; unset or blank takes the checkpoint's `engram_table_dir` |
| `TF_DSV41_ENGRAM_VERIFY` | DeepSeek-V4.1 on Metal: exactly `sample` or `full`; `full` also checks every byte of the Engram tables against their manifest at boot; default `sample` |
| `TF_DSV41_ENGRAM_CACHE` | DeepSeek-V4.1 on Metal: whole u32 count of Engram records a layer kept cached; default 65536 |
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
`TENSORFOLD_CUDA_KERNELS` is an exception to the blank rule: the main CUDA open takes a blank value as a folder name and refuses it (it holds no `aot.json`), and the Qwen4 and Drama 3 opens refuse a value of only whitespace; Qwen3 reads a blank value as unset.
The two `TF_DRAMA_*` settings are another: their old reader refused text of only tabs or line breaks, and now reads it as unset.

An explicit `--device` or `--segments` wins over its environment value, even if the ignored value is malformed.
`--carveout` likewise enables the carveout regardless of its enable variable.
The carveout size is checked only when the carveout is enabled, and still rounds to framebuffer rows.
`TF_NO_TREES` retains precedence over `TF_DRAFT_NODES`.
Backend-specific settings are checked when that backend starts; slide settings require learning to be enabled.
The `TF_GLM_*` rows are checked when GLM's CUDA family opens, on every rank, so they never stop another family.
Common server settings are checked before checkpoint resolution, including when no request would use them yet.
Direct family readers also refuse malformed values when called without the server.

## Flash Next

Flash Next checks these 29 settings when its engine loads, before any device or checkpoint work: a malformed value
stops the start with the variable, the supplied text and the accepted forms.

| Variable | Accepted explicit value |
| --- | --- |
| `TF_FLASHNEXT_PROMPT_BF16` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_WINDOW_GRAPH` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_LANE_PARTS` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_KEEP_INPASS` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_FIRST_EARLY` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_GPU_TOPK` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_GDN_FUSED` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_GDN_LANES` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_BATCH_HEADS` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_SAMPLED_HEADS` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_SHARED_DRAFTS` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_SHARED_SIDE` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_HC_FUSE` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_KEEP_LATE` | Exactly `0` or `1`; unset or blank leaves the choice to the loader |
| `TF_FLASHNEXT_DECODE_GRAPH` | Exactly `0` or `1`; default off |
| `TF_FLASHNEXT_ATTN_LANES` | Exactly `0` or `1`; default off |
| `TF_FLASHNEXT_WARM_AHEAD` | Exactly `0` or `1`; default off |
| `TF_FLASHNEXT_TABLE_WARM` | Exactly `0` or `1`; default off |
| `TF_FLASHNEXT_PROFILE` | Exactly `0` or `1`; default off |
| `TF_FLASHNEXT_VERIFY_HOST_LOGITS` | Exactly `0` or `1`; default off |
| `TF_FLASHNEXT_DRAFTS` | Whole draft count; default 6, then clamped to 1 through 15 |
| `TF_FLASHNEXT_DRAFTS_GREEDY` | Whole draft count; default 8, then clamped to 1 through 15 |
| `TF_FLASHNEXT_GREEDY_WIDE` | Whole round width; default 4 |
| `TF_FLASHNEXT_WINDOW_ROWS` | Whole row count; default the shared round's rows, capped by them |
| `TF_FLASHNEXT_READOUT_ROWS` | Whole row count; default 64; `0` never takes the tiles |
| `TF_FLASHNEXT_FACT_ROWS` | Whole row count; default 64; `0` never takes the tiles |
| `TF_FLASHNEXT_KV_PAGE_TEST` | Whole page size the kernels take: whole blocks of the indexer's ratio up to the window; unset or blank leaves the plain caches |
| `TF_FLASHNEXT_CONFIDENCE` | Finite number, negative values included; default -0.4, the chain's own stop |
| `TF_FLASHNEXT_PROMPT` | Exactly `rows` for the rows path; unset or blank keeps the plain path |

An on-by-default switch keeps the same kernels and bits as off; only the launch count changes.
The two row counts and the two draft counts keep their existing clamps after parsing.

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
- Flash Next keeps its lenient readers where they are: `TF_FLASHNEXT_TP` and `TF_FLASHNEXT_LOGITS` take a path,
  `TF_FLASHNEXT_OWNED` a list, `TF_FLASHNEXT_DEPTH_LOG` and `TF_TP_TRACE`/`TF_TP_LOCAL`/`TF_TP_STATS` select by presence,
  and the tool-only sampling settings of the MTP check and the `FZ_` developer switches keep their semantics.

The profile registry reports selected arithmetic/experimental overrides; it does not change their parsing or defaults.
See [profiles](profiles.md) for classes, scopes and grouped slide-learning names.
`capabilities --json` no longer advertises `TENSORFOLD_NO_UPDATE_CHECK` or `HF_HUB_OFFLINE`: the native binary reads neither.
The command-line `--no-update-check` remains an accepted no-op.
