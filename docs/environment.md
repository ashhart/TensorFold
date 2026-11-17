# Native environment validation

Malformed numeric settings are refused at startup with the variable, supplied value and accepted forms.
The same parser is used by direct CUDA entry points. An unset variable keeps its existing default.
A valid value keeps its existing meaning, including documented caps and clamping.
Diagnostics quote and escape the supplied text so it cannot create another terminal line.

| Variable | Accepted explicit value |
| --- | --- |
| `TF_CACHE_EVICT` | `main` (default) or `decayed_resume`. The opt-in gives recently resumed RAM snapshots bounded recency credit, within the same cache budget. It changes victim choice, not stored state or token arithmetic. Paged stores keep their own policy. Blank keeps `main`; other values are refused at server startup. |
| `TENSORFOLD_VIDEO_TOKENS` | Nonnegative whole video-token budget; default 16384; stored as an unsigned 64-bit count |
| `TENSORFOLD_MAX_PDF_PAGES` | Whole count 1 through 4294967295, capped by `--vision-max-images`; default min(16, image limit) |
| `TENSORFOLD_PARALLEL` | Whole count 1 through 4096, or `auto` (`auto` = 8); default: what `--parallel` says. Read only when `--parallel` is not on the command line, so the flag wins; blank keeps the default; other values are refused at server startup |
| `TENSORFOLD_PRESSURE_CACHE_PERCENT` | Whole percentage 0 through 100; default 50 |
| `TENSORFOLD_PREFIX_WAIT_MS` | Whole milliseconds 0 through 4294967295; zero disables waiting; default 60000 |
| `TENSORFOLD_RESTORE_PREFETCH` | Whole unsigned 64-bit count, capped at 65536; zero disables reading ahead; default 2 |
| `TENSORFOLD_RESTORE_PREFETCH_GIB` | Whole unsigned 64-bit GiB count, capped at 16777216; default 16 |
| `TENSORFOLD_RESTORE_PREFETCH_MS` | Whole unsigned 64-bit milliseconds, capped at 1099511627776; default 30000 |
| `TENSORFOLD_TOOL_ID_KEY_FILE` | The path of the file that holds the key for tool-call ids ([API](api.md#messages-and-tools)). The file must be a regular file, not a link, that only its owner can read or write (`chmod 600`), with 32 to 4096 bytes in it, and the path must not contain control characters. When this is unset, each server process makes its own random key. If the file is wrong, the server does not start; the message names this setting but never shows the path or what the file holds |
| `TF_STARTUP_RECEIPT` | Exactly `0` or `1`; default off. `1` writes the startup and first completion receipts |
| `TENSORFOLD_DEVICE_CHECK` | Exactly `0` or `1`; default 1. `0` skips serve's pre-load check against the saved [device profile](device.md) |
| `TF_SHARED_PRICE_LOG` | Exactly `0` or `1`; default off. `1` logs, at an engine's first priced shared round, the load-time shared forward costs and head steps, then one `shared price: round` line a priced shared round: its trial flag, rows, time, each stream's drafts and next depth, the plain and row prices and each stream's acceptance estimates |
| `TF_CUDA_DEVICE` | Whole ordinal 0 through 4294967295, or empty; default 0 |
| `TF_CUDA_SEGMENTS` | Whole count 1 through 4, or empty; default 1 |
| `TENSORFOLD_CUDA_KERNELS` | A path without control characters, used as given: a captured kernel set's folder, or `native` for the engine's own kernels. Unset uses the binary's bundled set for the GPU when it has one, else the engine's own kernels. A family that runs only on the engine's own kernels (Qwen3 among them) refuses any other value; Qwen3 reads a blank value as unset |
| `TF_CUDA_CARVEOUT` | Exactly `0` or `1`; default off |
| `TF_CUDA_CARVEOUT_MIB` | Positive whole MiB count whose byte size fits the platform, or empty for 1792 MiB |
| `TF_Q27_COPY_ROWS` | Whole u32 count; zero disables copied rows; default 48 |
| `TF_Q27_DRAFT_SLOTS` | Whole u32 count, clamped to 1 through the lane stream limit; default 8. DFlash2 can draft for at most the smaller of this count and the admitted streams, and none without a drafter. |
| `TF_MTP_DRAFTS` | Whole u32 count, clamped to 1 through the drafter's window less one; existing depth when unset |
| `TF_TIMING` | Exactly `0` or `1`; default off (each call's wall time on stderr while set) |
| `TF_NO_TREES` | Exactly `0` or `1`; default off. `1` keeps the CUDA drafter's chains only, and retains precedence over `TF_DRAFT_NODES` |
| `TF_MTP_TREES` | Exactly `0` or `1`; default off (1 turns the MTP drafter's trees on) |
| `TF_MTP_BATCH` | Exactly `0` or `1`; default on. `0` keeps one stream at a time |
| `TF_QWEN_KEEP_LATE` | Exactly `0` or `1`; default off |
| `TF_QWEN_KV_PAGE_TEST` | Whole page size from 1 to 2147483647; unset off. Each cache's own buffer reads through a table of n-position pages in reverse order (checks) |
| `TF_QWEN_POOL_LOG` | Exactly `0` or `1`; default off (the pool's census after a prompt pass) |
| `TF_COST_KEY_TRACE` | Exactly `0` or `1`; default off (each cost key's parts on stderr) |
| `TF_Q27_POCKET_OFF` | `all`, or a list of `layer:neuron` pairs; default off (pocket diagnostics) |
| `TF_Q27_TAIL` | Exactly `off`; default off — each nudged sequence's tail rows are the backward's rows |
| `TF_Q27_GEMM` | Exactly `narrow`; default the 128x128 tiles |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | A finite GiB amount; the HIP admission's floor (0.5, or 2 on shared memory) and the memory's size still bound it |
| `TENSORFOLD_HIP_MEMORY_LIMIT_GB` | A positive finite GiB amount whose byte count fits a 64-bit size |
| `TF_DSPINE_WINDOW` | Qwen3.8-27B on CUDA with a DSpine `--drafter`: whole row count 16 through 8192 that narrows the drafter's sliding window; it must be at least the block (16) and at most the checkpoint's own `sliding_window`, else the drafter is refused when it opens; unset or blank keeps the checkpoint's (8192 for tencent/Qwen3.8-27B-DSpine-b16) |
| `TF_DRAFT_NODES` | Whole native-size nonnegative count, capped by the drafter's window; zero disables trees |
| `TF_QMMF_HALF_BELOW` | Whole native-size nonnegative block count; zero disables half tiles; default 96 |
| `TF_HY4_MEM_FLOOR_GIB` | Whole GiB of MemAvailable below which `tf-hy4-cuda-generate`'s loader stops; default 10 |
| `TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS` | Exactly `0` or `1`, checked when the CUDA backend starts; default off ([profiles](profiles.md)) |
| `TF_PROMPT_INTERMEDIATE` | Exactly `0` or `1`, checked when the CUDA backend starts; default off. `1` keeps Flash Next's ladder of intermediate prompt-cache checkpoints under a long prompt's history |
| `TENSORFOLD_LFM2_KV_PAGES` | Exactly `0` or `1`, checked when the CUDA backend starts; unset leases pool pages ([profiles](profiles.md)) |
| `TENSORFOLD_UNIFIED_HOST_COUNTS` | Exactly `0` or `1`, checked when the CUDA backend starts. An integrated GPU on Linux admits against `/proc/meminfo` with `1` and against the CUDA pool with `0`; unset, it reads the pool under WSL2 (`WSL_DISTRO_NAME` set) and `/proc/meminfo` otherwise ([Windows](WINDOWS.md)) |
| `TF_HEAD_GEOMETRY` | Whole native-size nonnegative geometry count in the head benchmark |
| `TF_ROCE_GID_INDEX` | Whole index 0 through 2147483647; unset searches for a RoCE v2 IPv4 GID |
| `TF_QWEN35_CHUNK_SOLO` | Qwen 3.5/3.6/3.8 on CUDA: 4096 or 8192 rows a foreground prompt pass while no other stream decodes (a background warm pass keeps 4096); unset, empty or 4096 is off, with main's scratch. Checked when the model loads, with the family's own message |
| `TF_GEMMA4_CHUNK_SOLO` | Gemma 4 on CUDA: a multiple of 512 from 512 through 4096; unset, empty or 512 is off. Checked when the model loads, with the family's own message |
| `TF_DRAMA_PREFILL_ROWS` | Drama 3 on CUDA: whole prompt rows a chunk, 1 through 1024; default 1. Checked when the checkpoint opens, with its own message |
| `TF_DRAMA_REFERENCE_CACHE` | Drama 3 on CUDA: kept reference states, 0 (off) through 16; default 0. Checked when the checkpoint opens, with its own message |
| `TF_GLM_HOST_CACHE` | Exactly `0` or `1`; default off (kept GLM states on the device) |
| `TF_GLM_VISION_HOST` | Exactly `0` or `1`; default off (GLM vision weights on the device) |
| `TF_GLM_PROF` | Exactly `0` or `1`; default off (no GLM section timing). On, a prompt step is one chunk, with no pair or solo steps, so its prompt timings are of that shape |
| `TF_GLM_LINK_WAIT` | Exactly `0` or `1`; default off (short GLM steps go to the other ranks unwaited) |
| `TF_GLM_PDL` | Exactly `0` or `1`; default on (GLM's kernels and its qmmf linears use dependent launches) |
| `TF_GLM_PROMPT_STEPS` | Exactly `0` or `1`; default on (GLM prompt passes in chunks); read on rank 0 |
| `TF_GLM_SHARED` | Exactly `0` or `1`; default on (GLM windows share a forward); read on rank 0 |
| `TF_GLM_DSEL` | Exactly `0` or `1`; default on (a GLM decode window's DSA select runs on many CTAs; 0 keeps one CTA a row) |
| `TF_GLM_DSCORES` | Exactly `0` or `1`; default on (a GLM decode window's DSA index scores run on glm_index_scores; 0 keeps tf_glm_scores) |
| `TF_GLM_DFLASH2_LOG` | Whole unsigned 64-bit count of DFlash2 rounds whose drafts are logged; default 0 |
| `TF_GLM_DRAFT_CONFIDENCE` | Finite number 0 through 1 where a DFlash2 chain's chance product stops it; default 0.7 on CUDA, 0.5 on Metal |
| `TF_GLM_CONTEXT_CEILING` | GLM-5.3-Flash on CUDA: whole token count 1 through 2147483616, the window `--context` may ask for and `--context 0` serves, in place of config.json's 1048576; read at GLM's open. Past that length the model's quality is the operator's to check. Unset keeps config.json's window |
| `TF_GLM_EXPERT_FORM` | `classic` or `factored`: the expert form GLM's CUDA decode windows use at every width (the same bits either way), also at a load that times nothing; unset or blank races the two at each timed width when a build first loads with drafts and scratch context and keeps each width's winner, else classic; read on rank 0, which chooses every rank's forms |
| `TF_GLM_INDEX_CUDA` | GLM on CUDA: exactly `1` (both DSA indexer kernels), `0` (neither) or `scores` (scores only); unset or blank is `both` |
| `TF_GLM_KV_HOST_GIB` | GLM on CUDA: a finite GiB amount of at least 0 for the DSA layers' pinned host pages; unset or blank uses one full window and 16 rows a page |
| `TF_GLM_RANK` | GLM-5.3-Flash on CUDA: a whole rank index from 0 through 4294967295, set with `TF_GLM_WORLD`; unset with it gives one rank |
| `TF_GLM_WORLD` | GLM-5.3-Flash on CUDA: a whole rank count from 1 through 4294967295, set with `TF_GLM_RANK`; unset with it gives one rank |
| `TF_NEMOTRON_RANK` | Nemotron 3.5 Super on CUDA: a whole rank index from 0 through 4294967295, set with `TF_NEMOTRON_WORLD`; unset with it gives one rank |
| `TF_NEMOTRON_WORLD` | Nemotron 3.5 Super on CUDA: a whole rank count from 1 through 4294967295, set with `TF_NEMOTRON_RANK`; unset with it gives one rank |
| `TF_MINIMAX_RANK` | MiniMax-M3.1-Flash on CUDA: a whole rank index, 0 or 1 at two ranks, set with `TF_MINIMAX_WORLD`; unset with it gives one rank |
| `TF_MINIMAX_WORLD` | MiniMax-M3.1-Flash on CUDA: a whole rank count, 2 at CUDA, set with `TF_MINIMAX_RANK`; unset with it gives one rank |
| `TF_PROMPT_ROWS` | Native serving: a multiple of 16 from 128 through 2048; unset keeps the chip/family default |
| `TF_ANE_COLUMNS` | Whole u32 count; zero disables the experimental share; default zero |
| `TF_METAL4` | Metal: exactly `0` or `1`; `1` sends a family's GPU work through Metal 4 (macOS 26 and later) where the family supports it: the same kernels in the same order, with Metal 4's command queue, allocators, argument tables and barriers, and bit-identical outputs; `0` keeps Metal 3. DeepSeek-V4.1 defaults to `1` where the Mac has Metal 4 (its device test rigs to `0`); Macs without it stay on Metal 3 |
| `TF_DSV41_ROUND_PROF` | DeepSeek-V4.1 on Metal: exactly `0` or `1`; `1` logs decode rounds' host milliseconds (first encode to last wait) and GPU milliseconds by rows, every 1,024 rounds; default 0 |
| `TF_METAL4_TIMELINE` | Metal 4: a whole round count, default 0; `n` puts a GPU timestamp at every stage (between barriers) of `n` rounds of each kind a family names, after 8 warm ones, and logs their time by kernel and the longest stages. DeepSeek-V4.1 names its decode rounds by rows. Stamps cost GPU time, so stamped rounds read a little long |
| `TF_FLASHNEXT_SCALE_INDEX` | Flash Next on Metal before tensor units: exactly `0` or `1`; default 1, where the 4-bit expert kernels (decode, grouped and prompt gathers) read each group's scale and bias through a 16-bit index into its tensor's table of pairs, the same values from fewer weight bytes, and the routed experts' stored scales and biases are not kept in memory; `0` keeps and reads the stored arrays; ignored with `--ssd-experts` |
| `TF_FLASHNEXT_GROUPED_ROWS` | Flash Next on Metal without tensor units: `0`, or the rows (`1` through `32`) from which a decode window runs its 4-bit experts grouped by expert, with the same output bits; default 5; ignored with `--ssd-experts` and in speed-up mode |
| `TF_QWEN27_SHARED_PREFIX` | Qwen3.8-27B on Metal before M5: exactly `0` or `1`; `0` keeps cached keys on the tail attention kernel; unset or blank is `1` |
| `TF_QWEN27_SLOT_RESUME` | Qwen3.8-27B on Metal: exactly `0` or `1`; `0` stops slot 0 resuming a later prompt; unset or blank keeps it on |
| `TF_QWEN27_METAL_KV_PAGES` | Qwen3.8-27B on Metal: exactly `0` or `1`; `1` puts keys and values in core-pool pages (refused on Macs with tensor units); unset or blank is `0` |
| `TF_QWEN27_JOIN` | Qwen3.8-27B on Metal before M5: exactly `0` or `1`; `1` runs a window's rows in pairs over the keys below it in the decode tail attention (same replies); unset or blank is `1` |
| `TF_QWEN35_METAL_KV_PAGES` | Qwen 3.5 on Metal: exactly `0` or `1`; `1` puts the streams' keys and values in core-pool pages; unset or blank is `0` |
| `GLM_EHPROJ_NAX` | GLM-5.3-Flash on Metal: exactly `0` or `1`; `0` serves the MTP head's prompt eh_proj on the scalar kernel; default 1, the tile kernel where it builds |
| `TF_DSV41_SMALL` | DeepSeek-V4.1 on Metal: exactly `0` or `1`; `1` reads the checkpoint's config at its own small shapes (the served smoke test's tiny model); default 0 |
| `TF_DSV41_ASSETS` | DeepSeek-V4.1 on Metal: a directory path without control characters that holds the Engram token map; unset or blank reads it beside the checkpoint |
| `TF_DSV41_ENGRAM_DIR` | DeepSeek-V4.1 on Metal: a directory path without control characters that holds the packed Engram tables; unset or blank takes the checkpoint's `engram_table_dir` |
| `TF_DSV41_CALIB_FILE` | DeepSeek-V4.1 on Metal: a cost table file (JSON: `verify`, each window's ms by its rows from 1, and `draft`, the drafter pass's ms) that prices draft depth instead of the built-in table. Measure it on the machine that serves; pricing never changes a token. Unset: the built-in table. |
| `TF_DSV41_ENGRAM_VERIFY` | DeepSeek-V4.1 on Metal: exactly `sample` or `full`; `full` also checks every byte of the Engram tables against their manifest at boot; default `sample` |
| `TF_DSV41_ENGRAM_CACHE` | DeepSeek-V4.1 on Metal: whole u32 count of Engram records a layer kept cached; default 65536 |
| `TF_DSV41_PREFILL_PIECES` | DeepSeek-V4.1: exactly `0` or `1`, default 0. On CUDA, `1` lets the round planner run prompt pieces and decode rows in one pass; on Metal, `1` runs a prompt a chunk at a time between the other streams' rounds (as `--decode-share` above 0, `TENSORFOLD_PREFILL_ORDER=srpt` or a `TENSORFOLD_PREFILL_POLICY`, `_SHARE` or `_BUDGET` also do), replies taking `--decode-share`, else `TENSORFOLD_PREFILL_SHARE`, else 0.25 of wall time beside it; malformed text stops the start naming it |
| `XPU_A8_BITS` | Exactly `4` or `8`, for the XPU quantization study; unset keeps the study's existing setting |
| `LFM3_FUSE` | Whole u32 fusion mask; default 7 |
| `TF_SLIDE_POCKET_KEY` | Exactly `auto`, `reply`, `subject` or `both`; default `auto` |
| `TF_SLIDE_FIT_GIB` | Whole native-size nonnegative GiB count; default 4 |
| `TF_SLIDE_TAKE_OUT_CHECK` | Exactly `0` or `1`, read once when the server starts with `--slide`; `1` takes every kept lesson out again to check that the facts before it come back as they were; default off |
| `TF_SLIDE_TAP_CHECK` | Exactly `0` or `1`; `1` checks a pass's tapped rows against a fresh run's; default off |
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
The `TF_GLM_*` rows are checked when GLM's CUDA family opens, on every rank, so they never stop another family; on Metal `TF_GLM_DRAFT_CONFIDENCE` is read and validated when GLM loads `--drafter`.
Common server settings are checked before checkpoint resolution, including when no request would use them yet.
Direct family readers also refuse malformed values when called without the server.

## Flash Next

Flash Next checks these 35 settings when its engine loads, before any device or checkpoint work: a malformed value
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
| `TF_FLASHNEXT_READOUT_FUSED` | Exactly `0` or `1`; default off |
| `TF_FLASHNEXT_KEEP_LATE` | Exactly `0` or `1`; unset or blank leaves the choice to the loader |
| `TF_FLASHNEXT_DECODE_GRAPH` | Exactly `0` or `1`; default off |
| `TF_FLASHNEXT_SPLIT_DENSE` | Exactly `0` or `1`; default on: two ranks split the BF16 dense projections |
| `TF_FLASHNEXT_SHARD_HEAD` | Exactly `0` or `1`; default off: each rank keeps its half of the output head |
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
| `TF_FLASHNEXT_CHUNK` | Whole prompt rows a pass, 1 through 16384; default 3072 |
| `TF_FLASHNEXT_CHUNK_SOLO` | Whole prompt rows a pass with no stream decoding, from the chunk `TF_FLASHNEXT_CHUNK` resolves to through 16384; unset or blank is the larger of that chunk and 8192 on a server of one stream, and that chunk elsewhere |
| `TF_FLASHNEXT_FACT_ROWS` | Whole row count; default 64; `0` never takes the tiles |
| `TF_FLASHNEXT_KV_PAGE_TEST` | Whole page size the kernels take: whole blocks of the indexer's ratio up to the window; unset or blank leaves the plain caches |
| `TF_FLASHNEXT_CONFIDENCE` | Finite number, negative values included; default -0.4, the chain's own stop |
| `TF_FLASHNEXT_PROMPT` | Exactly `rows` for the rows path; unset or blank keeps the plain path |
| `TF_FLASHNEXT_DRAFT_HEAD` | Exactly `bf16` or `affine4`; unset or blank is `affine4`, a bf16 language head's draft rows held as 4-bit affine groups of 32. `bf16` keeps the rows as stored. Drafts only, so replies are unchanged |

An on-by-default switch keeps the same kernels and bits as off; only the launch count changes.
The two row counts and the two draft counts keep their existing clamps after parsing.
The two chunk settings keep their cross-setting rule in the family, which the table cannot state on its own: the family
checks a solo chunk reaches at least the chunk `TF_FLASHNEXT_CHUNK` resolves to, and its refusal names that setting and
the rows it must reach. An unset or blank solo chunk is the larger of that chunk and 8192 on a server of one stream
(`--parallel 1` on one rank), and that chunk elsewhere, where solo passes are off. An empty or whitespace-only value of
either setting takes the unset default above, like every other setting here.

## Existing lenient switches

These settings retain their existing boolean policies rather than gaining new aliases or refusals:

- `TF_GRAPH_CACHE`: exact `0` or case-insensitive `off` disables graphs; every other value leaves the family's settings unchanged.
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
The fabric/tp config helper keeps its runtime-name lookup instead of gaining a startup row: it carries the lenient switches above and transport names read only by tools, which no server would set.

The profile registry reports selected arithmetic/experimental overrides; it does not change their parsing or defaults.
See [profiles](profiles.md) for classes, scopes and grouped slide-learning names.
`capabilities --json` no longer advertises `TENSORFOLD_NO_UPDATE_CHECK` or `HF_HUB_OFFLINE`: the native binary reads neither.
The command-line `--no-update-check` remains an accepted no-op.

### DeepSeek V4.1 runtime settings

These settings are checked when the DeepSeek family opens, before assets or GPU resources.
The parser treats unset or blank text as absent, except `TF_DSV41_STALL_EXIT_S`, which refuses blank.
Readers retain their own defaults and presence checks: blank `TF_DSV41_LOOKUP` disables lookup, and presence readers see the original text.
`TF_DSV41_BOOT_MEASURED` retains its legacy fallback to automatic selection for unrecognized text.
Paths keep each reader's original trimming; control characters are refused without printing the path.
Tool controls for decode-TP gates, KV checks and the session-slots CLI stay local to those tools.

| Setting | Accepted values |
| --- | --- |
| `TF_DSV41_AOT_TRACE` | a path without control characters |
| `TF_DSV41_GRAMMAR_TOKENIZER` | a path without control characters |
| `TF_DSV41_ACCEPT_LOG` | a path without control characters |
| `TF_DSV41_PHASES_OUT` | a path without control characters |
| `TF_DSV41_PREPARED` | a path without control characters |
| `TF_DSV41_PROFILE` | a path without control characters |
| `TF_DSV41_TRACE_TOKENS` | a path without control characters |
| `TF_DSV41_CALIB_DIR` | a path without control characters |
| `TF_DSV41_CALIB_FILE` | a path without control characters |
| `TF_DSV41_SESSION_DISK` | a path without control characters |
| `TF_DSV41_PF_DENSE_TABLE` | a path without control characters |
| `TF_TP_KERNEL_IMAGE` | a transport kernel image path without control characters |
| `TF_DSV41_BRANCHES` | 0 or off or serial or e4 or 1 or on or streams (case-insensitive) |
| `TF_DSV41_BRANCHES_ROWS` | a whole count 1..4096 (default 64) |
| `TF_DSV41_BRANCHES_PRIO` | main or side or equal |
| `TF_DSV41_DRAFT_GRAPHS` | 0 or device or 1 |
| `TF_DSV41_DRAFT_GRAPHS_MAX` | a whole count 0..4294967295 (default 32) |
| `TF_DSV41_GRAPH_FLOOR` | shed or hold |
| `TF_DSV41_DRAFT_CHAIN` | 0 or 1 |
| `TF_DSV41_DRAFT_OVERLAP` | 0 or 1 |
| `TF_DSV41_DRAFT_CAPTURE` | 0 or 1 |
| `TF_DSV41_SAMP_BATCH` | 0 or 1 |
| `TF_DSV41_DRAFT_BATCH_PROJ` | 0 or 1 or check |
| `TF_DSV41_SLOT_DRAFT_GROUP` | a whole count 1..4294967295 (default automatic) |
| `TF_DSV41_ENGRAM_GATE` | 0 or 1 |
| `TF_DSV41_ENGRAM_GATE_ROWS` | a whole count 1..4294967295 (default 128) |
| `TF_DSV41_ENGRAM_GATE_TIMEOUT_S` | a finite number from 0 to 9223372036 (default 30) |
| `TF_DSV41_ENGRAM_WAIT_S` | a finite number from 0 to 18446744073 (default 300) |
| `TF_DSV41_GRAPHS_MAX` | a whole count 0..2147483647 (default 48) |
| `TF_DSV41_GRAPH_BUCKET` | a whole count 1..4294967295 (default 2048) |
| `TF_DSV41_GRAPH_BUCKET_GROW` | a finite number from 1 to 16 (default 1.25) |
| `TF_DSV41_GRAPH_ROWS_MAX` | a whole count 0..18446744073709551615 (default 64) |
| `TF_DSV41_LIN2` | 0 or off or 1 or on or lanes or linear (case-insensitive) |
| `TF_DSV41_LIN2_PF` | 0 or 1 |
| `TF_DSV41_LIN2_CLUSTER` | 0 or 1 |
| `TF_DSV41_LIN2_WK` | a whole count 4 / 8 (default both) |
| `TF_DSV41_L2PF` | 0 or off or none or false or 1 or on or true or bulk (case-insensitive) |
| `TF_DSV41_L2PF_ROWS` | a whole count 0..4294967295 (default 16) |
| `TF_DSV41_L2PF_MB` | any finite number (default emitter value) |
| `TF_DSV41_L2PF_CHUNK_KB` | any finite number (default emitter value) |
| `TF_DSV41_L2PF_PACE_GBPS` | any finite number (default emitter value) |
| `TF_DSV41_L2PF_PACE_CTAS` | any finite number (default emitter value) |
| `TF_DSV41_L2PF_DELAY_US` | any finite number (default emitter value) |
| `TF_DSV41_L2PF_SITES` | comma-separated x, o or f, three entries (case-insensitive) |
| `TF_DSV41_L2PF_PACE_SITES` | comma-separated x or f, two entries (case-insensitive) |
| `TF_DSV41_L2PF_AT` | one comma-separated before entry (any case) (case-insensitive) |
| `TF_DSV41_PROMPT_PROF` | 0 or 1 |
| `TF_DSV41_PHASES` | 0 or 1 or sync |
| `TF_DSV41_PREPARED_WRITE` | 0 or 1 |
| `TF_DSV41_PREPARED_VERIFY` | 0 or 1 |
| `TF_DSV41_ROUND_GRAPH` | 0, 1 or split and gate separated by comma or space |
| `TF_DSV41_ROUND_GRAPH_HEAD` | a whole count 1..4294967295 (default existing head) |
| `TF_DSV41_SAMP_STATS` | split or row |
| `TF_DSV41_PIECES_FLOOR` | 0 or 1 |
| `TF_DSV41_SESSION_DISK_MIN` | a whole count 0..18446744073709551615 (default existing minimum) |
| `TF_DSV41_SESSION_RESIDENT` | a whole count 0..4294967295 (default existing count) |
| `TF_DSV41_SESSION_RAM_MIB` | a whole count 0..18446744073709551615 (default existing budget) |
| `TF_DSV41_SESSION_STAGING_MIB` | a whole count 0..4294967295 (default existing staging) |
| `TF_DSV41_SESSION_DISK_GIB` | a finite number from 0 to 17179869183 (default existing budget) |
| `TF_DSV41_SESSION_RESIDENT_GIB` | a finite number from 0 to 17179869183 (default existing budget) |
| `TF_DSV41_ENGRAM_WARM` | 0 or 1 |
| `TF_DSV41_STALL_EXIT_S` | a whole count 0..18446744073 (default 0) |
| `TF_DSV41_GM_GU2` | 0 or 1 |
| `TF_DSV41_PF_4K` | 0 or 1 |
| `TF_DSV41_PF_4K_MEMORY_OK` | 0 or 1 |
| `TF_DSV41_PF_OVERLAP_SITE` | 0 or 1 |
| `TF_DSV41_PF_TBO` | 0 or 1 |
| `TF_DSV41_INDEX_BOUND` | 0 or 1 |
| `TF_DSV41_X3LD_EPI` | 0 or 1 |
| `TF_DSV41_PF_DENSE` | off or 0 or no or false or fused (case-insensitive) |
| `TF_DSV41_GM_V2` | 0 or 1 or split |
| `TF_DSV41_PF_DENSE_CFG` | configuration id 0-7, optionally followed by group 1-1024 (default 8) |
| `TF_DSV41_PF_OVERLAP` | a whole count 0..8 (default 0) |
| `TF_DSV41_MHC_SITE_ROWS` | a whole count 0 or 256..4096, a multiple of 16 (default 0) |
| `TF_DSV41_PF_TBO_GM_SMS` | a whole count 0 or 8..48 (default 0) |
| `TF_DSV41_STREAM_RB` | a whole count 0 / 1 / 2 / 4 (default 0) |
| `TF_DSV41_MHC_PFDEC` | a whole count 0..64 (default 0) |
| `TF_DSV41_PREFILL` | full or replay |
| `TF_DSV41_PROMPT_TAIL` | verify or prefill |
| `TF_DSV41_MHC_DEFER_AT` | 0 or off or exchange (case-insensitive) |
| `TF_DSV41_MHC_DEFER_AT_ROWS` | a whole count 0..64 (default 0) |
| `TF_DSV41_ATTN_SPLIT` | a whole count 0..18446744073709551615 (default 0) |
| `TF_DSV41_ATTN_ROPE` | 0 or 1 |
| `TF_DSV41_MHC_TAIL` | 0 or 1 |
| `TF_DSV41_MOE_BF16` | 0 or 1 |
| `TF_DSV41_PRUNE_KIT` | 0 or 1 |
| `TF_DSV41_RG_PRUNE` | 0 or 1 |
| `TF_DSV41_RMS_FOLD` | 0 or 1 |
| `TF_DSV41_CALIB` | cached or default or real or measure or zig (case-insensitive) |
| `TF_DSV41_CALIB_SHAPE` | raw or fit (case-insensitive) |
| `TF_DSV41_CALIB_STAT` | median or min (case-insensitive) |
| `TF_DSV41_CALIB_REPS` | a whole count 1..50 (default 5) |
| `TF_DSV41_CALIB_WARM` | a whole count 0..20 (default 3 raw / 2 fit) |
| `TF_DSV41_CALIB_RECHECK` | a whole count 0..8 (default 4 raw / 2 fit) |
| `TF_DSV41_CALIB_CYCLE` | a whole count 0..8 (default 4 raw / 0 fit) |
| `TF_DSV41_CALIB_DEEP_REPS` | a whole count 1..50 (default 5 raw / 3 fit) |
| `TF_DSV41_CODE_ACCEPT` | 0, 1, on, off, true or false (any case) |
| `TF_DSV41_DRAFT_SKIP` | 0, 1, on, off, true or false (any case) |
| `TF_DSV41_SPEC_DRAFT` | 0, 1, on, off, true or false (any case) |
| `TF_DSV41_SPEC_NUCLEUS` | 0, 1, on, off, true or false (any case) |
| `TF_DSV41_DEPTH` | cost or static (case-insensitive) |
| `TF_DSV41_DRAFT_DEPTH` | a whole count 0..15 (default 5 cost / 3 static) |
| `TF_DSV41_DEPTH_JOINT` | 0 or off or false or 1 or on or true or 2 (case-insensitive) |
| `TF_DSV41_DEPTH_FAIR` | a finite number from 0 to 4 (default existing fair term) |
| `TF_DSV41_DEPTH_HOST_MS` | a finite number from 0 to 100 (default existing host term) |
| `TF_DSV41_DRAFT_PLAIN_STREAMS` | a whole count 0..16 (default 0) |
| `TF_DSV41_LOOKUP_MIN` | a whole count 1..64 (default 4) |
| `TF_DSV41_MAX_ROWS` | a whole count 2..16 (default 16) |
| `TF_DSV41_TREE` | a whole count 0 / 2 / 3 / 4 (default 0) |
| `TF_DSV41_TREE_DUP_MS` | a finite number from 0 (default 0.4) |
| `TF_DSV41_TREE_ROWS` | a whole count 1..15 (default 4) |
| `TF_DSV41_TREE_PC` | 0 or off or false |
| `TF_DSV41_POST_BOOT_GIB` | a finite number from 0 (default existing budget) |
| `TF_DSV41_BOOT_GROWTH_GIB` | a finite number from 0 (default existing budget) |
| `TF_DSV41_SERVE_DIP_GIB` | a finite number from 0 (default existing budget) |
| `TF_DSV41_BOOT_MEASURED` | auto or 1 or on or yes or true or 0 or off or no or false (case-insensitive) |
| `TF_DSV41_KV_SPLIT_COMPACT` | 0 or off or no or false or 1 or on or yes or true or force (case-insensitive) |
| `TF_DSV41_PREFILL_CHUNK` | a whole count 32..4096, a multiple of 16 (default automatic) |

`TF_DSV41_GRAPH_ROWS_MAX` keeps both readers: graph capture requires 1–64, while draft pricing clamps to its window.
`TF_DSV41_PF_4K=1` still requires `TF_DSV41_PF_4K_MEMORY_OK=1`; configured prompt rows must then equal 4096.
Without PF_4K, prompt chunks and rows use multiples of 16 in 32–2048; the lower configured size wins.
L2 prefetch numeric overrides must match the active emitter; its site lists keep their original fixed entry counts.
Accept logging and prompt profiling retain presence semantics: a present `0` still enables the marker.
