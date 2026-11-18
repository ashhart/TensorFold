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
| `TENSORFOLD_PRIORITY_AGE_MS` | Whole milliseconds 0 through 4294967295; default 0. Above 0, a background request queued for at least this long may be admitted before foreground requests. Eligibility, memory and lane limits still apply; 0 keeps foreground-first FIFO |
| `TENSORFOLD_PULL_CONNECT_SECONDS` | Whole seconds 1 through 4294967295; how long `pull` waits for a connection to the hub, the storage host or the proxy; default 30 |
| `TENSORFOLD_PULL_STALL_SECONDS` | Whole seconds 1 through 4294967295; how long `pull` waits for the next byte before it resumes the file with a range request (at most 3 times); default 60 |
| `TENSORFOLD_RESTORE_PREFETCH` | Whole unsigned 64-bit count, capped at 65536; zero disables reading ahead; default 2 |
| `TENSORFOLD_RESTORE_PREFETCH_GIB` | Whole unsigned 64-bit GiB count, capped at 16777216; default 16 |
| `TENSORFOLD_RESTORE_PREFETCH_MS` | Whole unsigned 64-bit milliseconds, capped at 1099511627776; default 30000 |
| `TENSORFOLD_TOOL_ID_KEY_FILE` | The path of the file that holds the key for tool-call ids and cache salts ([API](api.md#messages-and-tools)). The file must be a regular file, not a link, that only its owner can read or write (`chmod 600`), with 32 to 4096 bytes in it, and the path must not contain control characters. When this is unset, each server process makes its own random key. If the file is wrong, the server does not start; the message names this setting but never shows the path or what the file holds |
| `TF_STARTUP_RECEIPT` | Exactly `0` or `1`; default off. `1` writes the startup and first completion receipts |
| `TENSORFOLD_DEVICE_CHECK` | Exactly `0` or `1`; default 1. `0` skips serve's pre-load check against the saved [device profile](device.md) |
| `TF_SHARED_PRICE_LOG` | Exactly `0` or `1`; default off. `1` logs, at an engine's first priced shared round, the load-time shared forward costs and head steps, then one `shared price: round` line a priced shared round: its trial flag, rows, time, each stream's drafts and next depth, the plain and row prices and each stream's acceptance estimates |
| `TF_CUDA_DEVICE` | Whole ordinal 0 through 4294967295, or empty; default 0 |
| `TF_CUDA_SEGMENTS` | Whole count 1 through 4, or empty; default 1 |
| `TENSORFOLD_CUDA_KERNELS` | A path without control characters, used as given: a captured kernel set's folder, or `native` for the engine's own kernels. Unset uses the binary's bundled set for the GPU when it has one, else the engine's own kernels. A family that runs only on the engine's own kernels (Qwen3 among them) refuses any other value; Qwen3 reads a blank value as unset |
| `TF_CUDA_CARVEOUT` | Exactly `0` or `1`; default off |
| `TF_CUDA_CARVEOUT_MIB` | Positive whole MiB count whose byte size fits the platform, or empty for 1792 MiB |
| `TF_Q27_COPY_ROWS` | Whole u32 count; zero disables copied rows; default 48 |
| `TF_Q27_DRAFT_SLOTS` | Whole u32 count, clamped to 1 through the lane stream limit; default 8. DFlash2 can draft for at most the smaller of this count and the admitted streams, and none without a drafter. Under `--mtp-drafts` the MTP head drafts every stream and this does nothing. |
| `TF_Q27_DFLASH_ADAPTIVE_BLOCK` | Qwen3.8-27B on Metal: exactly `0` or `1`; default 0. `1` lets a lone stream's DFlash2 block narrow from 16 to 8 rows after 8 consecutive rounds landing at most 4 tree drafts, and return to 16 after a round landing 6 or more. Only the drafter's proposal changes. |
| `TF_Q27_DFLASH_ABSORB_BATCH` | Qwen3.8-27B on Metal: exactly `0` or `1`; default `0`. `1` absorbs a shared round's new DFlash2 taps for up to 128 rows across streams in one pass, each stream's results in its own caches. |
| `TF_MTP_DRAFTS` | Whole u32 count, clamped to 1 through the drafter's window less one; existing depth when unset |
| `TF_TIMING` | Exactly `0` or `1`; default off (each call's wall time on stderr while set) |
| `TF_NO_TREES` | Exactly `0` or `1`; default off. `1` keeps the CUDA drafter's chains only, and retains precedence over `TF_DRAFT_NODES` |
| `TF_DFLASH_QL_SPAN` | Exactly `0` or `1`; default off. `1` drafts a ModelOpt NVFP4 or FP8 head over the DFlash2 vocabulary spans; other head formats stay whole |
| `TF_DFLASH_GRAPH` | Exactly `0` or `1`; default off. `1` replays DFlash2 holds and live absorbs from CUDA graphs, with the same kernels and arguments in the same order |
| `TF_MTP_TREES` | Exactly `0` or `1`; default off (1 turns the MTP drafter's trees on) |
| `TF_MTP_BATCH` | Exactly `0` or `1`; default on. `0` keeps one stream at a time |
| `TF_QWEN_KEEP_LATE` | Exactly `0` or `1`; default off |
| `TF_Q35_GROUP_PROJ` | Exactly `0` or `1`, read when a ModelOpt checkpoint loads on CUDA (sm_90 or newer); default off. `1` runs Qwen 3.5 same-input decode projections (MLP gate and up, GDN `in_proj_qkv` and `in_proj_z`, `in_proj_a` and `in_proj_b`, attention q, k and v) as one launch per group, each projection with its own tile and K slices, so outputs keep their bits |
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
| `TF_QMMF_HALF_BELOW` | Whole native-size nonnegative block count; zero disables half tiles; default 96, or two blocks per SM with `TF_QMMF_HALF_BN=1` |
| `TF_QMMF_HALF_BN` | Exactly `0` or `1`, read when a CUDA lane loads; default off. `1` also gives FP4, FP8 and FP8G decode calls of 16 rows or fewer the 32-column half tile, below `TF_QMMF_HALF_BELOW` blocks (default two per SM, so 340 on 170 SMs and 96 on 48). Columns split, never K, so every output keeps its bits |
| `TF_HY4_MEM_FLOOR_GIB` | Whole GiB of MemAvailable below which `tf-hy4-cuda-generate`'s loader stops; default 10 |
| `TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS` | Exactly `0` or `1`, checked when the CUDA backend starts; default off ([profiles](profiles.md)) |
| `TF_PROMPT_INTERMEDIATE` | Exactly `0` or `1`, checked when the CUDA backend starts; default off. `1` keeps Flash Next's ladder of intermediate prompt-cache checkpoints under a long prompt's history |
| `TF_Q35_DFLASH_ABSORB_BATCH` | Exactly `0` or `1`, checked when the CUDA backend starts; default off. `1` runs a shared round's live DFlash2 absorbs (two or more streams) as one pass over every stream's kept rows, each row's arithmetic unchanged |
| `TENSORFOLD_LFM2_KV_PAGES` | Exactly `0` or `1`, checked when the CUDA backend starts; unset leases pool pages ([profiles](profiles.md)) |
| `TENSORFOLD_UNIFIED_HOST_COUNTS` | Exactly `0` or `1`, checked when the CUDA backend starts. An integrated GPU on Linux admits against `/proc/meminfo` with `1` and against the CUDA pool with `0`; unset, it reads the pool under WSL2 (`WSL_DISTRO_NAME` set) and `/proc/meminfo` otherwise ([Windows](WINDOWS.md)) |
| `TF_HEAD_GEOMETRY` | Whole native-size nonnegative geometry count in the head benchmark |
| `TF_ROCE_GID_INDEX` | Whole index 0 through 2147483647; unset searches for a RoCE v2 IPv4 GID |
| `TF_QWEN35_ATTN_ALL_ROWS` | Qwen3.8 on CUDA: exactly `0` or `1`; unset, empty or `0` keeps the current BF16 decode attention launches. `1` uses the existing rows kernel at every context length and skips the separate partial launch for decode windows wider than one row. The kernel contract compares output bytes against the original dual-launch path |
| `TF_QWEN35_CHUNK_SOLO` | Qwen 3.5/3.6/3.8 on CUDA: 4096 or 8192 rows a foreground prompt pass while no other stream decodes (a background warm pass keeps 4096); unset, empty or 4096 is off, with main's scratch. Checked when the model loads, with the family's own message |
| `TF_GEMMA4_CHUNK_SOLO` | Gemma 4 on CUDA: a multiple of 512 from 512 through 4096; unset, empty or 512 is off. Checked when the model loads, with the family's own message |
| `TF_GEMMA4_ARGMAX_GRAPH` | Gemma 4 on CUDA: exactly `0` or `1`; default off. `1` captures a greedy verify round's argmax (the same `tf_round_bf16_argmax` over the same logits) in the target forward's graph, skips the drawn-token download no greedy entry reads, and stages each forward's ids, positions and slots in pinned memory; target tokens keep their bits |
| `TF_DRAMA_PREFILL_ROWS` | Drama 3 on CUDA: whole prompt rows a chunk, 1 through 1024; default 1. Checked when the checkpoint opens, with its own message |
| `TF_DRAMA_REFERENCE_CACHE` | Drama 3 on CUDA: kept reference states, 0 (off) through 16; default 0. Checked when the checkpoint opens, with its own message |
| `TF_GLM_HOST_CACHE` | Exactly `0` or `1`; default off (kept GLM states on the device) |
| `TF_GLM_VISION_HOST` | Exactly `0` or `1`; default off (GLM vision weights on the device) |
| `TF_GLM_VISION_FUSED` | Exactly `0` or `1`; default off (the GLM vision tower's attention as cuBLASLt products around a softmax kernel) |
| `TF_GLM_PAIR_EXPERTS` | Exactly `0` or `1`; default off. On, GLM paired CUDA prompt chunks share 128-pair routed NVFP4 expert items, with separate per-chunk outputs and unchanged slot combines |
| `TF_GLM_PROF` | Exactly `0` or `1`; default off (no GLM section timing). On, all four prompt traces are created before readiness; startup, gate, final-draw, cache-admission and memory-counter records go to the log. A prompt step is one chunk, with no pair or solo steps, so its prompt timings are of that shape |
| `TF_GLM_LINK_WAIT` | Exactly `0` or `1`; default off (short GLM steps go to the other ranks unwaited) |
| `TF_GLM_PDL` | Exactly `0` or `1`; default on (GLM's kernels and its qmmf linears use dependent launches) |
| `TF_GLM_INDEX_SPLIT` | A whole prompt position from `0` to `4294967295`; default `2048` (from that position on, rank 0 splits a prompt chunk's indexer rows at TP=2 or more; `0` never splits) |
| `TF_GLM_PROMPT_STEPS` | Exactly `0` or `1`; default on (GLM prompt passes in chunks); read on rank 0 |
| `TF_GLM_SHARED` | Exactly `0` or `1`; default on (GLM windows share a forward); read on rank 0 |
| `TF_GLM_DSEL` | Exactly `0` or `1`; default on (a GLM decode window's DSA select runs on many CTAs; 0 keeps one CTA a row) |
| `TF_GLM_DSCORES` | Exactly `0` or `1`; default on (a GLM decode window's DSA index scores run on glm_index_scores; 0 keeps tf_glm_scores) |
| `TF_GLM_MOE_DIRECT` | Exactly `0` or `1`; default off. On, a GLM decode window's EXL3 experts read the top-k kernel's flat picks and leave their slot rows in place, summed with the shared expert's row by one kernel: no row copies and no slot write, the same bytes (#3570) |
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
| `TF_NEMOTRON_WIDE_ROUNDS` | Exactly `0` or `1`, checked when the CUDA backend starts; default off. `1` lets Nemotron's shared rounds hold 16 streams and 32 rows on GB10 (one rank, non-latent, not Bolt, no `--precision checkpoint`, no bf16 experts, no Sliding Weights or saved lessons; refused with a log line naming why), each row's bits as in two 16-row rounds; a lone stream's window is unchanged |
| `TF_MINIMAX_RANK` | MiniMax-M3.1-Flash on CUDA: a whole rank index, 0 or 1 at two ranks, set with `TF_MINIMAX_WORLD`; unset with it gives one rank |
| `TF_MINIMAX_WORLD` | MiniMax-M3.1-Flash on CUDA: a whole rank count, 2 at CUDA, set with `TF_MINIMAX_RANK`; unset with it gives one rank |
| `TF_PROMPT_ROWS` | Native serving: a multiple of 16 from 128 through 2048; unset keeps the chip/family default |
| `TF_ANE_COLUMNS` | Whole u32 count; zero disables the experimental share; default zero |
| `TF_LFM3_METAL_WARM_TURNS` | Exactly `0` or `1`; default off. Uses one-row LFM3 Metal prefill even with caching disabled. A prompt cache enables background warm turns. |
| `TF_METAL4` | Metal: exactly `0` or `1`; `1` sends a family's GPU work through Metal 4 (macOS 26 and later) where the family supports it: the same kernels in the same order, with Metal 4's command queue, allocators, argument tables and barriers, and bit-identical outputs; `0` keeps Metal 3. DeepSeek-V4.1 defaults to `1` where the Mac has Metal 4 (its device test rigs to `0`); Macs without it stay on Metal 3 |
| `TF_DSV41_ROUND_PROF` | DeepSeek-V4.1 on Metal: exactly `0` or `1`; `1` logs decode rounds' host milliseconds (first encode to last wait) and GPU milliseconds by rows, every 1,024 rounds; default 0 |
| `TF_METAL4_TIMELINE` | Metal 4: a whole round count, default 0; `n` puts a GPU timestamp at every stage (between barriers) of `n` rounds of each kind a family names, after 8 warm ones, and logs their time by kernel and the longest stages. DeepSeek-V4.1 names its decode rounds by rows. Stamps cost GPU time, so stamped rounds read a little long |
| `TF_FLASHNEXT_SCALE_INDEX` | Flash Next on Metal: exactly `0` or `1`; default 1, where the 4-bit expert kernels (decode, grouped and prompt gathers) read each group's scale and bias through a 16-bit index into its tensor's table of pairs, the same values from fewer weight bytes, and the routed experts' stored scales and biases are not kept in memory; `0` keeps and reads the stored arrays; ignored with `--ssd-experts` |
| `TF_FLASHNEXT_GROUPED_ROWS` | Flash Next on Metal without tensor units: `0`, or the rows (`1` through `32`) from which a decode window runs its 4-bit experts grouped by expert, with the same output bits; default 5; ignored with `--ssd-experts` and in speed-up mode |
| `TF_FN_GX_GATEUP_MB2` | Flash Next on Metal, grouped experts only: exactly `0` or `1`; `1` has gate/up decode an expert's 4-bit weights once for two rows, with each row's arithmetic and order unchanged; default 0 |
| `TF_QWEN27_SHARED_PREFIX` | Qwen3.8-27B on Metal before M5: exactly `0` or `1`; `0` keeps cached keys on the tail attention kernel; unset or blank is `1` |
| `TF_QWEN27_SLOT_RESUME` | Qwen3.8-27B on Metal: exactly `0` or `1`; `0` stops slot 0 resuming a later prompt; unset or blank keeps it on |
| `TF_QWEN27_METAL_KV_PAGES` | Qwen3.8-27B on Metal: exactly `0` or `1`; `1` puts keys and values in core-pool pages (refused on Macs with tensor units); unset or blank is `0` |
| `TF_Q27_MTP_GPU_PICK` | Qwen3.8-27B with `--mtp-drafts` on Metal: exactly `0` or `1`; `1` runs each MTP absorb on one command buffer and picks drafts on the GPU; unset or blank is `0` |
| `TF_QWEN27_JOIN` | Qwen3.8-27B on Metal before M5: exactly `0` or `1`; `1` runs a window's rows in pairs over the keys below it in the decode tail attention (same replies); unset or blank is `1` |
| `TF_QWEN27_TIED_HEAD` | Dense Qwen on Metal: exactly `0` or `1`; `1` lets a checkpoint with a tied head load on the 27B engine, its head read from the embedding (one Mac, no draft head, qualified shapes only); unset or blank is `0` |
| `TF_QWEN35_METAL_KV_PAGES` | Qwen 3.5 on Metal: exactly `0` or `1`; `1` puts the streams' keys and values in core-pool pages; unset or blank is `0` |
| `GLM_EHPROJ_NAX` | GLM-5.3-Flash on Metal: exactly `0` or `1`; `0` serves the MTP head's prompt eh_proj on the scalar kernel; default 1, the tile kernel where it builds |
| `TF_DSV41_SMALL` | DeepSeek-V4.1 on Metal: exactly `0` or `1`; `1` reads the checkpoint's config at its own small shapes (the served smoke test's tiny model); default 0 |
| `TF_DSV41_ASSETS` | DeepSeek-V4.1 on Metal: a directory path without control characters that holds the Engram token map; unset or blank reads it beside the checkpoint |
| `TF_DSV41_ENGRAM_DIR` | DeepSeek-V4.1 on Metal: a directory path without control characters that holds the packed Engram tables; unset or blank takes the checkpoint's `engram_table_dir` |
| `TF_DSV41_ENGRAM_VERIFY` | DeepSeek-V4.1 on Metal: exactly `sample` or `full`; `full` also checks every byte of the Engram tables against their manifest at boot; default `sample` |
| `TF_DSV41_ENGRAM_CACHE` | DeepSeek-V4.1 on Metal: whole u32 count of Engram records a layer kept cached; default 4194304 (about 1.2 GiB a layer) |
| `TF_DSV41_MARKOV_GPU` | DeepSeek-V4.1 on Metal: `1` draws DSpark drafts (greedy, and keyed ones whose sampling races the whole row) on the GPU right behind the drafter's head from the Markov-biased logits over the whole vocabulary, the release's rule; `0` draws them on the host from each row's 64 best logits; drafts only, replies are unchanged; default 1 |
| `TF_DSV41_DSPARK_BLOCK` | DeepSeek-V4.1 on Metal: drafts a DSpark pass makes, 1 to 15, after a window that verified the checkpoint's whole block of 5 (a pass after any other window makes 5); a block under 5 applies to every pass, and the depth may verify up to all of them; drafts only, replies are unchanged; `0` keeps the checkpoint's block; default 7 |
| `TF_DSV41_PREFILL_PIECES` | DeepSeek-V4.1: exactly `0` or `1`, default 0. On CUDA, `1` lets the round planner run prompt pieces and decode rows in one pass; on Metal, `1` runs a prompt a chunk at a time between the other streams' rounds (as `--decode-share` above 0, `TENSORFOLD_PREFILL_ORDER=srpt` or a `TENSORFOLD_PREFILL_POLICY`, `_SHARE` or `_BUDGET` also do), replies taking `--decode-share`, else `TENSORFOLD_PREFILL_SHARE`, else 0.25 of wall time beside it; malformed text stops the start naming it |
| `TF_DSV41_CALIB` | DeepSeek-V4.1 draft pricing, any case: `cached`, `default`, `real`, `measure` or `zig`; unset or blank is `cached` on CUDA, and on Metal reads `TF_DSV41_CALIB_FILE` when set or `default` otherwise. `default` prices rounds from the built-in table with nothing timed; `measure` times a table at start, reports it and prices the rounds from it; `zig` reads this engine's stored table for the same shape, and times and stores one when there is none; `cached` and `real` read a stored table. On Metal a stored table also matches the device and checkpoint that stored it. Checked at startup on Metal; on CUDA a malformed value is refused when the engine opens. [Metal measurements](dsv41-metal-draft-pricing.md) |
| `TF_DSV41_CALIB_DIR` | DeepSeek-V4.1 draft pricing: a directory path without control characters where stored tables are read and written; unset or blank is `~/.cache/tensorfold/dsv41-calib`. Checked at startup on Metal; unchecked on CUDA |
| `TF_DSV41_CALIB_FILE` | DeepSeek-V4.1 draft pricing: a path without control characters to one cost table file (JSON: `verify`, `draft`, optional `slot`) read directly instead of searching the directory, including on Metal when `TF_DSV41_CALIB` is unset or blank; unset or blank searches (or keeps `default` on Metal when `TF_DSV41_CALIB` is unset or blank). Checked at startup on Metal; unchecked on CUDA |
| `TF_DSV41_CALIB_SHAPE` | DeepSeek-V4.1 draft pricing, any case: `raw` or `fit`; naming it times every row from 1 to 16 as CUDA does (slow on Metal); unset times a sparse set on Metal and `raw` on CUDA |
| `TF_DSV41_CALIB_STAT` | DeepSeek-V4.1 draft pricing, any case: `median` or `min`, the statistic over a window's timed runs; unset is `min` on Metal and `median` on CUDA |
| `TF_DSV41_CALIB_REPS` | DeepSeek-V4.1 draft pricing: whole number 1 through 50 of timed runs per entry; unset is 3 passes on Metal (cut by the budget) and 5 on CUDA |
| `TF_DSV41_CALIB_WARM` | DeepSeek-V4.1 draft pricing: whole number 0 through 20 of untimed runs before an entry; unset is 3 on CUDA; ignored by Metal's default method |
| `TF_DSV41_CALIB_RECHECK` | DeepSeek-V4.1 draft pricing: whole row count 0 through 8 retimed after the sweep; unset is 4 on CUDA; ignored by Metal's default method |
| `TF_DSV41_CALIB_CYCLE` | DeepSeek-V4.1 draft pricing: whole row count 0 through 8 timed in interleaved cycles; unset is 4 on CUDA; ignored by Metal's default method |
| `TF_DSV41_CALIB_DEEP_REPS` | DeepSeek-V4.1 draft pricing: whole number 1 through 50 of timed runs for rows past 8; unset is 5 on CUDA; ignored by Metal's default method |
| `TF_DSV41_CALIB_BUDGET_MS` | DeepSeek-V4.1 draft pricing on Metal: whole milliseconds 0 through 600000 after which later passes of the start-up measurement are dropped, each entry keeping its first timed run (0: no limit); unset is 1000. Read and checked on CUDA but not used there |
| `XPU_A8_BITS` | Exactly `4` or `8`, for the XPU quantization study; unset keeps the study's existing setting |
| `LFM3_FUSE` | Whole u32 fusion mask; default 7 |
| `TF_SLIDE_POCKET_KEY` | Exactly `auto`, `reply`, `subject` or `both`; default `auto` |
| `TF_SLIDE_FIT_GIB` | Whole native-size nonnegative GiB count; default 4 |
| `TF_SLIDE_TAKE_OUT_CHECK` | Exactly `0` or `1`, read once when the server starts with `--slide`; `1` takes every kept lesson out again to check that the facts before it come back as they were; default off |
| `TF_SLIDE_TAP_CHECK` | Exactly `0` or `1`; `1` checks a pass's tapped rows against a fresh run's; default off |
| `TF_SLIDE_POCKET_DUMP` | A directory path without control characters, used as given; each lesson's read rows are written there for offline study. Unset or blank writes none |
| `TF_SLIDE_FORGET_FAIL` | Whole u32 fact index, zero off; only checked in a slide-test-hooks build |
| `TF_DSV41_COPY_DRAFTER` | DeepSeek-V4.1 on CUDA: exactly `0` or `1`; enables the lookup planner (regardless of `TF_DSV41_LOOKUP_PLAN`) with continuation-ranked band matching, adaptive doubling (`8 -> 15`), and expected-yield gating; refused when `TF_DSV41_LOOKUP` is blank or off; default off |

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

## Pull behind a proxy

`tensorfold pull` reads the proxy variables the Hugging Face CLI reads: `HTTPS_PROXY` for https requests,
`HTTP_PROXY` for http ones and `ALL_PROXY` for either. The lower-case spelling wins when both are set, as in curl.
Every request goes through the proxy, including the storage host a hub redirect names; each hop takes the proxy for its own scheme.
`NO_PROXY` sends a host direct: `*`, an exact host, a domain suffix (`.example.org`, `*.example.org` or `example.org`)
or any of those with `:port`. Address ranges (CIDR) are not matched.
Only `http://` proxies are supported; pull opens a CONNECT tunnel through them, so the proxy must allow CONNECT to the hub's port.
Credentials in the proxy URL (`http://user:password@host:port`) are sent to the proxy only; messages show `http://host:port`.
Pull waits `TENSORFOLD_PULL_CONNECT_SECONDS` for each connection and `TENSORFOLD_PULL_STALL_SECONDS` for each next byte.
A stalled file resumes from its `.incomplete` bytes with a range request at most 3 times, then pull stops and names the proxy in use.
Windows uses the same proxy routes and timeout settings through native asynchronous sockets.

## Flash Next

Flash Next checks these 39 settings when its engine loads, before any device or checkpoint work: a malformed value
stops the start with the variable, the supplied text and the accepted forms.

| Variable | Accepted explicit value |
| --- | --- |
| `TF_FLASHNEXT_PROMPT_BF16` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_WINDOW_GRAPH` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_LANE_PARTS` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_KEEP_INPASS` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_FIRST_EARLY` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_GPU_TOPK` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_BATCH_GDN_WIDE` | Exactly `0` or `1`; default off. Metal batch segments of 8 or 16 rows use non-replay GDN when enabled |
| `TF_FLASHNEXT_GDN_FUSED` | Exactly `0` or `1`; default on |
| `TF_FLASHNEXT_GDN_CHAIN_TPR` | Exactly `0` or `1`; default off: opt into the two-thread reduction for ordinary recurrence rows below 64 |
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
| `TF_FLASHNEXT_FUSED_SPLIT` | Exactly `0` or `1`; default off. `1` runs CUDA split-K lane calls of 2,048 columns or more on the fused form, with the same bits; Metal and HIP refuse `1` |
| `TF_FLASHNEXT_ATTN_LANES` | Exactly `0` or `1`; default off |
| `TF_FLASHNEXT_WARM_AHEAD` | Exactly `0` or `1`; default off |
| `TF_FLASHNEXT_TABLE_WARM` | Exactly `0` or `1`; default off |
| `TF_FLASHNEXT_GATHER_EARLY` | Exactly `0` or `1`; default off. `1` gathers a shared pass's n-gram rows for every stream in one pool call, so the pass shares one distinct-row read and one set of parked readers; the rows, their order and every reply are the per-stream call's |
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


### Qwen mapped-host KV capacity mode

`TF_QWEN_KV_HOST=1` opts the dense and MoE CUDA hybrids into mapped pinned-host KV pages (#2637).
It requires `TF_QWEN_KV_PAGES` and `TF_QWEN_KV_PAGES_ALL=1`, `--parallel 1`, and a discrete GPU.
State and verify runs remain on the device. Spill, disk snapshots and `TF_QWEN_KV_SHARE` are refused by name.
The default path, with the switch unset, retains its existing plan, allocations and execution.

| Setting | Accepted value | Default | Effect |
|---|---|---|---|
| `TF_QWEN_KV_HOST_GIB` | Positive whole GiB, at most 17179869183 | Unset or blank | Optional upper limit; below a whole KV window including lookahead is refused. Runs are not charged to this cap. |

This is a capacity tradeoff, not a speed claim: attention reads mapped host KV over the device interconnect.
Set `--prompt-cache-gib` explicitly to bound the separate host prompt cache.
Host-on inference equality and performance on a discrete GPU remain device checks, not host-test results.
