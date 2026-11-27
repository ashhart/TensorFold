# Environment variable names

Every environment variable the engine's Zig source reads, where it is read, and what setting it can do.
[Environment validation](environment.md) documents each validated setting's accepted values; this page lists every name,
validated or not, and is what to read when auditing what a build can be told to do from its environment.

## How the list stays complete

`python3 tools/zig/env_names.py` rebuilds the list from the source and fails when this page disagrees with it. It finds
reads by the lookup call, not by the name's spelling, in the non-test sources under `zig/src` (test files, `test`
blocks and comments skipped):

- `getenv` under any receiver (the C library's, `sys.getenv`, a family's own), the test runner's `testing.environ`,
  and `get` on an environment map (a value typed `Environ.Map` or `environment.Process`, or one unwrapped from it);
- calls through a parameter typed as a function from a name to its value, such as `get: *const fn ([:0]const u8) ?[]const u8`;
- the wrappers and validators `tools/zig/env_lookups.py` lists: each reads the variable one argument names, so its
  callers' names are reads too, including calls through a constant that aliases the wrapper function itself.

The name argument must be written out: a string, an array of strings, or a constant naming one. Any other lookup is a
computed one, and the check fails unless `env_lookups.py` lists it, either as a listed wrapper's read of its own name
parameter (the parameter whole, its `.ptr`, or `buf[0..name.len :0]`) or with the names it reads; any other computed
lookup inside a wrapper still fails. A listed array constant must hold only strings, or the check fails. As a
cross-check, every literal shaped like a variable name (capitals, digits and an underscore) that no lookup reads must
be explained in `env_lookups.py` (an error code, a kernel constant, a name set only for a child process). A name
assembled at run time from pieces, or read only by a library the engine loads (NCCL, MCDMA), is not found.

`--write` adds new reads as `untagged` and drops names nothing reads any more, keeping every other row's tags; the check
then fails until each new row is tagged. When the scan itself has errors, `--write` leaves the page as it is. An
allowlist entry nothing matches only warns, so a branch and the main it will be rebased onto can both pass.

## Scope

A row's scope lists the programs whose build reaches a read of it. `zig build -Denv-graph=FILE` writes every compile
step's modules and imports without building anything, and the check follows `@import`s from each step's root, test
blocks counting only for tests. It configures `dist` (with a placeholder version, SDK path and host-only CUDA
archives), so every host sees the same three release servers. It configures a second time with the opt-in backends
(`-Dhip=true`), whose `native-hip` server and tools only that graph has; that graph adds tool and test scopes only:

| Scope | Meaning |
| --- | --- |
| `release` | A `tensorfold-native` server the `dist` archives ship: macOS arm64 and both Linux CUDA builds. |
| `tool` | Another executable: the `tf-*` developer programs and runners, and any server no archive ships. |
| `test` | Only test builds: test steps, the test runner's environment, or a function only a test build option calls (`forgetFail`, behind `-Dslide-test-hooks`). |

A read in a file no step reaches on the host running the check counts as `tool` when the file has `main`, else `test`.
The validation table's startup sweep and the profile report read every row they list; they set a name's scope only when
nothing else reads it. The check requires `release` exactly where a release build reaches a read, which every host sees
alike, and `tool` and `test` wherever the host's own build graph finds them; some tool and test steps are configured
on one host only, so `--write` keeps the `tool` and `test` scopes a row already has.

## Tags

| Tag | Meaning |
| --- | --- |
| `writes-content` | Writes request or reply text, token ids, or model state computed from a request (logits, hidden rows, KV) to files: the value is where, or turns the writer on. |
| `writes` | Writes other files: timings, kernel traces, caches of weights or downloads. |
| `logs-content` | Adds lines to standard error or output that can carry request text or token ids. |
| `logs` | Adds lines to standard error or output with counts, sizes, timings or settings only. |
| `loads` | Opens a library, kernel image, directory, socket or other file the value names, or found under it. |
| `learns` | Turns learning from chat turns on or widens it (with `--slide`). |
| `other` | Changes sizes, limits, schedules, credentials or choices without writing, logging, loading a path or learning. |

Every row was tagged by reading its reads and what the value then does. A line that only refuses a malformed value, or
reports the setting's own value, does not count as logging.

## Variables

| Name | Scope | Tags |
| --- | --- | --- |
| `ALL_PROXY` | release, tool, test | other |
| `ARC_STOP_FILE` | tool, test | loads |
| `BRAVE_SEARCH_API_KEY` | release, tool, test | other |
| `COLUMNS` | release, tool, test | other |
| `COUNT` | tool | other |
| `CUDA_VISIBLE_DEVICES` | release, tool, test | other |
| `DRAMA_LOAD_LIMIT_GB` | release, tool, test | other |
| `DYLD_FALLBACK_LIBRARY_PATH` | release, tool, test | loads |
| `DYLD_LIBRARY_PATH` | release, tool, test | loads |
| `FLASHNEXT_LOAD_LIMIT_GB` | release, tool, test | other |
| `FLASHNEXT_NO_LONE` | release, tool, test | other |
| `FLASHNEXT_NO_RESIDENCY` | release, tool, test | other |
| `FLASHNEXT_SLOT_CONTEXTS` | release, tool, test | other |
| `FLASHNEXT_SLOT_GROW` | release, tool, test | other |
| `FLINT_LOAD_LIMIT_GB` | release, tool, test | other |
| `FN_ANSWER` | tool | other |
| `FN_LAYERS` | tool | other |
| `FN_ONLY` | tool | other |
| `FN_TRACE` | tool | logs |
| `FN_WINDOWS` | tool | other |
| `FZ_ATTN2` | release, tool, test | other |
| `FZ_BATCH_LOG` | release, tool, test | logs |
| `FZ_BATCH_MTP_CHAIN` | release, tool, test | other |
| `FZ_BATCH_MTP_ROWS` | release, tool, test | other |
| `FZ_BATCH_MTP_SERIAL` | release, tool, test | other |
| `FZ_BATCH_UNPRICED` | release, tool, test | other |
| `FZ_BUSY` | release, tool, test | logs |
| `FZ_CALL_LOG` | release, tool, test | logs |
| `FZ_DRAFT_N` | release, tool, test | other |
| `FZ_FEED` | release, tool, test | loads |
| `FZ_GATE` | release, tool, test | other |
| `FZ_GATE_FIX` | release, tool, test | other |
| `FZ_GATE_ROWS` | release, tool, test | other |
| `FZ_GATE_STEPS` | release, tool, test | other |
| `FZ_GDN` | release, tool, test | other |
| `FZ_KEYED_DRAFTS` | release, tool, test | other |
| `FZ_LANE` | release, tool, test | other |
| `FZ_LONE_UNPRICED` | release, tool, test | other |
| `FZ_M3_LANES` | release, tool, test | other |
| `FZ_PAIR_MIN` | release, tool, test | other |
| `FZ_PRICE_BATCHED` | release, tool, test | other |
| `FZ_ROUND_LOG` | release, tool, test | logs |
| `FZ_ROUND_TIMES` | release, tool, test | logs |
| `FZ_SETTLE_SYNC` | release, tool, test | other |
| `FZ_SKIP` | release, tool, test | other |
| `FZ_TREE` | release, tool, test | other |
| `FZ_VERIFY_OVERLAP` | release, tool, test | other |
| `GEMMA_DRAFT_BF16` | release, tool, test | other |
| `GEMMA_DRAFT_TRACE` | release, tool, test | logs-content |
| `GLM_CHUNK` | release, tool, test | other |
| `GLM_COPY` | release, tool, test | other |
| `GLM_CUTS` | release, tool, test | other |
| `GLM_DRAFT_HEAD_Q4` | release, tool, test | other |
| `GLM_DRAFT_VOCAB` | release, tool, test | other |
| `GLM_DRY` | release, tool, test | other |
| `GLM_EHPROJ_NAX` | release, tool, test | other |
| `GLM_EP` | release, tool, test | loads |
| `GLM_EP_DELAY_US` | release, tool, test | other |
| `GLM_EP_SPLIT` | release, tool, test | other |
| `GLM_EP_TRACE` | release, tool, test | logs |
| `GLM_LAYERS` | release, tool, test | other |
| `GLM_LOAD_LIMIT_GB` | release, tool, test | other |
| `GLM_MTP` | release, tool, test | other |
| `GLM_NO_WARMUP` | release, tool, test | other |
| `GLM_POOL` | release, tool, test | other |
| `GLM_POOL_TOKENS` | release, tool, test | other |
| `GLM_PROFILE_AT` | release, tool, test | logs |
| `GLM_PROFILE_ROWS` | release, tool, test | logs |
| `GLM_PROFILE_TAPE` | release, tool, test | logs |
| `GLM_PROMPT` | release, tool, test | other |
| `GLM_RANKS` | release, tool, test | logs |
| `GLM_RESIDENCY` | release, tool, test | other |
| `GLM_ROUND_PROF` | release, tool, test | logs |
| `GLM_ROUTE` | release, tool, test | other |
| `GLM_SAMPLING` | release, tool, test | other |
| `GLM_SPARSE_NAX` | release, tool, test | other |
| `GLM_TP` | release, tool, test | other |
| `GLM_WINDOWS` | release, tool, test | logs |
| `HF_ENDPOINT` | release, tool, test | other |
| `HF_HOME` | release, tool, test | writes, loads |
| `HF_HUB_CACHE` | release, tool, test | writes, loads |
| `HF_TOKEN` | release, tool, test | other |
| `HF_TOKEN_PATH` | release, tool, test | loads |
| `HIP_VISIBLE_DEVICES` | release, tool, test | other |
| `HOME` | release, tool, test | writes-content, loads |
| `HTTPS_PROXY` | release, tool, test | other |
| `HTTP_PROXY` | release, tool, test | other |
| `HUGGING_FACE_HUB_TOKEN` | release, tool, test | other |
| `INKLING2_LOAD_LIMIT_GB` | release, tool, test | other |
| `LFM3_FUSE` | tool, test | other |
| `LLAMA_LOAD_LIMIT_GB` | release, tool, test | other |
| `MCDMA_RPCD_SOCKET` | release, tool, test | loads |
| `MCDMA_RPC_BOX_DIR` | release, tool, test | writes-content, loads |
| `MCDMA_RPC_LIBRARY` | release, tool, test | loads |
| `MINIMAX_DRAFTER` | release, tool, test | other |
| `MINIMAX_LOAD_LIMIT_GB` | release, tool, test | other |
| `MUSE_CAPTURE` | tool | writes-content |
| `MUSE_CONTEXT` | tool, test | other |
| `MUSE_DRAFTER` | tool, test | loads |
| `MUSE_DRAFT_TAPS` | tool, test | other |
| `MUSE_DUMP_TAPS` | tool | writes-content |
| `MUSE_GRAPHS` | tool | other |
| `MUSE_GREEDY` | tool, test | other |
| `MUSE_LAYERS` | tool, test | other |
| `MUSE_PAIR` | tool | other |
| `MUSE_PROMPT_PLANS` | tool, test | other |
| `MUSE_ROWS` | tool | other |
| `MUSE_SLOTS` | tool | other |
| `MUSE_TEACHER` | tool | other |
| `NCCL_ALGO` | release, tool, test | other |
| `NCCL_COMM_ID` | release, tool, test | other |
| `NCCL_CUMEM_ENABLE` | release, tool, test | other |
| `NCCL_CUMEM_HOST_ENABLE` | release, tool, test | other |
| `NCCL_DEBUG` | release, tool, test | logs |
| `NCCL_DEBUG_FILE` | release, tool, test | writes |
| `NCCL_DEBUG_SUBSYS` | release, tool, test | logs |
| `NCCL_IB_HCA` | release, tool, test | other |
| `NCCL_IB_TC` | release, tool, test | other |
| `NCCL_MAX_NCHANNELS` | release, tool, test | other |
| `NCCL_MIN_NCHANNELS` | release, tool, test | other |
| `NCCL_P2P_DISABLE` | release, tool, test | other |
| `NCCL_P2P_LEVEL` | release, tool, test | other |
| `NCCL_P2P_READ_ENABLE` | release, tool, test | other |
| `NCCL_PROTO` | release, tool, test | other |
| `NCCL_RUNTIME_CONNECT` | release, tool, test | other |
| `NCCL_SHM_DISABLE` | release, tool, test | other |
| `NCCL_SOCKET_IFNAME` | release, tool, test | other |
| `NO_PROXY` | release, tool, test | other |
| `PATH` | release, tool, test | loads |
| `PLE_LAYERS` | tool, test | logs |
| `SPARK_LOAD_LIMIT_GB` | release, tool, test | other |
| `STEP5_LOAD_LIMIT_GB` | release, tool, test | other |
| `TENSORFOLD_AOTRITON` | release, tool, test | loads |
| `TENSORFOLD_API_KEY` | release, tool, test | other |
| `TENSORFOLD_BENCHMARK_TOKEN` | release, tool, test | other |
| `TENSORFOLD_CHECKPOINT_SHA256` | release, tool, test | other |
| `TENSORFOLD_CUDA_KERNELS` | release, tool, test | loads |
| `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` | release, tool, test | other |
| `TENSORFOLD_DEVICE_CHECK` | release, tool | other |
| `TENSORFOLD_DORY_GRAPHS` | release, tool, test | other |
| `TENSORFOLD_DORY_KV_PAGES` | release, tool, test | other |
| `TENSORFOLD_GEMMA4_GRAPHS` | release, tool, test | other |
| `TENSORFOLD_GEMMA4_KV_PAGES` | release, tool, test | other |
| `TENSORFOLD_HIP` | tool, test | loads |
| `TENSORFOLD_HIP_MEMORY_LIMIT_GB` | tool, test | other |
| `TENSORFOLD_KOLIBRI_GRAPHS` | release, tool, test | other |
| `TENSORFOLD_KOLIBRI_KV_PAGES` | release, tool, test | other |
| `TENSORFOLD_LAUNCH_URL` | release, tool, test | other |
| `TENSORFOLD_LFM2_KV_PAGES` | release, tool, test | other |
| `TENSORFOLD_LFM3_KV_PAGES` | release, tool, test | other |
| `TENSORFOLD_LIBPDFIUM` | release, tool, test | loads |
| `TENSORFOLD_LIBTURBOJPEG` | release, tool, test | loads |
| `TENSORFOLD_LIBWEBP` | release, tool, test | loads |
| `TENSORFOLD_LOG_TIME` | release, tool, test | logs |
| `TENSORFOLD_MAX_PDF_PAGES` | release, tool, test | other |
| `TENSORFOLD_MEMORY_PRESSURE` | release, tool, test | other |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | release, tool, test | other |
| `TENSORFOLD_NO_LIVE` | release, tool, test | other |
| `TENSORFOLD_P2P_DIAGNOSTICS` | release, tool, test | logs |
| `TENSORFOLD_P2P_PROBE` | release, tool, test | logs |
| `TENSORFOLD_PARALLEL` | release, tool | other |
| `TENSORFOLD_PHASE_METRICS` | release, tool, test | other |
| `TENSORFOLD_PREFILL_BUDGET` | release, tool, test | other |
| `TENSORFOLD_PREFILL_ORDER` | release, tool, test | other |
| `TENSORFOLD_PREFILL_POLICY` | release, tool, test | other |
| `TENSORFOLD_PREFILL_SHARE` | release, tool, test | other |
| `TENSORFOLD_PREFILL_TOKENS` | release, tool, test | other |
| `TENSORFOLD_PREFIX_WAIT_MS` | release, tool, test | other |
| `TENSORFOLD_PRESSURE_CACHE_PERCENT` | release, tool, test | other |
| `TENSORFOLD_PRIORITY_AGE_MS` | release, tool, test | other |
| `TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS` | release, tool, test | other |
| `TENSORFOLD_PULL_CONNECT_SECONDS` | release, tool, test | other |
| `TENSORFOLD_PULL_STALL_SECONDS` | release, tool, test | other |
| `TENSORFOLD_REQUEST_JOURNAL` | release, tool, test | writes-content |
| `TENSORFOLD_REQUEST_LOG` | release, tool, test | writes-content |
| `TENSORFOLD_REQUEST_LOG_REPLIES` | release, tool, test | writes-content |
| `TENSORFOLD_RESTORE_PREFETCH` | release, tool, test | other |
| `TENSORFOLD_RESTORE_PREFETCH_GIB` | release, tool, test | other |
| `TENSORFOLD_RESTORE_PREFETCH_MS` | release, tool, test | other |
| `TENSORFOLD_ROWS_SYNC` | release, tool, test | other |
| `TENSORFOLD_SEED_SALT` | release, tool, test | other |
| `TENSORFOLD_SPARK25_KV_PAGES` | release, tool, test | other |
| `TENSORFOLD_SPARK_GRAPHS` | release, tool, test | other |
| `TENSORFOLD_TOOL_GRAMMAR` | release, tool, test | other |
| `TENSORFOLD_TOOL_ID_KEY_FILE` | release, tool, test | loads |
| `TENSORFOLD_UNIFIED_HOST_COUNTS` | release, tool, test | other |
| `TENSORFOLD_VIDEO_TOKENS` | test | other |
| `TENSORFOLD_WEB_NO_PROMPT` | release, tool, test | other |
| `TENSORFOLD_WIRE` | release, tool, test | other |
| `TENSORFOLD_XPU_MTP` | tool, test | other |
| `TF_ANE_COLUMNS` | release, tool, test | other |
| `TF_ATTENTION_PV_FP32` | release, tool, test | other |
| `TF_CACHE_EVICT` | release, tool, test | other |
| `TF_CHECKPOINT_REVISION` | release, tool, test | other |
| `TF_COMM_BACKEND` | release, tool, test | other |
| `TF_COOP_SPLIT` | release, tool, test | other |
| `TF_COST_KEY_TRACE` | release, tool, test | logs |
| `TF_CUDA_CARVEOUT` | release, tool, test | other |
| `TF_CUDA_CARVEOUT_MIB` | release, tool, test | other |
| `TF_CUDA_DEVICE` | release, tool, test | other |
| `TF_CUDA_MEMORY_RECEIPTS` | release, tool, test | logs |
| `TF_CUDA_POOL_AFTER_WARMUP` | release, tool, test | other |
| `TF_CUDA_SCHED` | release, tool, test | other |
| `TF_CUDA_SEGMENTS` | release, tool, test | other |
| `TF_CUDA_TEST_ATTN_MT2` | release, tool, test | other |
| `TF_CUDA_TEST_MLP128` | release, tool, test | other |
| `TF_DFLASH_GRAPH` | release, tool, test | other |
| `TF_DFLASH_QL_SPAN` | release, tool, test | other |
| `TF_DORY_COPIES` | release, tool, test | other |
| `TF_DRAFT_NODES` | release, tool, test | other |
| `TF_DRAIN_S` | release, tool, test | other |
| `TF_DRAMA_PREFILL_ROWS` | release, tool, test | other |
| `TF_DRAMA_REFERENCE_CACHE` | release, tool, test | other |
| `TF_DRM_CARD` | release, tool, test | loads |
| `TF_DSPINE_WINDOW` | release, tool, test | other |
| `TF_DSV41_ACCEPT_LOG` | tool, test | writes |
| `TF_DSV41_ADMIT_CACHE_KEEP_GB` | tool, test | other |
| `TF_DSV41_ADMIT_FREE_FLOOR_GIB` | tool, test | other |
| `TF_DSV41_AOT_GUARD` | tool, test | other |
| `TF_DSV41_AOT_TRACE` | tool, test | writes |
| `TF_DSV41_ARENA` | tool, test | other |
| `TF_DSV41_ASSETS` | tool, test | loads |
| `TF_DSV41_ATTN_ROPE` | tool, test | other |
| `TF_DSV41_ATTN_SPLIT` | tool, test | other |
| `TF_DSV41_BATCH_SHORT` | tool, test | other |
| `TF_DSV41_BIAS_VL` | tool, test | other |
| `TF_DSV41_BOOT_GROWTH_GIB` | tool, test | other |
| `TF_DSV41_BOOT_MEASURED` | tool, test | other |
| `TF_DSV41_BOOT_TRACE` | tool, test | logs |
| `TF_DSV41_BRANCHES` | tool, test | other |
| `TF_DSV41_BRANCHES_PRIO` | tool, test | other |
| `TF_DSV41_BRANCHES_ROWS` | tool, test | other |
| `TF_DSV41_CALIB` | tool, test | writes |
| `TF_DSV41_CALIB_BUDGET_MS` | tool, test | other |
| `TF_DSV41_CALIB_CYCLE` | tool, test | other |
| `TF_DSV41_CALIB_DEEP_REPS` | tool, test | other |
| `TF_DSV41_CALIB_DIR` | tool, test | writes, loads |
| `TF_DSV41_CALIB_FILE` | tool, test | loads |
| `TF_DSV41_CALIB_RECHECK` | tool, test | other |
| `TF_DSV41_CALIB_REPS` | tool, test | other |
| `TF_DSV41_CALIB_SHAPE` | tool, test | other |
| `TF_DSV41_CALIB_STAT` | tool, test | other |
| `TF_DSV41_CALIB_WARM` | tool, test | other |
| `TF_DSV41_CODE_ACCEPT` | tool, test | other |
| `TF_DSV41_COEF_LATE` | tool, test | other |
| `TF_DSV41_CONTEXT` | tool, test | other |
| `TF_DSV41_COPY_DRAFTER` | tool, test | other |
| `TF_DSV41_DECODE_TP_LANES` | tool | other |
| `TF_DSV41_DECODE_TP_PREFILL` | tool | other |
| `TF_DSV41_DECODE_TP_PROF_ROWS` | tool, test | other |
| `TF_DSV41_DECODE_TP_TOPP` | tool | other |
| `TF_DSV41_DEPTH` | tool, test | other |
| `TF_DSV41_DEPTH_FAIR` | tool, test | other |
| `TF_DSV41_DEPTH_HOST_MS` | tool, test | other |
| `TF_DSV41_DEPTH_JOINT` | tool, test | other |
| `TF_DSV41_DRAFTS` | tool, test | other |
| `TF_DSV41_DRAFT_BATCH_PROJ` | tool, test | other |
| `TF_DSV41_DRAFT_CAPTURE` | tool, test | other |
| `TF_DSV41_DRAFT_CHAIN` | tool, test | other |
| `TF_DSV41_DRAFT_DEPTH` | tool, test | other |
| `TF_DSV41_DRAFT_GRAPHS` | tool, test | other |
| `TF_DSV41_DRAFT_GRAPHS_MAX` | tool, test | other |
| `TF_DSV41_DRAFT_OVERLAP` | tool, test | other |
| `TF_DSV41_DRAFT_PLAIN_STREAMS` | tool, test | other |
| `TF_DSV41_DRAFT_SKIP` | tool, test | other |
| `TF_DSV41_DRAFT_TIMED` | tool, test | other |
| `TF_DSV41_DSPARK_BLOCK` | tool, test | other |
| `TF_DSV41_ENGRAM_AIO` | tool, test | other |
| `TF_DSV41_ENGRAM_CACHE` | tool, test | other |
| `TF_DSV41_ENGRAM_DIR` | tool, test | loads |
| `TF_DSV41_ENGRAM_GATE` | tool, test | other |
| `TF_DSV41_ENGRAM_GATE_ROWS` | tool, test | other |
| `TF_DSV41_ENGRAM_GATE_TIMEOUT_S` | tool, test | other |
| `TF_DSV41_ENGRAM_POLL` | tool, test | other |
| `TF_DSV41_ENGRAM_PREFETCH` | tool, test | other |
| `TF_DSV41_ENGRAM_QD` | tool, test | other |
| `TF_DSV41_ENGRAM_READERS` | tool, test | other |
| `TF_DSV41_ENGRAM_URING` | tool, test | other |
| `TF_DSV41_ENGRAM_VERIFY` | tool, test | other |
| `TF_DSV41_ENGRAM_WAIT_S` | tool, test | other |
| `TF_DSV41_ENGRAM_WARM` | tool, test | other |
| `TF_DSV41_EXPERT_TOPP` | tool, test | other |
| `TF_DSV41_EX_STACKS` | tool, test | other |
| `TF_DSV41_FLOAT_LINEAR` | tool, test | other |
| `TF_DSV41_FLOOR_GIB` | tool, test | other |
| `TF_DSV41_FLOOR_HARD_GIB` | tool, test | other |
| `TF_DSV41_FLOOR_PF_CHUNK_GIB` | tool, test | other |
| `TF_DSV41_FLOOR_PF_HOST_GIB` | tool, test | other |
| `TF_DSV41_FLOOR_PF_INDEX_GIB` | tool, test | other |
| `TF_DSV41_FLOOR_PF_OTHER_GIB` | tool, test | other |
| `TF_DSV41_FLOOR_PRICE` | tool, test | other |
| `TF_DSV41_FLOOR_REFUSE_S` | tool, test | other |
| `TF_DSV41_FORMS_INPLACE` | tool, test | other |
| `TF_DSV41_GEN_STREAMS` | tool, test | other |
| `TF_DSV41_GM_GU2` | tool, test | other |
| `TF_DSV41_GM_V2` | tool, test | other |
| `TF_DSV41_GM_V3` | release, tool, test | other |
| `TF_DSV41_GRAMMAR_TOKENIZER` | tool, test | loads |
| `TF_DSV41_GRAPHS` | tool, test | other |
| `TF_DSV41_GRAPHS_MAX` | tool, test | other |
| `TF_DSV41_GRAPH_BUCKET` | tool, test | other |
| `TF_DSV41_GRAPH_BUCKET_GROW` | tool, test | other |
| `TF_DSV41_GRAPH_COST_MIB` | tool, test | other |
| `TF_DSV41_GRAPH_FLOOR` | tool, test | other |
| `TF_DSV41_GRAPH_FLOOR_GIB` | tool, test | other |
| `TF_DSV41_GRAPH_LOG` | tool, test | logs |
| `TF_DSV41_GRAPH_ROWS_MAX` | tool, test | other |
| `TF_DSV41_GREEDY_GPU` | tool, test | other |
| `TF_DSV41_IMAGES` | tool, test | other |
| `TF_DSV41_INDEX_BOUND` | tool, test | other |
| `TF_DSV41_INDEX_BUDGET_MIB` | tool, test | other |
| `TF_DSV41_JOIN_DRAFTS` | tool, test | other |
| `TF_DSV41_KV_CHECK_PARK` | tool | other |
| `TF_DSV41_KV_CHECK_POOL_TOKENS` | tool | other |
| `TF_DSV41_KV_CHECK_TOPP` | tool | other |
| `TF_DSV41_KV_NORM_STORE` | tool, test | other |
| `TF_DSV41_KV_SPLIT` | tool, test | other |
| `TF_DSV41_KV_SPLIT_COMPACT` | tool, test | other |
| `TF_DSV41_KV_SPLIT_DEDUP` | tool, test | other |
| `TF_DSV41_KV_SPLIT_EARLY` | tool, test | other |
| `TF_DSV41_KV_SPLIT_HISTORY` | tool, test | other |
| `TF_DSV41_KV_SPLIT_UNION_MIB` | tool, test | other |
| `TF_DSV41_L2PF` | tool, test | other |
| `TF_DSV41_L2PF_AT` | tool, test | other |
| `TF_DSV41_L2PF_CHUNK_KB` | tool, test | other |
| `TF_DSV41_L2PF_DELAY_US` | tool, test | other |
| `TF_DSV41_L2PF_MB` | tool, test | other |
| `TF_DSV41_L2PF_PACE_CTAS` | tool, test | other |
| `TF_DSV41_L2PF_PACE_GBPS` | tool, test | other |
| `TF_DSV41_L2PF_PACE_SITES` | tool, test | other |
| `TF_DSV41_L2PF_ROWS` | tool, test | other |
| `TF_DSV41_L2PF_SITES` | tool, test | other |
| `TF_DSV41_LAYERS` | tool, test | other |
| `TF_DSV41_LIN2` | tool, test | other |
| `TF_DSV41_LIN2_CLUSTER` | tool, test | other |
| `TF_DSV41_LIN2_PF` | tool, test | other |
| `TF_DSV41_LIN2_WK` | tool, test | other |
| `TF_DSV41_LONG_PROMPT` | tool, test | other |
| `TF_DSV41_LOOKUP` | tool, test | other |
| `TF_DSV41_LOOKUP_MIN` | tool, test | other |
| `TF_DSV41_LOOKUP_PLAN` | tool, test | other |
| `TF_DSV41_MARKOV_GPU` | tool, test | other |
| `TF_DSV41_MAX_ROWS` | tool, test | other |
| `TF_DSV41_MHC_DEFER` | tool, test | other |
| `TF_DSV41_MHC_DEFER_AT` | tool, test | other |
| `TF_DSV41_MHC_DEFER_AT_ROWS` | tool, test | other |
| `TF_DSV41_MHC_PFDEC` | tool, test | other |
| `TF_DSV41_MHC_SITE_ROWS` | tool, test | other |
| `TF_DSV41_MHC_TAIL` | tool, test | other |
| `TF_DSV41_MOE_BF16` | tool, test | other |
| `TF_DSV41_NEEDS_POOL` | tool, test | other |
| `TF_DSV41_NUCLEUS` | tool, test | other |
| `TF_DSV41_OWN_PREFILL` | tool, test | other |
| `TF_DSV41_PF_4K` | tool, test | other |
| `TF_DSV41_PF_4K_MEMORY_OK` | tool, test | other |
| `TF_DSV41_PF_DENSE` | tool, test | other |
| `TF_DSV41_PF_DENSE_CFG` | tool, test | other |
| `TF_DSV41_PF_DENSE_TABLE` | tool, test | loads |
| `TF_DSV41_PF_OVERLAP` | tool, test | other |
| `TF_DSV41_PF_OVERLAP_SITE` | tool, test | other |
| `TF_DSV41_PF_TBO` | tool, test | other |
| `TF_DSV41_PF_TBO_GM_SMS` | tool, test | other |
| `TF_DSV41_PF_WS_FLOOR_GIB` | tool, test | other |
| `TF_DSV41_PF_WS_SHARE` | tool, test | other |
| `TF_DSV41_PHASES` | tool, test | logs |
| `TF_DSV41_PHASES_OUT` | tool, test | writes |
| `TF_DSV41_PIECES_FLOOR` | tool, test | other |
| `TF_DSV41_PIECE_RUNS` | tool, test | other |
| `TF_DSV41_POOL_TOKENS` | tool, test | other |
| `TF_DSV41_POST_BOOT_GIB` | tool, test | other |
| `TF_DSV41_PREFETCH_AHEAD` | tool, test | other |
| `TF_DSV41_PREFILL` | tool, test | other |
| `TF_DSV41_PREFILL_ADAPT` | tool, test | other |
| `TF_DSV41_PREFILL_ADAPT_GIB` | tool, test | other |
| `TF_DSV41_PREFILL_ADAPT_MIN` | tool, test | other |
| `TF_DSV41_PREFILL_ADAPT_UP_GIB` | tool, test | other |
| `TF_DSV41_PREFILL_CHUNK` | tool, test | other |
| `TF_DSV41_PREFILL_CONCURRENT` | tool, test | other |
| `TF_DSV41_PREFILL_PIECES` | tool, test | other |
| `TF_DSV41_PREFILL_ROWS` | tool, test | other |
| `TF_DSV41_PREFILL_SHARE` | tool, test | other |
| `TF_DSV41_PREPARED` | tool, test | loads |
| `TF_DSV41_PREPARED_VERIFY` | tool, test | other |
| `TF_DSV41_PREPARED_WRITE` | tool, test | writes |
| `TF_DSV41_PROFILE` | tool, test | writes |
| `TF_DSV41_PROMPT_PROF` | tool, test | logs |
| `TF_DSV41_PROMPT_TAIL` | tool, test | other |
| `TF_DSV41_PRUNE_KIT` | tool, test | other |
| `TF_DSV41_R1` | tool, test | other |
| `TF_DSV41_REPLAY_RUNS` | tool, test | other |
| `TF_DSV41_RG_PRUNE` | tool, test | other |
| `TF_DSV41_RMS_FOLD` | tool, test | other |
| `TF_DSV41_ROUND_GRAPH` | tool, test | other |
| `TF_DSV41_ROUND_GRAPH_HEAD` | tool, test | other |
| `TF_DSV41_ROUND_PROF` | tool, test | logs |
| `TF_DSV41_ROUTER_GROUP_ROT` | tool, test | other |
| `TF_DSV41_ROWS_CAP` | tool, test | other |
| `TF_DSV41_SAMPLING` | tool, test | other |
| `TF_DSV41_SAMP_BATCH` | tool, test | other |
| `TF_DSV41_SAMP_STATS` | tool, test | other |
| `TF_DSV41_SCHEMA_PROMPT` | release, tool, test | other |
| `TF_DSV41_SERVE_DIP_GIB` | tool, test | other |
| `TF_DSV41_SESSIONS` | tool, test | other |
| `TF_DSV41_SESSION_DELTA` | tool, test | other |
| `TF_DSV41_SESSION_DISK` | tool, test | writes-content |
| `TF_DSV41_SESSION_DISK_GIB` | tool, test | other |
| `TF_DSV41_SESSION_DISK_MIN` | tool, test | other |
| `TF_DSV41_SESSION_RAM_MIB` | tool, test | other |
| `TF_DSV41_SESSION_RESIDENT` | tool, test | other |
| `TF_DSV41_SESSION_RESIDENT_GIB` | tool, test | other |
| `TF_DSV41_SESSION_SKIP_COVERED` | tool, test | other |
| `TF_DSV41_SESSION_SLOTS` | tool, test | other |
| `TF_DSV41_SESSION_SLOTS_FRESH` | tool | loads |
| `TF_DSV41_SESSION_SLOTS_MSG` | tool | other |
| `TF_DSV41_SESSION_SLOTS_SEED` | tool | other |
| `TF_DSV41_SESSION_SLOTS_STEPS` | tool | other |
| `TF_DSV41_SESSION_SLOTS_TIERS` | tool | logs |
| `TF_DSV41_SESSION_SLOTS_TURNS` | tool | other |
| `TF_DSV41_SESSION_STAGING_MIB` | tool, test | other |
| `TF_DSV41_SLOTS` | tool, test | other |
| `TF_DSV41_SLOT_DRAFTS` | tool, test | other |
| `TF_DSV41_SLOT_DRAFT_GROUP` | tool, test | other |
| `TF_DSV41_SMALL` | tool, test | other |
| `TF_DSV41_SPEC_DRAFT` | tool, test | other |
| `TF_DSV41_SPEC_NUCLEUS` | tool, test | other |
| `TF_DSV41_STALL_EXIT_S` | tool, test | other |
| `TF_DSV41_STREAM_LAZY` | tool, test | other |
| `TF_DSV41_STREAM_RB` | tool, test | other |
| `TF_DSV41_SYNC_EACH` | tool, test | other |
| `TF_DSV41_TAIL_JOIN` | tool, test | other |
| `TF_DSV41_TOOL_GRAMMAR` | release, tool, test | other |
| `TF_DSV41_TRACE_TOKENS` | tool, test | writes-content |
| `TF_DSV41_TREE` | tool, test | other |
| `TF_DSV41_TREE_DUP_MS` | tool, test | other |
| `TF_DSV41_TREE_PC` | tool, test | other |
| `TF_DSV41_TREE_ROWS` | tool, test | other |
| `TF_DSV41_WIN_PROF` | tool, test | logs |
| `TF_DSV41_X3LD_EPI` | tool, test | other |
| `TF_EXL3_PDL` | release, tool, test | other |
| `TF_EXPERIMENTAL` | tool, test | other |
| `TF_FLASHNEXT_ATTN_LANES` | release, tool, test | other |
| `TF_FLASHNEXT_BATCH_DRAWS` | release, tool, test | other |
| `TF_FLASHNEXT_BATCH_GDN_WIDE` | release, tool, test | other |
| `TF_FLASHNEXT_BATCH_HEADS` | release, tool, test | other |
| `TF_FLASHNEXT_CHUNK` | release, tool, test | other |
| `TF_FLASHNEXT_CHUNK_SOLO` | release, tool, test | other |
| `TF_FLASHNEXT_CONFIDENCE` | release, tool, test | other |
| `TF_FLASHNEXT_DECODE_GRAPH` | release, tool, test | other |
| `TF_FLASHNEXT_DEPTH_LOG` | release, tool, test | logs |
| `TF_FLASHNEXT_DRAFTS` | release, tool, test | other |
| `TF_FLASHNEXT_DRAFTS_GREEDY` | release, tool, test | other |
| `TF_FLASHNEXT_DRAFT_HEAD` | release, tool, test | other |
| `TF_FLASHNEXT_DUMP` | release, tool, test | loads |
| `TF_FLASHNEXT_FACT_ROWS` | release, tool, test | other |
| `TF_FLASHNEXT_FIRST_EARLY` | release, tool, test | other |
| `TF_FLASHNEXT_FUSED_SPLIT` | release, tool, test | other |
| `TF_FLASHNEXT_GATHER_EARLY` | release, tool, test | other |
| `TF_FLASHNEXT_GDN_CHAIN_TPR` | release, tool, test | other |
| `TF_FLASHNEXT_GDN_FUSED` | release, tool, test | other |
| `TF_FLASHNEXT_GDN_LANES` | release, tool, test | other |
| `TF_FLASHNEXT_GPU_TOPK` | release, tool, test | other |
| `TF_FLASHNEXT_GRAPH_TAIL` | release, tool, test | other |
| `TF_FLASHNEXT_GREEDY_WIDE` | release, tool, test | other |
| `TF_FLASHNEXT_GROUPED_ROWS` | release, tool, test | other |
| `TF_FLASHNEXT_HC_FUSE` | release, tool, test | other |
| `TF_FLASHNEXT_KEEP_INPASS` | release, tool, test | other |
| `TF_FLASHNEXT_KEEP_LATE` | release, tool, test | other |
| `TF_FLASHNEXT_KV_PAGES` | release, tool, test | other |
| `TF_FLASHNEXT_KV_PAGES_ALL` | release, tool, test | other |
| `TF_FLASHNEXT_KV_PAGE_TEST` | release, tool, test | other |
| `TF_FLASHNEXT_LANE_PARTS` | release, tool, test | other |
| `TF_FLASHNEXT_LOAD_THREADS` | release, tool, test | other |
| `TF_FLASHNEXT_LOGITS` | release, tool, test | writes-content |
| `TF_FLASHNEXT_OWNED` | release, tool, test | other |
| `TF_FLASHNEXT_PLE_IO` | release, tool, test | other |
| `TF_FLASHNEXT_PREFILL_MTP_LEAN` | release, tool, test | other |
| `TF_FLASHNEXT_PROFILE` | release, tool, test | logs |
| `TF_FLASHNEXT_PROMPT` | release, tool, test | other |
| `TF_FLASHNEXT_PROMPT_BF16` | release, tool, test | other |
| `TF_FLASHNEXT_PROMPT_EXPERT_TILE` | release, tool, test | other |
| `TF_FLASHNEXT_READOUT_FUSED` | release, tool, test | other |
| `TF_FLASHNEXT_READOUT_ROWS` | release, tool, test | other |
| `TF_FLASHNEXT_REPEAT` | release, tool, test | other |
| `TF_FLASHNEXT_REPEAT_CACHE` | release, tool, test | other |
| `TF_FLASHNEXT_SAMPLED_HEADS` | release, tool, test | other |
| `TF_FLASHNEXT_SCALE_INDEX` | release, tool, test | other |
| `TF_FLASHNEXT_SEED` | release, tool, test | other |
| `TF_FLASHNEXT_SHARD_HEAD` | release, tool, test | other |
| `TF_FLASHNEXT_SHARED_SIDE` | release, tool, test | other |
| `TF_FLASHNEXT_SPLIT_DENSE` | release, tool, test | other |
| `TF_FLASHNEXT_TABLE_WARM` | release, tool, test | other |
| `TF_FLASHNEXT_TEMPERATURE` | release, tool, test | other |
| `TF_FLASHNEXT_TOP_K` | release, tool, test | other |
| `TF_FLASHNEXT_TOP_P` | release, tool, test | other |
| `TF_FLASHNEXT_TP` | release, tool, test | loads |
| `TF_FLASHNEXT_VERIFY_HOST_LOGITS` | release, tool, test | other |
| `TF_FLASHNEXT_WARM_AHEAD` | release, tool, test | other |
| `TF_FLASHNEXT_WINDOW_CHECK` | release, tool, test | other |
| `TF_FLASHNEXT_WINDOW_GRAPH` | release, tool, test | other |
| `TF_FLASHNEXT_WINDOW_GRAPH_PARITY` | release, tool, test | other |
| `TF_FLASHNEXT_WINDOW_ROWS` | release, tool, test | other |
| `TF_FN_GX_GATEUP_MB2` | release, tool, test | other |
| `TF_FN_GX_SHARED` | release, tool, test | other |
| `TF_GEMMA4_ARGMAX_GRAPH` | release, tool, test | other |
| `TF_GEMMA4_CHUNK_SOLO` | release, tool, test | other |
| `TF_GEMMA4_COPIES` | release, tool, test | other |
| `TF_GLM_CHUNK_SOLO` | release, tool, test | other |
| `TF_GLM_CLEAR_THINKING` | release, tool, test | other |
| `TF_GLM_CONTEXT_CEILING` | release, tool, test | other |
| `TF_GLM_COPIES` | release, tool, test | other |
| `TF_GLM_DENSE_BITS` | release, tool, test | other |
| `TF_GLM_DFLASH2_LOG` | release, tool, test | logs-content |
| `TF_GLM_DFLASH2_PROF` | release, tool, test | logs |
| `TF_GLM_DRAFT_CONFIDENCE` | release, tool, test | other |
| `TF_GLM_DRAFT_SPLIT` | release, tool, test | other |
| `TF_GLM_DSCORES` | release, tool, test | other |
| `TF_GLM_DSEL` | release, tool, test | other |
| `TF_GLM_EXPERT_FORM` | release, tool, test | other |
| `TF_GLM_HEAD_BITS` | release, tool, test | other |
| `TF_GLM_HOST_CACHE` | release, tool, test | other |
| `TF_GLM_INDEX_CUDA` | release, tool, test | other |
| `TF_GLM_INDEX_SPLIT` | release, tool, test | other |
| `TF_GLM_KDA_DECODE_WIDE` | release, tool, test | other |
| `TF_GLM_KDA_PAD` | release, tool, test | other |
| `TF_GLM_KEEP_SEQS` | release, tool, test | other |
| `TF_GLM_KV_HOST` | release, tool, test | other |
| `TF_GLM_KV_HOST_GIB` | release, tool, test | other |
| `TF_GLM_KV_HOST_SHARED` | release, tool, test | other |
| `TF_GLM_KV_LIVE_HALF` | release, tool, test | other |
| `TF_GLM_KV_SHARE` | release, tool, test | other |
| `TF_GLM_L2PF` | release, tool, test | other |
| `TF_GLM_L2PF_CTAS` | release, tool, test | other |
| `TF_GLM_L2PF_MB` | release, tool, test | other |
| `TF_GLM_L2PF_PACE_GBPS` | release, tool, test | other |
| `TF_GLM_LAYERS` | release, tool, test | other |
| `TF_GLM_LINK_WAIT` | release, tool, test | other |
| `TF_GLM_LOAD_PACK` | release, tool, test | other |
| `TF_GLM_LOAD_PREFETCH` | release, tool, test | other |
| `TF_GLM_MM` | release, tool, test | other |
| `TF_GLM_MOE_DIRECT` | release, tool, test | other |
| `TF_GLM_OVERLAP_PIECES` | release, tool, test | other |
| `TF_GLM_PAIR_EXPERTS` | release, tool, test | other |
| `TF_GLM_PDL` | release, tool, test | other |
| `TF_GLM_PREFILL_REDUCE` | release, tool, test | other |
| `TF_GLM_PROF` | release, tool, test | logs |
| `TF_GLM_PROMPT_K` | release, tool, test | other |
| `TF_GLM_PROMPT_PAIR` | release, tool, test | other |
| `TF_GLM_PROMPT_STEPS` | release, tool, test | other |
| `TF_GLM_PROMPT_TAIL` | release, tool, test | other |
| `TF_GLM_RANK` | release, tool, test | other |
| `TF_GLM_REQUEST_STARTS` | release, tool, test | other |
| `TF_GLM_ROCE_MULTI` | release, tool, test | other |
| `TF_GLM_SHARED` | release, tool, test | other |
| `TF_GLM_VISION_FUSED` | release, tool, test | other |
| `TF_GLM_VISION_HOST` | release, tool, test | other |
| `TF_GLM_WIDE_EXPERT_TILE` | release, tool, test | other |
| `TF_GLM_WORLD` | release, tool, test | other |
| `TF_GRAPH_CACHE` | release, tool, test | other |
| `TF_GRAPH_LOG` | release, tool, test | logs |
| `TF_HEAD_GEOMETRY` | release, tool, test | other |
| `TF_HEAT_HIGH` | release, tool, test | other |
| `TF_HEAT_LOW` | release, tool, test | other |
| `TF_HEAT_ROOT` | release, tool, test | loads |
| `TF_HY4_DRAFT_BITS` | release, tool, test | other |
| `TF_HY4_ENGRAM` | release, tool, test | other |
| `TF_HY4_MEM_FLOOR_GIB` | tool, test | other |
| `TF_IMPRINT_VERIFY` | release, tool, test | other |
| `TF_INKLING2_DRAFT_HEAD_Q4` | release, tool, test | other |
| `TF_INKLING2_EXL3_DIR` | release, tool, test | loads |
| `TF_KOLIBRI_COPIES` | release, tool, test | other |
| `TF_KV_PAGES` | release, tool, test | other |
| `TF_KV_POISON` | release, tool, test | other |
| `TF_KV_SHARE` | release, tool, test | other |
| `TF_LANE_TRACE` | release, tool, test | writes |
| `TF_LFM3_METAL_WARM_TURNS` | release, tool, test | other |
| `TF_LOAD_DIRECT` | release, tool, test | other |
| `TF_LOAD_PREFETCH` | release, tool, test | other |
| `TF_METAL4` | tool, test | other |
| `TF_METAL4_TIMELINE` | release, tool, test | logs |
| `TF_MINIMAX_COPIES` | release, tool, test | other |
| `TF_MINIMAX_GRAPHS` | release, tool, test | other |
| `TF_MINIMAX_RANK` | release, tool, test | other |
| `TF_MINIMAX_WORLD` | release, tool, test | other |
| `TF_MTP_BATCH` | release, tool, test | other |
| `TF_MTP_BITS` | release, tool, test | other |
| `TF_MTP_DRAFTS` | release, tool, test | other |
| `TF_MTP_PLAIN_GUARD` | release, tool, test | other |
| `TF_MTP_TREES` | release, tool, test | other |
| `TF_NCCL_LIB` | release, tool, test | loads |
| `TF_NEMOTRON_METAL_KV_PAGES` | release, tool, test | other |
| `TF_NEMOTRON_RANK` | release, tool, test | other |
| `TF_NEMOTRON_TRACE` | tool, test | logs-content |
| `TF_NEMOTRON_WIDE_ROUNDS` | release, tool, test | other |
| `TF_NEMOTRON_WORLD` | release, tool, test | other |
| `TF_NO_TREES` | release, tool, test | other |
| `TF_NVFP4_DECODE_WARPS` | release, tool, test | other |
| `TF_POLICY` | tool, test | other |
| `TF_PROMPT_INTERMEDIATE` | release, tool, test | other |
| `TF_PROMPT_ROWS` | release, tool, test | other |
| `TF_Q27_COPY_ROWS` | release, tool, test | other |
| `TF_Q27_DFLASH_ABSORB_BATCH` | release, tool, test | other |
| `TF_Q27_DFLASH_ADAPTIVE_BLOCK` | release, tool, test | other |
| `TF_Q27_DRAFT_SLOTS` | release, tool, test | other |
| `TF_Q27_GEMM` | release, tool, test | other |
| `TF_Q27_IDLE_RELEASE` | release, tool, test | writes |
| `TF_Q27_KV_GROW` | release, tool, test | other |
| `TF_Q27_MTP_GPU_PICK` | release, tool, test | other |
| `TF_Q27_PASS_PROFILE` | release, tool, test | logs |
| `TF_Q27_POCKET_OFF` | release, tool, test | other |
| `TF_Q27_PROFILE` | release, tool, test | logs |
| `TF_Q27_TAIL` | release, tool, test | other |
| `TF_Q35_DFLASH_ABSORB_BATCH` | release, tool, test | other |
| `TF_Q35_DFLASH_TRACE` | release, tool, test | other |
| `TF_Q35_GROUP_PROJ` | release, tool, test | other |
| `TF_QMMF_HALF_BELOW` | release, tool, test | other |
| `TF_QMMF_HALF_BN` | release, tool, test | other |
| `TF_QWEN27_COPIES` | release, tool, test | other |
| `TF_QWEN27_FIT_FOOTPRINT_BYTES` | release, tool, test | other |
| `TF_QWEN27_FIT_RAM_BYTES` | release, tool, test | other |
| `TF_QWEN27_JOIN` | release, tool, test | other |
| `TF_QWEN27_LANES` | release, tool, test | other |
| `TF_QWEN27_MESSAGE_STARTS` | release, tool, test | other |
| `TF_QWEN27_METAL_KV_PAGES` | release, tool, test | other |
| `TF_QWEN27_SHARED_PREFIX` | release, tool, test | other |
| `TF_QWEN27_SLOT_RESUME` | release, tool, test | other |
| `TF_QWEN27_TIED_HEAD` | release, tool, test | other |
| `TF_QWEN27_WINDOW` | release, tool, test | other |
| `TF_QWEN27_WORKING_SET_BYTES` | release, tool, test | other |
| `TF_QWEN35_ATTN_ALL_ROWS` | release, tool, test | other |
| `TF_QWEN35_CHUNK` | release, tool, test | other |
| `TF_QWEN35_CHUNK_SOLO` | release, tool, test | other |
| `TF_QWEN35_METAL_KV_PAGES` | release, tool, test | other |
| `TF_QWEN_KEEP_LATE` | release, tool, test | other |
| `TF_QWEN_KV_HOST` | release, tool, test | other |
| `TF_QWEN_KV_HOST_GIB` | release, tool, test | other |
| `TF_QWEN_KV_PAGES` | release, tool, test | other |
| `TF_QWEN_KV_PAGES_ALL` | release, tool, test | other |
| `TF_QWEN_KV_PAGE_TEST` | release, tool, test | other |
| `TF_QWEN_KV_SHARE` | release, tool, test | other |
| `TF_QWEN_KV_SNAPSHOT_GPU` | release, tool, test | other |
| `TF_QWEN_PDL` | release, tool, test | other |
| `TF_QWEN_POOL_LOG` | release, tool, test | logs |
| `TF_QWEN_REQUEST_STARTS` | release, tool, test | other |
| `TF_ROCE` | release, tool, test | other |
| `TF_ROCE_GID_INDEX` | release, tool, test | other |
| `TF_ROCE_HCA` | release, tool, test | other |
| `TF_ROCE_MAX_KB` | release, tool, test | other |
| `TF_ROUND_LOG` | release, tool, test | logs |
| `TF_SHARED_PRICE_LOG` | release, tool, test | logs |
| `TF_SLIDE_AUTO` | release, tool, test | learns |
| `TF_SLIDE_AUTO_ALL` | release, tool, test | learns |
| `TF_SLIDE_EDIT` | release, tool, test | other |
| `TF_SLIDE_FAST_CUES` | release, tool, test | other |
| `TF_SLIDE_FIT_GIB` | release, tool, test | other |
| `TF_SLIDE_FIT_REDUCE` | release, tool, test | other |
| `TF_SLIDE_FORGET_FAIL` | test | other |
| `TF_SLIDE_LEGACY` | release, tool, test | other |
| `TF_SLIDE_PLACEMENT` | release, tool, test | other |
| `TF_SLIDE_POCKET_DUMP` | release, tool, test | writes-content |
| `TF_SLIDE_POCKET_GROUP` | release, tool, test | other |
| `TF_SLIDE_POCKET_KEEPS` | release, tool, test | other |
| `TF_SLIDE_POCKET_KEY` | release, tool, test | other |
| `TF_SLIDE_POCKET_LEADS` | release, tool, test | other |
| `TF_SLIDE_POCKET_REPLAY` | release, tool, test | other |
| `TF_SLIDE_POCKET_SAME` | release, tool, test | other |
| `TF_SLIDE_POCKET_VALUES` | release, tool, test | other |
| `TF_SLIDE_PROBE` | release, tool, test | other |
| `TF_SLIDE_ROUTED` | release, tool, test | other |
| `TF_SLIDE_TAKE_OUT_CHECK` | release, tool, test | other |
| `TF_SLIDE_TAP_CHECK` | release, tool, test | logs |
| `TF_SLIDE_TRACE` | release, tool, test | logs-content |
| `TF_SPARK_COPIES` | release, tool, test | other |
| `TF_SSE_KEEPALIVE_S` | release, tool, test | other |
| `TF_STAGE_PEERS` | release, tool, test | other |
| `TF_STAGE_RANK` | release, tool, test | other |
| `TF_STAGE_SECRET_FILE` | release, tool, test | loads |
| `TF_STAGE_TIMEOUT_S` | release, tool, test | other |
| `TF_STAGE_WORLD` | release, tool, test | other |
| `TF_STARTUP_RECEIPT` | release, tool | logs |
| `TF_TEST_RECEIPTS` | test | logs |
| `TF_TIMING` | release, tool, test | logs |
| `TF_TP_ALLOW_HOST_ONLY` | tool | other |
| `TF_TP_DEVICE` | release, tool, test | other |
| `TF_TP_FAILFAST` | release, tool, test | other |
| `TF_TP_FAILFAST_GRACE_S` | release, tool, test | other |
| `TF_TP_KERNEL_IMAGE` | release, tool, test | loads |
| `TF_TP_LINK` | release, tool, test | other |
| `TF_TP_LOCAL` | release, tool, test | other |
| `TF_TP_MAILBOX_MAX_KB` | release, tool, test | other |
| `TF_TP_MAILBOX_TIMEOUT_S` | release, tool, test | other |
| `TF_TP_MASTER` | release, tool, test | other |
| `TF_TP_PEER_MEM_S` | release, tool, test | other |
| `TF_TP_PLAN_LINK` | release, tool, test | other |
| `TF_TP_PLAN_PIN` | release, tool, test | other |
| `TF_TP_PLAN_PORT` | release, tool, test | other |
| `TF_TP_PLAN_SPIN_US` | release, tool, test | other |
| `TF_TP_PLAN_TIMEOUT` | release, tool, test | other |
| `TF_TP_PORT` | release, tool, test | other |
| `TF_TP_RANK` | release, tool, test | other |
| `TF_TP_ROCE_BLOCKS` | release, tool, test | other |
| `TF_TP_ROCE_CHUNKED_KB` | release, tool, test | other |
| `TF_TP_ROCE_CPU` | release, tool, test | other |
| `TF_TP_ROCE_FALLBACK` | release, tool, test | other |
| `TF_TP_ROCE_FAST` | release, tool, test | other |
| `TF_TP_ROCE_HCA` | release, tool, test | other |
| `TF_TP_ROCE_HCAS` | release, tool, test | other |
| `TF_TP_ROCE_POLLERS` | release, tool, test | other |
| `TF_TP_ROCE_RELAY` | release, tool, test | other |
| `TF_TP_ROCE_RELAY_MAX_KB` | release, tool, test | other |
| `TF_TP_ROCE_STAGGER_NS` | release, tool, test | other |
| `TF_TP_ROCE_TC` | release, tool, test | other |
| `TF_TP_ROCE_WIRE` | release, tool, test | other |
| `TF_TP_STATS` | release, tool, test | logs |
| `TF_TP_TRACE` | release, tool, test | logs |
| `TF_TP_WORLD` | release, tool, test | other |
| `TMPDIR` | release, tool, test | writes |
| `USERPROFILE` | release, tool, test | writes-content, loads |
| `WSL_DISTRO_NAME` | release, tool, test | other |
| `XPU_A8` | tool, test | other |
| `XPU_A8_BITS` | tool, test | other |
| `XPU_A8_PROJ` | tool | other |
| `XPU_A8_ROT` | tool, test | other |
| `XPU_A8_SITES` | tool, test | other |
| `XPU_A8_SITES4` | tool, test | other |
| `XPU_A8_ZEBIN` | tool | loads |
| `XPU_BLOCK` | tool, test | other |
| `XPU_CHUNK` | tool, test | other |
| `XPU_CHUNK_SOLO` | tool, test | other |
| `XPU_COPY_MATCH` | tool, test | other |
| `XPU_CTX` | tool | other |
| `XPU_DEVICE_ID` | tool, test | other |
| `XPU_DRAFTER` | tool | loads |
| `XPU_DUMP` | tool, test | writes-content |
| `XPU_DUMP_H` | tool, test | writes-content |
| `XPU_EXL3_PROBE` | tool | other |
| `XPU_FIDELITY` | tool | logs-content |
| `XPU_FIDELITY_CACHE` | tool | writes-content, loads |
| `XPU_FLASHNEXT_WARM` | tool, test | other |
| `XPU_FP64_REF` | tool | logs |
| `XPU_GEMM_ZEBIN` | tool | loads |
| `XPU_KV8` | tool, test | other |
| `XPU_KV_INT8` | tool, test | other |
| `XPU_MTP` | tool, test | other |
| `XPU_MTP_CONF` | tool, test | other |
| `XPU_MTP_DEPTH` | tool, test | other |
| `XPU_MULTI_TEST` | tool | other |
| `XPU_OVERLAP_PROBE` | tool | other |
| `XPU_PREFILL_INT4` | tool, test | other |
| `XPU_PREFILL_INT4_SITES` | tool, test | other |
| `XPU_PREFILL_INT8` | tool, test | other |
| `XPU_PROF` | tool, test | logs |
| `XPU_PROFILE` | tool, test | logs |
| `XPU_ROWS_PROBE` | tool | other |
| `XPU_TIMING` | tool | logs |
| `XPU_TP` | tool, test | other |
| `XPU_TREE` | tool, test | other |
| `XPU_VERIFY_TEST` | tool | other |
| `all_proxy` | release, tool, test | other |
| `http_proxy` | release, tool, test | other |
| `https_proxy` | release, tool, test | other |
| `no_proxy` | release, tool, test | other |
