# Dynamic KV paging for long contexts (KVP)

## Experimental rank-shared BF16 block arena

The default remains a private pager per sequence. `TF_FLASHNEXT_KVPAGE=1` with
`TF_FLASHNEXT_KVPAGE_SHARED=1` opts into a fixed rank-owned allocation. Set
`TF_FLASHNEXT_KVPAGE_SHARED_LANES` to the admitted lane count (default 8, maximum
32). The existing window, rows and slots settings retain their meanings; slots
now form one pool across every lane and attention layer. Ordinary-context TP2
hardware gates pass within the scope below; a repeatable performance improvement
and 200k-context capacity remain unqualified.

Runtime source `ccb32fc3` passes 153 EXL3 host tests and real CUDA sm_120 builds.
Matched private/shared GPU fixtures force sparse paging with 1,024 slots and
compare 54 per-step logit hashes and six logical-state records per rank. They
also check exact restored state, owned-entry payload/ages, recycled and
asymmetric TP generations, and a foreign request's windows/history/continuation.
Each restored graph phase replays 201 prefill and seven decode graphs per rank.
Two concurrent 16,424-token HTTP histories with MTP and graphs pass cold, warm,
branch and rewind against the private-pager reference, reusing 16,407–16,417
tokens. This is not an independent golden-model check or a 200k-capacity gate.

A separate same-binary eight-lane allocation diagnostic on source `0de41d37`
measures 920 MiB lower sampled physical GPU peaks per rank. It uses 1,024 input
and 128 output tokens per request; do not relabel it as the later source or
long-context performance evidence. Full settings, hashes and limitations are in
the [qualification receipt](https://github.com/ashhart/TensorFold/pull/1113#issuecomment-6325790752).

The slot count is global in shared mode, not per sequence: eight lanes with
16,384 slots share 16,384 entries rather than reserving eight private pools.
This shares allocation capacity, not identical-prefix payloads. Reduced total
residency can increase misses; memory savings are not a throughput claim. Fixed
windows are prepaid for every configured lane, so low-concurrency operation can
use more resident memory than private, lazily created pagers. Compare physical
peaks with the same active concurrency, window geometry and expert pools.

Each K/V side is `[all lane/layer windows | shared block slots]`. A pager uses
its own window base and a checked relative offset to the shared slots, retaining
the upstream BF16 attention load path. Canonical RAM, window origins, logical
history, indexer keys and recurrent state remain sequence-owned. Every sequence
must use the engine's common admitted capacity (`context` plus MTP slack, bounded
by the model window). A mismatched capacity or arena geometry is rejected.

Startup allocates and charges all fixed windows, shared payload, global mapping
tables and paging work once, before expert-pool admission. Per-sequence admission
charges the remaining sequence buffers. The lane limit also caps startup room.
Request teardown releases only its generation's tags; it does not free the arena
or enable paged-sequence recycling. New requests retain the global addresses and
receive a fresh lane generation. This does not recycle sequence-owned buffers.

`Engine.forward` and `Engine.round` acquire a rank-wide batch lease containing all
participating sequence generations. Pager calls validate membership, and the
lease covers paging plus every attention consumer, including captured/replayed
rounds. Normal release acknowledges the existing `finishPaged`/collective
completion boundary and introduces no additional device-wide wait. A submitted
failure poisons touched targets and retains the lease until draining completes.

Private snapshots retain their exact version 1 contract. Shared snapshots use
explicit version 2: canonical BF16 prefixes, complete owned windows, ordinary
state, local marks/counters and only the sequence's resident blocks. Restore
assigns destination slots and generations, invalidates evicted inverse maps, and
preserves owned block ages against a monotonic shared clock. Global physical slot
numbers and another lane's payload are not restored. Unrelated windows and
canonical history remain untouched. If residency is full, ordinary eviction can
remove a foreign cached block while preserving its canonical history. Snapshot
admission includes the checked maximum owned-entry payload and metadata.

## Snapshot and cache reuse validation (2026-10-20)

Version 1 snapshots preserve canonical BF16 RAM rows after flushing every dirty window, the full compact device
windows and slot payloads, both mapping directions, marks, clocks, epochs and LRU stamps. They also retain recurrent
and convolution state, partial raw index keys, complete pooled blocks, PLE tail/history, next logits, and an optional
settled MTP head. Restore copies into destination allocations; source lifetime does not constrain snapshot lifetime.
Transient MTP verification generations and captured sequence graphs are invalidated before restored state is used.

The pager configuration is frozen on the loaded engine. Paged serving lanes allocate the configured context capacity
already included in stream admission, so different prompt lengths use compatible table strides. Snapshot admission
includes its fixed windows, tables and header in addition to logical history; paged snapshots are charged to host RAM.
Capacity or geometry changes are rejected before mutation. This format preserves the warm pager cache, rather than
reconstructing a cold cache.

TP2 restore validates both ranks, copies and waits on both ranks, then commits. The owning scheduler remains excluded
until acknowledgements complete. A copy or peer failure poisons the session; a partially restored destination cannot
execute. Failed asynchronous submissions drain before their memory is freed, following the existing CUDA completion
policy (terminate the rank if neither its stream nor context can drain).

Current scope: snapshot/cache reuse correctness and prefill/decode graph replay with two ordinary-length lanes.
The two concurrent 200,000-token gate is deferred. Its later sizing candidate is `--context 200256 --parallel 2`,
`TF_FLASHNEXT_EXL3_STREAMS=2`, BF16, host IKC and snapshots, paging window 4096 / slots 49152 / rows 2048, engine chunk
512 and deep prefill rows 2048. The serving client must assert the exact prompt-token count and return nonzero on any
failed check. Configuration is a sizing candidate, not a validated capacity claim.

Validation: focused host/lifecycle coverage passes, including restore-copy submission failures, preflight rejection,
source drop/repeated restore, MTP stale-publication rejection, TP2 prepare/copy/commit ordering, and failed-completion
draining. CUDA lifecycle tests pass at boundaries 0, 1, 3, 4, 2051, 2052, 4096, 4099 and 6147, checking windows,
slots and mapping tables. The TP2 two-GPU graph test passes with two paged BF16 lanes and exact state/logit parity;
each rank replayed prefill graphs 72 times and decode graphs 5 times after restore. A real TP2 HTTP chat-history gate
with BF16 paging, graphs, MTP drafting and two concurrent requests passed warm, branch and rewind reuse at 12,040
prompt tokens: warm/rewind reused 12,033 tokens, branch reused 12,023, and all response hashes matched the cold
baseline. With capture-after=1, each rank observed 72 prefill graph replays and 216 captures. With the default
capture-after=2, each rank observed 0 prefill replays, 192 eager layer executions and 96 captures, with lower
one-time capture cost. Decode logs expose host preparation and capture time; capture-after=2 reduced cumulative
capture time versus capture-after=1. Paired DPLAN-vs-graph gates preserved exact hashes, but timings show no
repeatable graph speedup. This proves ordinary-length cache reuse, not 200k capacity.
Live-rank overflow/failure injection remains open. The user explicitly defers 200k × 2 until graphs and cache reuse
are optimized. Int8 KV is out of scope; BF16 is the required primary format.

Status: active implementation, 2026-10-20. Branch `exl3-c-main`. Task: prove a dynamic KV
paging cache at long context on upstream's attention kernels: a sequence's full bf16 K/V in pinned host memory, the
card holding a recent dense window plus an LRU cache of the 4-key blocks the sparse indexer selects.

Every number is tagged: **measured** (par1-cs13, one idle RTX 5060 Ti under `~/bin/gpurun`, sm_120), **replay**
(offline simulation over measured selection traces), or *estimate* (arithmetic, not a run).

## 1. What the attention reads (measured traces, replayed)

Tool: `tf-cuda-test exl3-kvpage DIR trace IDS N,N.. DECODE OUT prompt` runs the family's Engine (EXL3 offload pool,
expert-major deep prefill, as serve attaches it) over the first N ids of `~/pf/long_ids.bin` (306k ids of real text:
the repo's docs tokenized), then DECODE greedy rounds, and records after every `fn_qsa_select` each row's 512
selected blocks (`cuda_exl3_kvpage.Trace`, a one-line hook at the top of `cuda_batch.zig` attnRows). Traces:
32,768 / 65,536 / 200,000 prompt tokens, every prompt row of every attention layer plus 256 real decode rows each
(2.40 M rows at 200k, 2.3 GiB). `tools/exl3_kvpage_replay.py` replays them.

Selection churn (replay over the decode rows, mean over the 12 layers):

| prompt tokens | blocks in context a layer | overlap with the previous row | new blocks a row a layer | union of a 4-row MTP window | share of the 512 in the last 4,096 tokens |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 8,192 | 0.62 (3 layers) | 197 | 1,002 | 0.21 |
| 65,536 | 16,384 | 0.55 | 230 | 1,093 | 0.27 |
| 200,000 | 50,000 | 0.49 (0.39-0.60 by layer) | 263 | 1,175 | 0.11 |

K10's 0.80-0.88 (6.5k tokens) does not hold at long context: at 200k consecutive decode rows share half their blocks.

Prompt rows: a 256-row attention call's selections cover (union, blocks) 4,068 mean / 7,202 max at <32k, 9,538 /
12,136 at 32-64k, 11,692 / 16,699 at 64-128k and 14,175 / 18,852 at 128-200k (layer 0; layer 1 is the widest: max
24,564). Outside a 4,096-token window the max is 23,595 blocks (184 MiB of K/V) at 200k.

### Cache replay

One slot pool shared by the 12 layers (keys layer × blocks + block; exact LRU at call granularity; calls in
execution order: the prompt layer-major inside each deep chunk, then decode 12 layers a token), window W tokens a
layer always on the card. Hit rate = selected blocks found on the card (window or slots) / selected; bytes = misses ×
8 KiB (4 keys × K and V × 2 heads × 256 × bf16). MTP: 4 consecutive decode rows as one call a layer.

| prompt | W | slots (MiB) | decode hit | misses a token (p95) | MiB a token | MTP-4 hit | MiB a 4-row window | prefill copies | per prompt row |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 4,096 | 12,288 (96) | 0.819 | 1,111 (1,990) | 8.7 | 0.640 | 34.7 | 2.6 GiB | 83 KiB |
| | | 24,576 (192) | 0.907 | 572 (1,223) | 4.5 | 0.813 | 18.0 | 2.5 GiB | 80 KiB |
| | | 49,152 (384) | 0.963 | 228 (569) | 1.8 | 0.925 | 7.2 | 2.0 GiB | 63 KiB |
| 65,536 | 4,096 | 12,288 (96) | 0.779 | 1,358 (2,115) | 10.6 | 0.596 | 41.3 | 12.4 GiB (overflow) | 199 KiB |
| | | 24,576 (192) | 0.863 | 839 (1,598) | 6.6 | 0.747 | 25.9 | 11.0 GiB | 176 KiB |
| | | 49,152 (384) | 0.913 | 532 (1,118) | 4.2 | 0.837 | 16.7 | 10.5 GiB | 168 KiB |
| | | 98,304 (768) | 0.954 | 280 (734) | 2.2 | 0.914 | 8.8 | 8.2 GiB | 131 KiB |
| 200,000 | 4,096 | 24,576 (192) | 0.764 | 1,447 (2,478) | 11.3 | 0.585 | 45.7 | 104.7 GiB | 549 KiB |
| | | 49,152 (384) | 0.831 | 1,039 (1,930) | 8.1 | 0.704 | 32.6 | 90.9 GiB | 477 KiB |
| | | 98,304 (768) | 0.884 | 712 (1,598) | 5.6 | 0.798 | 22.3 | 88.6 GiB | 465 KiB |
| | 8,192 | 98,304 (768) | 0.890 | 679 (1,504) | 5.3 | 0.807 | 21.2 | 84.2 GiB | 442 KiB |
| | 2,560 | 49,152 (384) | 0.827 | 1,062 (1,980) | 8.3 | 0.698 | 33.3 | 92.7 GiB | 486 KiB |

(Per-layer pools of the same total do about the same: 8,192 a layer at 200k: decode hit 0.897.) 12,288 slots
overflow during the 200k prompt (a call's union does not fit); 24,576 is the floor at 200k with 256-row calls.

Reading: the cache is not a "hit rate 99%" cache at 200k; it is a bandwidth reducer. At 0.38-0.77 GiB of slots a
decode token moves 5.6-8.1 MiB (all 12 layers) instead of the 48 MiB its 12 × 512 selected blocks hold, and the card
does not hold the 4.58 GiB of a 200k stream's full K/V. At PCIe ~12-13 GB/s (K10: one pinned H2D 13.9
GB/s, kernel gather from mapped memory 12.5 GB/s) that is 0.45-0.7 ms a token (*estimate*), next to a decode step of
~57-70 ms on this card (the expert pool's misses dominate). A 4-row MTP verify window moves 22-33 MiB (~2-2.7 ms,
*estimate*); the prompt moves ~0.45-0.55 MiB a row (90 GiB for 200k: ~7.5 s next to a ~170-250 s prefill,
*estimate*).

## 2. Design (built)

Files: `zig/src/families/flashnext/cuda_exl3_kvpage.zig` (trace, `Paged`, the attention rows over it),
`zig/kernels/cuda/flashnext_exl3_kvpage.cu` (`kvp_plan`, `kvp_fetch`, `kvp_remap`), `zig/tests/cuda/exl3_kvpage_tests.zig`
(`tf-cuda-test exl3-kvpage`), `tools/exl3_kvpage_replay.py`. Hooks (one line each): `cuda_batch.zig` attnRows
(`if (exl3_kvpage.takes(self.seq)) return exl3_kvpage.attnRows(...)`), `cuda_batch_state.zig` SeqState `.paged`,
`cuda_prompt.zig` Seq `.paged` / view / deinit / newSeq (`kvpage.forSeq`; kc/vc shrink to one row), `cuda_snap.zig`
(refuses a paged sequence), `cuda_lanes_load.zig` (stream budget), kernel list (2 lines), barrel, main.zig.
**No upstream `.cu` changed.**

Switch: `TF_FLASHNEXT_KVPAGE=1` (default off; all-on-device bf16 stays the default), sizes
`TF_FLASHNEXT_KVPAGE_WINDOW` (4096), `_SLOTS` (49152, shared by the 12 layers), `_ROWS` (max(1536, engine chunk)).

- **Host store**: one mapped pinned buffer a sequence, `[layer][K|V][cap][2][256]` bf16 (4.6 GiB at 200k).
- **Card**: one K buffer and one V buffer `[12 windows of wr rows | 4 × slots rows][2][256]`, wr = window + rows + 4;
  tables: blk2slot `[12][blocks]`, mark `[12][blocks]`, slot2blk / stamp `[slots]`, counters `[12][8]`.
- **Writes**: upstream's `fn_attn_prep` writes the rows into the layer's window (its K/V base pointer offset by
  −win0 rows: prep indexes by absolute position). When a call would run past the window, the window moves: the rows
  that leave are written to host memory (`cuMemcpyDtoDAsync` to the mapped pointer) and the last `window` tokens move
  down in non-overlapping pieces. So the prompt writes its K/V to host as it goes (one window step behind) and decode
  writes nothing per token.
- **Reads**: after upstream's `fn_qsa_select` lists a call's keys (positions), `kvp_plan` (one 1024-thread block)
  collects the call's unique blocks before the window, stamps the hits with the call's epoch, takes for the misses the
  least recently used slots (exact LRU: a 4-pass radix selection of the smallest stamps, as `fn_qsa_select` selects the
  largest scores), and updates the tables; `kvp_fetch` copies each miss's 8 KiB from mapped host memory into its slot
  (a kernel, 128 blocks); `kvp_remap` rewrites each sparse row's key list from positions to rows of the compact
  buffer (window rows, or slot base + 4 × slot + offset). Upstream's `fn_attn_chunks` / `fn_attn_merge` then run
  unchanged on the compact buffer. Dense rows (<= 2,051 keys) read rows 0..n−1, which is the window while it has not
  moved (it moves only past position window + rows > 2,051).
- **Shared slots**: one pool for all layers, because a prompt runs layer-major (one layer's calls take the whole
  pool: a 256-row call at 200k needs up to 23.6k blocks) and decode spreads evenly (replay: same hit rate as per-layer
  pools of the same total).
- Everything runs on the compute stream (plan → fetch → remap → attention): the misses are only known after the
  layer's own selection, and the next layer depends on this one, so there is nothing to overlap the fetch with inside
  a token (see §5).

## 3. Proof (measured, `tf-cuda-test exl3-kvpage`)

**Kernels** (`exl3-kvpage none kernel 6144|49152`, synthetic key lists, 60 calls of 1 or 64 rows with 15% drift a
row, window start at 40,000): 898,338 keys remapped, **every one reads its own host row** (K and V), 0 bad; misses
GPU 13,657 vs host LRU model 13,666 (6,144 slots) and 9,642 vs 9,642 (49,152 slots); no overflow.

**Whole model, one card** (`exl3-kvpage DIR cmp IDS N.. DECODE 4096 49152 [NEEDLE]`; the family's Engine with the EXL3
offload pool and the deep prefill as serve attaches them, `TF_FLASHNEXT_CHUNK=512 TF_FLASHNEXT_EXL3_PREFILL_ROWS=4096`;
each prompt through an all-on-device sequence, then the same Engine with a paged sequence; real text from
`~/pf/long_ids.bin`):

| prompt | last-row logits | greedy tokens | K/V rows equal (12 layers × K,V) | prefill ms all-on-device / paged | decode ms/token all-on-device / paged |
|---:|---|---|---|---:|---:|
| 32,768 | bit-equal | 128/128 equal | 24/24 (32,896 rows) | 34,605 / 34,828 (+0.6%) | 63.48 / 63.95 (+0.47) |
| 65,536 | bit-equal | 128/128 equal | 24/24 | 71,075 / 72,594 (+2.1%) | 64.33 / 65.20 (+0.87) |
| 200,000 | bit-equal | 128/128 equal | 24/24 (200,128 rows) | 244,232 / 254,237 (+4.1%) | 69.44 / 70.61 (+1.17) |
| 65,536 needle at 0.2 | bit-equal | 24/24 equal | 24/24 | 75,385 / 76,342 | 51.93 / 50.76 |
| 131,072 needle at 0.2 | bit-equal | 24/24 equal | 24/24 | 159,563 / 164,551 | 49.17 / 49.63 |
| 65,536 needle at 0.35 | bit-equal | 24/24 equal | 24/24 | 75,384 / 76,314 | 59.65 / 58.89 |
| 98,304 needle at 0.35 | bit-equal | 24/24 equal | 24/24 | 115,648 / 118,685 | 50.21 / 49.83 |

Copies and memory (paged runs):

| prompt | prefill hit / fetched | decode hit (outside the window) | fetched a token | sequence on the card all-on-device / paged |
|---:|---|---:|---:|---|
| 32,768 | 93.7% / 1.9 GiB | 94.1% | 2.30 MiB | 5.484 / 1.391 GiB (cap 200k: freed 4.09 GiB) |
| 65,536 | 90.8% / 10.3 GiB | 86.9% | 4.40 MiB | same allocation |
| 200,000 | 84.4% / 90.4 GiB | 80.5% | 8.26 MiB | same allocation |

The paged side itself is 0.486 GiB (windows 0.10, slots 0.375, tables); the rest of the 1.39 GiB is what stays as
upstream has it (indexer keys 0.57 and pooled keys 0.14 at 200k, GDN states, PLE tail). Pinned host: 4.6 GiB a 200k
stream. The measured copies match the replay (200k: 8.26 vs 8.1 MiB a token, 80.5% vs 80.9% outside-window hit,
prefill 90.4 vs 90.9 GiB).

**Needle**: a sentence with a 5-digit code inserted at 20% of the text and a question appended. At 131,072 tokens
both caches answer "70259" (the code; position ~26k, far outside the window and long since evicted from and fetched
back into the slots). At 65,536 both caches emit `<|im_end|>` and start a chat turn instead of answering (identical
tokens: the model's behaviour on this prompt, not the cache's). With the code at 35% of the text both caches answer "48613" at 65,536 tokens and
"70259" at 98,304 tokens.

**tp=2 serve** (`tensorfold-native serve --tp 2 --context 65536 --prompt-cache-gib 0`, both cards, upstream's
expert-parallel EXL3 path; `~/tensorfold-c-kvpage/tpkv.sh`): two real-text prompts of 27,632 and 57,546 tokens,
greedy 32 tokens, `TF_FLASHNEXT_KVPAGE` off then on: **the completions are identical** on both prompts. With paging
the stream budget is smaller, so the expert pools grow from 52 / 59 to 63 / 71 lines a layer (rank 0 / rank 1) and
the requests take 123 s / 260 s instead of 132 s / 278 s (prefill plus 32 tokens, one run each). Not instrumented:
per-rank paging counters at tp=2 (the hit/miss numbers above are tp=1); the tp=2 equality is over 2 × 32 tokens.

## 4. Cost (why it is cheap, and what it is not)

Decode: +0.5 / +0.9 / +1.2 ms a token at 32k / 65k / 200k on 63-69 ms tokens (+0.7-1.7%). 8.3 MiB at ~12.5 GB/s is
~0.7 ms; the plan/remap launches the rest. Prefill: +0.6% / +2.1% / +4.1% (90 GiB of fetches at 200k ≈ 7.5 s at PCIe
rate, 10 s measured, of 244 s). Memory: −4.09 GiB a 200k stream on the card for 4.6 GiB of pinned host memory.
Not measured: MTP verify windows (the tp=2 serve run had drafts at the server default; only its completions were compared) through the paged cache in the engine (the replay gives 22-33 MiB a 4-row window at
200k, ≈ 2-2.7 ms, *estimate*); decode rounds of several lanes (the hook runs per lane, not exercised here); expert-pool
contention for PCIe beyond what the decode numbers above contain.

## 5. Limits and next steps

1. A call whose union of blocks outside the window exceeds the free slots must fail closed. Device-side overflow
   signalling and a host-side
   check before consuming the result are wired into the forward path; synthetic and 16k TP=2 deep-prefill tests
   pass. The planned 200k × 2 stress/overflow-injection gate remains open. Slots must be >= ~24k at 200k with
   256-row prompt calls (replay max 23.6k).
2. Paged prompt-cache snapshots now flush and preserve BF16 host K/V, device windows, table/LRU metadata, recurrent,
   indexer, PLE and MTP state. TP2 restore validates and commits both ranks before resuming. Ordinary-context two-lane
   warm/branch/rewind cache reuse passes with exact cold-output parity; deterministic failure injection on live ranks
   remains open.
3. Paging captures and replays both deep-prefill layer graphs and decode graphs at TP=2. The 16,384-row CUDA test
   verifies exact graph-vs-eager state/logit parity; the 12,040-token two-request HTTP test records actual prefill
   graph replay and paging snapshots. These establish correctness, not a speed benefit. Paired DPLAN-vs-graph runs
   remain within variation; optimize/profile before claiming graphs improve performance.
4. The fetch is on the compute stream. Overlap needs prediction (prefetch for row t+1 from row t's selection), but at
   200k consecutive rows share only half their blocks, so most misses are new blocks; not worth it at +1 ms a token.
5. int8 K/V (K10) is out of scope here (bf16 only, as asked); the paging and int8 would compose (the host store and
   slots would halve).

## 6. Acceptance gates (current requirement)

BF16 remains the production KV format. RAM paging is a capacity tier for the same BF16 rows, not a reason to switch
to int8. The single-card 200k parity result and the short TP=2 parity run above are prototype evidence only; they do
not satisfy the following acceptance gates:

1. **Graph performance (immediate):** keep the DPLAN-vs-graph A/B on the same TP2, two-lane BF16-paged workload;
   profile capture/replay, host prepare time and eager fallback by phase. Optimize prefill and decode separately,
   preserving exact outputs and NCCL overlap. Do not enable graphs by default until repeated runs show a benefit.
2. **Safety:** deterministically inject slot overflow and restore/copy failures on TP2; fail closed before attention
   consumes invalid rows, drain both ranks, and prove the session is discarded cleanly.
3. **Cache lifecycle:** retain exact warm/branch/rewind parity and exercise eviction/reload and sequence reuse with
   MTP draft rejection/rollback. Existing ordinary-context two-lane tests pass; failure cleanup remains open.
4. **Long context (later):** only after graph and cache reuse are optimized, prove the 200k × 2 TP2 capacity target
   with BF16 paging and graphs enabled. Keep a resident-BF16 comparison where it fits; otherwise compare every
   fetched/evicted K/V row against the resident oracle in bounded fixtures plus full-request logits/tokens.
5. **Observability:** expose per-rank overflow, hit/miss, pinned-RAM and device-byte counters in the acceptance logs.

Until these pass, paging stays opt-in and must not be described as production-ready. CPU expert execution and int8 KV
remain out of scope for this BF16 paging milestone.

## 7. Integration audit (updated 2026-10-20)

The allocator/admission mismatch is fixed in commit `8528ca2c`: when host indexer keys are enabled, the
placement-aware `newSeq` path honors the BF16 pager instead of allocating resident K/V, and the admission budget
uses the same sequence layout, configured chunk and one-row placeholders. Quantized KV and split-head placement are
rejected when paging is enabled. Later work adds pager-aware snapshot state and TP2 graph replay counters.

Latest hardware gates (par1-cs13, both 5060 Ti cards, TP=2): graph snapshot/restore parity passes on both ranks.
The 12,040-token two-request HTTP gate passes with BF16 paging, MTP, graph mode and exact warm/branch/rewind cache
parity; each rank logs prefill replays and captures. Four alternating 4,139-token DPLAN/graph runs also preserve exact
cross-mode hashes and cache hits, but do not show a repeatable graph speedup. The 16,384-row deep-prefill graph gate
continues to pass eager parity. This is ordinary-context correctness, not 200k × 2 capacity.

Remaining blockers:

- **Graph performance:** repeated same-workload TP2 A/B and phase-level profiling are still needed; graphs are correct
  but have not demonstrated a speed win.
- **Overflow stress:** deterministic TP2 overflow/failure cleanup and clean collective drain remain open.
- **Mixed serving:** two-lane cache reuse passes at 12k tokens with MTP and graphs; long-context paging and mixed
  failure/rollback under cache eviction remain unproven.

The next priority is graph profiling/optimization and deterministic TP2 overflow/failure cleanup. The 200k × 2
capacity run is deferred until those are complete. For resident parity, use replicated attention
(`TF_FLASHNEXT_ATTN_SPLIT=0`) so a different attention reduction order does not confound the cache comparison.
