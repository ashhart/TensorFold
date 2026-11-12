# Optimization ledger

This page shows which speed and memory optimization applies to which model family, and where each one stands.
A change proven on one family should be ported to the others, not invented again, so each row names the PR that did
it first and the families still to do. The cross-family parts live in shared helpers, so a port is mostly the family's
own hook on top of them. Update a cell in the same PR that changes it.
How to approach a speed change, from bounding it to measuring it, is in [optimizing](optimizing.md).

Shared helpers that hold the cross-family parts:

- `zig/src/core/kv_pool.zig`: the core KV page pool, its leases and promises. CUDA's plan for it is in
  `zig/src/native/cuda_kv_pool.zig`, and the page reads in `zig/kernels/cuda/common/kv_pages.cuh`.
- `zig/src/core/prompt_radix.zig` and `zig/src/core/kv_prompts.zig`: one prompt store over the pool's pages
  (`TF_KV_SHARE`), with the CUDA adapter in `zig/src/native/cuda_prompt_pages.zig`.
- `zig/src/core/kv_ledger.zig`: the admission ledger the lane host checks for every request.
- `zig/src/core/lane_host.zig` and `zig/src/core/lanes/`: rounds, shared drafts, copy drafts (`context_copy.zig`,
  with the search in `zig/src/core/copy_index.zig`) and `--decode-share`.
- `zig/src/core/checkpoint_prefetch.zig` (`TF_LOAD_PREFETCH`), `zig/src/core/direct_gather.zig` and
  `zig/src/core/direct_io.zig`: load read-ahead and reads past the page cache.
- `zig/src/native/disk_spill.zig` and `zig/src/native/cuda_disk.zig`: kept prompt states on disk.
- `zig/src/cuda/graph_cache.zig`, `zig/src/cuda/grouped.zig` and `zig/src/cuda/experts.zig`: CUDA graphs, the
  expert plan and the grouped expert call.
- `zig/src/cuda/window_attention.zig` `promptRows`: the shared prompt attention over a window.
- `zig/kernels/cuda/common/`: shared CUDA headers (`fp_pairs.cuh`, `cp_async.cuh`, `reduce.cuh`, `vec.cuh` and others).
- `zig/src/core/bf16_round.zig`: the one fp32-to-bf16 rounding every host copy uses.
- `zig/src/core/profile.zig` and [profiles](profiles.md): the `--profile` switch registry.
- `zig/src/core/contract/` and the [kernel contract](kernel-contract.md): the gates a new or ported kernel passes.
- `zig/src/core/prompt_tail.zig`: the last layer's kept tile ([#2229]) and the chunks nothing reads ([#2432]).
- `zig/src/core/state_walk.zig`: one walk for kept prompt states. Not on main yet: it is on [#2215].

## The table

Each cell is one of these:

- `yes`: applies, not done yet.
- `no`: does not apply, with a short reason.
- `done`: on main. A closed PR that the maintainers landed on main counts as done.
- `failed`: tried and measured, with no gain, or rejected.
- `open`: a PR is open.
- `?`: can't tell from the code and the PRs.

`done¹` means the code is on main but no single PR for it was found. `C`, `M`, `H` and `X` name the backend (CUDA,
Metal, HIP, XPU) where it matters; with no letter, the cell is about CUDA. The table is split into parts of five
columns or fewer to keep lines short. The first three parts hold the native families, the most active first. The
last two hold the families with only a Metal, HIP or scoring backend.

| Optimization | q3.5 | fnext | glm | gem4 | nemo |
| --- | --- | --- | --- | --- | --- |
| 1. Prompt tail | done [#2229] | yes | yes | open [#2432] | open [#2432] |
| 2. Paged KV | done C[#1750] M[#1294] | done [#1169] | done [#1163] | done [#1619] | done C[#935] M[#1537] |
| 3. Prompt store | done [#1357] | yes (host store) | done [#1732] | done [#1883] | yes C open M[#1850] |
| 4. iGPU states | done [#2113] | done¹ | yes | done¹ | done¹ |
| 5. State walk | open C[#2392] | open C[#2392] | open [#2215] | open C[#2392] | open C[#2392] |
| 6. KV ledger | done [#2127] | done [#2127] | done [#2127] | done [#2127] | done [#2127] |
| 7. Host KV tier | yes | open [#1501] | done [#1795] | yes | yes |
| 8. Disk cache | done C[#1358] | done C[#1359] M[#519] | done M¹ open C[#1411] | open C[#3135] | done C[#1413] M¹ |
| 9. FP8 KV | done [#886] | yes | done¹ | yes | yes |
| 10. FP4/FP8 decode | yes | yes | yes | no (affine) | yes |
| 11. Fused proj. | done [#1783] | done¹ | done¹ | done [#1783] | done [#1783] |
| 12. Add + norm | done H[#2099] | no (bound <1 %) | yes | done¹ | done [#796] |
| 13. MoE grouped | done [#605] | done [#1351], open M[#2784] | done [#864] | no (dense) | done [#558] |
| 14. CUDA graphs | done¹ | done [#1786] | open [#2279] | done [#1174] | done [#615] |
| 15. MTP | done [#1562] | done [#1925] | done [#1585]; wide C open [#2257] | done [#1385] | done [#997] |
| 16. DFlash(2) | done [#1465] | yes | done [#979] | done C¹ M[#879] | yes |
| 17. DSpark | yes | yes | yes | yes | yes |
| 18. Copy drafts | done [#1477] | done [#468] | done [#2146]; wide C open [#2257] | done [#2022] | done [#1974] |
| 19. Shared drafts | done [#1562] | no (null [#1762]) | done [#1779]; cap C open [#2257] | done [#1779] | done [#768] |
| 20. Solo chunks | yes | done C[#1733] M[#2167] | done [#1823] | done C[#2391] | yes |
| 21. Read-ahead | done [#1928] | no (row 22) | done [#1616] | done [#1928] | done [#2033] |
| 22. Direct load | done [#2282] | done [#1901] | yes | done [#2282] | done [#458] |
| 23. Page drop | done [#1884] | done [#1880] | done [#600] | done [#1884] | done [#1884] |
| 24. L2 prefetch | failed [#2014] | no (null) | done [#2162] | no (null) | failed [#2014] |
| 25. TP ranks | done H[#655] | done C[#1771] H[#817] | done [#855] | yes | done¹ |
| 26. Pipeline | open [#2115] | yes | yes | yes | yes |
| 27. Profiles | done [#1944] | done [#1944] | done [#1959] | done [#1944] | done [#2061] |
| 28. Scale index | ? | open M[#2760] | ? | ? | ? |
| 29. DSpine | done C[#2633] | no (no drafter) | no (no drafter) | no (no drafter) | no (no drafter) |

| Optimization | kol1 | spk25 | dory | mm3 | dsv41 |
| --- | --- | --- | --- | --- | --- |
| 1. Prompt tail | yes | yes | yes | yes | yes |
| 2. Paged KV | done [#1620] | done [#1653] | done [#1284] | yes | yes (own pool) |
| 3. Prompt store | done [#1755] | done [#1756] | open [#2430] | yes | yes |
| 4. iGPU states | done¹ | done¹ | done¹ | done¹ | yes |
| 5. State walk | yes | yes | yes | yes | yes |
| 6. KV ledger | done [#2127] | done [#2127] | done [#2127] | yes | yes |
| 7. Host KV tier | yes | yes | yes | yes | yes |
| 8. Disk cache | yes | open [#3135] | open [#3135] | yes | done¹ |
| 9. FP8 KV | yes | yes | yes | yes | no (FP8 by design) |
| 10. FP4/FP8 decode | failed [#1448] | no (bf16) | yes | no (EXL3) | yes |
| 11. Fused proj. | done¹ | done [#1783] | yes | done¹ | done¹ |
| 12. Add + norm | done¹ | done¹ | done [#1256] | done¹ | done M[#1714] ? C |
| 13. MoE grouped | done¹ | no (dense) | no (dense) | done [#1388] | done M[#1707] |
| 14. CUDA graphs | done¹ | done¹ | done¹ | yes | done¹ |
| 15. MTP | no (no head) | no (no head) | no (no head) | no (DSpark) | no (DSpark) |
| 16. DFlash(2) | yes | yes | yes | done¹ | yes |
| 17. DSpark | yes | yes | yes | done¹ | done [#1751] |
| 18. Copy drafts | done [#2022] | done [#2022] | done [#2022] | done [#2022] | done¹ |
| 19. Shared drafts | no (copies) | no (copies) | no (copies) | open [#1426] | done¹ (opt-in) |
| 20. Solo chunks | yes | yes | yes | yes | yes |
| 21. Read-ahead | done [#2033] | done [#2033] | yes | done [#2033] | done [#2033] |
| 22. Direct load | yes | open [#2449] | open [#2449] | yes | open [#1924] |
| 23. Page drop | done [#1884] | done [#1884] | done [#1884] | done [#1884] | done [#1884] |
| 24. L2 prefetch | yes | yes | yes | yes | done¹ |
| 25. TP ranks | yes | no (4B) | yes | done [#1388] | done¹ |
| 26. Pipeline | yes | no (4B) | yes | yes | yes |
| 27. Profiles | done [#1944] | done [#1944] | open [#2306] | yes | yes |
| 28. Scale index | ? | ? | ? | ? | ? |
| 29. DSpine | no (no drafter) | no (no drafter) | no (no drafter) | no (no drafter) | no (no drafter) |

| Optimization | lfm2m | lfm3 | qwen4 | ling |
| --- | --- | --- | --- | --- |
| 1. Prompt tail | no (no CUDA) | no (no CUDA) | no (n/a) | no (no engine) |
| 2. Paged KV | open [#1503] | yes | yes | yes |
| 3. Prompt store | open [#1503] | yes | yes | yes |
| 4. iGPU states | yes | yes | yes | yes |
| 5. State walk | yes | yes | yes | yes |
| 6. KV ledger | yes | yes | yes | yes |
| 7. Host KV tier | yes | yes | yes | yes |
| 8. Disk cache | yes | yes | yes | yes |
| 9. FP8 KV | yes | yes | yes | yes |
| 10. FP4/FP8 decode | no (affine) | yes | yes | no (EXL3) |
| 11. Fused proj. | yes | done¹ | done¹ | yes |
| 12. Add + norm | done M[#822] | done X[#1696] | yes | yes |
| 13. MoE grouped | done M[#822] | no (dense) | no (dense) | done M[#1836] |
| 14. CUDA graphs | yes | yes | yes | yes |
| 15. MTP | no (no head) | no (no head) | no (no head) | yes |
| 16. DFlash(2) | yes | yes | yes | yes |
| 17. DSpark | open [#1394] | yes | yes | yes |
| 18. Copy drafts | done M¹ | done X[#1696] | done¹ | yes |
| 19. Shared drafts | done [#1779] | no (no head) | no (no head) | yes |
| 20. Solo chunks | yes | yes | yes | yes |
| 21. Read-ahead | no (no CUDA) | yes | yes | yes |
| 22. Direct load | yes | yes | yes | yes |
| 23. Page drop | done M[#1885] | yes | done [#1887] | yes |
| 24. L2 prefetch | yes | yes | yes | yes |
| 25. TP ranks | yes | no (1.7B) | yes | yes [#850] |
| 26. Pipeline | yes | no (1.7B) | yes | yes |
| 27. Profiles | yes | yes | yes | yes |
| 28. Scale index | ? | ? | ? | ? |
| 29. DSpine | no (no drafter) | no (no drafter) | no (no drafter) | no (no drafter) |

q3.5 `qwen3_5` (C, M, H, X), fnext `flashnext` (C, M, H), glm `glm` (C, M), gem4 `gemma4` (C, M, X),
nemo `nemotron` (C, M, H, X), kol1 `kolibri1` (C), spk25 `spark25` (C, M),
dory `dory` (C), mm3 `minimax_m3` (C, M), dsv41 `deepseek_v41` (its own CUDA binary, not the release's CUDA server;
M), lfm2m `lfm2_moe` (M; C on [#1492]), lfm3 `lfm3` (X; C on [#1508]), qwen4 `qwen4` (C, M),
ling `ling` (host package only, no engine yet; CUDA plan in [#850]).

| Optimization | q27 | lfm2 | llama | step5 |
| --- | --- | --- | --- | --- |
| 1. Prompt tail | done M[#2436] | no (n/a) | no (Metal) | no (Metal) |
| 2. Paged KV | done M[#2283] | no (scores) | yes | yes |
| 3. Prompt store | yes | no (scores) | yes | yes |
| 4. iGPU states | no (unified) | no (scores) | no (unified) | no (unified) |
| 5. State walk | yes | no (scores) | yes | yes |
| 6. KV ledger | yes | no (scores) | yes | yes |
| 7. Host KV tier | yes | no (scores) | yes | yes |
| 8. Disk cache | done¹ | no (scores) | yes | yes |
| 9. FP8 KV | yes | no (scores) | yes | yes |
| 10. FP4/FP8 decode | no (Metal) | no (bf16) | no (Metal) | no (Metal) |
| 11. Fused proj. | done¹ | yes | yes | done¹ |
| 12. Add + norm | done M¹ | done¹ | done¹ | yes |
| 13. MoE grouped | done [#1125] | no (dense) | no (dense) | done¹ |
| 14. CUDA graphs | no (Metal) | no (scores) | no (Metal) | no (Metal) |
| 15. MTP | open M[#2464] | no (scores) | no (no head) | done¹ |
| 16. DFlash(2) | done [#578] | no (scores) | yes | yes |
| 17. DSpark | yes | no (scores) | yes | yes |
| 18. Copy drafts | done¹ | no (scores) | done M¹ | done M¹ |
| 19. Shared drafts | done [#578] | no (scores) | no (no head) | yes |
| 20. Solo chunks | yes | no (scores) | yes | yes |
| 21. Read-ahead | no (Metal) | yes | no (Metal) | no (Metal) |
| 22. Direct load | yes | yes | yes | yes |
| 23. Page drop | done M[#1885] | done [#1884] | done M[#1885] | done M[#1885] |
| 24. L2 prefetch | no (Metal) | no (scores) | no (Metal) | no (Metal) |
| 25. TP ranks | open [#1434] | no (small) | yes | yes |
| 26. Pipeline | open M[#2375] | no (small) | yes | yes |
| 27. Profiles | yes | yes | yes | yes |
| 28. Scale index | ? | ? | ? | ? |
| 29. DSpine | open M[#2634] | no (scores) | no (no drafter) | no (no drafter) |

| Optimization | gdsa | kimi | qwen3 |
| --- | --- | --- | --- |
| 1. Prompt tail | no (Metal) | no (Metal) | no (HIP only) |
| 2. Paged KV | yes | yes | no (prompt) |
| 3. Prompt store | yes | yes | no (prompt) |
| 4. iGPU states | no (unified) | no (unified) | no (prompt) |
| 5. State walk | yes | yes | no (prompt) |
| 6. KV ledger | yes | yes | no (prompt) |
| 7. Host KV tier | yes | yes | no (prompt) |
| 8. Disk cache | yes | yes | no (prompt) |
| 9. FP8 KV | yes | yes | no (prompt) |
| 10. FP4/FP8 decode | no (Metal) | no (Metal) | no (bf16) |
| 11. Fused proj. | yes | yes | yes |
| 12. Add + norm | done¹ | done M¹ | yes |
| 13. MoE grouped | done¹ | done¹ | no (dense) |
| 14. CUDA graphs | no (Metal) | no (Metal) | no (prompt) |
| 15. MTP | done¹ | no (DSpark) | no (prompt) |
| 16. DFlash(2) | yes | yes | no (prompt) |
| 17. DSpark | yes | yes | no (prompt) |
| 18. Copy drafts | done M¹ | done [#2221] | no (prompt) |
| 19. Shared drafts | yes | yes | no (prompt) |
| 20. Solo chunks | yes | yes | no (prompt) |
| 21. Read-ahead | no (Metal) | no (Metal) | yes |
| 22. Direct load | yes | yes | yes |
| 23. Page drop | done M[#1888] | yes | yes |
| 24. L2 prefetch | no (Metal) | no (Metal) | no (prompt) |
| 25. TP ranks | yes | yes | no (small) |
| 26. Pipeline | yes | yes | no (small) |
| 27. Profiles | yes | yes | yes |
| 28. Scale index | ? | ? | ? |
| 29. DSpine | no (no drafter) | no (no drafter) | no (prompt) |

q27 `qwen27` (M: Qwen3.8-27B and Qwen3.5-MoE), lfm2 `lfm2` (C: LFM2-VL, scores decisions and writes no text),
llama `llama` (M), step5 `step5` (M), gdsa `glm_dsa` (M), kimi `kimi_k3` (M, native serving on [#2221]),
qwen3 `qwen3` (H: embeddings and rerankers, prompt only).

Not in the table: `h3` (MiniMax H3 denoising blocks), `drama3` (speech), `moss_image`, `qwen_image`, `radio`, and
the HIP OCR families `dots_mocr` and `glm_ocr`. They are not autoregressive text servers on the lane host, so few rows
apply.

## Rough gain

A rough size of each optimization's effect where it applies, before measuring a family: `low` under 3 %, `medium`
3–15 %, `big` over 15 %, or for memory, what lets a case fit at all. Measured numbers and their PRs are in the list
below; these are estimates for planning a port, not results.

| Optimization | Prefill | Decode | Memory | Where it shows |
| --- | --- | --- | --- | --- |
| 1. Prompt tail | low | none | none | a cold prompt, about 1/layers |
| 2. Paged KV | none | none | big | streams take pages as they grow |
| 3. Prompt store | big | none | medium | resumed turns skip their prefix; cold turns 3–6 % |
| 4. iGPU states | low | none | low | integrated GPUs only |
| 5. State walk | none | none | none | shared code; enables pool pages |
| 6. KV ledger | none | none | medium | more streams admitted in the same memory |
| 7. Host KV tier | medium | low (a cost) | big | resumes past the card's memory |
| 8. Disk cache | big | none | big | resumes across restarts and long sessions |
| 9. FP8 KV | low | medium | big | half the KV; decode gains at long context |
| 10. FP4/FP8 decode | low | low | none | 4–5 % of the FP4 prompt kernels |
| 11. Fused proj. | medium | medium | none | +3..+13 % measured |
| 12. Add + norm | low | low | none | one launch and one read fewer per layer |
| 13. MoE grouped | big | medium | none | MoE families only |
| 14. CUDA graphs | none | medium | none | launch time at one to few streams |
| 15. MTP | none | big | low | +30..+70 % one stream, acceptance-bound |
| 16. DFlash(2) | none | big | low | drafter weights; acceptance-bound |
| 17. DSpark | none | big | low | drafter ships with the release |
| 18. Copy drafts | none | medium | none | only where the reply repeats the context |
| 19. Shared drafts | none | big | none | +18..+38 % at 2–8 streams; none at one |
| 20. Solo chunks | medium | none | none | first token −5..−10 % |
| 21. Read-ahead | none | none | none | cold load about −50 % |
| 22. Direct load | none | none | low | cold load; no page-cache copy |
| 23. Page drop | none | none | medium | frees the checkpoint's page cache |
| 24. L2 prefetch | none | low | none | +1..+4 % on GLM TP=3; a cost on Qwen |
| 25. TP ranks | big | big | big | scale-out: models past one box |
| 26. Pipeline | none | none | big | fits models across nodes; adds hops |
| 27. Profiles | none | none | none | names the opt-in switches |
| 28. Scale index | none | low | 5 % of the weights once prompts read it too | one stream's decode before tensor units |
| 29. DSpine | none | medium | low | Qwen3.8-27B; agent 2–20 % faster, chat and code 10–19 % slower |

## The optimizations

1. **Prompt tail on the last layer.** A prompt chunk keeps one row's logits, so past the last layer's keys and values
   only the kept row's 128-row tile needs q, attention, o and the MLP ([#2165]). It saves about 1/layers of a cold
   prompt, less the k/v share. A family is blocked where something else reads every row of the last layer: MTP taps,
   drafter taps, hidden readouts, or a fused projection that makes k/v and q in one GEMM. Where only the prompt's last
   chunk has its logits read, the other chunks keep no tile and stop at their last state write.

   | Family | Last layer, layers | Main blocker | Gain | Effort |
   | --- | --- | --- | --- | --- |
   | qwen3_5 (27B) | full attention, 64 | MTP taps the last layer | −1.2..−1.6 % TTFT | done [#2229] |
   | qwen3_5 (35B-A3B) | full attention, 40 | MTP taps the last layer | −1.3..−2.9 % TTFT (no drafter) | done [#2229] |
   | gemma4 | global attention, 42–60 | drafter taps copy every row | ~1.5 % | easy |
   | dory | an MLP sublayer, 72 sublayers | none | ~1.3 % | easy |
   | spark25 | full attention, 36 | fused QKVG GEMM | ~2 % | medium |
   | kolibri1 | full attention, 50 | fused QKV; MoE plus shared expert | ~1.7 % | medium |
   | minimax_m3 | sparse attention plus indexer, 60 | `prep` fuses q with k/v; taps; TP gathers | ~1.5 % | medium |
   | nemotron | MoE block, 88 | MTP reads every row's hidden state | ~1.1 % | medium |
   | flashnext | attention plus indexer, 48 | MTP, hyper-connections, `.all` logits | ~2 % | hard |
   | deepseek_v41 | compressed attention, 40 | DSpark taps, mHC, emitted programs | ~2 % | hard |
   | glm | KDA linear layer, 45 | must write the recurrent state for every row | small | hard |
   | qwen4, lfm2 | | | n/a | |

   The qwen3_5 numbers are measured. The others are estimates (`~`): about 1/layers, minus the k/v share.

2. **KV pages from the core pool.** Streams lease KV pages from one `kv_pool` as their rows grow, instead of each
   reserving a whole `--context` window, so more streams fit and a short request holds little ([#1156], [#781]'s
   follow-ups). Sliding layers keep their rings. Qwen's pages are opt-in (`TF_QWEN_KV_PAGES`, `_ALL`); [#2011] keeps
   them opt-in, makes `--parallel` a cap with pages set, and brings them to the window path's speed (2k to 32k prompts
   within +0.3..+0.7 %). Dory's are opt-in
   (`TENSORFOLD_DORY_KV_PAGES=1`), Flash Next's experimental. Metal's port is tracked in [#1293].

3. **Shared paged prompt store.** A kept prompt stays as the pool's own pages in one `prompt_radix` store, so a resume
   copies nothing and the cache and the streams draw on the same memory (`TF_QWEN_KV_SHARE`, and `TF_KV_SHARE` on
   the core adapter, [#1657], which GLM uses). On Qwen3.8-27B a cold first turn is 3–6 % faster with it on, and the
   second turn level ([#1357]). [#2211] moved GLM onto the shared store. It can't be combined with
   `--learn-conversations` today. Flash Next already resumes a prefix from the host store's whole-state copies; on the
   shared store a kept 32k state would stop duplicating about 0.9 GiB of attention rows (27 KiB a token).

   Qwen CUDA's host-copy path can instead opt into message-start cuts with `TF_QWEN_REQUEST_STARTS=1`
   (#2176). Later turns closer than 256 tokens now keep their own history cuts. On a GB10 with Qwen3.8-27B 4-bit and
   its DFlash2 drafter, turns 2 to 6 of a six-turn chat reused 1,450 to 1,635 prompt tokens and took 0.72 to 1.33 s
   against 1.78 to 2.10 s unset. Prompt passes are cut at replies, so one of the six replies differed from the unset
   path; resumed and fresh replies matched with it set. This mode remains incompatible with `TF_QWEN_KV_SHARE`.

4. **Prompt states in device memory on integrated GPUs.** On a GB10 the device and host share memory, so a kept
   state can stay where it is instead of being copied through pinned host memory on every save ([#2043]). On Qwen
   CUDA this takes 1.6–2.1 % off 8k and 32k prompts ([#2113]). Metal is unified already.

5. **Core state walk for kept states.** About 25 snapshot files each had their own loop that splits a state's live
   parts at page ends and copies them. `core/state_walk.zig` holds that loop once, and a family keeps only its parts
   list and a one-copy hook ([#2215], GLM first on CUDA and Metal). It saves code, not time: every later layout change
   is made once.

6. **KV admission ledger.** A request reserves its prompt and reply, not a whole window a `--parallel` stream
   ([#1943]). [#1945] added the core ledger, and [#2127] counts every paged pool's promises in it, with no change in
   behaviour. A stream paid for at load (the engine's own sequence) is a standing lease in it, so the ledger admits the
   window plan's streams and not one fewer; no family on main reaches that yet. The unpaged families still convert one
   by one, as #1943 plans.

7. **KV pages in pinned host memory.** Pages that don't fit on the card live in pinned host memory, so a long
   conversation's next turn still resumes. GLM keeps its latent rows there with `TF_GLM_KV_HOST` ([#1795]) and its kept
   states with `TF_GLM_HOST_CACHE` ([#1736]). [#1501] adds a host tier to the core pool itself.

8. **Prompt cache on disk.** `--learn-conversations` and `--spill-gib` keep prompt states on disk under a probed
   identity, so a conversation resumes after a restart or after its RAM slot was evicted. `--checkpoint-slots` bounds
   what stays in RAM beside it.

9. **FP8 KV cache.** `--kv-cache fp8` stores keys and values as e4m3 with a scale a row, which halves KV memory for
   longer windows or more streams ([#886] on Qwen). GLM has an opt-in FP8 latent cache; its quality qualification is
   pending. DeepSeek-V4.1's compressed attention is FP8 by design.

10. **Native FP4/FP8 pair decode.** nvcc 13.2 (PTX ISA 9.2) decodes an FP4 or FP8 pair to a bf16 pair in one
    instruction on `sm_120a`/`sm_121a`, where TF decodes in software today (`fp4pair`, `fp8pair` in `fp_pairs.cuh`,
    [#1555]). On the 27B's prompt16 FP4 projections it took 4–5 % off the kernel time with bit-equal output ([#1448]).
    The FP8 block experts gained nothing, as decode there is bandwidth-bound, so FP8-only families are `failed`. [#2238]
    lands it behind `-Dnative-fp-decode`, off by default; the release toolchain stays on nvcc 13.0 until its own PR.

11. **Projections that share an input, in one launch.** Projections that read the same input (q/k/v/gate, gate/up)
    run as one grouped launch or one fused GEMM, byte-equal to separate launches. [#1783] does this for affine-4
    decode and verify on GB10; [#1439] runs up to four BF16 prompt projections in one grid on Qwen3.8.

    | Family | Gain (measured) | Where |
    | --- | --- | --- |
    | nemotron | +5.9 % (2 streams), +13.3 % (8 streams) | [#1783] |
    | qwen3_5 (27B, DFlash2) | +3.5..+5.0 % decode | [#1783] |
    | qwen3_5 (Qwen3.8, BF16) | +9.5 % prefill one stream, +8.2 % at 8 | [#1439] |
    | gemma4 (31B), spark25 | level | [#1783] |

12. **Residual add plus norm in one launch.** The residual add and the next RMS norm run as one kernel, which saves a
    read and a write of the hidden rows a layer. Nemotron's is on the shared norm kernel ([#796]); HIP Qwen3.5's
    prompt does it after the mixer ([#2099]).

13. **MoE grouped plans and expert kernels.** One expert plan sorts a step's rows by expert, and one grouped call runs
    every expert's rows ([#558], [#605], NVFP4 in [#629], EXL3 in [#741]). A family adds only its router and its weight
    format. Wide router calls run on the shared register tiles in `zig/kernels/cuda/common/router_tiles.cuh`, the
    same fma chain a logit: Nemotron's past 32 rows, and qwen3_5's from 257 rows since [#2499].
    Flash Next on Metal before tensor units groups a decode window's 4-bit experts from five rows up, each
    (row, expert) pair keeping the per-row kernels' sums ([#2784], plan in [#2764]); its shared rounds do not yet.

14. **CUDA graphs for decode and verify.** Each decode or verify shape is captured once and replayed, which removes the
    host's launch cost from every step. `TENSORFOLD_GEMMA4_GRAPHS=0` (and the same switch for Kolibri, Spark
    and Dory) keeps those eager. GLM's capture is [#2279] (open), for [#1490].

15. **MTP drafts.** The checkpoint's own MTP head drafts tokens that the target verifies in one pass, with the same
    replies as serial decode. `--mtp-drafts` and `--mtp-confidence` size the drafts on Nemotron and GLM.

    GLM CUDA's proposed 32-row prices, three-draft shared MTP cap and wider copy windows are tracked in [#2257].
    The TP=2 receipt on two GB10s compares main 52427c023e with an earlier revision of the proposed branch: one-stream code 55.4 to 73.7 tok/s
    (+33.1%), two-stream chat +3.2%, one-stream chat +1.0%, two-stream code +1.3%, prefill within 0.3% and
    replies 52/0/0 equal/unequal/unverified. The receipt does not identify the drafter mode, so it qualifies
    neither drafter separately. TP=1, TP=4, sampled runs, copies on and the shared MTP cap on another chip
    remain open device cells. The shared step-price correction needs no separate device cell: GLM's shared draft
    allocator uses the measured overhead including its head time, so changing the modeled step only redistributes
    that same total between fixed and head components. Its row learning, granted-depth and shared-depth limits are
    off. The preserved shared step values are still checked on the host; the other proposed behavior retains its gates.

16. **DFlash and DFlash2 drafters.** An external block drafter reads the target's hidden taps and drafts a block (or a
    tree) a step. It needs a trained drafter for each target; see [training a drafter](recipes/training-a-drafter.md).

17. **DSpark drafters.** A drafter that ships with the release (DeepSeek-V4.1's `mtp.i` blocks, MiniMax-M3.1's, Kimi
    K3's) drafts from the target's taps with a confidence schedule. Applies wherever a DSpark drafter exists for the
    target.

18. **Copy drafts.** Where a reply repeats earlier text, the lane host drafts the continuation from the context
    (`copy_index.zig`) and the target verifies it. It needs no drafter weights; [#2022] brought verified root and
    context copies to five CUDA lane hosts at once (opt-in). GLM's issue is [#1910].

    | Family | Gain (measured) | Where |
    | --- | --- | --- |
    | nemotron | level from 2K to 64K (−1.4..+1.0 %) on prompts without repeats | [#1974] |

19. **Shared draft rounds.** Several streams' drafts run in one batched pass, and a shared round prices drafts first
    and skips proposals it won't verify ([#1779]).

    | Family | Gain (measured) | Where |
    | --- | --- | --- |
    | qwen3_5 (Qwen3.6 MTP) | +18 % (2 streams), +32 % (4), +38 % (8) aggregate | [#1562] |
    | nemotron | −3.6..−7.5 % time at 2–8 streams | [#768] |

20. **Prompt chunk size and interleaving.** A prompt pass takes a wider chunk while no other stream decodes, and the
    normal chunk once one does, so a cold prompt is faster without longer decode waits ([#1554] option C).
    `--decode-share` ([#1327]) sizes a filling prompt's chunks beside live replies on every lane family.

    | Family | Gain (measured) | Where |
    | --- | --- | --- |
    | flashnext (C) | first token −5..−10 % at 2k–32k, decode waits unchanged | [#1733] |
    | glm (C) | cold first token −6..−7 % at TP=2, −7..−8 % at TP=3 | [#1823] |
    | flashnext (M) | 1,024-token reply beside a 64K prompt 48 → 8 s, prompt's first token 90 → 54 s | [#2167] |
    | flashnext (C), one GPU | prompts 15–23 % faster at `--parallel 1`, up to 8,192 rows a pass | [#2246] |
    | glm (C), TP=2 | prompts about 2 % sooner at 8K–128K, two chunks a step, replies unchanged | [#1423] |
    | gemma4 (C) | first token 31B −4..−5 %, E4B −9..−10 %, opt-in with `TF_GEMMA4_CHUNK_SOLO` | [#2391] |

    `TF_GLM_CHUNK_SOLO=4096` is slower with GLM's NVFP4 checkpoint on two GB10s ([#2288]).

21. **Load read-ahead.** Reader threads read a layer's checkpoint into the page cache while the loader packs the
    previous one (`TF_LOAD_PREFETCH`, 16 readers, on by default where it landed). [#2033] ports it to six more
    loaders, opt-in. The readers start only on Linux, so a family on Metal alone is `no`; Metal's shared loader
    reads large tensors on eight threads of its own.

    | Family | Gain (measured) | Where |
    | --- | --- | --- |
    | glm | cold load −52.6 % (TP=2), −53.9 % (TP=3) | [#1616] |
    | qwen3_5 (27B NVFP4) | cold load 25 / 22 → 15 s | [#1928] |
    | gemma4 (31B) | cold load 18 → 11 s | [#1928] |

22. **Direct load past the page cache.** Weights are read with direct I/O into the loader's buffers, so a cold load
    reads at the disk's speed and leaves no page cache behind ([#1873]).

    | Family | Gain (measured) | Where |
    | --- | --- | --- |
    | flashnext | cold load 74 → 48 s | [#1901] |
    | flashnext | 50 → 43.5 s, next chunk gathered while one packs | [#1960] |
    | spark25 (4B 8-bit) | cold load 5.0 → 2.5 s | [#2449] |
    | dory (NVFP4) | cold load 4.8 → 3.8 s | [#2449] |

23. **Page-cache drop after load.** Every CUDA loader drops its checkpoint's pages once the weights are up, so the
    memory goes back to KV and the prompt cache on unified-memory boxes ([#1884]). TP loaders wait for every rank
    ([#1887]). Metal's loaders drop theirs from the file cache ([#1885], [#1888]).

24. **L2 prefetch.** A side stream reads a decode window's next weights into L2 ahead of the kernels that use them.

    | Family | Gain (measured) | Where |
    | --- | --- | --- |
    | qwen3_5 (27B), nemotron | −3.5 % (a cost) with exact tokens, closed | [#2014] |
    | glm | decode +3.0..+4.1 % at TP=3, +0.8..+1.9 % at TP=2 | [#2162] |
    | flashnext, gemma4 | null: +0.45 % and +0.15 % at one stream, inside the runs' spread, same replies; not proposed | measured on a GB10, no PR |

25. **Tensor parallel ranks.** A model splits over two or more GPUs or hosts, so a model that doesn't fit one box
    runs, and decode reads each rank's share of the weights. Mixed Metal and CUDA ranks are tracked in [#1236].

26. **Pipeline stages.** Layer ranges on different nodes, one hidden row a token over Ethernet, so a GB10 and a Mac
    can serve one model ([#2005]). The stage link and its protocol are [#2012] and [#2060]; the 27B as a stage is
    [#2115], logits byte-equal to the whole model at two and three stages.

27. **`--profile` switches.** One registry names each opt-in switch with its backend, families, class and evidence,
    and the startup line prints what differs from the defaults ([#1660], [#1944]). Only `stable` is served;
    `fast-exact` is refused by name until switches qualify for it, and `fast` is not planned: it changes output tokens.
28. **Scale index.** An affine checkpoint stores a bf16 scale and bias beside every group's codes, a fifth of the
    bytes at 4 bits in groups of 32, and a tensor holds few distinct pairs. A 16-bit index a group into the tensor's
    table of pairs gives a kernel the same two values from fewer bytes. Flash Next's 4-bit expert kernels read it on
    Macs before tensor units behind `TF_FLASHNEXT_SCALE_INDEX` ([#2760], plan in [#2751]); the prompt gathers and the
    dense rows still read the stored arrays. `?` cells: the same count has not been run on those checkpoints.

29. **DSpine drafter.** Tencent's DSpine-b16 for Qwen3.8-27B, opt-in by `--drafter` with its folder, never the
    default. It reuses DFlash2's taps and tap ring and adds a code bridge and an exact-sender table.
    `core/dspine.zig` and the tree walker in `core/lanes/draft_tree.zig` are on main through the CUDA binding
    ([#2633]); the Metal binding ([#2634]) is still open. On a GB10, against DFlash2: agent prompts decode 2–20 %
    faster, chat and code 10–19 % slower. It needs the checkpoint's own DSpine drafter, so other families have none.

[#458]: https://github.com/ashhart/TensorFold/pull/458
[#468]: https://github.com/ashhart/TensorFold/pull/468
[#519]: https://github.com/ashhart/TensorFold/pull/519
[#558]: https://github.com/ashhart/TensorFold/pull/558
[#578]: https://github.com/ashhart/TensorFold/pull/578
[#600]: https://github.com/ashhart/TensorFold/pull/600
[#605]: https://github.com/ashhart/TensorFold/pull/605
[#615]: https://github.com/ashhart/TensorFold/pull/615
[#629]: https://github.com/ashhart/TensorFold/pull/629
[#655]: https://github.com/ashhart/TensorFold/pull/655
[#741]: https://github.com/ashhart/TensorFold/pull/741
[#768]: https://github.com/ashhart/TensorFold/pull/768
[#781]: https://github.com/ashhart/TensorFold/issues/781
[#796]: https://github.com/ashhart/TensorFold/pull/796
[#817]: https://github.com/ashhart/TensorFold/pull/817
[#822]: https://github.com/ashhart/TensorFold/pull/822
[#850]: https://github.com/ashhart/TensorFold/issues/850
[#855]: https://github.com/ashhart/TensorFold/pull/855
[#864]: https://github.com/ashhart/TensorFold/pull/864
[#879]: https://github.com/ashhart/TensorFold/pull/879
[#886]: https://github.com/ashhart/TensorFold/pull/886
[#935]: https://github.com/ashhart/TensorFold/pull/935
[#979]: https://github.com/ashhart/TensorFold/pull/979
[#997]: https://github.com/ashhart/TensorFold/pull/997
[#1125]: https://github.com/ashhart/TensorFold/pull/1125
[#1156]: https://github.com/ashhart/TensorFold/issues/1156
[#1163]: https://github.com/ashhart/TensorFold/pull/1163
[#1169]: https://github.com/ashhart/TensorFold/pull/1169
[#1174]: https://github.com/ashhart/TensorFold/pull/1174
[#1236]: https://github.com/ashhart/TensorFold/issues/1236
[#1256]: https://github.com/ashhart/TensorFold/pull/1256
[#1284]: https://github.com/ashhart/TensorFold/pull/1284
[#1293]: https://github.com/ashhart/TensorFold/issues/1293
[#1294]: https://github.com/ashhart/TensorFold/pull/1294
[#1327]: https://github.com/ashhart/TensorFold/pull/1327
[#1351]: https://github.com/ashhart/TensorFold/pull/1351
[#1357]: https://github.com/ashhart/TensorFold/pull/1357
[#1358]: https://github.com/ashhart/TensorFold/pull/1358
[#1359]: https://github.com/ashhart/TensorFold/pull/1359
[#1385]: https://github.com/ashhart/TensorFold/pull/1385
[#1388]: https://github.com/ashhart/TensorFold/pull/1388
[#1394]: https://github.com/ashhart/TensorFold/pull/1394
[#1411]: https://github.com/ashhart/TensorFold/pull/1411
[#1413]: https://github.com/ashhart/TensorFold/pull/1413
[#3135]: https://github.com/ashhart/TensorFold/pull/3135
[#1423]: https://github.com/ashhart/TensorFold/pull/1423
[#1426]: https://github.com/ashhart/TensorFold/pull/1426
[#1434]: https://github.com/ashhart/TensorFold/pull/1434
[#1439]: https://github.com/ashhart/TensorFold/pull/1439
[#1448]: https://github.com/ashhart/TensorFold/issues/1448
[#1465]: https://github.com/ashhart/TensorFold/pull/1465
[#1477]: https://github.com/ashhart/TensorFold/pull/1477
[#1490]: https://github.com/ashhart/TensorFold/issues/1490
[#1492]: https://github.com/ashhart/TensorFold/pull/1492
[#1501]: https://github.com/ashhart/TensorFold/pull/1501
[#1503]: https://github.com/ashhart/TensorFold/pull/1503
[#1508]: https://github.com/ashhart/TensorFold/pull/1508
[#1537]: https://github.com/ashhart/TensorFold/pull/1537
[#1554]: https://github.com/ashhart/TensorFold/issues/1554
[#1555]: https://github.com/ashhart/TensorFold/pull/1555
[#1562]: https://github.com/ashhart/TensorFold/pull/1562
[#1585]: https://github.com/ashhart/TensorFold/pull/1585
[#1616]: https://github.com/ashhart/TensorFold/pull/1616
[#1619]: https://github.com/ashhart/TensorFold/pull/1619
[#1620]: https://github.com/ashhart/TensorFold/pull/1620
[#1653]: https://github.com/ashhart/TensorFold/pull/1653
[#1657]: https://github.com/ashhart/TensorFold/issues/1657
[#1660]: https://github.com/ashhart/TensorFold/issues/1660
[#1696]: https://github.com/ashhart/TensorFold/pull/1696
[#1707]: https://github.com/ashhart/TensorFold/pull/1707
[#1714]: https://github.com/ashhart/TensorFold/pull/1714
[#1732]: https://github.com/ashhart/TensorFold/pull/1732
[#1733]: https://github.com/ashhart/TensorFold/pull/1733
[#1736]: https://github.com/ashhart/TensorFold/pull/1736
[#1750]: https://github.com/ashhart/TensorFold/pull/1750
[#1751]: https://github.com/ashhart/TensorFold/pull/1751
[#1755]: https://github.com/ashhart/TensorFold/pull/1755
[#1756]: https://github.com/ashhart/TensorFold/pull/1756
[#1762]: https://github.com/ashhart/TensorFold/issues/1762
[#1771]: https://github.com/ashhart/TensorFold/pull/1771
[#1779]: https://github.com/ashhart/TensorFold/pull/1779
[#1783]: https://github.com/ashhart/TensorFold/pull/1783
[#1786]: https://github.com/ashhart/TensorFold/pull/1786
[#1795]: https://github.com/ashhart/TensorFold/pull/1795
[#1823]: https://github.com/ashhart/TensorFold/pull/1823
[#1836]: https://github.com/ashhart/TensorFold/pull/1836
[#1850]: https://github.com/ashhart/TensorFold/pull/1850
[#1873]: https://github.com/ashhart/TensorFold/issues/1873
[#1880]: https://github.com/ashhart/TensorFold/pull/1880
[#1883]: https://github.com/ashhart/TensorFold/pull/1883
[#1884]: https://github.com/ashhart/TensorFold/pull/1884
[#1885]: https://github.com/ashhart/TensorFold/pull/1885
[#1887]: https://github.com/ashhart/TensorFold/pull/1887
[#1888]: https://github.com/ashhart/TensorFold/pull/1888
[#1901]: https://github.com/ashhart/TensorFold/pull/1901
[#1910]: https://github.com/ashhart/TensorFold/issues/1910
[#1924]: https://github.com/ashhart/TensorFold/pull/1924
[#1925]: https://github.com/ashhart/TensorFold/pull/1925
[#1928]: https://github.com/ashhart/TensorFold/pull/1928
[#1943]: https://github.com/ashhart/TensorFold/issues/1943
[#1944]: https://github.com/ashhart/TensorFold/pull/1944
[#1945]: https://github.com/ashhart/TensorFold/pull/1945
[#1959]: https://github.com/ashhart/TensorFold/pull/1959
[#1960]: https://github.com/ashhart/TensorFold/pull/1960
[#1974]: https://github.com/ashhart/TensorFold/pull/1974
[#2005]: https://github.com/ashhart/TensorFold/issues/2005
[#2011]: https://github.com/ashhart/TensorFold/pull/2011
[#2012]: https://github.com/ashhart/TensorFold/pull/2012
[#2014]: https://github.com/ashhart/TensorFold/pull/2014
[#2022]: https://github.com/ashhart/TensorFold/pull/2022
[#2033]: https://github.com/ashhart/TensorFold/pull/2033
[#2043]: https://github.com/ashhart/TensorFold/issues/2043
[#2060]: https://github.com/ashhart/TensorFold/pull/2060
[#2061]: https://github.com/ashhart/TensorFold/pull/2061
[#2099]: https://github.com/ashhart/TensorFold/pull/2099
[#2113]: https://github.com/ashhart/TensorFold/pull/2113
[#2115]: https://github.com/ashhart/TensorFold/pull/2115
[#2127]: https://github.com/ashhart/TensorFold/pull/2127
[#2146]: https://github.com/ashhart/TensorFold/pull/2146
[#2162]: https://github.com/ashhart/TensorFold/pull/2162
[#2165]: https://github.com/ashhart/TensorFold/issues/2165
[#2432]: https://github.com/ashhart/TensorFold/pull/2432
[#2499]: https://github.com/ashhart/TensorFold/issues/2499
[#2167]: https://github.com/ashhart/TensorFold/pull/2167
[#2211]: https://github.com/ashhart/TensorFold/pull/2211
[#2215]: https://github.com/ashhart/TensorFold/pull/2215
[#2221]: https://github.com/ashhart/TensorFold/pull/2221
[#2229]: https://github.com/ashhart/TensorFold/pull/2229
[#2238]: https://github.com/ashhart/TensorFold/pull/2238
[#2246]: https://github.com/ashhart/TensorFold/pull/2246
[#2279]: https://github.com/ashhart/TensorFold/pull/2279
[#2282]: https://github.com/ashhart/TensorFold/pull/2282
[#2283]: https://github.com/ashhart/TensorFold/pull/2283
[#2288]: https://github.com/ashhart/TensorFold/pull/2288
[#2306]: https://github.com/ashhart/TensorFold/pull/2306
[#2391]: https://github.com/ashhart/TensorFold/pull/2391
[#2449]: https://github.com/ashhart/TensorFold/pull/2449
[#2430]: https://github.com/ashhart/TensorFold/pull/2430

[#2257]: https://github.com/ashhart/TensorFold/pull/2257
[#2760]: https://github.com/ashhart/TensorFold/pull/2760
[#2751]: https://github.com/ashhart/TensorFold/issues/2751
[#2392]: https://github.com/ashhart/TensorFold/pull/2392
[#2436]: https://github.com/ashhart/TensorFold/pull/2436
[#2464]: https://github.com/ashhart/TensorFold/pull/2464
[#2375]: https://github.com/ashhart/TensorFold/pull/2375
[#2633]: https://github.com/ashhart/TensorFold/pull/2633
[#2634]: https://github.com/ashhart/TensorFold/pull/2634
[#2784]: https://github.com/ashhart/TensorFold/pull/2784
[#2764]: https://github.com/ashhart/TensorFold/issues/2764
