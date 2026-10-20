# GLM-5.3-Flash FP4-Local on CUDA: final-source receipt

Tested implementation: `80517fc36ea6ee9183c5237ae92b1abd70619f75` (commit.txt has the tree and the eight commits).
Base: `v1.1.1` (`b10231b94f8adf3d4132acebdc0a1ac87e53a838`); baseline binaries are plain v1.1.1 built the same way (`bin-b10231b9`).
Test date: 2026-10-20. Platform: NVIDIA DGX Spark (GB10, sm_121, 128 GB shared memory), two at TP=2 over a direct
RoCE link (two rails), three at TP=3 over a ring of direct links. Containers pinned to the efficiency cores
(`--cpuset-cpus=0-4,10-14`, as 1.1.1's runbook in #1033).
Toolchain: Zig 0.17.0, nvcc 13.0 (V13.0.88) for sm_121; runtime NCCL 2.30.7. binaries.txt has every binary's sha256.
Checkpoints: `nvidia/GLM-5.3-Flash-FP4-Local` at `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`,
`nvidia/GLM-5.3-Flash-NVFP4` at `423acf37583782c51c142d145aef733d72943d93`.
Grammar: `lib/libtfgrammar.so` from the v1.1.1 release archive (a source build makes no `lib/`); lib/ has its sha256.

## Scope

This qualifies FP4-Local loaded as stored and served at TP=2 and TP=3: the loader gate on every layer, the lane
invariants, served exactness (drafted = `"draft": false`, resumed = cold, concurrent = alone, prompts admitted
together = alone), media, structured output and tool calls, prompt logprobs against a BF16 reference, per-rank load
and peak memory, the kernel contract and glm-ref, the no-GPU CI steps, NVFP4 before and after against plain v1.1.1,
and FP4-Local against NVFP4 on matched inputs. It does not qualify Metal, or a comparison with vLLM.

## Results

| Check | Result |
| --- | --- |
| Loader gate, TP=2, every layer (`tp2/gate/all-layers-*`) | Ranks 0 and 1: 894 uploads as the checkpoint cut, 486 as the host packers make them, 0 differ (3 left out: the draft head's GPU-quantized tiles) |
| Loader gate, TP=3, every layer (`tp3/gate/all-layers-*`) | Ranks 0, 1 and 2: 894 and 486, 0 differ (TP=3's FP8 shares start inside a 128 block: #1024's repacking from the origin, per rank) |
| Loader gate, six layers (0, 3, 4, 43, 44, MTP) | TP=2 and TP=3: 115 and 72, 0 differ |
| `glm-lanes-cover`, `glm-lanes-concurrent`, `glm-shared-check` | TP=2 and TP=3: 40/40, 38/38, and 108 rows of 5 streams over 12 rounds equal to their lone twins byte for byte |
| Golden ids (`tp2/serve/ex-*`) | NVFP4 on this head equals plain v1.1.1 on all three prompts (count `6faafd98`, prose `b82116bd`, code `ddd42119`), drafts and tokens per round included; FP4-Local drafted = `"draft": false` on all three; at TP=3 (`tp3/serve/ex3-*`) drafted = `"draft": false` on both checkpoints |
| Concurrent (`bench_concurrent.py --alone --serial`, levels 1, 2, 4) | 12 cells a checkpoint at TP=2 and TP=3, 0 unequal, 0 failed |
| Resume (a chat of 6,593 to 6,595 tokens resumed from its kept state, `"draft": false`, two at once) | Every reply equals the cold one, both checkpoints, TP=2 and TP=3 |
| Prompts sent together beside a decode (`batch`) | Four fresh prompts, each equal to its alone run, both checkpoints, TP=2 and TP=3 (GLM fills them in chunks between rounds by default) |
| 1.1.1's batched prompt pass (`bt-*`, `TF_GLM_PROMPT_STEPS=0`) | Three fresh prompts shared one pass (first tokens 0.28 to 0.53 ms apart, `prefill_seconds` within 43 µs), each equal to its alone run: plain v1.1.1 and this head on NVFP4, this head on FP4-Local (its MXFP8 `lm_head` through `cuda_shared.heads`) |
| Media, media resume (`--vision`, `assets/tensorfold-hero.png`) | Drafted = `"draft": false`, repeated, three at once = alone; resumed past the image = cold: FP4-Local and NVFP4 at TP=2, FP4-Local at TP=3, plain v1.1.1 for resume |
| Structured output and tools (`s9/`, #1017's and #1018's items) | At TP=2, 26/26 on each checkpoint with drafts: schemas with thinking off and on, a schema plain already meets, `"draft": false`, seeded sampling twice, nested, resume from cache, required, named, parallel, single, nested and empty tool calls each again with `"draft": false`, mixed streams and prompts before and behind a constrained one; `--no-drafts` serves: 22/22; `compare`: all 22 shared greedy replies equal, and the resumed reply equals a cold one |
| Prompt logprobs against BF16 (`s5/`, 144,285 positions) | Mean NLL: BF16 1.00691, FP4-Local 1.01811 (+1.11%), NVFP4 1.02069 (+1.37%); top-1 equal to BF16's 0.9282 and 0.9268; KL over BF16's top 20 (an approximation) 0.0528 and 0.0537 |
| Memory (`memory/summary.json`, `memory_receipts.py`) | Every rank's load and cleanup receipts recorded; TP=2 FP4-Local 90.27 GiB at load end, 92.96 GiB peak, 0.00 after cleanup; TP=3 63.01, 62.88 and 57.74 GiB at load end |
| TP=3 rank loss (`tp3/serve/sp3-fl`) | Rank 2 killed: ranks 0 and 1 left 7 s later |
| kernel contract (`single/`) | `SUMMARY 91 kernels: 91 pass gates 1-4, 0 fail, 0 known findings; 7 controls: 7 red as planted, 0 missed; 1533 limit problems`; plain v1.1.1 (`single-v1.1.1/`): 85 kernels, 85 pass, the same 1533 (the contract counts each output: the two new kernels add six, rmsnorm's rows and 64-sums and hc_finish's rows, 64-sums, post and comb, and the launches go from 1101 to 1103) |
| kernel-contract-limits, sm_80 to sm_121 | 877 problems, the same 877 on plain v1.1.1; this branch's two kernels (`tf_glm_rmsnorm_4096_f32w`, `tf_glm_hc_finish_f32w`) pass on all five (3000 PASS lines against 2990) |
| glm-ref | 12 pass, every RESULT line equal to plain v1.1.1's |
| No-GPU CI (`ci/`) | See ci/ci-summary.txt: every step passes; with fresh caches upstream's `cuda_cost_identity` mtime test fails here, on plain v1.1.1 too |

Speed, medians (`tp2/serve/sp*`, `tp3/serve/sp3-*`); at TP=2 each binary and checkpoint served twice, alternating
(both values given), at TP=3 once:

| TP=2 | plain v1.1.1 | this head |
| --- | --- | --- |
| NVFP4, drafted decode, fibonacci-raw t=0 / t=1 (tok/s) | 56.2, 56.1 / 36.0, 35.9 | 56.1, 56.0 / 36.0, 35.9 |
| NVFP4, drafted decode, gpu-chat-no-think t=0 / t=1 (tok/s) | 30.6, 30.7 / 28.3, 28.3 | 30.6, 30.6 / 28.4, 28.4 |
| NVFP4, cold prefill of 18,789 tokens (s) | 14.19, 14.13 | 14.14, 14.10 |

| Matched inputs, this head | NVFP4 | FP4-Local |
| --- | --- | --- |
| TP=2, `"draft": false` decode, fibonacci-raw and gpu-chat-no-think (tok/s) | 18.3, 18.3 | 24.5 to 24.6 |
| TP=2, cold prefill of 9,600 and 38,400 token ids (s) | 7.22 and 28.46, 7.21 and 28.35 | 7.22 and 28.52, 7.24 and 28.53 |
| TP=3, `"draft": false` decode (tok/s) | 21.8 to 21.9 | 31.4 to 31.6 |
| TP=3, cold prefill of 9,600 and 38,400 token ids (s) | 6.33 and 25.06 | 5.93 and 23.44 |

## Reproduction

Every serve: `tensorfold-native serve CKPT --name local-model --parallel 4 --context 40960 --port 8095`, drafts on,
one rank a host (`TF_GLM_RANK`, `TF_GLM_WORLD`), each host's page cache of both checkpoints dropped first; `md*` and
`mr-*` add `--vision`, `x9-*-nd` add `--no-drafts`, and `bt-*` set `TF_GLM_PROMPT_STEPS=0`. The checks are tools/:
`golden_http.py` on dc.json, upstream's `bench_concurrent.py`, `bench_openai.py` and `bench_cache.py`,
`openai_aligned.py` (the `"draft": false` decode), `resume_http.py`, `batch_http.py` (`one-pass` for `bt-*`),
`media_http.py`, `media_resume_http.py`, `x9_live.py` (and its `compare`), `s5_prompt_lp.py` (prompt_logprobs 20 on the
128 held-out sequences) with `s5_compare.py` against `s5_bf16_ref.py`'s reference. `mkprompts.py` makes the matched
token-id files. runs.log has one line a check.

## Remaining coverage limits

- TP=3 needs `TF_GLM_MM=0` on every rank: the default glm_mm refuses KDA's input stack there with `BadShape`
  (#1067, whose thread has plain 1.1.1's run; here `tp3/serve/smoke3-mm1`, this branch with the bootstrap on FP4-Local),
  so TP=3 has no batched prompt pass (it needs glm_mm's tiles).
- TP=3's ranks meet through a local bootstrap commit that is not part of this branch (`tp3/bootstrap.patch`); the
  gate and the lanes at TP=3 ran this branch's own binaries.
- 1.1.1's batched prompt pass runs only with `TF_GLM_PROMPT_STEPS=0` on rank 0; `bt-*` set it, the other serves ran
  the default.
- The KL is over BF16's top 20 only, and both sides' logits are bf16, so best tokens tie; `s5-compare.json` also
  gives top-1 equal within ties.
- runs.log's FAIL lines: `smoke3-mm1` is the #1067 probe above; the first `x9-fl` ran before `libtfgrammar.so` was
  in place (HTTP 400 naming it) and its retake is the one published; `single` is the limits line above, on this head
  and on plain v1.1.1 alike.
- Metal and MLX were not run (no Mac). No vLLM comparison is claimed.

## Publication handling

Machine names, internal addresses, home paths and working-session names replaced, CRLF line ends made LF and trailing blanks dropped; result
values, token ids and token hashes kept.
`redaction-report.json` counts the replacements a file. The memory wrapper's receipts are raw for TP=2 `ex-fl` only;
`memory/summary.json` summarizes every receipt with its original's sha256. The prompt-logprob rows (87 MB a
checkpoint) are not included; `s5/plp-sha256.txt` has their sha256. SHA256SUMS pins every published file except
itself.
