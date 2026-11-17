# GLM-5.3-Flash on CUDA: its kernels as dependent launches, receipt

| TP=2 on 2 × GB10, same checkpoint, both serves on the factored expert form | main | this branch | change |
| --- | --- | --- | --- |
| Plain decode, 128 tokens, one stream (tok/s; serves `fb1` `fb2`, `fh1` `fh2`) | 24.60, 24.38 | 26.11, 26.20 | +6.8% |
| Time between tokens, p50 | 40.45, 40.83 ms | 38.13, 38.01 ms | −6.3% |
| Drafted decode, 256 tokens, one stream (tok/s; `db1` `db2`, `dh1` `dh2`) | 43.45, 42.70 | 45.17, 44.45 | +4.0% |
| Four streams at once, summed (tok/s, median of 4 prompts) | 98.55, 99.85 | 103.40, 104.05 | +4.6% |
| Cold 8k prompt, first token (s) | 5.666, 5.659 | 5.609, 5.610 | −0.9% |
| Cold 32k prompt, first token (s) | 22.83, 22.80 | 22.58, 22.73 | −0.7% |
| Tokens | | | equal in every serve |
| 30 / 60 cold 8k replies of 16 tokens, drafts off | 1 distinct of 30 | 1 distinct of 60, main's | the first version: 4 distinct of 30 |

Tested implementation: binaries built from `60e673799f` (the branch before its rebase onto `3f1290ffe7`; the rebase
changed no line of the branch and the 23 commits it crossed touch no GLM or qmmf source). Base: `077998f7d9`.
Test date: 2026-11-17. Platform: 2 × GB10 (sm_121, 128 GB shared memory) at TP=2 over a direct RoCE link,
driver 580.178.04, Ubuntu 24.04, NCCL 2.30. Toolchain: Zig 0.17.0, nvcc 13.5 (V13.5.37) for the served binaries.
`binaries.txt` has their sha256. Checkpoint: `nvidia/GLM-5.3-Flash-FP4-Local` at
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`.

## Scope

Every GLM kernel and qmmf's decode linears launch as dependent launches (default; `TF_GLM_PDL=0` launches them
plain). This receipt qualifies served speed and exactness at TP=2 on GB10: plain and drafted decode, four streams,
cold 8k and 32k prompts, drafted = plain, concurrent = alone, resumed = cold, determinism after a long prompt, the
kernel contract and the early-load check. It does not qualify sm_90, sm_100 or sm_120 GPUs on hardware (no such GPU
here; their images pass the SASS check), TP=3 or one GPU.

## Results

| Check | Result |
| --- | --- |
| Plain decode and cold prompts, alternated main, head, head, main, then the head with `TF_GLM_PDL=0` (`runs/fb1` `fh1` `fh2` `fb2` `foff`) | Table above; `TF_GLM_PDL=0` 24.21 tok/s, 41.12 ms, 5.681 s and 22.74 s, main's tokens. Every serve's tokens equal: plain `1d7c9d905ba7`, 8k `43b05393cf25`, 32k `2e98fab70f4c` |
| Drafted decode and four streams, alternated (`runs/db1` `dh1` `dh2` `db2`) | Table above; drafted tokens `3fcd20ce2fb0` in all four, 2.94 to 3.01 tokens a chunk on both |
| Exactness on each binary (`runs/re-h`, `runs/re-b`) | `golden_http.py`: drafted = plain on 3 prompts, the same 3 token sha256s on head and main; `bench_concurrent.py --alone --serial`, 1, 2 and 4 streams, greedy and sampled: 84/84 equal their alone runs and 30/30 their serial runs on each; `resume_http.py`: repeated, `"draft": false` and two at once equal the cold reply (the prompt holds a clock nonce, so its reply differs between the two serves) |
| Determinism after a cold 8k prompt (`runs/determinism.txt`) | 16-token replies, drafts off: head 60/60 one reply, main 30/30 the same reply. The first version (`79c9ffe97f`, inputs `const __restrict__`): 24/30 that reply and 3 others (3, 2 and 1) |
| Kernel contract on GB10 (`contract/`) | `tf-cuda-test kernel-contract`, nvcc 13.3, sm_121: `SUMMARY 661 kernels: 661 pass gates 1-4, 0 fail, 8 known findings; 422 controls: 422 red as planted, 0 missed; 387 limit problems`, exit 1 from the limit problems alone; the 8 known findings are the registered ones (GLM's relu as `fmaxf`, qwen35's NaN-chunk merge). The limit problems are stale nvcc 13.3 records, none a launch past a limit: `kernel-contract-limits` on sm_80/86/89/120/121 fatbins finds 834 on main and 832 on this branch, every branch one also stale on main (`contract/limits-stale.txt`; 5,330 PASS, 0 FAIL). The contract ran before `1240c1908f`, which keeps main's records of 4 qmmf cluster tiles; the limits run on it |
| Dependent launch order on GB10 (`contract/glm-pdl-order.log`) | `tf-cuda-test glm-pdl-order`: 13 PASS; each GLM and qmmf kernel launched dependent after a slow writer equals its plain launch, and the planted read before the wait sees the poison |
| Early loads (`contract/check-pdl-loads.log`) | `zig build check-pdl-loads -Dsm=80,120,121`, nvcc 13.5: 0 early reads in 402 functions of 9 images (the first version: 12 in 201 functions, `contract/sass-early-loads.txt`) |

## The race the SASS check guards

A first version launched the same kernels dependent with their inputs `const __restrict__`. nvcc gives those
inputs non-coherent loads (`ld.global.nc`), and ptxas scheduled them (`LDG.E.CONSTANT`) above
`griddepcontrol.wait` in 12 kernels: the DSA selectors (`dsel_scan`, `dsel_merge`, `index_pools`), `dsa_tokens`,
seven KV-host kernels and `df_attn`. The PTX rules allow it, as `.nc` data must stay unchanged for the kernel's whole
life. A selector then read a position or a score before the kernel writing it had ended, and replies after a long
prompt varied while the first token did not. Those inputs drop `__restrict__`; outputs keep it.
`zig build check-pdl-loads -Dnvcc=…` disassembles the GLM images and qmmf and fails any sm_90 or later function
that reads global memory before its wait, or that has no wait and is not a plain-only entry.

## The expert-form confounder

The GLM load times its two expert forms once a binary and keeps the faster (`window cost` cache, keyed by the
executable). The head's binary kept `classic`, main's `factored`; per kernel (`runs/kprof-*.csv`, nsys over 3 × 128
tokens, both plain launches) the forms moved plain decode by about 3% on their own. The first A/B (`runs/rb1` `rh1`
`rh2` `rb2` `roff`) ran with them mixed and showed +3.5%. For the A/B above, the head's kept flag was set to `factored`,
as main's, so only the change under test differs; each serve's `form.txt` has its line. The form choice is outside this
change.

## Reproduction

Every serve, rank 0 and rank 1 (`TF_GLM_RANK`): `tensorfold-native serve CKPT --host 127.0.0.1 --port 8090 --name
local-model --no-update-check --context 65536 --parallel 4 --prompt-cache-gib 0` (`ab`, `dc`), `--context 40960
--parallel 4` (`ex`). In each serve's folder: `tools/suites.sh ab|dc|ex REPO prompts`. Table: `python3
tools/pdl_ab.py runs fb1 fh1 fh2 fb2 foff`, `… db1 dh1 dh2 db2`. Determinism: `python3 tools/first.py 8090
prompts/p8192.json 16 60` against a fresh serve with `--prompt-cache-gib 0`.
Contract, nvcc 13.3 (`-Dnvcc=/usr/local/cuda-13.3/bin/nvcc`): `zig build kernel-contract -Dnvcc=…`; `zig build fatbins
kernel-contract-fatbins -Dnvcc=… -Dsm=80,86,89,120,121`, then `tf-cuda-test kernel-contract-limits zig-out/fatbin` on
this branch and on main; `tf-cuda-test glm-pdl-order`. Early loads: `zig build check-pdl-loads -Dnvcc=…
-Dsm=80,120,121`.
