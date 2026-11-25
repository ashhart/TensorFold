# Full GLM-5.3 on CUDA at TP=3 (#4900): receipt

Tested implementation: `f3bed06e…` (commit.txt has the commit and its tree): this pull request's 15 commits on
`main` `ac79d6db6` (the 15th adds the own-partial residual's byte comparison to `glm-ref-dsa`). The receipt's commit
adds this folder and changes no source.
Test date: 2026-11-25. Platform: three DGX Spark (NVIDIA GB10, sm_121, 128 GB shared memory), one rank a host,
200G RoCE between them; ranks in `nvcr.io/nvidia/pytorch:26.07-py3` containers; NCCL over the RoCE links.
Toolchain: Zig 0.17.0, nvcc 13.0 (V13.0.88) for sm_121. binaries.txt has both binaries' sha256.
Checkpoint: `Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold` at `2d747d0e30eca6e3fe37ba63c471cdf9172faa83`, the same 146
files on all three hosts (checkpoint-sha256.txt: every file's sha256, as Hugging Face stores it).
KV cache: BF16 (the default; no `--kv-cache` flag anywhere in this receipt).

## Launch settings

Every server rank (launch-tp3-serve.sh, the `--binary` that `tools/conformance.py` starts):
`tensorfold-native serve CKPT --backend cuda --context 32768 --parallel 2 --no-thinking --tp 3 --rank R` plus the
profile's `--prompt-cache-gib 0` (cache-off server) or `2` (cached server), with `TF_GLM_RANK=R`, `TF_GLM_WORLD=3`,
`NCCL_COMM_ID=<link-a>:29651`, `TENSORFOLD_MEMORY_RESERVE_GIB=6`, drafts on (MTP + copies). Rank 0 serves; ranks 1 and
2 follow. The `tf-cuda-test` checks: launch-tp3-test.sh (`glm-cap-check CKPT dc.json RANK 3 IDFILE` on the three
hosts) and launch-one-gpu.sh (`glm-ref-dsa`, `kernel-contract glm mla` on one GB10).

## Results

| Check | Result |
| --- | --- |
| `tools/conformance.py tools/conformance_profiles/glm53-cuda.json` at TP=3, cache-off and cached servers one after the other (`records/conformance.json`, `logs/conformance.log`) | 7 passed, 0 failed, 0 skipped. drafted == plain 10/10 (401 of 683 drafts accepted); repeat == first 10/10 (another seed differs); concurrent == solo 28/28 (levels 1 and 2, drafted and plain, 16 with a long prompt among the decoders; 2 to 4 chat requests open at once); resumed == fresh 6/6 (4,096 tokens resumed each); refusals 4/4; cancel 4/4 equal after a cut, census back to idle; evict 3/3 against cache-off (kept, resumed, evicted; cached 4,096 -> 0 after 10 fill prompts). `records/conformance.json` keeps all 103 requests with the sha256 of their token ids, prompt and cached counts and draft counts (its `binary_sha256` is the launcher's, launch-tp3-serve.sh; the engine it starts is binaries.txt's `tensorfold-native`) |
| `glm-cap-check` at TP=3 (`logs/glm-cap-check.r*.log`) | 5 prompts, with and without request starts, stepped under changing caps (16 steps ending inside a cold chunk) against their cold passes: the first token, the next window's logits and tokens, and every state byte but the index rings' (at pool 1 they hold a step's keys until the pooled keys copy them) equal |
| `glm-ref-dsa`, one GB10 (`logs/glm-ref-dsa.log`) | 11 bf16 projection classes at both TP=3 shares, prompt and decode rows, and the router at 6144, the worst column at 0.714 of its float64 tolerance |
| Own-partial residual, byte for byte (in `glm-ref-dsa`) | `tf_glm_residual_add_own` against `tf_glm_residual_add` on the same inputs, with NaN in the rank's own slot of the gathered buffer and its partial read in place: 40 cases (worlds 2 and 3, every rank, 1, 7, 70 and 1,024 rows, widths 6144 and 2048), the same bytes in every case. Split prefill's FP32 sum keeps its order: rank 0 first, each rank's partial added in turn |
| `kernel-contract glm mla`, one GB10 (`logs/kernel-contract.log`) | 62 kernels pass gates 1-4, 0 fail, 5 known findings; 15 controls red as planted, 0 missed; 4 limit problems, explained below (`main` shows the same four on this machine) |

## The kernel contract's four limit problems, a symbol at a time

`kernel-contract glm mla` on this head reports the same four limit problems as `main` on the same machine, and as
the sm_120 run posted on the pull request. All four are planted-control entrypoints in the `contract_controls` image,
whose records are `main`'s (`zig/src/core/contract/glm_f32w_pending.zig`, recorded 2026-10-24); this pull request
does not change them:

| symbol | registers live / recorded | static shared live / recorded | declared in the source |
| --- | --- | --- | --- |
| `tf_ctl_glm_old_rmsnorm_512` | 22 / 22 | 128 / 16 | `float smem[4]` (4 warps) |
| `tf_ctl_glm_old_rmsnorm_1536` | 40 / 40 | 128 / 16 | `float smem[4]` |
| `tf_ctl_glm_old_rmsnorm_4096` | 39 / 39 | 128 / 32 | `float smem[8]` (8 warps) |
| `tf_ctl_glm_old_hc_finish` | 79 / 79 | 128 / 48 | `float smem[8]`, `float pre[4]` |

The records match the source. The live 128 bytes come from the image: since `542deb953` (2026-11-14) it also holds
`contract_hcn_controls.cuh`, whose `flashnext_hcn.cuh` declares `extern __shared__ __align__(128) unsigned char
buf[]`. Dynamic shared memory then starts on a 128-byte boundary for every kernel of the image, and nvcc pads each
kernel's static block up to it. Shown offline with nvcc 13.0 for sm_121 (`cuobjdump -res-usage`, the 1024 bytes the
system reserves included):

- `contract_controls.cu` alone: the four at 1040, 1040, 1056 and 1072 bytes (1024 + 16, 16, 32, 48);
- with each family header added in turn: unchanged, except with `contract_hcn_controls.cuh`: 1152 for all four;
- `contract_controls.cu` plus one kernel declaring `extern __shared__ __align__(128)`: 1152 for all four; with
  `__align__(64)` instead, `tf_ctl_glm_old_rmsnorm_4096` at 1088 (1024 + 64); with `__align__(16)`, 1056 (its own 32).

The new norms of this pull request (`tf_glm_rmsnorm_6144`, `tf_glm_rmsnorm_2048`) and `main`'s replacements live in
the `glm_glue` image and pass with their recorded 32 and 16 bytes. The four records need re-measuring against the
image as it is built now. That is `main`'s and fits a separate change.

## Files

- commit.txt: the tested commit, its tree, the base, and this pull request's commits
- binaries.txt: sha256 of `tensorfold-native` and `tf-cuda-test` (built on host-c from commit.txt's commit: `zig build
  -Dgpu=spark -Dnvcc=... -Doptimize=fast` and `zig build compile-cuda-tests -Dnvcc=...`)
- checkpoint-sha256.txt: every checkpoint file's sha256 at `2d747d0e30eca6e3fe37ba63c471cdf9172faa83`
- launch-tp3-serve.sh, launch-tp3-test.sh, launch-one-gpu.sh, dc.json: how every server and check started
- records/conformance.json: the conformance receipt (profile, gates, servers, 103 request rows)
- logs/: conformance, both conformance servers' rank-0 logs, `glm-cap-check` (each rank), `glm-ref-dsa`, the
  kernel contract

## Publication handling

CRLF line ends made LF and trailing blanks dropped; machine names, internal addresses, home and temporary paths, and
user and container names replaced with placeholders (host-a, host-b and host-c run ranks 0, 1 and 2; `<link-a>`
names the RoCE link; `redaction-report.json` counts them a file); result values, token ids and token hashes kept.
SHA256SUMS pins every published file except itself.
