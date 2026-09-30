# DeepSeek V4 CUDA port — T01 scope / context summary

Authoritative sources (read these before editing):
- CUDA plan: `docs/recipes/deepseek-v4-cuda-plan.md` (rev `e65c8d9fb2ef1308b5bc44164a4dd6629dc3d232`). Defines R1–R8, G0–G11, milestones 0–7. Serial build-ready comes first; DSpark (milestone 6, R8/G11) is an optional follow-up.
- Worker prompt: `/home/josh/Documents/Projects/TensorFold/docs/recipes/deepseek-v4-kanban.md` (rev `afee846`). Keep live services up; commit locally; log blockers precisely.

## T01 scope (this card feeds t_bf4e4540/T01)
Gates: **G0** (scope/reuse, R1–R3) — import/discovery + backend-option tests: CPU family discovery imports neither torch backend nor MLX; existing MLX selection stays valid; unsupported ranks/features fail before loading. Record donor and upstream revisions. Requirements in force for T01: **R1, R2, R3, R7** (below). Milestones 0–5 then 7 for first serial release; G10 (real checkpoint) stays pending if full-model cannot fit → artifact labeled "build ready, full-model qualification pending".

## Requirements (R1–R3, R7)
- **R1** — existing DeepSeek endpoint, Hunyuan, moderation keep running; build/tests never manage their lifecycle or client routes.
- **R2** — read the named 0731 GGUF unchanged; IQ2_XXS/Q2_K/Q8_0/F16/F32 validated from actual tensor descriptors. No full BF16 expert expansion, replacement quant, or download.
- **R3** — native TensorFold family CUDA engine, `tp=1`, one request, no drafting for first release; preserve MLX family and shared server contracts.
- **R7** — build artifact, checkpoint identity, tokenizer provenance, toolchain, and test results reproducible and recorded.

## Upstream references
- **issue #14** — DeepSeek support issue; release follow-up shipped Mac support, left CUDA pending. Confirm whether the maintainer has unpublished CUDA work before starting a port.
- **PR #119** — numpy/mmap GGUF reader for GLM Q8_0 conversion. Its metadata parsing is a reuse candidate; its GLM mappings and Q8_0 conversion do NOT implement DeepSeek IQ2_XXS/Q2_K CUDA execution.

## Installed ds4 donor (MIT-licensed, audit before reuse)
- Entrpi/ds4 → `/home/josh/code/ds4` (rev `d183482`, remote github.com/Entrpi/ds4.git).
- Entrpi/ds4-on-spark → `/home/josh/Documents/Projects/ds4` (rev `4cb79fd`).
Pin a donor revision, preserve license notices, and test its arithmetic against TensorFold's contract before importing kernels/readers. Wrapping a separate ds4 HTTP server does not establish a TensorFold engine or qualify its exactness contract.

## 0731 architecture differences
Target file `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` (~81 GiB) uses mixed layout: IQ2_XXS routed-expert gate/up, Q2_K routed-expert down, Q8_0 dense projections, F16/F32 tensors. Validate actual tensor types/dimensions from the file — a filename suffix is NOT sufficient evidence of generation/head compatibility; unsupported architecture differences fail explicitly. Existing MLX conversion keeps ~151 GiB resident and cannot substitute for this checkpoint on one Spark. Exact 0731-vs-MLX differences are T01's audit deliverable.

## Operational constraints (verified live)
- Live services: `:8000` = ds4 `deepseek-v4-flash` (`/v1/models`: ds4.c, context 262144, no drafting); `:8100` = Hunyuan (`/health`: pipeline_loaded=true); `:8101` = CUDA lane (`/health`: model_loaded=true, device cuda). All three probed OK — do NOT stop/restart or bind these ports.
- Candidate port is loopback `18000` only; no production cutover, no redirect/restart of live services.
- Treat live ds4 binary, installed launcher, service units, and checkpoint files as read-only. No edits to launcher/GGUF/binary.
- Isolated caches: separate checkout + virtualenv/container with its own build, extension, and model-state caches. Read GGUF in place; write generated config/tokenizer metadata into a candidate model directory.
- At most **two compiler jobs**; bound CPU parallelism and host memory; one GPU test worker; ~256 MiB device allocation ceiling per process (lower it if the live stack has less headroom). Tests skip with a clear reason when the live workload leaves insufficient headroom; a GPU skip = pending qualification, never a pass claim.
- Commit locally: leave an auditable commit on the implementation branch when tests pass; include the hash in the card result. Never merge/push.
- Log blockers precisely: record the exact blocker when an input, upstream decision, resource budget, or required measurement is missing.
- `COMPANION_RESERVE_GIB`: no invented default; select from measured companion peaks and headroom before the helper command is usable. `--preflight-only` validates/reports without install or compile.

## On-disk assets (present, verified)
`/home/josh/gguf/` holds the 0731 target GGUF plus MTP (`...MTP-Q4K-Q8_0-F32.gguf`, 3.6G), `DSpark-drafter-Q2K-Q8-0731.gguf` (6.5G), and `Huihui-...-BF16-abliterated-ds4-Q2.gguf` (81G).
