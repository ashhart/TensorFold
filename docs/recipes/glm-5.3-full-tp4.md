# Full GLM-5.3 on four DGX Sparks (tensor parallel 4)

TensorFold's `glm_moe_dsa` family serves the full GLM-5.3 (753B total, ~40B active; `model_type: glm_moe_dsa`) across
four DGX Sparks (GB10, 128 GB unified memory each), one rank per Spark, over their ConnectX-7 200 GbE RoCE fabric.
It is TensorFold's first four-rank engine (the CLI now takes `--tp 1 | 2 | 4`, per family).

Everything TensorFold promises holds: a drafted reply is bit-identical to a serial one (every verify row has a serial
step's bits: row-invariant kernels, rank-order fp32 sums), and every rank computes the same bits.

## Measured (four DGX Sparks, 2.75 bpw EXL3 checkpoint)

Single stream, 32K-token context, temperature 1.0 / top-p 0.95, 512 tokens (the vLLM rows: the same checkpoint and
hardware, vLLM with EXL3 kernels, MTP k = 2, CUDA graphs, RoCE all-reduce):

| | Prose | Code | Step | Prefill 8K / 32K | TTFT 32K |
|---|---|---|---|---|---|
| TensorFold, MTP k = 2 | **24.6 tok/s** | 27.3 tok/s | 78.5 ms | 773-783 / 705 tok/s | 46 s |
| TensorFold, DFlash2 (depth 7, confidence 0.4) | 23.0 | **32.4** | 87-101 ms | | |
| vLLM (same checkpoint, tuned) | 23.9 | 30.3 | 77-78 ms | ~755 / 749 tok/s | 43.6 s |

Short context, greedy: MTP 28.8 prose / 31.7 code; DFlash2 28.3 / **40.0** (3.6 tokens a verify step).
Checkpoint quality vs BF16 (exllamav3 `model_diff`, 65,536 tokens): KL 0.124 nats, top-1 agreement 89.6 %.

## Run

Weights: an EXL3 GLM-5.3 checkpoint, e.g. [keys-GLM-5.3-EXL3-2.75BPW](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW)
(routed experts 2/3/4-bit per expert, mean 2.75; 5-bit non-expert layers; 8-bit MTP), at the same path on all four
nodes (an NFS export works; each rank reads only its share through safetensors slices).

```sh
export NODES="spark1 spark2 spark3 spark4"   # fabric addresses, rank 0 first (serves HTTP)
export IMAGE=<CUDA 13 + torch image>          # + cuda-exl3 and vLLM for the fast prompt path (optional)
export CKPT=/models/GLM-5.3-EXL3-2.75BPW
export GIDS="3 3 3 3"                         # RoCE v2 GID index of each rank's port
tools/tp4_run.sh comm                         # fabric check: exact reduce-scatter bits, collective latencies
tools/tp4_run.sh serve --context 36864        # OpenAI-compatible server on rank 0, :8890
```

DFlash2 drafts: add `DOCKER_ENV="-e TF_GLM53_DFLASH=/models/GLM-5.3-DFlash2"` (e.g.
[incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2)); requests choose with `"tf_mtp"`:
`"normed/normed"` (MTP, default), `"dflash"`, or `"auto"` (MTP or DFlash2 each round, whichever is emitting faster).
`~/tf-glm53/DFLASH_CFG` on every node (`{"depth": 7, "confidence": 0.4}`) tunes DFlash2 at run time.

Node settings that matter on GB10:

- `sysctl vm.compaction_proactiveness=0` on every node: proactive memory compaction migrates pages under the GPU and
  stalls a rank ~130 ms at a time (a quarter of decode rounds took 200 ms instead of 70 before this).
- `NCCL_MAX_NCHANNELS=2` (the launcher's default): with NCCL's default channels a prompt chunk's all-reduce cannot run
  beside compute; with two it hides ~2/3 of its time.
- Keep bulk transfers (NFS copies, uploads) off the RoCE port while serving: they add decode latency.

## How it works

- `fused.py`: the whole forward on static buffers and device positions (CUDA-graph capturable), every kernel
  row-invariant: MLA with RoPE on the 576-wide latent (absorbed queries, chunked attention over per-row key lists),
  the V3.2 token-level indexer (top-2048, ties broken by position so the chosen set never depends on the window),
  EXL3 linears (the universal kernel; a decode-once tensor-core GEMM for prompt chunks), routed experts (TensorFold's
  universal experts kernel), RMSNorm/router/top-k/residual kernels from `glm5_next`.
- Collectives: decode windows reduce with a one-shot RoCE all-reduce (b12x RoCEnante, rank-order sums: the same bits
  as NCCL's all-gather + rank-order sum) when available; prompt chunks use NCCL's ring all-reduce of bf16 partials,
  as two micro-batches whose all-reduces overlap each other's compute.
- `experts_cx.py`: with cuda-exl3 available, each MoE layer's experts live once in cuda-exl3's per-width stacks; prompt
  chunks run its grouped GEMM and decode windows read the same memory through TensorFold's experts kernel
  (`gu_stride`: gate and up as column blocks of the fused trellis).
- `runner.py`: prompt chunks (equal sizes up to 4096 rows), then MTP rounds whose cache refresh merges with the first
  draft, DFlash2 rounds, or auto; every decode graph captured at startup.
- `dflash.py`: `glm5_next`'s DFlash2 drafter over this family (a bf16 vocabulary-share head).

Design notes and the history of each measurement: `docs/design/glm-moe-dsa-tp4.md`.
