# Qwen3.8-27B

This branch adds experimental [packed affine formats](../quantization.md), including 8-bit and mixed layer precision.

The `qwen3_5` family combines Gated DeltaNet and full attention. The standard recipe below uses its
4-bit/group-64 checkpoint; this branch also adds the affine formats listed in the quantization guide.

```bash
tensorfold pull Vontra/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --name bench
```

DFlash2 is used automatically once pulled. On MLX, `--drafter none` disables that draft model;
`--no-drafts` disables all drafts on either backend. CUDA requires DFlash2 unless `--no-drafts` is set.
The target verifies every proposed token against its own serial sample.

## MLX

M5 tensor-unit GPUs run the lane decoder with draft trees. M1 through M4 use `row_forward` and the
packed row decoder with windows of up to 16 rows by default. Its existing 4-bit formats use `simd_qmm`,
and other supported affine formats use the general packed kernel. Both paths use the same arithmetic for serial
and drafted calls. Load-time checks determine usable window widths and shared-forward support.

The engine can share a round across requests while keeping each stream's attention, recurrent state and
sampling independent. DeltaNet commits replay the accepted path; attention commits retain only its keys.
Prompt chunks start at detected assistant-message boundaries and the second message when these are at
least 256 tokens beyond the previous chunk start, or after 2,048 tokens if no earlier boundary qualifies.
Prefix reuse resumes only at these chunk starts, so a follow-up can reuse the state before its previous
reply. The plan comes from rendered tokens; a template without detected markers uses 2,048-token chunks.

## Weights other than 4-bit

The M5 lane kernels accept MLX affine 2-, 3-, 4-, 5-, 6- and 8-bit projections in groups of 64.
They widen packed values for the tensor operations without changing those values. Mixed-width stacks
keep separate calls where a fused projection needs one width. Examples include
`Vontra/Qwen3.8-27B-oQ2` and `Vontra/Qwen3.8-27B-oQ4`.

The packed row readers on Apple Silicon and the CUDA readers cover MLX affine 2/3/4/5/6/8-bit projections
with groups of 32/64/128, including mixed layers. CUDA also reads [EXL3 packs](#exl3-checkpoints-experimental).
A fused projection keeps its members separate where their bit width or group size differs. Unsupported formats
and tied embedding heads are refused from `config.json` before weight downloads and again at load; loaded
projections must also be covered by the selected decoder. `--lane-kernels on` requires M5 tensor units and
formats they read; `auto` falls back to the packed row decoder for the rest.
Lower weight precision does not guarantee faster decode or a fitting context. Release memory and
quality comparisons are TBD [release-0.3.5].

## CUDA

Use the [CUDA container setup](../../RUNBOOK.md#dgx-spark). One or two ranks are supported.
Pull the model and drafter on every rank, then start rank 1 before rank 0:

```bash
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --tp 2 --rank 1 --master 192.0.2.1
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --tp 2 --rank 0 --master 192.0.2.1 --name bench --host 0.0.0.0
```

The verify matmul fixes reduction order by weight shape. Tree attention reads only committed keys and the
node's own path; recurrent commits replay that path. Two-rank reductions gather fp32 partials and add in
rank order. Each rank count has its own serial reference. See the
[CUDA kernel map](../../src/tensorfold/families/qwen3_5/cuda/README.md).

### EXL3 checkpoints (experimental)

The CUDA engine also reads turboderp's EXL3 packs of the model (`turboderp/Qwen3.8-27B-exl3`, a branch per size,
`mul1` codebook, 6-bit head) through the shared EXL3 module ([EXL3 weights](exl3.md)) and drafts with the same
`z-lab/Qwen3.8-27B-DFlash2`. One GPU: two ranks read the MLX checkpoint. Download a size by its branch, then
serve the folder:

```bash
python -c "from huggingface_hub import snapshot_download as d; d('turboderp/Qwen3.8-27B-exl3', revision='3.00bpw', local_dir='qwen27b-exl3-3.00bpw')"
tensorfold pull z-lab/Qwen3.8-27B-DFlash2
tensorfold serve qwen27b-exl3-3.00bpw --host 0.0.0.0 --port 8080
```

Verify windows use the row-invariant EXL3 linear, so drafted replies equal `"draft": false` ones; prompts use the
EXL3 prompt path (W_q decoded once a chunk, a fixed-tile GEMM), whose bits do not depend on chunking, so the
engine keeps prompt ends as it does for the MLX checkpoint. The drafter reads the target's 6-bit head over its
draft vocabulary by slicing the head's 128-column strips as stored: its logits are the target's, bit for bit.

Measured on one DGX Spark (GB10) through `tensorfold serve`, the 3.00bpw pack against the MLX 4-bit checkpoint on
the same engine and box, the [public benchmark command](README.md#measurements), medians of 15 runs a cell:

| Cell | EXL3 3.00bpw | MLX 4-bit | vLLM MTP=3 |
| --- | ---: | ---: | ---: |
| Code, sampled | 83.4 tok/s | 57.5 tok/s | 23.4 tok/s |
| Chat, sampled | 44.2 tok/s | 49.7 tok/s | 25.4 tok/s |
| Code, greedy | 64.5 tok/s | 53.6 tok/s | 25.8 tok/s |
| Chat, greedy | 39.9 tok/s | 50.0 tok/s | 24.7 tok/s |

A 12-row round costs 75 ms on the pack against 84 on the MLX checkpoint (10.1 GB of weights a token against
14.4). The chat cells keep fewer drafted tokens a round (3.0-3.3 against 4.2), since DFlash2 was trained on the
unquantized model. Cold prefill runs 880-970 tok/s from 2k to 16k and 720-890 at 32k-64k, about half the MLX
checkpoint's FP8 prompt path. The engine and drafter take 13.2 GiB after loading; a 64k prompt peaks at 28 GiB
allocated. Other branches of the pack load through the same path; only 3.00bpw is measured here.

### Concurrent requests

```bash
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --parallel 16 --context 8192 --name bench
```

`--parallel N` decodes up to N requests in shared rounds: each stream verifies its own DFlash2 tree in one
forward, and every reply equals the same request served alone. A new prompt prefills 1,024 tokens a round
while the other streams decode. States kept at message starts let prompts that share a system prompt resume
there, and each stream's caches are sized once, at admission.

On one DGX Spark, with the workload from issue #38 (48 chat requests of 3,000 to 5,100 tokens sharing a
2,900-token system prompt, up to 512 tokens each, greedy, all sent at once), TensorFold serves 161.7 tok/s at
16 in flight with a 25.4 GiB peak (nvidia-smi). vLLM with `nvidia/Qwen3.8-27B-NVFP4`, MTP=3, prefix caching
and `--max-num-seqs 16` serves 132.1 tok/s at 51.7 GiB on the same Spark. At 8 and 4 in flight TensorFold
serves 147.5 and 112.0 tok/s. `tools/shared_prefix_prompts.py` builds the workload and
`tools/shared_prefix_load.py` sends it (`--concurrency`, `--mem` for the memory peak).

### Historical public-fixture results

The earlier CUDA recipe reports these decode medians in NVIDIA's `pytorch:26.07-py3` container on GB10.
They are retained as historical results, not measurements of the merged 0.3.5 release.

| Ranks | Code sampled | Chat sampled | Code greedy | Chat greedy |
| --- | ---: | ---: | ---: | ---: |
| One | 49.6 tok/s | 45.8 tok/s | 49.2 tok/s | 45.9 tok/s |
| Two | 82.4 tok/s | 58.9 tok/s | 76.2 tok/s | 71.1 tok/s |

Reproduce the workload with the checkpoint above, default drafting and the
[public benchmark command](README.md#measurements). Its fixed prompts, 64-token replies, seeds 1234
through 1238 and sampling settings define these cells. For one rank, omit the tensor-parallel flags. Record the runtime
and model revision with any new result; these historical rates are not predictions for another runtime.

## Calibration and checks

The draft calibration metadata names public prompts. When regenerating it, start its server with
`--port 8473 --name qwen27` so the collection client reaches the named endpoint. Pass the saved full
provenance object with `--source` to `tools/fit_draft_calibration.py`; retain model and drafter revisions.

Kernel tests check rows alone and in windows, tree paths and committed state. Release checks must also
compare drafted/serial, resumed/fresh and concurrent/solo requests with thinking on and off and tools.
Decode rate, prefill, concurrency and peak-memory results are TBD [release-0.3.5].
