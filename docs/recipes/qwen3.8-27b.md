# Qwen3.8-27B

The `qwen3_5` family combines Gated DeltaNet and full attention. It serves the same MLX affine
4-bit/group-64 checkpoint on MLX and CUDA.

```bash
tensorfold pull Vontra/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --name bench
```

DFlash2 is used automatically once pulled. On MLX, `--drafter none` disables that draft model;
`--no-drafts` disables all drafts on either backend. CUDA requires DFlash2 unless `--no-drafts` is set.
The target verifies every proposed token against its own serial sample.

## MLX

M5 tensor-unit GPUs run the lane decoder with draft trees. M1 through M4 use `row_forward` and the
row-exact `simd_qmm` decoder with chains of up to 16 rows by default. Both paths use the same arithmetic for serial
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

On M1 through M4, only 4-bit/group-64 projections are supported. CUDA support here is also 4-bit/group-64;
the MLX lane-width list does not describe CUDA support. On MLX, unsupported projection formats and tied
embedding heads are refused from `config.json` before weight downloads and again at load; loaded
projections must also be covered by the selected decoder. `--lane-kernels on` requires M5 tensor units.
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
