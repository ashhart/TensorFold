# Qwen3.8-27B

Experimental [packed affine formats](../quantization.md) include 8-bit and mixed layer precision.

The `qwen3_5` family combines Gated DeltaNet and full attention. The standard recipe below uses its
4-bit/group-64 checkpoint; the quantization guide lists the other affine formats it reads.

```bash
tensorfold pull TensorFold/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --name bench
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
`TensorFold/Qwen3.8-27B-oQ2` and `TensorFold/Qwen3.8-27B-oQ4`.

The packed row readers on Apple Silicon and the CUDA readers cover MLX affine 2/3/4/5/6/8-bit projections
with groups of 32/64/128, including mixed layers. CUDA also reads [EXL3 packs](#exl3-checkpoints-experimental).
A fused projection keeps its members separate where their bit width or group size differs. Unsupported formats
and tied embedding heads are refused from `config.json` before weight downloads and again at load; loaded
projections must also be covered by the selected decoder. `--lane-kernels on` requires M5 tensor units and
formats they read; `auto` falls back to the packed row decoder for the rest.
Lower weight precision does not guarantee faster decode or a fitting context. Release memory and
quality comparisons are TBD [release-0.3.5].

## CUDA

On CUDA the 27B serves NVFP4 and EXL3 checkpoints, and the MLX 4-bit checkpoint as the portable option: the same
files a Mac serves, and the only format two ranks read. `tensorfold serve` loads the checkpoint you name; it picks
none by itself. Use the [CUDA container setup](../../RUNBOOK.md#dgx-spark) for any of them. Prompts take bf16
activations by default; what that costs against the FP8 prompt path (`--prefill-fp8`) depends on the format
([prompt precision](cuda.md#prompt-precision)):

| Checkpoint | Weights | Ranks | bf16 prompts against `--prefill-fp8` |
| --- | --- | --- | --- |
| `nvidia/Qwen3.8-27B-NVFP4` | NVFP4 MLP and head, FP8 attention and DeltaNet | one | not measured yet |
| `turboderp/Qwen3.8-27B-exl3` (3.00bpw) | EXL3 | one | unchanged: EXL3 prompts never took FP8 activations |
| `TensorFold/Qwen3.8-27B-MLX-4bit` | MLX affine 4-bit, groups of 64 | one or two | 0.73-0.82x from 2k to 128k |

Mia-AiLab publishes EXL3 packs of the model (`Mia-AiLab/Qwen3.8-27B-EXL3`, `Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw`,
`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`) and an EXL3 DFlash2 drafter (`Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw`); none
has been loaded here yet.

### NVFP4 checkpoints

The CUDA engine reads NVIDIA's ModelOpt export of the model (`nvidia/Qwen3.8-27B-NVFP4`: NVFP4 MLP and head, FP8
attention and DeltaNet projections, bf16 embedding and gates) as it ships. The reader also takes
compressed-tensors NVFP4 and FP8 exports, checked on synthetic tensors only. Each projection is read by its tensors: NVFP4 codes with their e4m3 block scales and
FP8 bytes go to the device unchanged, and the lane matmuls turn them into exact bf16 operands (an e2m1 code times
its block scale fits bf16), so drafted replies equal `"draft": false` ones and prompts keep their bits in any
chunking. `--parallel` serves concurrent requests as on the MLX checkpoint, each reply equal to the same request
alone. One GPU: `--tp 2` stops at startup (two ranks read the MLX checkpoint), and so does `--vision` until image
input is qualified on this checkpoint.

```bash
tensorfold pull nvidia/Qwen3.8-27B-NVFP4 z-lab/Qwen3.8-27B-DFlash2
tensorfold serve nvidia/Qwen3.8-27B-NVFP4 --host 0.0.0.0 --port 8080
```

Prompts take bf16 rows on a prompt GEMM that reads the stored bytes: each FP8 byte and each NVFP4 code times its
block scale is exact in bf16, summed in fp32 over the inputs, the tensor scale last. `--prefill-fp8` restores the FP8
prompt matmul (FP8 projections as stored, NVFP4 ones staged to e4m3 once a chunk, a step that rounds by 2^-4 at most,
and e4m3 activations). Decode, measured on one DGX Spark (GB10) through `tensorfold serve` against the MLX 4-bit
checkpoint on the same engine and box, alternating (MLX, NVFP4, NVFP4, MLX), 64 tokens, five seeds, medians:

| Cell | NVFP4 | MLX 4-bit | vLLM MTP=3 (NVFP4) |
| --- | ---: | ---: | ---: |
| Code, sampled | 47.1 tok/s | 57.8 tok/s | 23.4 tok/s |
| Chat, sampled | 38.2 tok/s | 50.0 tok/s | 25.4 tok/s |
| Code, greedy | 47.2 tok/s | 53.9 tok/s | 25.8 tok/s |
| Chat, greedy | 38.3 tok/s | 50.1 tok/s | 24.7 tok/s |

The table predates a fix to the FP8 lane matmul (its pipeline stages now start on 128-byte lines), which took a
12-row verify from 103 ms to 85 against the MLX checkpoint's 84, although the attention and DeltaNet weights are 8-bit
(16.3 GB read a token against 14.4). Tokens a round match (4.0-5.2). Cold prefill ran 1,834 / 1,872 / 1,779 / 1,563 /
1,240 tok/s at 2k / 8k / 16k / 32k / 64k on the FP8 prompt path (today's `--prefill-fp8`), level with the MLX
checkpoint's. The startup estimate is 62.5 GiB at the 262,144-token window.

### EXL3 checkpoints (experimental)

The CUDA engine also reads turboderp's EXL3 packs of the model (`turboderp/Qwen3.8-27B-exl3`, a branch per size,
`mul1` codebook, 6-bit head) through the shared EXL3 module ([EXL3 weights](exl3.md)) and drafts with
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
unquantized model. Cold prefill runs 880-970 tok/s from 2k to 16k and 720-890 at 32k-64k, about two thirds of the
MLX checkpoint's bf16 prompts and half its `--prefill-fp8` ones. The engine and drafter take 13.2 GiB after loading;
a 64k prompt peaks at 28 GiB allocated. Other branches of the pack load through the same path; only 3.00bpw is
measured here.

### MLX 4-bit, one or two ranks

One or two ranks are supported. Pull the model and drafter on every rank, then start rank 1 before rank 0:

```bash
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --tp 2 --rank 1 --master 192.0.2.1
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --tp 2 --rank 0 --master 192.0.2.1 --name bench --host 0.0.0.0
```

The verify matmul fixes reduction order by weight shape. Tree attention reads only committed keys and the
node's own path; recurrent commits replay that path. Two-rank reductions gather fp32 partials and add in
rank order. Each rank count has its own serial reference. See the
[CUDA kernel map](../../src/tensorfold/families/qwen3_5/cuda/README.md).

### Concurrent requests

```bash
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --parallel 16 --context 8192 --name bench
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

### Structured output

With `pip install 'tensorfold[grammar]'` (xgrammar), serving on one or two GPUs enforces `response_format` and the
`guided_*` fields: drafts the grammar rejects are cut and rows are masked by their paths' grammar before sampling, so
drafted, serial and concurrent replies stay equal. See [the API reference](../api.md#structured-output).

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

### A 64 GB M5 Pro on 0.3.5.1

@benwilson measured these on real 64 GB hardware for issue #70. They describe TensorFold 0.3.5.1, not a later
release; the issue's first comment has the archive of logs, request bodies and the fixture builder.

| | |
| --- | --- |
| Machine | MacBook Pro Mac17,9, Apple M5 Pro, 20-core GPU, 64 GB, macOS 26.5.2 (25F84), on AC power |
| Budget | 44.8 GiB: 70% of 64 GB, under the 55 GiB Metal working set (`iogpu.wired_limit_mb=56320`) |
| Runtime | TensorFold 0.3.5.1 (`beddbb7`, from the tag), Python 3.12.13, mlx and mlx-metal 0.31.2, mlx-lm 0.31.3 |
| Checkpoints | `TensorFold/Qwen3.8-27B-MLX-4bit@70ae7fac`, `z-lab/Qwen3.8-27B-DFlash2@50307d4c`, `TensorFold/Qwen3.8-27B-oQ2@8cf0a7da` |
| Launch | `tensorfold serve <model> --name bench`, plus `--drafter none` or `--no-drafts` where named |
| Peak memory | `ri_lifetime_max_phys_footprint` from `proc_pid_rusage` |

| Configuration | Fitted context | Peak footprint | Workload that set the peak |
| --- | ---: | ---: | --- |
| Qwen3.8-27B + DFlash2 | 140,288 | 43.69 GiB | prompts up to 139,922 tokens, cold and resumed |
| Qwen3.8-27B, `--drafter none` | 152,576 | 39.87 GiB | a 152,210-token prompt, cold and resumed |
| Qwen3.8-27B-oQ2 + DFlash2 | 172,032 | 43.81 GiB | a 171,667-token prompt, cold and resumed |

Long prompts used the source of `ggml-org/llama.cpp@90c26fcd` (`src/*.cpp`, then `ggml/src/*.c*`, sorted),
tokenized by the served model and cut to N tokens behind a nonce line. Replies were 64 tokens, greedy, with
`ignore_eos` and thinking off; the resumed request adds one assistant and one user turn.

| Rendered prompt | Cold TTFT | Prefill | Decode at depth | Resumed: cached, TTFT |
| ---: | ---: | ---: | ---: | --- |
| 8,224 | 19.5 s | 423 tok/s | 25.5 tok/s | 8,192, 0.25 s |
| 16,417 | 41.0 s | 401 tok/s | 29.8 tok/s | 16,384, 0.31 s |
| 32,800 | 90.5 s | 363 tok/s | 24.4 tok/s | 32,768, 0.37 s |
| 65,569 | 221.6 s | 296 tok/s | 20.5 tok/s | 65,536, 0.61 s |
| 98,337 | 395.1 s | 249 tok/s | 15.8 tok/s | 98,304, 0.69 s |
| 131,106 | 578.4 s | 227 tok/s | 15.7 tok/s | 0, 574 s (#71) |
| 139,922 | 640.5 s | 219 tok/s | 16.3 tok/s | 0, 644 s (#71) |

Decode medians from the [public benchmark command](README.md#measurements), in tok/s for code sampled, chat
sampled, code greedy and chat greedy: 62.8, 45.9, 61.2 and 56.2 with DFlash2; 16.6, 16.1, 16.2 and 16.3 with
`--no-drafts`; 16.2, 16.2, 16.6 and 16.4 with `--drafter none`; 41.8, 37.2, 92.1 and 47.9 for oQ2 with DFlash2.
`tools/bench_concurrent.py --alone --serial` at 1, 2, 4 and 8 streams found all 180 concurrent replies equal to
their solo runs and every solo run equal to `"draft": false` (60 of 60 for oQ2 at 1 and 4). Code greedy served
78.9, 117.7, 158.2 and 201.4 tok/s in aggregate at 1, 2, 4 and 8 streams. The release checks (drafted against
serial, resumed against fresh, by `token_sha`) matched on 8 of 8 short-prompt cells and 4 of 4 cells at 8,192
tokens with a real resume, and replays after a restart matched too. A conversation grown by 6,144 tokens a turn
to the 140,288 window kept swap at 653 to 656 MB, with a 43.57 GiB peak footprint.

### A 48 GB M4 Pro on 0.6.2 (emulated memory classes and the tuned default)

Measured on a 48 GB Apple M4 Pro (10P/4E cores, no tensor units, macOS 27.0, AC power) to fill the 32, 36 and
48 GB rows of the README's memory-class table. The 32 and 36 GB cells are emulated budgets on this host
(`TENSORFOLD_MEMORY_LIMIT_GB` at the class ceiling); the 48 GB cell is the machine's own default budget, plus
its raised-budget variant. Weights were 16.3 GiB resident (11.8 GiB for oQ2, 15.1 GiB with `--drafter none`);
startup reported a stream at 166 MB for 64 tokens, 391 MB at 2,112, then 64.0 KB a token, and a shared round up
to 2.75 GiB at 8 streams.

| | |
| --- | --- |
| Machine | Apple M4 Pro, 48 GB, macOS 27.0, on AC power; no thermal limit recorded during any run |
| Runtime | TensorFold 0.6.2 (`56e2e3e`, from the tag), Python 3.12.13, mlx and mlx-metal 0.32.3 |
| Checkpoints | `Vontra/Qwen3.8-27B-MLX-4bit@22d8d538`, `z-lab/Qwen3.8-27B-DFlash2@50307d4c`, `Vontra/Qwen3.8-27B-oQ2@086dcb02` |
| Launch | `tensorfold serve <model> --name bench --context 0`, `TENSORFOLD_MEMORY_LIMIT_GB` where named |
| Peak memory | `phys_footprint_peak` from `footprint(1)` sampled at 2 s with `vm_stat` and `pmset -g therm` |

| Budget | Configuration | Fitted capacity | Peak footprint |
| --- | --- | --- | --- |
| 22.4 GiB (32 GB class) | Qwen3.8-27B + DFlash2 | Refused at startup: model and one prompt chunk need about 21.0 GiB against 19.4 GiB for MLX; 0 streams of 8,192 fit | not reached |
| 25.2 GiB (36 GB class) | Qwen3.8-27B + DFlash2 | 6,912 tokens retained; 1 stream of 8,192 fits | 26 GiB (host has more RAM than the class) |
| 25.2 GiB (36 GB class) | Qwen3.8-27B, `--drafter none` | 17,920 tokens retained; 3 streams of 8,192 fit | not measured separately |
| 33.6 GiB (48 GB default) | Qwen3.8-27B + DFlash2 | 11 streams of 8,192 fit | 26 GiB |
| 37.4 GiB (48 GB raised) | Qwen3.8-27B + DFlash2 | 92,672 tokens retained; 15 streams of 8,192 fit | 26 GiB |

Decode medians from the [public benchmark command](README.md#measurements) at the 48 GB default budget, in
tok/s for code sampled, chat sampled, code greedy and chat greedy: 43.5, 31.5, 43.4 and 33.2 with DFlash2;
at 256-token replies the code cells rise to 53.0 sampled and 47.8 greedy. Per-round telemetry: 4.0 tokens a
round at 91 ms, 47 of 109 drafts accepted, 7.8 rows a forward. `tools/bench_concurrent.py --alone --serial`
at 1, 2, 4 and 8 streams (128-token replies) found every concurrent reply equal to its solo run and every
solo run equal to serial; aggregate code-sampled throughput was 42.0, 53.2, 60.4 and 61.5 tok/s
(steady-state 62.8, 69.4 and 72.0 at 2, 4 and 8). No swap activity fell inside any benchmark window and
`pmset -g therm` reported no CPU or IO speed limit.

oQ2 (2-bit) on this host decoded at 16.4-19.7 tok/s with 2.9 rows a round and 31 of 60 drafts accepted:
without tensor units the 2-bit pack runs the general packed row decoder and drafting acceptance collapses,
a 2.4x regression against the 4-bit checkpoint that inverts the oQ2 speedups the 64 GB M5 Pro row records.
A 3-bit mixed-width oQ checkpoint (`Toomanydatsuns/Qwen3.8-27B-oQ3.5e-fp16-mtp@75c55831`) failed at startup
with an MLX kernel-build error (`thread bfloat` from `const device float16_t` in the row decoder's JIT), so
no 3-bit cell is reported.

## Calibration and checks

The draft calibration metadata names public prompts. When regenerating it, start its server with
`--port 8473 --name qwen27` so the collection client reaches the named endpoint. Pass the saved full
provenance object with `--source` to `tools/fit_draft_calibration.py`; retain model and drafter revisions.

Kernel tests check rows alone and in windows, tree paths and committed state. Release checks must also
compare drafted/serial, resumed/fresh and concurrent/solo requests with thinking on and off and tools.
Decode rate, prefill, concurrency and peak-memory results are TBD [release-0.3.5].
