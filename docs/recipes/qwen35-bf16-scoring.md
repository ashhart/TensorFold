# Qwen3.5-27B: plain BF16 CUDA prompt scores

This candidate adds a dense, unquantized BF16 format to the existing native Qwen3.5 CUDA engine.
It serves nonstreamed `/v1/completions` requests with `prompt_logprobs: 0..20` through the normal lane scheduler.
It does not introduce a separate inference worker. This remains an experimental text/scoring format.
The [qualification receipt](../receipts/qwen35-bf16-scoring/README.md) retains API, FP64, complete CUDA,
served/cold, cache/cancellation, existing-format and memory results. The complete sm_120 CUDA run passes
with six unchanged known findings and zero launch-limit problems. BF16 prefill and concurrency remain limiting;
model equivalence, production and performance qualification are not established.

## Checkpoint and precision

Use `Qwen/Qwen3.5-27B` at revision `fc05daec18b0a78c049392ed2e771dde82bdf654`, including its tokenizer.
Weights stay in their stored format: BF16 projections and embeddings, and the checkpoint's FP32 GDN normalization scales.
The loader applies the existing host-side, BF16-rounded shifted-norm convention to the HF shifted norms.

Activations and ordinary generation logits retain the engine's BF16 interfaces. Dense dots use the existing fixed-order
FP32 accumulator. The scoring head uses the same row arithmetic with FP32 output, followed by a full-vocabulary FP32
normalizer, selected-token logit, rank and at most 20 top entries. Row arithmetic is independent of batch width.
The FP32 scoring head does not change the ordinary generation/sampling head's output type.

FP32 GDN scales use the existing shared FP32-weight RMS operator and BF16 SiLU product. They cannot enter a fused
post operator that expects BF16 weights. Existing quantized formats keep their original operators and admission.
Vision, int8 prompt precision and learning are explicitly refused for the new BF16 format pending separate qualification.
The checkpoint's declared vision tower is not evidence that this text-only path supports images.

## Build and run

Use the exact Zig version in `.zig-version`. On an authorized CUDA machine, generate the maintained asset set and build:

```sh
zig build fatbins kernel-contract-fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=80,86,89,120,121
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=80,86,89,120,121 -Dfatbins=zig-out/fatbin -Dcontract-controls=zig-out/fatbin/contract_controls.fatbin -Doptimize=ReleaseFast
zig build native qwen35-score-reference -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=80,86,89,120,121 -Dfatbins=zig-out/fatbin -Doptimize=ReleaseFast
```

Use the native binary's `serve --help` for startup flags. Keep the listener on loopback and run under the existing GPU
lock and admission guard, with one model process per test machine. For cache-off checks use `--prompt-cache-gib 0`.
A `prompt_logprobs` request intentionally recomputes the full prompt and reports zero reused prefix tokens.
The first prompt entry is null; each later entry scores that token from its preceding normalized hidden row.
The 43-word GPU output layout is per row; it remains disjoint at rows 4, 5 and 128.

## Reproduce the checks

The standard clients remain the performance and ordinary-token checks:

```sh
python3 tools/bench_openai.py http://127.0.0.1:8080 local-model --tokens 128 --reps 3 --temperatures 0 --output decode.json
python3 tools/bench_concurrent.py http://127.0.0.1:8080 local-model --tokens 128 --reps 3 --temperatures 0,0.7 --levels 1,4 --alone --serial --output concurrent.json
python3 tools/zig/check_qwen35_prompt_scores.py http://127.0.0.1:8080 local-model fixtures.json --output scores.json
zig-out/bin/tf-cuda-test kernel-contract
zig-out/bin/tf-cuda-test kernel-contract-limits zig-out/fatbin
```

`fixtures.json` is an array of public-tokenizer token arrays. Retain its exact bytes and hash with the result.
Use lengths 1, 32, 129, 257 and 4097 for empty-score, batch-boundary and prompt-chunk-seam coverage.
The score client checks drafted/plain tokens, repeatability, concurrent/solo FP32 score hashes, top-k admission and
recovery after refused requests. Its throughput fields describe that scoring workload, not an ordinary decode benchmark.

The finite capture tool runs production forwards and records normalized BF16 input rows, FP32 head logits and raw stats.
It is a qualification tool, not a server or an alternative inference runtime:

```sh
zig-out/bin/tf-qwen35-score-reference MODEL reference-fixtures.json captures
python3 tools/zig/qwen35_scores_fp64.py MODEL captures --revision fc05daec18b0a78c049392ed2e771dde82bdf654 --output fp64.json
```

The independent reference reads the same stored BF16 weights and computes FP64 head dots and full-vocabulary
logsumexp on identical native normalized rows. It reports prompt and decode separately, worst row/token errors,
bound ratios, exact selected/top/rank fields, and argmax differences with FP64 margins. This is not a full-model FP64
forward. Additional projection captures exercise each named checkpoint matrix with deterministic synthetic BF16 inputs;
they do not claim that every layer's intermediate activation matches a separate model implementation.

The head bound uses `gamma(ceil(hidden/2048)*8+13) * sum(abs(x*w))` plus output rounding. The reduction bound uses
`ceil(vocab/256)+8` depth and 12 post-reduction operations. Logprob bounds include per-token and row-max head errors,
reduction error and FP32 subtraction rounding. The script records the formulas and every capture's result.
The stats stress fixture requires both smaller accumulators to miss by at least 100x in all three simulated orders.
Both the omitted-vocabulary-tail and FP16-head-accumulator GPU controls must fail their intended gates.

## Qualification scope

The initial device is one RTX PRO 6000 Blackwell Max-Q workstation GPU (sm_120), rank 1, on a 128 GB-class Linux host.
A four-CPU/32 GiB container budget is not evidence of fit on a physical 32 GiB machine.
Compiled images for sm_80/86/89/120/121 do not qualify execution on those other devices.
Metal, HIP, XPU, multiple ranks, image inputs, int8 prompts and learning remain unqualified for this new format.
The receipt retains matched MLX/Qwen3.5 and ModelOpt/Qwen3.8 parent/candidate tokens and descriptive performance.
The public EXL3 export is rejected by both builds because its embeddings are F16; EXL3 inference is unqualified.
FP8 KV and optional grammar/tool generation are also unqualified for this new format.

The receipt binds exact source/binary identities, complete CUDA SUMMARY and exit status, live and five-SM compile
checks, host/plain builds, zero-new lean findings, public fixture hashes, FP64 reports and served measurements.
Cold 2k/8k/32k/64k arrays and requests are retained. Memory separates Linux RSS, container budgets, CUDA-owned peaks
and sampled pinned maxima. SGLang HTTP cache counters are absent; observed scheduler counts remain separate evidence.

The [issue gate](https://github.com/ashhart/TensorFold/issues/1136#issuecomment-6553918935) requires same-weight FP64
worst-row/token evidence and unchanged existing formats; it explicitly allows the other-engine comparison to fail.
The actual served-versus-Transformers experiment reports both measured BF16/FP32 directions without widening a tolerance.
Both precision arms cover the 4097-ID seam, and the reference restores all 96 original FP32 GDN/A_log parameters.
This experimental format's functional review does not establish model equivalence or performance qualification:
BF16 prefill and multi-stream throughput remain slow. Read the current receipt's results and limitations.

```sh
python3 tools/zig/check_qwen35_served_reference.py http://127.0.0.1:8091 qwen35-bf16 fixtures.json transformers.json --output served-reference.json
```
