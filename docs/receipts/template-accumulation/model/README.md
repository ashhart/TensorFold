# Qwen chat model check

This gate uses the installed public `nvidia/Qwen3.8-Flash-Next-FP4-Local` checkpoint at revision `b9ce18a6887e18c089fdea8509a33e63429a3300`. The selected `chat_template.jinja` is 8,952 bytes with SHA256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`; its config model type is `qwen4_exp`. Chat requests reach `HfText.renderFn` and the changed Jinja evaluator. This template emits messages directly and does not accumulate a growing namespace string, so these are model regression checks, not evidence of the targeted accumulator speedup.

[Environment and artifact hashes](environment.json) pin both commits, native binaries, all 103 shared flat GPU images, checkpoint assets and request settings. The baseline was built with nvcc 13.3.73. Four independently compiled candidate GPU images differed despite unchanged kernel sources. The measured candidate was therefore rebuilt with `-Dfatbins` using the baseline's exact images; its independent build was preserved but not measured. The native servers use Zig 0.17.0 ReleaseFast, one DGX Spark GB10, context 8,192, two lanes, bf16 KV, zero prompt-cache retention, a 9 GiB reserve and serial weight loading.

## Protocol

The [client](client.py) sends genuine chat arrays: ordinary prose, code and a synthetic 34-message dialogue. It measures 64 reply tokens at temperatures zero and one (top-k 20, top-p 0.95), with seeds 1234/1235, one and two streams and three repetitions. Each server begins with the multi-turn request before tokenizer warmups. Subsequent measurements are KV-cold, with zero reported cached tokens; they are not fresh-process measurements. Solo and `draft:false` references are checked before the concurrent cells.

The [independent comparator](compare.py) reconstructs full token and text hashes, validates every response and rechecks raw timings. A complete ABAB gate requires 73 responses and 12 cells per run, 292 responses total. Tables compare arithmetic means of the two per-run medians on each side. First-request TTFT is reported separately. Raw completion benchmarks are excluded because they bypass chat rendering.

[Template byte parity](template-parity.json) compares these exact three fixtures with and without the generation suffix against Jinja2 3.1.6. Both pinned revisions match 6/6 cases, 7,219 output bytes each, in ReleaseFast.

## Results

All four runs pass: **292 complete 64-token responses, zero unequal or unverified responses**, and zero cached tokens. Each run contains 12 cells and nine solo/plain reference pairs. The six generation/history tokenization snapshots and all 18 fixture/temperature/seed/draft output keys agree across all four runs. The [independent summary](summary.json) reconstructs every hash and timing from the [A1](A1.json), [B1](B1.json), [A2](A2.json) and [B2](B2.json) receipts. JSON whitespace was compacted for publication; parsed values are unchanged and the originally recorded byte hashes are retained in `environment.json`.

Decode changes range from +0.01% to +0.80%; repeated-request TTFT changes range from −2.33% to +1.67%. These small movements do not establish a model-speed gain. Baseline A1-to-A2 decode drift ranges from −0.36% to +0.91%, and its TTFT drift from −1.01% to +2.24%. The largest absolute TTFT increase is 5.03 ms (+1.00% for the greedy 34-message chat at one stream). No large regression is evident in this bounded gate.

### Aggregate decode rate

| Fixture | T | Streams | Baseline (tok/s) | This PR (tok/s) | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 0 | 1 | 55.78 | 55.89 | +0.21% |
| Code | 0 | 2 | 97.64 | 97.96 | +0.32% |
| Code | 1 | 1 | 63.93 | 64.11 | +0.28% |
| Code | 1 | 2 | 74.21 | 74.73 | +0.70% |
| 34-message chat | 0 | 1 | 49.71 | 49.71 | +0.01% |
| 34-message chat | 0 | 2 | 58.18 | 58.39 | +0.36% |
| 34-message chat | 1 | 1 | 46.91 | 47.12 | +0.45% |
| 34-message chat | 1 | 2 | 52.51 | 52.60 | +0.17% |
| Prose | 0 | 1 | 48.17 | 48.46 | +0.60% |
| Prose | 0 | 2 | 80.62 | 81.23 | +0.76% |
| Prose | 1 | 1 | 41.61 | 41.65 | +0.08% |
| Prose | 1 | 2 | 55.95 | 56.40 | +0.80% |

### Repeated-request TTFT, zero cached tokens

| Fixture | T | Streams | Baseline (ms) | This PR (ms) | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 0 | 1 | 113.80 | 111.14 | -2.33% |
| Code | 0 | 2 | 170.03 | 166.71 | -1.96% |
| Code | 1 | 1 | 113.33 | 112.25 | -0.96% |
| Code | 1 | 2 | 166.84 | 166.88 | +0.02% |
| 34-message chat | 0 | 1 | 503.70 | 508.73 | +1.00% |
| 34-message chat | 0 | 2 | 786.51 | 785.40 | -0.14% |
| 34-message chat | 1 | 1 | 506.69 | 511.40 | +0.93% |
| 34-message chat | 1 | 2 | 787.94 | 789.76 | +0.23% |
| Prose | 0 | 1 | 120.59 | 119.98 | -0.50% |
| Prose | 0 | 2 | 181.31 | 178.64 | -1.48% |
| Prose | 1 | 1 | 123.59 | 123.45 | -0.11% |
| Prose | 1 | 2 | 183.43 | 186.50 | +1.67% |

### First request on each fresh server

The first request is the 764-token multi-turn prompt. These are fresh-process, KV-cold measurements; filesystem cache is not flushed.

| Side | Run 1 (ms) | Run 2 (ms) | Mean (ms) | Change |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 589.55 | 561.45 | 575.50 | — |
| This PR | 574.38 | 577.27 | 575.82 | +0.06% |

The 0.32 ms difference between means is negligible beside the baseline first-request variation of −4.77%. End-to-end throughput is also available in the machine-readable summary; changes range from −0.16% to +0.70%.

### Memory and cleanup

The [window outcome](window-outcome.json) records successful cleanup and eight memory observations, at healthy and client-finished boundaries. These include the server's reported lifetime peak resident/device counters, not continuous whole-host physical-footprint sampling. Device and host resident gauges are separate and must not be added on this unified-memory GPU.

| Run | Device peak bytes | Pinned bytes after requests | Reported process resident peak bytes |
| --- | ---: | ---: | ---: |
| A1 | 79,342,702,460 | 12,851,216 | 5,254,488,064 |
| B1 | 79,342,702,460 | 12,851,216 | 5,256,142,848 |
| A2 | 79,342,702,460 | 12,851,216 | 5,256,511,488 |
| B2 | 79,342,702,460 | 12,851,216 | 5,255,016,448 |

The existing six-GiB guard remained active throughout each load and request batch. No guard/OOM event was observed. The test container was removed and production was restored and health-checked. An earlier attempt loaded the baseline but sent no inference requests because the controller expected the production server's health-response shape; it was stopped and production restored. The corrected test-only readiness check passed eleven offline cases before this successful ABAB window.

## Limits

This bounded Qwen check does not replay the original multi-rank long-context report. Resumed-cache equality is not tested because this protocol disables prefix retention. No CUDA kernel changed; these model checks do not substitute for the full device-contract suite, Metal qualification or other checkpoint gates.

## Build and server commands

In the pinned dev image, from the baseline checkout, build into its isolated prefix:

```sh
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /baseline -j2 fatbins native
```

From the candidate checkout, mount the baseline assets read-only and omit `-Dnvcc` when selecting prebuilt images:

```sh
zig build -Dfatbins=/baseline/fatbin -Doptimize=fast --prefix /candidate -j2 fatbins native
```

Each fresh server uses the following command, substituting the baseline binary on A runs. Run only one model process, under the machine's existing GPU admission guard; the builds used CPU-only containers capped at 12 GiB and two CPUs.

```sh
HF_HUB_OFFLINE=1 TENSORFOLD_NO_UPDATE_CHECK=1 TENSORFOLD_MEMORY_RESERVE_GIB=9 TF_FLASHNEXT_LOAD_THREADS=1 \
/candidate/native/bin/tensorfold-native serve /checkpoint --name Qwen3.8-Flash-Next \
  --host 127.0.0.1 --port 30001 --context 8192 --parallel 2 --kv-cache bf16 \
  --max-tokens 256 --temperature 1.0 --top-p 0.95 --top-k 20 --thinking --prompt-cache-gib 0
```

The host is Ubuntu 24.04.5 LTS, kernel `7.0.0-1019-nvidia`, driver `580.178.04`. The native process uses the CUDA driver API and maps the container's `libcuda.so.610.43.02` forward-compatibility library; `cuDriverGetVersion` returns 13030 (CUDA 13.3), success code zero. No `libcudart` mapping was observed. These loaded-library facts are separate from the nvcc compiler version.

## Reproduction

With both pinned commits available locally, Zig 0.17.0 and Python 3.11+ with Jinja2 3.1.6:

```sh
python3 docs/receipts/template-accumulation/model/verify_template.py --template /checkpoint/chat_template.jinja --zig zig --out /tmp/model-template-parity
python3 docs/receipts/template-accumulation/model/client.py --url http://127.0.0.1:30001 --tools tools --output A1.json
python3 docs/receipts/template-accumulation/model/compare.py /path/to/four-receipts --output model-summary.json
```

Start a fresh pinned server before each client run, in baseline/PR/baseline/PR order, with the recorded identical settings. Use the output names `A1.json`, `B1.json`, `A2.json` and `B2.json`. A successful client exit qualifies only that run; the final comparator must pass before claiming cross-build parity or a speed comparison. Model startup and service restoration are environment-specific and must use the machine's existing guard.
