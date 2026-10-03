# CUDA sleep/wake reference implementations on 0.6.4

The dense Qwen and Nemotron-H adapters share lifecycle control, checkpoint identity,
transactional disk storage and bounded tensor transport. Family hooks retain their
native cache selection and state restoration. See the
[family integration guide](../recipes/adding-a-cuda-family.md#sleepwake-reference-implementations)
and [usage guide](../model-sleep.md).

The tested source is `a72d4d8`, which includes upstream TensorFold 0.6.4 at `6ea5ade`.
The merge retains both lifecycle metrics and the release's process-footprint metric.
All runtime results below use this combined source, including the updated attention
kernels. The earlier 0.6.3 receipts remain historical evidence.

## Checkpoints and settings

| Reference | Checkpoint | Revision | Preserved state |
| --- | --- | --- | --- |
| Dense Qwen | `nvidia/Qwen3.8-27B-NVFP4` | `482ca0f3832238542f8f5295dde86b5f22711d80` | Attention K/V, gated-delta and convolution state, optional DFlash context |
| Qwen drafter | `z-lab/Qwen3.8-27B-DFlash2` | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | Included with the target prefix |
| Nemotron-H | `Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` | `d9d758fb83953437f7263256b0d96157e2a348b8` | Attention K/V, Mamba and convolution state, retained hidden row, optional integrated MTP state |

The Nemotron checkpoint uses affine 4-bit/group-64 weights. Both families run on
single-device CUDA with unified memory, compute capability 12.1, PyTorch
2.13.0+cu130 and CUDA 13.0. Checkpoint precision and default prompt precision are
unchanged. File hashes, output hashes, counters and timings are retained in the
[machine-readable results](model-sleep-reference-results.json).

Direct qualification repeats token IDs 1–128, with a per-stream offset, and generates
32 reply tokens using seed 1234 plus that offset, temperature 1, top-k 20 and top-p
0.95. EOS stopping is disabled. These prompts test state restoration rather than
model quality. HTTP qualification and performance comparisons use public text.

## Sleep, restoration and latency

All four full-model configurations pass two sleep/wake cycles. Serial, drafted,
warm-cache and restored output tokens match within each configuration. Each cycle
releases the old engine, retains the frontend and returns both CUDA allocated and
reserved bytes to zero. Wake leaves prefix tensors on disk; the matching request
loads them with no restore failures or memory-admission misses.

| Configuration | Context / prompt tokens | Prefixes | Snapshot MiB | Cached tokens per request after wake | Median sleep / wake |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nemotron + MTP | 4096 / 2048 | 1 | 78.24 | 2047 | 8.52 / 21.42 s |
| Nemotron, drafts disabled | 4096 / 2048 | 1 | 76.24 | 2047 | 8.50 / 20.84 s |
| Qwen + DFlash, one stream | 4096 / 2048 | 1 | 314.75 | 2047 | 12.20 / 29.56 s |
| Qwen + DFlash, two streams | 16384 / 8192 | 2 | 1397.53 | 8191 | 14.09 / 32.98 s |

Snapshot sizes include headers and all retained prefixes. Sleep includes checkpoint
hashing and snapshot writing/validation. Wake includes reloading weights and checking
snapshots, before lazy tensor restoration. The served HTTP context stays at the
requested limit even when native buffers are rounded upward; reload uses the original
admitted context rather than that rounded capacity.

Time to first token includes lazy disk loading and, for concurrent requests, queueing:

| Request | Cold | Prefix in memory | After wake, cycle 1 | After wake, cycle 2 |
| --- | ---: | ---: | ---: | ---: |
| Nemotron + MTP, 2048 tokens | 0.271 s | 0.022 s | 0.077 s | 0.070 s |
| Nemotron without MTP, 2048 tokens | 0.269 s | 0.020 s | 0.071 s | 0.063 s |
| Qwen one stream, 2048 tokens | 0.738 s | 0.097 s | 0.282 s | 0.279 s |
| Qwen concurrent stream 1, 8192 tokens | 2.820 s | 0.136 s | 0.946 s | 0.953 s |
| Qwen concurrent stream 2, 8192 tokens | 6.741 s | 0.135 s | 0.945 s | 0.952 s |

Serial reference requests warm kernels before measurement. These few repetitions
are observations, not a performance guarantee. The concurrent engine's `prefill_s`
counter excludes lazy restoration, so the table uses externally timed first-token
latency instead.

## HTTP conversation continuity

The HTTP probe checks authorization, browser-origin refusal, unsupported sleep-level
refusal, streamed-request draining, new-request admission, transition conflicts,
discovery and stored-response reads while asleep. It then continues the same
`previous_response_id` after wake and compares the output hash, history length and
cached-token count. CUDA allocated and reserved memory must both be zero asleep,
and wake must leave prefixes unloaded until a matching request arrives.

With cache preservation required, the unrelated drain probe uses the serial
reference engine. An unrelated drafted request would evict Nemotron's live
conversation prefix before sleep under its existing retention rule. The initial
probe exposed that test assumption; runtime eviction behavior was left unchanged.
Drafted replies are separately compared with serial replies before and after wake.

Both servers pass with a 16384-token served context: Nemotron with one stream and
Qwen with two. Nemotron restores 58 cached tokens for the 59-token continuation;
Qwen restores 55 for its 56-token continuation. Their output hashes and history
lengths stay unchanged. Each reports zero eager prefix loads at wake and no lazy
restore failures. Normal shutdown removes the process's snapshot files.

## Comparison with unmodified 0.6.4

Each family uses the same checkpoint and a 16384-token context in four phases:
unmodified `6ea5ade`, sleep support enabled before sleeping, the same server after
wake, and unmodified `6ea5ade` again. Qwen uses two stream slots and DFlash; Nemotron
uses its serial request execution and MTP. A 64-token sampled completion probe has
the same serial/drafted token hash in every phase.

Decode measurements use `tools/bench_openai.py`, 64 tokens, greedy sampling and
three measured repetitions after warm-up for each public prompt. Cold-prefill
measurements use `tools/prefill_cold.py`, three distinct prompts at each of 2048 and
8192 tokens, a 1024-token warm-up and two output tokens. Prompt content comes from
the Python standard library; actual lengths differ from the nominal target by at
most one token. The alternating phases reduce drift; they do not establish a
statistical bound on small timing differences.

| Reference and metric | Release before | Enabled before sleep | After wake | Release after |
| --- | ---: | ---: | ---: | ---: |
| Nemotron, Fibonacci decode (tokens/s) | 134.07 | 133.76 | 133.83 | 133.77 |
| Nemotron, GPU explanation decode (tokens/s) | 152.13 | 151.37 | 152.01 | 151.77 |
| Nemotron, 2K cold first token (s) | 0.277 | 0.276 | 0.279 | 0.273 |
| Nemotron, 8K cold first token (s) | 1.132 | 1.127 | 1.139 | 1.126 |
| Qwen, Fibonacci decode (tokens/s) | 50.41 | 50.40 | 50.30 | 50.32 |
| Qwen, GPU explanation decode (tokens/s) | 40.92 | 40.95 | 40.84 | 40.91 |
| Qwen, 2K cold first token (s) | 0.792 | 0.786 | 0.790 | 0.791 |
| Qwen, 8K cold first token (s) | 2.901 | 2.869 | 2.884 | 2.889 |

These are per-phase medians. For each decode prompt, the four medians vary by less
than 1%; cold-prefill medians vary by less than 3% at each measured length. The results
file retains individual repetitions and the tools' reported precision.

## Host and device checks

The integration suite passed **345 tests, 14 skipped** across lifecycle, storage,
CLI, HTTP, Responses, metrics and the new `plan` command. The real CPU tensor,
codec, storage and lifecycle suite passed **83 tests**. An additional capacity,
cache, communicator and attention-metadata suite passed **200 tests, 108 skipped**.
These suites overlap and their counts should not be added.

The compact Nemotron GPU test passes with and without MTP. Each case restores into
a fresh engine and compares both an identical prompt and a changed suffix against
serial generation. It was invoked directly in the CUDA environment, which does not
have pytest installed. Host regressions also cover the served context being smaller
than rounded runtime capacity, native-context admission, and reader cleanup after
failed weight loading. Qwen's snapshot bytes remain compatible with the previous
codec after extracting shared transport.

The full host run was stopped after 5m29s with **1,846 passed, 424 skipped, 17 failed
and six subtests passed**. Fifteen failures report unavailable Metal or TUI
dependencies. The two remaining GLM checks reproduce on unmodified `6ea5ade`.
The full suite is not claimed green or complete. New implementation and test files
pass lint; changed existing files introduce no new lint diagnostics.

## Reproduction and limits

```bash
PYTHONPATH=src python tools/qualify_sleep.py --model /path/to/nemotron \
  --preserve-cache --context 4096 --prompt-tokens 2048 --tokens 32 --seed 1234 \
  --cycles 2 --output nemotron-mtp.json
PYTHONPATH=src python tools/qualify_sleep.py --model /path/to/nemotron --no-drafts \
  --preserve-cache --context 4096 --prompt-tokens 2048 --tokens 32 --seed 1234 \
  --cycles 2 --output nemotron-no-mtp.json
PYTHONPATH=src python tools/qualify_sleep.py --model /path/to/qwen --draft /path/to/dflash \
  --preserve-cache --context 4096 --prompt-tokens 2048 --tokens 32 --seed 1234 \
  --cycles 2 --output qwen-single.json
PYTHONPATH=src python tools/qualify_sleep.py --model /path/to/qwen --draft /path/to/dflash \
  --preserve-cache --parallel 2 --context 16384 --prompt-tokens 8192 --tokens 32 --seed 1234 \
  --cycles 2 --output qwen-concurrent.json
python tools/qualify_sleep_http.py http://127.0.0.1:8080 qualification \
  --require-cache --output http.json
```

This qualifies the pinned CUDA checkpoints and retained text prefixes within one
process. It does not establish Level 1, Metal, tensor parallelism, image-state or
restart persistence. Nemotron executes requests serially; Qwen also supports its
concurrent decoder. Existing eviction rules still apply. Snapshots cannot recover
prefixes evicted before sleep or suspend an active generation.
In particular, Nemotron clears its live prefix cache on an unrelated drafted prompt;
its two retained boundaries are not two independent conversation slots.

The runs use warm filesystem caches. Cold-storage latency, disk contention, longer
contexts, aggregate throughput and GPU allocation-failure injection remain
unmeasured. Lazy restoration uses synchronous I/O and can delay other admitted work.
Zero PyTorch allocator bytes excludes driver context and other library allocations.
