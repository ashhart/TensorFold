# Disk prefix preservation validation

The CUDA dense-Qwen sleep adapter can preserve retained text prefixes through
`--sleep-cache-dir`. This extends [weights sleep/wake](model-sleep-cuda-validation.md)
with attention K/V, convolution history, recurrent state and DFlash2 context. The
frontend still owns conversation history; saved prefixes accelerate its next prefill.
See the [usage guide](../model-sleep.md) for storage lifetime and failure behavior.
The [machine-readable results](model-sleep-cache-results.json) retain checkpoint
hashes, output hashes, memory counters, timings and HTTP observations.

## Configuration and correctness

Full-checkpoint checks use the same pinned target and drafter as the weights receipt:

- `nvidia/Qwen3.8-27B-NVFP4`, revision
  `482ca0f3832238542f8f5295dde86b5f22711d80`.
- `z-lab/Qwen3.8-27B-DFlash2`, revision
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.
- CUDA compute capability 12.1, unified memory, PyTorch 2.13.0+cu130, CUDA 13.0.
- Checkpoint precision, default prompt precision; temperature 1, top-k 20, top-p
  0.95, seed 1234 plus stream offset, 32 reply tokens with EOS stopping disabled.
- Direct prompts repeat token IDs 1–128 with a per-stream offset. They check exact
  state restoration, not model quality. The HTTP check uses public text prompts.

One stream uses a 4096-token context and a 2048-token prompt; two streams use a
16384-token context and 8192-token prompts. Both configurations pass repeated
sleep/wake cycles with identical serial/drafted token output. Every cycle returns
PyTorch allocated and reserved bytes to zero, releases the old engine, retains the
frontend and leaves prefixes unloaded until a matching request arrives.

| Configuration | Prefixes saved | Snapshot data including headers | Cached tokens after wake, per request |
| --- | ---: | ---: | ---: |
| One stream + drafter | 1 | 314.75 MiB | 2047 / 2048 |
| Two streams + drafter | 2 | 1397.53 MiB | 8191 / 8192 |

A small affine fixture also passes two cycles with one stream and two with four
streams. The latter retains four prefixes and restores 31 of 32 prompt tokens for
every request. The qualification tool reserves enough checkpoint slots for its
measured working set; production eviction policy is unchanged.

## Latency and memory

The two-stream run measures time from each runtime request to its first token,
including queueing and lazy disk restoration. Both requests use distinct prefixes.
The in-memory baseline repeats them while awake; each restored measurement follows
a complete sleep/wake cycle.

| 8192-token request | Cold | Cache already in memory | After wake, cycle 1 | After wake, cycle 2 |
| --- | ---: | ---: | ---: | ---: |
| First stream | 2.785 s | 0.136 s | 0.951 s | 0.930 s |
| Second stream | 6.433 s | 0.136 s | 0.951 s | 0.931 s |

The server's concurrent `prefill_s` statistic starts after admission and excludes
lazy restore, so it is not used for this comparison. `first_token_s` in the runtime
tool includes that work. Serial reference requests warm kernels before these
measurements; the table does not represent the first invocation of an unwarmed
runtime. These few repetitions are observations, not a statistical performance
guarantee. Cold requests can queue behind each other's prompt work.

For the one-stream run, prefill including restoration took 0.278–0.286 seconds,
versus 0.729 seconds cold and 0.094 seconds with the prefix already in memory.
The one-stream engine includes prefix loading in its `prefill_s` measurement.

| Configuration | Median sleep | Median wake | Available-memory increase asleep |
| --- | ---: | ---: | ---: |
| One stream + drafter | 12.19 s | 29.57 s | 22.12–22.95 GiB |
| Two streams + drafter | 13.79 s | 32.52 s | 24.50–24.88 GiB |

Sleep timings include checkpoint hashing, prefix writing and validation. Wake
includes weight reload and snapshot verification, but not lazy tensor restoration.
Linux can retain reclaimable file-cache pages. Zero PyTorch allocator bytes does
not imply zero process or CUDA-context memory.

## HTTP conversation continuity

The real server with `--parallel 2 --context 16384 --sleep-cache-dir` passes
`tools/qualify_sleep_http.py --require-cache`. It checks authorization, stream
draining, admission and transition conflicts, reads while asleep, and stored
`previous_response_id` continuation. The follow-up uses the same 56-token history,
the same output hash, and 55 cached tokens before and after wake. Wake reports zero
eager prefix loads; the follow-up loads its matching snapshot without errors.
Allocated and reserved memory are zero asleep. Normal server shutdown removes its
private snapshot directory.

## Failure and format checks

Host tests cover transactional publication, disk-write rollback with the original
runtime still usable, repeated generations, lazy longest-prefix selection, memory
admission misses, corrupt-file rejection and retryable wake. Missing/corrupt old
disk-only prefixes are reported unavailable during a later save; destination write
failures still abort that save. Cleanup leaves unrelated files untouched.

Real CPU tensor tests check exact bf16/fp32 bytes, including non-finite values and
signed zero; valid-row trimming, strided tensors, all recurrent/drafter fields and
None slots; bounded save/load staging; malformed identity/schema/offset/shape/dtype
rejection before allocation; and compatibility with the safetensors reader. These
tests do not substitute for device-memory measurements.

The final affected lifecycle, CLI, HTTP, Responses and storage run passed **267 tests,
four skipped**. The real CPU tensor/storage/lifecycle run passed **46 tests**; these
overlap and should not be added together. An additional **247 existing Qwen cache
checks** passed. The broader suite limitations remain documented in the
[weights validation](model-sleep-validation.md#host-checks-and-review).

New files pass lint. Changed existing files introduce no new lint diagnostics;
45 pre-existing diagnostics remain, compared with 47 at the base revision.

## Reproduction and limits

```bash
PYTHONPATH=src python tools/qualify_sleep.py --model /path/to/target --draft /path/to/draft \
  --preserve-cache --context 4096 --prompt-tokens 2048 --tokens 32 --seed 1234 \
  --cycles 2 --output cache-single.json
PYTHONPATH=src python tools/qualify_sleep.py --model /path/to/target --draft /path/to/draft \
  --preserve-cache --parallel 2 --context 16384 --prompt-tokens 8192 --tokens 32 --seed 1234 \
  --cycles 2 --output cache-concurrent.json
python tools/qualify_sleep_http.py http://127.0.0.1:8080 qualification \
  --require-cache --output cache-http.json
```

This qualifies the pinned NVFP4 target/drafter and the synthetic affine fixture.
It does not qualify other full-size checkpoint formats, image-state preservation,
Metal, Level 1 or tensor parallelism. Snapshots preserve already-retained prompt
boundaries; previously evicted conversations cannot be recovered. They do not
suspend an active request or survive a process restart. The runs use warm filesystem
caches; cold-storage latency, disk contention and GPU allocation-failure injection
remain unmeasured. Lazy restore performs synchronous I/O on the serving execution
path, so a large snapshot can delay other admitted streams.
