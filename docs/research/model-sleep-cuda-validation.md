# Full-checkpoint Level 2 validation

The implementation through `710ed91` passes three sleep/wake cycles with one stream
and three with two concurrent streams using the same pretrained target and drafter.
Each run compares drafted output with serial output, then checks every token again
after each wake. Both PyTorch allocated and reserved bytes reach zero during every
sleep. [Machine-readable results](model-sleep-cuda-results.json) include checkpoint
file hashes, token hashes, allocator counters and system-memory readings.

## Configuration

- Target: `nvidia/Qwen3.8-27B-NVFP4`, revision
  `482ca0f3832238542f8f5295dde86b5f22711d80`.
- Drafter: `z-lab/Qwen3.8-27B-DFlash2`, revision
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.
- CUDA compute capability 12.1, unified memory, driver 580.173.02, CUDA 13.0,
  PyTorch 2.13.0+cu130, Python 3.12.14. Checkpoint precision, default prompt precision.
- Sampling: seed 1234 plus the stream offset, temperature 1, top-k 20, top-p 0.95;
  32 reply tokens with EOS stopping disabled.
- One stream: context 4096, prompt 512 tokens. Two streams: context 16384,
  prompts 2048 tokens. The direct runtime fixture cycles through token IDs 1–128;
  these are exactness and reclamation checks, not model-quality evaluations.

## Memory release and timing

| Configuration | Allocated while loaded | Allocated/reserved asleep | Increase in available system memory | Median sleep | Median wake |
| --- | ---: | ---: | ---: | ---: | ---: |
| One stream + drafter | 19.99 GiB | 0 / 0 | 21.92–22.49 GiB | 11.66 s | 29.35 s |
| Two streams + drafter | 20.55 GiB | 0 / 0 | 23.47–23.51 GiB | 11.71 s | 32.13 s |

Old engine weak references expire in every cycle; app, tokenizer and template
identities remain stable. Linux `MemAvailable` rises after sleep, independently of
the allocator counters. The process and CUDA context remain alive, so zero PyTorch
allocation does not imply zero process or driver memory.

Transition times include full checkpoint hashing. Sleep verifies the backing files;
wake verifies them before and after reconstruction. These cycles use a warm
filesystem cache. Linux retains reclaimable cached checkpoint pages; no CPU tensor
backup of the runtime is created. These numbers do not predict cold-storage wake
latency or other quantizations.

Commands, using local checkpoint directories in a CUDA-enabled environment:

```bash
PYTHONPATH=src python tools/qualify_sleep.py --model /path/to/target --draft /path/to/draft \
  --context 4096 --prompt-tokens 512 --tokens 32 --seed 1234 --cycles 3 --output single.json
PYTHONPATH=src python tools/qualify_sleep.py --model /path/to/target --draft /path/to/draft \
  --parallel 2 --context 16384 --prompt-tokens 2048 --tokens 32 --seed 1234 --cycles 3 --output concurrent.json
```

## Scope

This qualifies weight/runtime reconstruction for the pinned NVFP4 checkpoint and
drafter at the settings above. It does not preserve KV, recurrent or drafter prefix
caches; those are reconstructed by prefilling after wake. EXL3, full-size affine
checkpoints, image input, other precision modes, longer contexts, Metal and tensor
parallelism remain unqualified by this run. Level 1 remains a separate adapter for
devices with separate GPU and CPU memory.

## HTTP behavior

The real server, started with `--enable-sleep-mode --parallel 2 --context 16384`,
passes `tools/qualify_sleep_http.py`:

- Unauthorized control requests, browser Origin requests and Level 1 requests are refused.
- A 64-token stream completes successfully when sleep is requested during streaming.
  New generation requests receive 503 while draining/asleep; a conflicting wake returns 409.
- Health, metrics, discovery and stored-response reads remain available while asleep.
- Allocated and reserved CUDA bytes both reach zero through the HTTP endpoint.
- Sampled chat output matches serial output before and after wake.
- A stored `previous_response_id` remains usable. Repeating the same follow-up after
  wake produces the same token hash and the same 56-token conversation input.

The measured HTTP sleep took 13.01 seconds, including the accepted stream's drain;
wake took 32.73 seconds. The persistent object is the conversation history; runtime
prefix caches are rebuilt. This does not provide process-restart persistence.

```bash
python tools/qualify_sleep_http.py http://127.0.0.1:8080 qualification \
  --output http-sleep.json
```

The client reads the bearer secret from `TENSORFOLD_SLEEP_TOKEN` by default. Host
tests cover failed reloads, cleanup failures and timeouts; this hardware run does
not inject GPU OOM or storage corruption.
