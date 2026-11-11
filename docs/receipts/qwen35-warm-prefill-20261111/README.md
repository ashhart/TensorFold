# Qwen 3.5 CUDA foreground-prefill yield

## Scope

This receipt covers the foreground-yield fix in this pull request. Before another chunk starts, `cuda_prefill.chunk` checks `stream.yields()`. When foreground work is waiting, it preserves the completed prefix with `keepInterruptedPrefill(f.at)` and returns `error.Yielded`. No new forward starts after the yield is observed.

On the final rebase base, `93a2e93ec`, `cuda_native.prefillFn` routes a normal whole prompt through `prefill.run`; `run` calls `chunk` until the prompt completes. The warm pass uses this whole-prefill path, so it reaches the yield check.

## Host regression

On the narrowed head, `zig build test-qwen35-prefill-host --summary all` passed 17/17 tests. The new test confirms one completed chunk, retention of its exact prefix, and no second forward after yield.

Both negative controls failed as intended:

- Without the yield check, the test failed with `expected error.Yielded, found true`.
- With the yield check but without `keepInterruptedPrefill`, the test failed with `expected 1, found 0` for the saved-prefix callback.

## GPU integration context

These integration measurements were collected on 2026-11-11 against main base `ccad6e6cddf1536d14834b59e666b9d59bcb0c3c`, using combined candidate source `708f26effc0525623c909b3af37683e52124db33`. That source still contained the warm-turn enablement and other hunks now dropped from this PR. The measurements demonstrate the warm-prefill scenario, but are not measurements of this narrowed head and do not isolate the yield change's latency impact. They also predate the new yield-side `keepInterruptedPrefill` call; the host regression below tests that behavior.

| Path | Cached tokens | Engine prefill | Client TTFT | Yields |
| --- | ---: | ---: | ---: | ---: |
| Main, cold continuation | 0 | 2.565895 s | 2.567245 s | 0 |
| Earlier combined branch, foreground request after 200 ms | 4,096 | 0.163107 s | 2.534763 s | 1 |

The continuation token and output hashes matched. TTFT stayed near the cold latency because the chunk already in progress completed before foreground prefill resumed. A separate warm-boundary case reused 4,096 tokens and measured 2.576211 s cold prefill versus 0.163060 s after the warm pass, with matching hashes.

The raw JSON and SHA-256 manifest for the full earlier qualification are in this [unlisted gist](https://gist.github.com/tacosta/ad6fd030862b9966791acf3d540a1591). Its records include exact token/output hashes, prompt lengths, cached-token counts, and engine prefill time distinct from client TTFT.

## Environment and limits

The earlier integration run used an NVIDIA GeForce RTX 5090 Laptop GPU (SM120, 24,463 MiB), Ubuntu 26.04 on Linux 7.0.0-43-generic, 32 GB RAM, CUDA 13.3, and Zig 0.17.0. The target was dense Qwen3.8-27B NVFP4. No immutable checkpoint revision was available; these file hashes identify the tested checkpoint:

- `config.json`: `71fa01bb64de20971045cd82a82c0ebb25a54e1b0a3f09ef23cf5479e9c54df8`
- `tokenizer.json`: `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`
- `tokenizer_config.json`: `e5d078b00e6c1223b32444db8c1001dc71d86ceef8ee706b5bf084c3a43a1f9c`
- `model.safetensors.index.json`: `7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2`

No Qwen MoE fit this GPU: the smallest available MoE weights were 22.02 GiB with a 24.06 GiB runtime floor, above 23.89 GiB total device memory. This PR adds no model arithmetic or kernel changes. Main-versus-head performance for the narrowed change was not remeasured; the previous sub-1% deltas were too small to draw a conclusion and do not establish the strict no-slower gate.
