# Native Metal coverage

This fork ports the inference hosts to Zig and embeds the original author's Metal
kernels. MLX-C supplies arrays, scheduling, safetensors, and general operations.
This matrix distinguishes exercised behavior from physical-device validation.

## Model paths

| Path | Native implementation | Executed checks on M5 Max |
| --- | --- | --- |
| Qwen3.8-27B target | Embedding, quantized projections, norms, GDN, attention, gated MLP, head | 993,280 Python logits exact; sampled and greedy completions |
| Dense tree verification | Parent/depth metadata, recurrent replay, KV gather and partial commit | 513-token prefix, branched rows, continuation and all 64 caches exact |
| Dense SIMD | simdgroup quantized matmul and row attention | Forced SIMD tree/cache check; 128-token GPU-sampled DFlash2 equals serial; physical M1–M4 untested |
| DFlash2 | Five layers, hidden taps, dynamic convolution, selector codebooks, candidate tree | 128 sampled tokens match Python serial; GPU radix top-k preserves output |
| Context copy | Suffix matching and target-verified proposals | Host tests and generation path |
| Nemotron target | Mamba convolution/SSM, grouped norm, NoPE attention, top-6 routing, ReLU² experts, shared experts | 524,288 Python logits exact; verified rows and all 52 caches exact |
| Nemotron MTP | Embedding/hidden fusion, attention, MoE, cloned draft cache | 32 sampled tokens match Python serial with both CPU and GPU samplers |
| Nemotron SIMD | Row quantized projections and expert kernels; MLX SDPA | Forced SIMD verified rows, rollback and all 52 caches exact |
| Nemotron long context | 128-wide tensor attention at 10K visible keys; SDPA otherwise | Kernel fixtures at 9,999 and 10,007 keys, serial/window exact |
| Flash Next target | Four residual streams, hyper-connections, GDN, sparse attention, top-10 experts/shared gate | 248,320 Python logits exact; verified rows and all 48 caches exact |
| Flash Next PLE | N-gram hashing/EOS reset, lazy quantized shards, gate, dilated convolution | Shipped hash constants checked; full-model parity and rollback history/conv cache exact |
| Flash Next MTP | Embedding and stream fusion, HC attention/MoE, separate output mixer | 16 sampled tokens match Python serial |
| Flash sparse selection | Pool four keys, select top 512 blocks, include causal tail, merge attention | Prefixes 2,044/2,051/2,063; cold and populated pool, eight rows and rollback at every nonzero row |

## Shared Metal utilities

`test-metal` generates fixtures using the original Python wrappers and independently
loads/dispatches them through Zig/MLX-C. Tests fail on any bit mismatch.

| Utility | Cases |
| --- | --- |
| GPU sampling | BF16/f32 logits; greedy and sampled; seeds 0/1234/5678/UINT64_MAX; mapped and full token IDs; top-k 0/20/2048; top-p .8/.95/1; ties and dominant logits; positions through 262,144 |
| Radix top-k | Vocabularies 31/4,097; k 1/16/31/64; deterministic ties; BF16 conversion of f32 input |
| Tensor attention | D128/D256; 1/3/8 queries; 513/9,999/10,007 keys; cache views with unused trailing capacity; every batch row equals serial |
| Sparse attention | Original fused Python attention with deterministic synthetic weights at actual Flash dimensions; pooled keys and projected outputs match; rejected rows do not affect continuation |

The fixtures use synthetic inputs to reach expensive context branches cheaply.
The separate full-checkpoint tests establish model integration. Kernel export checks
ensure embedded sources remain verbatim; source export alone is not a numerical test.

## Reproduce

```sh
bash scripts/fetch-zig.sh
.zig-toolchain/zig build test -Doptimize=safe
.venv/bin/python tools/export_native_kernels.py --check
.zig-toolchain/zig build test-metal -Doptimize=safe -Dmetal-tensors=true
.zig-toolchain/zig build test-models -Doptimize=safe
```

The suite needs the MLX prefix described in README and Python development dependencies
in `.venv`. Omit `-Dmetal-tensors=true` on GPUs without tensor units. Model paths default
to `build/models`; override with `-Dmodel-root=/absolute/path`. All fixtures and weights
stay in ignored `build/` directories. `test-models` deliberately serializes model loads.

## Limits

Physical validation is on one M5 Max with 128 GiB, with both normal and forced SIMD
dispatch. Older Apple GPUs still require execution on those devices before claiming
hardware qualification. Flash Next's short tests fit through lazy PLE access; long
full-model context memory and throughput are not established on this machine.

This is functional coverage of the three upstream Metal inference recipes. Native
dispatch selects one implementation of each operation; historical or optional fused
optimization variants are not all separate native execution modes. CUDA, HTTP serving,
chat templates, vision, and disk prefix-cache persistence are outside this Metal port.
