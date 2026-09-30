# Proposal: DeepSeek V4 Flash GGUF on one DGX Spark

This is an implementation proposal, not a supported backend or a measured
TensorFold configuration. The target is one NVIDIA GB10 DGX Spark with 128 GB
nominal unified memory, sharing the machine with a loaded Hunyuan3D-2.1 shape
pipeline and CUDA image moderation. Two-rank execution is outside this proposal.

## Target checkpoint and baseline

The target file is
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`,
approximately 81 GiB, distributed through
[antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf).
Its mixed layout uses IQ2_XXS routed-expert gate/up weights, Q2_K routed-expert
down weights, Q8_0 dense projections, and F16/F32 tensors. Validate actual tensor
types and dimensions from the file rather than dispatching from its name.

The existing deployment uses [Entrpi/ds4](https://github.com/Entrpi/ds4), mapped
base weights, one session, a configured context of 262,144 tokens, an 8 GiB live
memory floor, and speculation disabled. These are deployment settings, not
TensorFold capacity or throughput claims. Admission must account for actual
available memory and a reserved companion budget; an 8 GiB free-memory floor
alone does not guarantee Hunyuan can complete generation.

TensorFold's existing MLX conversion keeps about 151 GiB resident and cannot be
substituted for this checkpoint on one Spark. Neither expanding all experts to
BF16 nor requiring that conversion meets this target.

## Implementation milestones

0. **Agree the integration boundary.** Extend the existing `deepseek_v4`
   family under `families/deepseek_v4/cuda/`, following the
   [CUDA family guide](adding-a-cuda-family.md) and
   [CUDA arithmetic rules](cuda.md). Review the reuse map below with maintainers
   before implementing new shared modules. Continue the CUDA work anticipated
   in [DeepSeek support issue #14](https://github.com/ashhart/TensorFold/issues/14),
   whose release follow-up shipped Mac support and left CUDA pending. Confirm
   whether the maintainer has unpublished CUDA work before starting a port.
   Audit the MIT-licensed ds4 backend
   as a candidate source of already working GGUF and GB10 kernels; pin a donor
   revision, preserve notices, and test its arithmetic against TensorFold's
   contract before importing components. Wrapping a separate ds4 HTTP server
   would not establish a TensorFold engine or qualify its exactness contract.

1. **Checkpoint inspection and validation.** Reuse or extend a GGUF reader that inspects
   metadata, tensor names, shapes, offsets and mixed quant types without loading
   weights. Validate the 0731 architecture, tokenizer and generation metadata.
   Reject unknown layouts, truncated data and incompatible heads before GPU
   allocation. Audit differences from the existing MLX family explicitly.
   Coordinate with [PR #119](https://github.com/ashhart/TensorFold/pull/119),
   which proposes a numpy/mmap GGUF reader for GLM Q8_0 conversion. Its metadata
   parsing is a reuse candidate; its GLM mappings and Q8_0 conversion do not
   implement DeepSeek IQ2_XXS/Q2_K CUDA execution. Prefer a model directory
   containing config/tokenizer files and a referenced unchanged GGUF, since
   current discovery and serving expect that directory layout. Do not silently
   treat a bare GGUF as an existing supported CLI input. Validate any generated
   sidecar metadata against the GGUF and pin its provenance.

2. **Packed CUDA weights.** Implement IQ2_XXS, Q2_K and Q8_0 projection/expert
   paths by adapting validated ds4/ggml kernels where suitable, with inline
   dequantization; retain F16/F32 tensors in their stored
   precision where appropriate. Candidate components include the existing CUDA
   affine infrastructure and ds4's GGUF/CUDA implementation. Existing affine,
   EXL3 and NVFP4 readers are not interchangeable with these GGUF formats.
   Preserve attribution and license notices for any imported implementation.

3. **Serial target forward.** Port the four residual streams and hyper-connection
   mixing, sliding-window attention, compression ratios 4 and 128, sparse
   indexer selection, RoPE/YaRN, token-table routing, sqrt-softplus routing and
   shared/routed experts. Begin with one request and no drafter. Use tiny
   synthetic checkpoints for correctness, then the target GGUF for quality.
   Start from the existing DeepSeek model decomposition and test fixtures.
   Reuse GLM CUDA hyper-connection primitives where equations, dtype and
   reduction order agree. Its sparse selector is a candidate for fixed-order
   top-k selection, but GLM's attention/pooling is not a drop-in V4 compressor.

4. **Caches and memory admission.** Implement key/projection rings, compressed
   pools and bounded prefill workspace. Choose and validate the cache precision
   explicitly; stored weight quantization does not define cache precision.
   Keep packed weights mapped or otherwise resident without a second full copy.
   Extend the shared `cuda/capacity.py` accounting and receipt rather than add
   a separate memory governor. Its current header reader sizes safetensors;
   supply verified GGUF weight sizes and DeepSeek cache geometry. Coordinate
   a companion reserve through the existing budget calculation, subtracting
   only the additional reservation not already reflected in available memory.
   Account for host mappings, staging, cache growth, graphs and runtime buffers
   in the same physical-memory budget on GB10. Reserve memory for companions
   and refuse an explicit context that cannot fit. Map retention, paging and
   any disk cache behavior must be measured rather than assumed free.

5. **TensorFold serving.** Add the family's `cuda_engine` entry point and CUDA
   quant declaration only after the serial backend works. Implement the
   [CUDA engine contract](adding-a-cuda-family.md), capacity reporting and
   cancellation; reuse DeepSeek prompt encoding and DSML parsing only after
   checking them against the 0731 tokenizer/template. Verify thinking, tools,
   streaming, EOS and request errors through the OpenAI-compatible server.
   Use `cuda/server.py`, existing request policies and keyed sampling. Supply a
   family `CUDA_APP` for the existing DeepSeek encoder if required, instead of
   writing another HTTP server or relying on the generic Jinja chat template.
   Keep torch imports inside backend modules and leave MLX discovery usable.

6. **Optional DSpark and exact verification.** Add
   `DSpark-drafter-Q2K-Q8-0731.gguf` after the serial coexistence target passes.
   Validate base/head generation compatibility; do not attach the legacy MTP
   GGUF to the 0731 base. Qualify every enabled verify width, routing ties,
   accepted-prefix commits and rollback of both target and drafter state.
   Keep drafting disabled if exactness or the companion memory budget fails.

## Reuse map and remaining work

Paths below are relative to `src/tensorfold/`; candidates need compatibility
tests before reuse, and are not promises that the current code accepts GGUF.

| Existing component | Reuse or adaptation | Remaining work |
| --- | --- | --- |
| `families/deepseek_v4/config.py`, `model.py`, `prompts.py` and tests | Architecture validation, model decomposition, prompt encoder and fixtures | Audit 0731 differences and supply CUDA tensors/cache operations |
| `families/glm5_next/cuda/glue.py` | RMSNorm, hyper-connections and combination primitives where compatible | V4 token-table and sqrt-softplus routing; different quant projections |
| `families/glm5_next/cuda/sparse.py` | Stable sparse-selection ordering and partitioning patterns | V4 sliding window, 4/128 compressors, overlap and RoPE/YaRN |
| `cuda/capacity.py` | Unified-memory availability, weight/cache geometry, fitting context and receipts | GGUF header sizing, companion reservation and mapped-weight qualification |
| `cuda/server.py`, `sampling.py`, request policy and health modules | HTTP lifecycle, admission hooks, keyed sampling and error reporting | Family engine and DeepSeek prompt adapter |
| `cuda/build.py`, `direct_read.py`, existing package-data declarations | Extension builds and bounded-loading patterns | GGUF mapping/lifetime support; direct reader currently targets safetensors |
| PR #119 and ds4/ggml GGUF implementations | Reader/quant fixtures and existing kernels after review | One agreed reader, V4 tensor mapping and exactness qualification |

The repository audit at base commit `9cd52ab` found an existing MLX DeepSeek
recipe but no DeepSeek CUDA engine or merged IQ2_XXS/Q2_K reader. The public PR
audit found #119's GLM GGUF converter proposal, not a DeepSeek CUDA port. Recheck
upstream before implementation, including the CUDA follow-up in #14; extract shared helpers only when actual reuse
justifies them, and avoid copying a complete GLM engine into a new family.

## Acceptance gates

- CPU tests cover metadata, malformed files, quant decoding against known
  fixtures, format rejection and architecture/head compatibility.
- GPU tests cover real head dimensions, compression boundaries, the 2,048-token
  sparse-selection transition, cache wrapping, partial keeps, and eager versus
  graph execution. Quality comparisons use a trusted forward on the same GGUF;
  serial self-consistency alone does not establish model fidelity.
- An explicit 262,144-token capacity must fit the whole deployment or be
  refused with a measured fitting limit. Exercise long prefill and decode with
  Hunyuan and moderation loaded, including an actual shape-generation request.
  Record process and system peak memory, paging, latency and failures.
- For any enabled drafting or prefix reuse, compare complete token sequences
  against the same engine's serial or fresh run under identical sampling.
  Add concurrency only after concurrent-versus-solo equality is established.
- Publish checkpoint hashes/revisions, runtime and CUDA toolchain pins, launch
  commands, rendered-token counts and benchmark outputs using the
  [public fixtures](README.md#measurements). Package all kernel sources in the
  wheel. No performance or coexistence claim precedes these measurements.

The first functional release gate is serial decoding of this GGUF beside
Hunyuan on one Spark. DSpark, batching and other quant formats are follow-ups.
