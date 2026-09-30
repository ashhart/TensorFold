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

1. **Checkpoint inspection and validation.** Add a GGUF reader that inspects
   metadata, tensor names, shapes, offsets and mixed quant types without loading
   weights. Validate the 0731 architecture, tokenizer and generation metadata.
   Reject unknown layouts, truncated data and incompatible heads before GPU
   allocation. Audit differences from the existing MLX family explicitly.

2. **Packed CUDA weights.** Implement IQ2_XXS, Q2_K and Q8_0 projection/expert
   paths with inline dequantization; retain F16/F32 tensors in their stored
   precision where appropriate. Candidate components include the existing CUDA
   affine infrastructure and ds4's GGUF/CUDA implementation. Existing affine,
   EXL3 and NVFP4 readers are not interchangeable with these GGUF formats.
   Preserve attribution and license notices for any imported implementation.

3. **Serial target forward.** Port the four residual streams and hyper-connection
   mixing, sliding-window attention, compression ratios 4 and 128, sparse
   indexer selection, RoPE/YaRN, token-table routing, sqrt-softplus routing and
   shared/routed experts. Begin with one request and no drafter. Use tiny
   synthetic checkpoints for correctness, then the target GGUF for quality.

4. **Caches and memory admission.** Implement key/projection rings, compressed
   pools and bounded prefill workspace. Choose and validate the cache precision
   explicitly; stored weight quantization does not define cache precision.
   Keep packed weights mapped or otherwise resident without a second full copy.
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

6. **Optional DSpark and exact verification.** Add
   `DSpark-drafter-Q2K-Q8-0731.gguf` after the serial coexistence target passes.
   Validate base/head generation compatibility; do not attach the legacy MTP
   GGUF to the 0731 base. Qualify every enabled verify width, routing ties,
   accepted-prefix commits and rollback of both target and drafter state.
   Keep drafting disabled if exactness or the companion memory budget fails.

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
