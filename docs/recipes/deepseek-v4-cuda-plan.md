# Proposal: DeepSeek V4 Flash GGUF on one DGX Spark

This is an implementation proposal, not a supported backend or a measured
TensorFold configuration. The target is one NVIDIA GB10 DGX Spark with 128 GB
nominal unified memory, sharing the machine with a loaded Hunyuan3D-2.1 shape
pipeline and CUDA image moderation. Two-rank execution is outside this proposal.

Development must keep the existing DeepSeek endpoint and its loaded model
available. The deliverable is a separately built TensorFold backend that can
later serve the same unchanged GGUF; developing it must not require replacing
the active ds4 installation or restarting the live stack.

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

7. **Build and launch handoff.** Deliver a documented build helper and an
   ordinary `tensorfold serve` invocation for this checkpoint. Package the
   CUDA sources, model-directory metadata adapter, and required dependencies;
   prebuild the extensions that can be compiled without loading the model.
   Document any remaining Triton specialization or graph capture performed on
   first launch. Installation, build and launch must not manage the existing
   DeepSeek/Hunyuan services or modify client routing.

## Development with DeepSeek still running

Keep this work in a separate checkout and virtual environment or container,
with its own build, extension and model-state caches. Read the GGUF in place;
write generated config/tokenizer metadata into a candidate model directory.
Treat the live ds4 binary, installed launcher, service units and checkpoint
files as read-only inputs. Do not rebuild into the live binary directory,
install into its environment, bind its port, or invoke stack stop/restart
targets. Failed builds and tests must leave the existing service usable.

Use these stages in order:

1. **Inspect and build without loading the full model.** Read GGUF headers and
   selected quant blocks, generate metadata, run CPU fixtures, build the wheel
   and compile CUDA extensions in the candidate environment. Bound compiler
   parallelism and host-memory use. A large file hash or scan also competes for
   storage bandwidth; perform those deliberately rather than on every edit.

2. **Run bounded CUDA tests.** Reuse tiny synthetic DeepSeek fixtures and small
   quant blocks, with an explicit allocation ceiling and one GPU test worker.
   Check the available-memory budget before starting. Tests must skip with a
   clear reason if the live workload leaves insufficient headroom. Compilation
   or a small kernel test can still contend for CPU/GPU time; keeping the live
   model loaded does not imply zero latency impact.

3. **Validate the candidate launch without allocation.** Provide a helper
   preflight mode that validates the GGUF/sidecars and reports packed weight,
   staging, cache/workspace and companion-reservation estimates. It must not
   load weights, allocate a full cache, evict live resources or start a server.
   Record build/runtime pins and the model identity with the candidate artifact.

4. **Attempt a separate full-model test only if the budget permits.** Use a
   loopback candidate port such as 18000. Count the live DeepSeek, Hunyuan and
   moderation footprint as occupied memory. Sharing a pathname or file-backed
   pages does not establish sharing of CUDA allocations, registered pages,
   graphs or caches. Refuse candidate startup before major allocation if it
   cannot fit alongside them; never stop the live service to make a test fit.
   An estimate is not a hard resource limit: retain headroom, use bounded
   allocations, and abort candidate loading if pressure rises.

5. **Keep full-model qualification pending when it cannot fit.** CPU and tiny
   GPU tests can make an implementation ready for launch, but cannot prove
   full-model quality, long-context capacity or production coexistence.
   Complete those measurements in a separately arranged final validation
   window. Any replacement of the live service is a distinct operator action
   after development, not a prerequisite or an automatic build/test step.

## Intended build and run interface

The following is the interface the implementation should deliver. The build
helper and GGUF adapter do not exist yet; these are acceptance targets, not
commands supported by the current branch. Run the build in the isolated,
documented NVIDIA toolchain environment:

```bash
# Build/install the candidate, prepare sidecars, and inspect the launch budget.
# The helper reads the existing GGUF and does not start or stop any service.
python tools/build_deepseek_v4_cuda.py \
  --gguf "$HOME/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf" \
  --model-dir ./candidate-model --jobs 2

# Launch the completed backend separately; refuse before loading if it cannot fit.
tensorfold serve ./candidate-model \
  --backend cuda --tp 1 --parallel 1 --no-drafts \
  --context 262144 --name deepseek-v4-flash \
  --host 127.0.0.1 --port 18000
```

The helper must also offer `--preflight-only` for stage 3. Select and record the
companion reservation in candidate metadata or a maintainer-approved shared
budget option, then require the loader to enforce that setting. The live
service remains on its original endpoint throughout development. The candidate
port does not trigger a production cutover, and a refused launch does not
change the active stack.

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
- Build and preflight work with the live stack running. Candidate tests use
  separate paths and endpoints; a failed build, insufficient-memory refusal
  or candidate exit leaves live DeepSeek serving requests. Confirm this with
  lightweight health/chat probes around bounded tests, and record any impact.
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
