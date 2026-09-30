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

Every milestone follows the test-first workflow and gates below. A milestone
is complete only when its required tests pass; a skipped GPU or full-model
gate remains pending. The milestone order is 0 through 5, then 7 for the first
serial release. Milestone 6 is a separate optional follow-up.

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
  --tokenizer-dir ./pinned-0731-tokenizer \
  --companion-reserve-gib "$COMPANION_RESERVE_GIB" \
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

`COMPANION_RESERVE_GIB` must be selected from measured companion peaks and
headroom before this command is usable; it has no invented default. The helper
rejects a missing, negative or non-finite value. `--tokenizer-dir` names an
already available, pinned 0731-compatible tokenizer/config source; preparation
must not silently download another model revision. A build can succeed while
its recorded launch preflight reports insufficient memory. `--preflight-only`
performs validation/reporting without installation or compilation.

## Fixed scope and interfaces

Use these requirements as the implementation checklist. Changing them requires
a documented design update and corresponding test changes before new behavior
is implemented.

| ID | Requirement |
| --- | --- |
| R1 | The existing DeepSeek endpoint, Hunyuan and moderation remain running during development; build/tests never manage their lifecycle or client routes. |
| R2 | Read the named 0731 GGUF unchanged, with IQ2_XXS/Q2_K/Q8_0/F16/F32 validated from actual tensor descriptors. No full BF16 expert expansion, replacement quant or download. |
| R3 | Native TensorFold family CUDA engine, `tp=1`, one request, no drafting for the first release. Preserve the existing MLX family and shared server contracts. |
| R4 | Companion-aware physical-memory accounting precedes full loading; oversized explicit context is refused. A configured 262,144-token context is a target awaiting measurement. |
| R5 | Build/install/prepare are separate from serve; CPU preparation does not create CUDA allocations, and startup uses its own candidate port and caches. |
| R6 | Serial forward passes independent quant/math/state fixtures and real-checkpoint quality gates. Exactness and model fidelity are separately evidenced. |
| R7 | Build artifact, checkpoint identity, tokenizer provenance, toolchain and test results are reproducible and recorded. |
| R8 | Optional DSpark requires the matching 0731 head, additional memory admission and drafted-versus-serial equality; legacy MTP is refused. |

Suggested file ownership, subject to the agreed upstream boundary:

- Shared GGUF container parsing belongs in one agreed reader coordinated with
  PR #119. Model-specific tensor aliases, shapes and semantics belong under
  `families/deepseek_v4/`; do not put V4 architecture decisions in a shared reader.
- Backend implementation belongs in `families/deepseek_v4/cuda/`: separate
  checkpoint/weights, forward/attention/compressor, cache, engine and app
  responsibilities. Exact filenames may follow donor or upstream conventions.
- GGUF size input extends `cuda/capacity.py`; serving, sampling and extension
  compilation use shared components. Declare compiled sources in package data.
- The proposed `tools/build_deepseek_v4_cuda.py` orchestrates isolated artifact
  creation and preflight; it does not become a second inference engine.
- CPU tests live in `tests/test_deepseek_v4_*.py`; device tests in
  `tests/cuda/test_deepseek_v4_*.py`. Reuse existing prompt/CLI/capacity fixtures
  and test helpers rather than copying their assertions into parallel suites.

The CPU inspection result must describe every tensor's canonical name, stored
name/type, shape, byte span, block size and required interpretation. Validate
unique names, offsets/alignment, shape products, quant block divisibility,
bounds against actual file size and required/optional tensor sets. Metadata
must retain the architecture, EOS IDs, maximum context and tokenizer identity.
Unsupported architecture differences fail explicitly; a filename suffix is
not sufficient evidence of generation/head compatibility.

Candidate metadata must carry a versioned GGUF storage descriptor separate
from architecture settings: source path, size, recorded immutable identity,
descriptor digest and tokenizer/config provenance. Use an existing verified
checksum or deliberately compute one once before final qualification; a
header digest plus size/mtime is only a fast consistency check, not a full
content identity. Keep generated files within `--model-dir`, write them
atomically, and require explicit replacement of incompatible existing
metadata. No modification of the source GGUF or live tokenizer directory.

The CUDA engine factory must match the existing family contract and expose
`eos`, effective context/capacity and
`generate(prompt, max_tokens, sampling, on_tokens, ...)`. Match the server's
return-statistics and cancellation contract using its existing engine fixtures.
Reject `tp != 1`, unsupported parallel settings, incompatible drafters and
unknown formats before loading. `ignore_eos`, stop strings and token limits
must use existing request policy. Prefix reuse and CUDA graphs may be disabled
initially; enabling either requires its equality tests. Do not advertise a
feature because the shared CLI can parse its flag.

## Test-first development protocol

For each behavior below:

1. Write an observable test that fails on the missing or incorrect behavior.
   Use a small independent fixture/oracle; do not generate expected results
   through the implementation under test. Record the failing command and
   failure reason. A fixture/import problem is not the intended failing test.
2. Implement the smallest supported behavior that makes the test pass. Run the
   targeted test, then the existing tests for any shared code it touched.
3. Refactor while those tests remain green. Preserve the same oracle and
   tolerance. Never loosen a numerical gate merely to accept a new kernel.
4. Record the test command, result, artifact revision and resource ceiling.
   Commit the test and implementation together in a reviewable increment.
5. Stop at the milestone's gate. Mark hardware-unavailable checks pending and
   continue independent CPU/build work; do not convert skipped checks into a
   claim of working CUDA or full-model support.

CPU tests may use numpy and minimal existing project dependencies, but must
not import MLX, initialize CUDA or require target weights. Mock the allocator
and weight loader so refusal tests fail if those paths are reached. CUDA tests
use deterministic tiny fixtures, fixed seeds and one worker. Suggested initial
test limits are 256 MiB device allocations per process and two compiler jobs;
include context/extension overhead in preflight and lower the limit if the live
stack has less headroom. These limits are development controls, not inferred
available-memory guarantees or production context limits.

## Test matrix and milestone gates

These are proposed test modules/behaviors to implement, not existing passing
tests. Prefer adding cases to existing suites when they cover the same contract.

| Gate | Tests written first | Required passing behavior |
| --- | --- | --- |
| G0: scope/reuse (R1–R3) | Import/discovery and backend-option tests | CPU family discovery imports neither torch backend nor MLX; existing MLX selection remains valid; unsupported ranks/features fail before loading. Record donor and upstream revisions. |
| G1: GGUF inspection (R2,R7) | `test_deepseek_v4_gguf.py`: tiny handcrafted containers; wrong magic/version, duplicate tensor names, malformed metadata, overflow/negative counts, overlap/out-of-bounds spans, invalid shapes, truncation and unsupported types | Header-only inspection returns exact byte spans and type/shape inventory, with descriptive failures and bounded reads. Required V4 tensor schema matches the fixture; missing or unexpected architecture fields are handled explicitly. |
| G2: sidecars/CLI (R2,R5,R7) | `test_deepseek_v4_prepare.py`: mismatched tokenizer/vocabulary/EOS, conflicting model directories, changed source identity, absent reserve and interrupted writes | Generated metadata reproduces the selected checkpoint's architecture/tokenizer, stays within candidate paths, is atomic and idempotent for identical inputs, and is detected by normal family discovery. Preflight does no install, GPU work or service launch. |
| G3: quant primitives (R2,R6) | CPU quant fixtures and `tests/cuda/test_deepseek_v4_quant.py`: zero/negative/subnormal scales, extreme codes, each IQ2 sign/grid pattern, Q2 subblocks, block boundaries, multiple expert IDs, strides and final tiles | CPU decode agrees with pinned ggml/ds4 fixtures; CUDA unpack agrees with that decode under declared output rounding. GEMV/GEMM checks cover relevant dimensions and tails; no whole-model dequantization or second expert copy is allocated. |
| G4: forward primitives (R3,R6) | `tests/cuda/test_deepseek_v4_forward.py`: RMSNorm, hc mixing, Sinkhorn, routing ties, token-table layers, shared expert, gated activation clamp, positional frequencies and output head | Each primitive matches an independent high-precision calculation within a preregistered dtype-specific tolerance. Stable tie ordering and fixed reduction partitions are tested separately by exact equality. |
| G5: attention/state (R3,R6) | `tests/cuda/test_deepseek_v4_cache.py` and `test_deepseek_v4_attention.py`: ratio 0/4/128, boundaries, window/ring wrap, sparse transition and partial keeps | Visibility includes only valid past/current rows; no future or stale pool row leaks. Fresh and resumed supported paths match; rejected rows never corrupt committed cache. V4 overlap/position rules match an independent miniature forward. |
| G6: startup memory (R1,R4) | CPU cases in `test_cuda_capacity.py` plus V4 geometry tests: fitting and nonfitting budgets, missing memory info, reclaimed pages, companion reserve, mapped/resident/staging bytes, changed pressure and explicit/default contexts | Shared receipt includes all live/candidate budgets once, accounts for prefill and graph peaks, and refuses before full load. No negative/overflow capacity; explicit requested context is never silently reduced. Insufficient memory does not trigger service management. |
| G7: serial engine (R3,R6) | Tiny end-to-end fixture tests: prefill + decode, zero/one/max reply, EOS/ignore-EOS, seeded sampling, callback cancellation, exception cleanup and repeated request | Sequence and committed state match the independent serial fixture; callback receives each committed token once; effective capacity bounds prompt plus reply; cleanup releases candidate-owned allocations/mappings. |
| G8: HTTP integration (R1,R3,R5) | Existing CUDA server tests with tiny V4 engine: models/health, completions/chat, streaming, thinking/DSML tools, stops, context refusal, disconnect and error paths | Reuse shared response schemas, finish reasons and policies. Bad requests fail before streaming as appropriate; disconnect ends candidate work; listen only on requested candidate port; no redirect or restart of live services. |
| G9: artifact/handoff (R5,R7) | `test_deepseek_v4_build.py`: subprocess commands stubbed; missing compiler, failed build, incompatible dependencies, wrong target architecture, absent kernel files and repeat builds | Wheel includes kernel sources/licenses; build uses candidate caches/env and requested job cap; clean wheel install and preflight work without the checkout; diagnostics distinguish build success from pending launch qualification. |
| G10: real checkpoint (R1,R2,R4,R6,R7) | Operator-run manifest-driven quality, long-context and coexistence fixtures, explicitly separated from routine unit tests | Same quant and tokenizer; real chat/tools work, long prefill/decode stays within admitted capacity, actual Hunyuan generation and CUDA moderation succeed, candidate errors do not disrupt live serving. Record system pressure and failure evidence. |
| G11: optional features (R8) | Window-width/drafted equality, base/head mismatch, partial accept/reject, resumed/fresh, eager/graph and concurrent/solo cases | Enable only independently qualified features. Legacy MTP is rejected for 0731; DSpark also fits the companion budget. All supported verify widths reproduce serial tokens and committed state. |

For cache tests, explicitly cover positions 0/1, 3/4/5, 127/128/129,
2,047/2,048/2,049 and multiple wraps. Include compression block completion,
partially filled blocks, overlap at ratio 4 and top-k score ties. The exact
first sparse-selection position follows validated V4 visibility semantics;
test both sides of that boundary rather than assuming a generic GLM cutoff.
Use real head dimensions in isolated tests where their bounded footprint fits.
Before enabling verify width W, test every accepted prefix from 0 through W,
including rejection on a compression/window boundary and decode immediately
after rollback. Default serial width is one until wider widths are qualified.

## Numerical oracles and quality qualification

Use three distinct kinds of evidence:

- **Stored values:** independent ggml/ds4 quant vectors with pinned revisions
  and expected decoded values. For identical declared output dtype and decode
  rounding, require exact equality; verify the quant codebook and scale rules
  rather than just matching two copies of the same decoder.
- **Model math:** a small CPU numpy/high-precision reference implementing the
  validated V4 equations on those stored values. Define absolute/relative
  tolerances per operation and activation dtype before testing; record them
  with fixtures, including near-zero and close-logit cases. The independent
  reference and tolerance review are prerequisites to the affected gate.
- **Runtime exactness:** bitwise equality of supported serial/verify cache
  states and complete token sequences under identical weights, settings,
  runtime, seed and absolute positions. No numerical tolerance substitutes for
  this gate. Cross-runtime ds4/TensorFold bitwise equality is not presumed.

For the real GGUF, pin a public prompt fixture containing plain chat, a coding
task, thinking, offered tools and a tool-result follow-up. Collect the trusted
same-quant reference's rendered tokens, selected layer outputs/logits where
available, top-token margins and quality/latency results. Specify the quality
thresholds in the fixture manifest before looking at candidate results. Close
logit disagreements require diagnosis; do not classify a broken attention or
router as acceptable rounding. A rendered prompt mismatch is an integration
failure and must be fixed before comparing inference quality. Continue using
the official TensorFold public benchmark separately for comparable throughput.

Full-model execution alongside two instances of DeepSeek is not a required
development assumption. If the running service leaves insufficient memory,
G10 remains pending and the artifact is labeled **build ready, full-model
qualification pending**. Final validation of TensorFold + Hunyuan + moderation
may require a later operator-arranged service replacement. That action is not
authorized or automated by this plan. No result from tiny fixtures can replace
the final target-hardware gate.

## Evidence, drift control and completion

Maintain a versioned fixture/result manifest containing:

- Requirement/gate IDs, test names and current state: pending, failed, passed
  or skipped with reason; skipped required tests block the relevant claim.
- TensorFold base/candidate commits, donor kernel/parser commits, copied
  source paths/licenses and coordinated upstream PR revisions.
- GGUF full identity and tensor inventory, sidecar provenance, tokenizer
  revision/hash, architecture differences and effective cache precision.
- CPU architecture, GB10 compute capability as queried from the target,
  Python/compiler/toolchain/container versions, package pins and build flags.
- Build commands, wheel hash, fixture seeds, rendered token hashes,
  preregistered tolerances/quality thresholds and complete output hashes.
- Requested/allocated context, reply limit, process/system peak memory,
  mapped/staging/cache/graph bytes, companion reserve, swap/page faults,
  benchmark commands and measured latency/throughput.
- Live endpoint probes before/after bounded tests and any observed impact.

Keep portable fixtures/results in the agreed test/benchmark locations, and
large machine-specific artifacts outside the repository. Do not publish
credentials, user prompts, workstation paths or private service data. Record
decisions as requirement changes with the reason and affected tests. Work on
new quants, two ranks, concurrency, new drafters or performance optimizations
only as separate follow-ups with their own gates. Recheck upstream/donor
changes before each shared-code increment; extend shared tests when behavior
changes for other families.

The serial implementation is **build ready** after G0–G9 and required existing
regression checks pass, the wheel installs cleanly and the two-command handoff
works through preflight. It is **qualified for this deployment** only after
G10 passes on the named GGUF with the measured companion budget. DSpark and
other optional features need G11 for each advertised feature. At handoff,
provide the exact pinned build/run commands, result manifest and remaining
pending gates; never describe a successful compilation as proof the model
runs correctly.

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
