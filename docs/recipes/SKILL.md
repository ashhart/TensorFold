---
name: tensorfold-contributor
description: Implement or review TensorFold native-engine contributions using the repository's model recipes, kernel contracts, exactness checks and reproducible performance gates. Use for model-family ports, GPU operators, weight loaders, native serving and their tests.
---

# TensorFold contributor

Produce a small, reviewable native contribution with evidence for every affected contract.
This skill defines the conditions a contribution must satisfy; it does not approve a merge, release or hardware qualification.

Before editing, follow the [ATC contributor instructions](../atc/SKILL.md), inspect the public radar and check or claim the issue.
Publish an anonymous session with `atc start <issue> --public https://tensorfold.dev --json` when reporting this work is authorized; show a GitHub name only when the human opts in.
Pulse only while working, review overlaps before a pull request, and land the session when it stops.

## Establish the contribution contract

Find the TensorFold checkout and its Git root before editing.
A native checkout has `build.zig`, `.zig-version`, `zig/src/native/` and `docs/recipes/`.
Read any AGENTS.md in the folders you touch (the app has its own), [CONTRIBUTING.md](../../CONTRIBUTING.md), [RUNBOOK.md](../../RUNBOOK.md) and the [recipe index](README.md).
For a GPU change, also read the complete [kernel contract](../kernel-contract.md) before choosing an implementation.
Target `main`, the only branch, and preserve unrelated changes; a different local TensorFold checkout may be the frozen Python engine.

Before implementation, write a short contract in the working notes or PR draft:

- The requested behavior, reproducer and existing failure, or the missing capability being added.
- The target commit, family, backend, checkpoint revision, quantization, precision mode, GPU class and rank count.
- The owning modules and existing shared operator or family adapter to extend.
- The applicable correctness checks, performance metric and agreed acceptance threshold, including cold-prefill requirements.
- The hardware available, required checks that cannot run there and the evidence needed from another tester.

New families, backends and arithmetic changes need the maintainer-agreed gate described in CONTRIBUTING.md.
If that gate is missing, identify the missing decision before presenting the work as ready to land.
Do not silently lower a threshold, change its workload or replace an unavailable test with an easier one.
The current `zig/gate.json` may have no entries; that is not permission to invent a release-wide speed target.
Creating an issue, sending a PR, uploading results or changing a shared machine still needs the user's authorization.
Keep one job per pull request; stack dependent ones, say what each sits on, and keep stacks short.
Maintainers rebase at landing, so don't force-push a branch under review just to follow main.

## Read the right recipe and the code it describes

Use [adding a Zig family](adding-a-zig-family.md) for native bring-up and optimization stages.
Read the affected family's page from the recipe index, then trace the actual native loader, backend registration, operator and tests.
For checkpoint arithmetic, also read [quantization](../quantization.md); for transport changes, read [speed-up settings](../speed-up-mode.md).

Separate native instructions from sections explicitly labelled Python 0.6.6, historical measurements or unproven next steps.
The Python engine is frozen; Python clients, fixture builders and trusted references under `tools/` are development tools, not native inference dependencies.
Do not copy a Python-only flag, environment variable, transport or support table into a native port.
Do not require every optimization stage for an unrelated bug fix or documentation change.

Some family-page introductions lag newer native code.
Check the target build's `--version`, `--help` and `capabilities --json`, its native registry and loader, and matching device/model receipts.
A registered family or compiling source proves neither whole-checkpoint support nor qualification on every GPU.
If documentation and implementation disagree, record the discrepancy and proposed correction; do not treat it as permission to relax a precision or exactness requirement.

## Keep ownership in the native architecture

| Responsibility | Inspect and extend |
| --- | --- |
| Request lifetime, events and advertised capabilities | [engine API](../../zig/src/core/engine_api.zig) and [server adapter](../../zig/src/server/engines.zig) |
| Shared scheduling, target verification, sampling and per-stream state | [lane core](../../zig/src/core/lanes/) |
| Backend selection, admission and family registration | [native adapters](../../zig/src/native/), including the relevant Metal, CUDA, HIP or XPU adapter |
| Family config, tensor mapping, layer graph and family-specific mixers | [families](../../zig/src/families/) |
| Reusable projections, layouts, caches and backend operators | [shared core](../../zig/src/core/), [CUDA operators](../../zig/src/cuda/) and [GPU kernels](../../zig/kernels/) |
| Build registration and deployable assets | [build.zig](../../build.zig), [build modules](../../zig/build/) and [packaging](../../packaging/README.md) |

Adapt the family to the existing engine interfaces; do not add a parallel scheduler or a second copy of shared verification/cache machinery.
An unsupported family is a bring-up task, not a reason to route production inference through Python, a CPU substitute or a serial-only workaround.
Plain decoding remains the correctness oracle; disabling drafts must not be required for an advertised drafted path to work.
Concurrent requests share rounds through the lane core, with drafting on for every stream the family admits; a drafts-off or serial fallback at some width or stream count is a bug, not a limit to document.
Test the path the native server will actually call, including capability reporting and intentional refusals.
A checkpoint whose config declares a vision tower needs image input in the engine's own kernels before the family is complete; until then, refuse image parts and `--vision` with a clear message and test that refusal.

Keep stored weights and tokenizer assets unchanged.
Respect each tensor's actual bit width, group size, strides, scales, bias convention and format metadata; mixed projections cannot share an incompatible fused kernel.
Reject unsupported layouts and malformed geometry explicitly, and test those refusals.
Keep scratch and prepared packs in the loader's owned cache, with its existing identity and atomic-publication rules.

## Register and prove a GPU kernel

A new or changed GPU operator must be reachable through its backend contract runner, not merely have a standalone unit test.
The registration types are [Spec](../../zig/src/core/contract/contract.zig) and [Kernel/VTable](../../zig/src/core/contract/gates.zig).
Use the shared affine, dense or rows reference helpers where their formulas match the operator.
Declare real served shapes, tile edges, output type, accumulation and rounding depth, NaN behavior, and permitted aliasing.
A bit-preserving replacement declares `replaces` and supplies the `replaced` hook on identical inputs; its contract requires byte equality.
If an intended reduction reorder changes bytes, stop treating it as that drop-in replacement and resolve the arithmetic/comparison contract with the maintainer before landing it.
An improved FP64 error alone does not waive replacement equality, authorize a new serial baseline or justify renaming the kernel to evade the gate.
Register the operator in [the Metal runner](../../zig/tests/contract/metal_main.zig) or [the CUDA runner](../../zig/tests/cuda/contract_tests.zig), as appropriate.
Register its launch limits and compile records separately in [registry.zig](../../zig/src/core/contract/registry.zig).

| Gate | Required evidence |
| --- | --- |
| Non-finite inputs | Every output class matches the FP64 formula or the operator's explicitly tested NaN rule. |
| Repeat and aliasing | Five byte-identical runs; unchanged inputs and guards; no undeclared output/input overlap. |
| Shapes | Served dimensions and valid tile edges meet the derived reference bound, or replacement outputs equal the replaced kernel's bytes. |
| Accumulation | The stress fixture distinguishes reduced-precision sums and proves the required accumulator; report stated limitations or genuine N/A cases. |
| Launch limits | Updated recorded compiles and device checks cover the relevant served chips, threads, registers, shared memory and clusters. |

Use [Metal contract tests](../../zig/tests/contract/) and [CUDA contract tests](../../zig/tests/cuda/contract_tests.zig) as concrete integration examples.
Every planted control must fail its intended gate; a control error, skip or unexpected pass proves nothing.
Attach the backend's complete `SUMMARY`, exit status, known findings and unrun chips.
Diagnostic `--only`, `--no-gates` or `--no-limits` runs are not the full qualification receipt.
Do not add a `known` entry to conceal a new regression; an existing known finding must remain reproducible until its fix removes the entry.
Metal/CUDA runner coverage does not establish HIP/XPU correctness; name missing backend coverage and agree its checks separately.

Keep checkpoint activation precision and the required accumulation precision.
Do not add FP8/int8 activations, lossy caches, reduced-precision sums or fast-math changes merely to clear a speed gate.
Checkpoint-defined arithmetic is a separate case: [Nemotron Super VL](nemotron-3.5-super-vl.md) explicitly defines `checkpoint` and `full` modes.
Qualify each claimed mode against its own plain output, label the mode in receipts, and measure fidelity between modes separately.
Do not turn an older recipe's broad precision statement into a ban on an already qualified checkpoint mode.

A changed reduction order also needs an independent FP64 reference for each affected projection class.
Native error must be no worse than the current reference.
Compare prompt and decode rows separately; retain worst-column errors and teacher-forced top-token differences and margins.
Same-engine drafted/plain equality cannot replace this independent model-fidelity evidence.
Do not widen a tolerance or apply a tie policy beyond its documented contenders and margins to manufacture a pass.

### Preserve device ordering and buffer lifetime

Trace the launcher as well as the shader.
For generated or captured kernels, update their owning source or generator and regenerate the matching assets and manifest; patching one generated copy does not update the runtime path.
The [Metal lane projection](../../zig/kernels/metal/core/lane_projection.metal) shares weight reads while preserving each row's sums; [CUDA lane GEMV](../../zig/kernels/cuda/lane_gemv.cu) preserves group and K-slice order.
Keep row arithmetic independent of verify width, grouping, prompt chunking and concurrent stream order wherever the selected contract requires it.
Check real dimensions, tails, sparse-attention boundaries and accepted-path commits, not only convenient full tiles.

GPU resources, borrowed views and per-call metadata must remain valid until their commands complete.
Preserve residency declarations and the existing Metal fence/event ordering when command buffers or queues change.
On CUDA, upload on the stream that consumes the data and preserve graph capture/replay ownership and stable buffers.
Do not introduce host readbacks or synchronizations into the hot path without measuring their dependent and served cost.
A chip-specific fast path must retain a valid capability-gated path for the older chips the contribution claims to support.

## Prove the engine contracts

At the same checkpoint, backend, runtime, rank count, precision and request settings, check every affected case:

- Drafted output equals `"draft": false` token for token at the supported windows/depths.
- Supported resumed, edited, regenerated and post-restart paths equal fresh execution of their full rendered prompts.
- Concurrent replies equal their solo runs, including backends that queue requests.
- Prompt/decode seams preserve required KV, recurrent, convolution, n-gram, drafter and final-logit state.
- Rejection, partial acceptance, cancellation and reset leave the next request valid.
- Advertised thinking, sampling, tools, structured output and media features reach the intended native path and preserve their existing API behavior.

Only claim supported combinations actually exercised; do not equate different backends, quantizations or precision modes by assumption.
Use fresh processes for cold checks and seams vulnerable to uninitialized state.
Retain before/after failures and token hashes or full token arrays, not just plausible text.
For cache changes, verify identity/invalidation, cached-token counts and genuinely cache-off controls.
For multi-host changes, include matching build/checkpoint/precision checks and peer failure/cleanup, using the affected transport's recipe.

## Select checks by the change

Use the exact Zig version in `.zig-version`; do not upgrade the toolchain as a porting shortcut.
The native-code baseline host checks are visible in [no-GPU CI](../../.github/workflows/no-gpu.yml):

```sh
zig build test test-kernel-contract test-golden fuzz-cortheon
git diff --check
python3 tools/zig/lean_check.py
```

Use [lean_new.py](../../tools/zig/lean_new.py) and the CI procedure to compare the same checker against the PR's merge base.
Inherited lean findings must be reported; the acceptance condition is zero new problems, not a fabricated clean exit.
Source modules stay at most 600 lines, with single-line comments/docstrings within 120 columns; put measurements in receipts or recipes.
Run additional family/operator tests selected from the actual build targets and the affected recipe.
CI runs `fuzz-cortheon` unconditionally; a focused local iteration is not the complete no-GPU receipt.
Run a plain `zig build` as well, because the named steps skip some tools and only the plain build compiles every target.
Use the existing packaging checks for changed assets or archives.

Mac builds need Xcode's Metal toolchain (`xcrun metal`); Command Line Tools alone can't build the metallibs.
On a Mac, a native build intended to run across supported Apple chips uses:

```sh
zig build native -Dcpu=apple_m1
zig build kernel-contract -Dcpu=apple_m1
```

Structured output and required tool calls need the release archive's `lib/` or `zig build grammar -Dxgrammar=DIR`; prepare DIR with `git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar` (use the native build's same `--prefix` for grammar).

The second command runs GPU gates and requires the test device to be available.
For compile-limit records use the Metal contract's `--offline DIR` workflow from the kernel-contract guide.
On an authorized CUDA test machine, with the qualified compiler path adjusted if necessary:

```sh
zig build kernel-contract -Dnvcc=/usr/local/cuda/bin/nvcc
zig build fatbins kernel-contract-fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=80,86,89,120,121
zig-out/bin/tf-cuda-test kernel-contract-limits zig-out/fatbin
```

Inspect [kernel-contract build registration](../../zig/build/kernel_contract.zig) and the target's build options before running a platform command.
Host tests, compile-limit records, a host-only archive and a fake-engine golden corpus do not prove real-model inference on a GPU.
Documentation-only changes need link/claim checks, not an invented model receipt or a new GPU workload.

## Attach a reproducible performance receipt

Use the native server, the issue's agreed workload and the existing [served](../../tools/bench_openai.py), [concurrent](../../tools/bench_concurrent.py) and [cold-cache](../../tools/bench_cache.py) clients.
Match the machine, baseline/candidate binary identities, checkpoint/tokenizer/drafter revisions, exact rendered prompt IDs, output length, seed and generation settings.
Record thinking mode, precision, rank count, cache state, concurrency, warmups and repetitions.
Measure ordinary greedy prose and code, and sampled or feature-specific cells when the changed path requires them.
Check that run's tokens before interpreting its speed.

Report decode rate, cold prefill/TTFT and aggregate concurrency separately.
For prompt processing, retain cold 2k/8k/32k/64k token arrays and request bodies where admitted, with longer model-native windows separately labelled.
Keep `--prompt-cache-gib 0` controls and fresh KV state for cold measurements; a warm prefix is not a cold-prefill improvement.
Alternate baseline/candidate runs for small differences, retain all attempts and explain interruptions or drift.
Do not call a change in output tokens or quantization an isolated kernel improvement.
Independent microbenchmarks supplement dependent operations and whole served requests; they cannot replace them.

Include peak physical process footprint and device/pinned allocations where available, plus the actual RAM class and context.
A configured budget on a bigger machine is not proof of fit on a smaller machine.
Use the existing GPU lock and admission guard, with one model process per test machine.
Do not stop somebody else's server, raise shared memory limits, change drivers or expose a listener merely to complete a receipt.
Keep receipts local unless upload has been explicitly authorized.

## Decide readiness without changing the gate

Follow the existing [PR template](../../.github/pull_request_template.md).
State the changed behavior, issue/gate, owning modules, commands, exit statuses, pass/fail/skip counts, exactness results, kernel summaries, before/after rates, cold-prefill impact and unrun platforms.
Update the relevant recipe or capability documentation only as far as the new evidence supports it.
Write the code for TensorFold rather than copying source, kernels or tables from other inference engines or projects.
If something has to come from elsewhere, name its source and license in the PR and wait for the maintainer's decision.
Preserve contributor authorship, required licenses and third-party notices; exclude private paths, hosts, credentials and confidential data from source, fixtures and public receipts.
Commit under the contributor's GitHub noreply address with no AI-tool co-author or "Generated with" lines; a PR whose commits carry them lands as new commits credited with Co-authored-by and shows as closed, not merged.

A contribution is ready for the claimed scope only when its required checks and agreed performance gates are met.
A result just below a minimum is still below it; a skip or unavailable device is not a pass.
Unqualified support can be presented as a draft or limited experiment, with the remaining gates explicit.
Report blockers and the smallest next check instead of claiming completion or weakening the test.
Do not merge, publish, assign a release version or deploy as a side effect of using this skill.
