# Experimental dense BF16 Qwen3.5 scoring on CUDA

This proposal adds dense BF16 loading and selected-token scoring to the existing Qwen3.5 CUDA family.
The intended initial geometry is Qwen3.5 27B. It is an engineering candidate, not a qualified checkpoint or release.
The arithmetic proposal and acceptance gates are tracked in [issue #1136](https://github.com/ashhart/TensorFold/issues/1136).
No public checkpoint revision has been qualified by this contribution, and no checkpoint performance claim is made.

## Scope and arithmetic

The loader accepts an unquantized dense configuration declaring `bfloat16`. It rejects unquantized MoE and other
activation dtypes. It reads HF `model.language_model.*` tensors, a tied embedding or separate `lm_head.weight`, and
`[channels, 1, taps]` convolution weights. Norm weights retain the checkpoint's offset representation: the CUDA
RMS and head-norm kernels add one in FP32 rather than rounding the transformed multiplier back to BF16 at load.
Gated DeltaNet's post-normalization weights retain their existing interpretation.

Dense projections use cuBLASLt with FP32 accumulation and BF16 output. Selected-token scoring gathers the final
normalized row of each independent segment, projects those rows to FP32 vocabulary logits, and reduces the entire
vocabulary to log-sum-exp and a deterministic argmax. Equal finite maxima select the lower token ID. Up to 16
selected logits per segment are returned; the normalizer is not restricted to those selected tokens.
The new reduction needs the kernel-contract registration, non-finite policy and device gates before acceptance.
Its finite-input reference utility below does not replace those gates or projection-class FP64 references.

## Development runner

Build with exactly the repository's `.zig-version` toolchain and a supported CUDA compiler:

```sh
zig build qwen35-score -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=120
zig build test-qwen35-bf16-host
zig-out/bin/tf-qwen35-score /models/qwen35-bf16 0
```

A complete, matching catalog of prebuilt fatbins can instead be supplied with `-Dfatbins=/opt/tensorfold/fatbins`.
Rebuild `qwen35.fatbin` from this source; the previous release image lacks the scoring symbol.
These paths are illustrative. Keep weights outside the checkout and observe the existing GPU admission guard.
The executable is a Linux JSONL development tool; it is not the native HTTP server or another serving scheduler.

Each newline-terminated input contains one to four independent, already-tokenized requests:

```json
{"requests":[{"input_ids":[1,2,3],"labels":[4,5]}]}
```

The IDs above illustrate the wire schema, not a model prompt fixture. After a readiness line, each batch returns
`results` containing the greedy `token`, selected full-vocabulary-normalized `logprobs` in label order and
`cached_tokens`, plus batch `engine_ms`. This time includes prefix work and device synchronization, but excludes
JSON parsing, output serialization, tokenization, network transport and external queueing.

The runner allocates four 65,536-token work caches. Prompts are limited to 65,530 tokens; that admission limit
is not a long-context qualification. Each operation resets its recurrent state and starts at token zero;
`cached_tokens` is always zero. Every prompt token and the vocabulary head execute on every operation.
Batches use at most 128 rows per forward. Prefix reuse belongs to the native shared cache and is deliberately
absent from this tool; there is no second prefix store or hand-written state-restore path.
Invalid input terminates the runner; cancellation, reset recovery, sampling, streaming and vision are unqualified.
The runner owns one device and does not implement tensor parallelism or a multi-device dispatcher.
The native server explicitly refuses this experimental BF16 format before loading weights; its Qwen CUDA
capability metadata remains unchanged. The development runner is the only proposed BF16 scoring entry point.

## Checks and receipt

Port base: `d1018b318fb51981c4a18f12e40269a4d3ee9713`, upstream `main`.
The following Linux x86_64 checks used Zig 0.17.0 and nvcc 13.3.73 for the changed CUDA image. They do not load a checkpoint:

```sh
zig build test test-golden test-cuda-host test-kernel-contract -j4 --summary all
zig build fuzz-cortheon -j4 --summary all
zig build -j4 --summary all
zig build qwen35-score test-qwen35-bf16-host -Doptimize=ReleaseFast -j4 --summary all
python3 tools/zig/lean_check.py
```

- Host/golden/limit suites: 227/227 build steps succeeded; 3,024 tests passed, 14 skipped, zero failed.
- Server corpus: 306/319 equal with 13 recorded known differences; separate corpus 26/26 equal;
  zero unexpected differences.
- A baseline-only BF16 configuration admission probe failed (0/1 passed); the candidate admits that same fixture.
- Final incremental recheck, including targeted tests: 231/231 steps succeeded; 219 tests executed and passed,
  three skipped, with other unchanged checks cached. Zero failures.
- Plain build: exit 0, 24/24 build steps succeeded. Fuzz: exit 0, 3/3 steps and 11/11 tests passed.
- Targeted admission, graph identity and label-range regressions: 3/3 passed, zero failed or skipped.
- Native scorer compiled both host-only and with real embedded CUDA images. The changed Qwen image was rebuilt
  with `nvcc -fatbin -arch=sm_120 -O3 -std=c++17`; other catalog images were retained assets, not rebuilt here.
  Compilation is not a device execution result. No generated binaries or kernel assets are contributed.
- Lean checker: 444 findings on the base and candidate, zero new findings; the absolute checker exit remains nonzero.
- Python reference utility: syntax passed; execution skipped without PyTorch. GPU execution is not claimed.

The finite-input diagnostic below uses synthetic logits only, covers vocabulary tails and ties, and reports each
row's normalizer error against FP64. It requires PyTorch and a free CUDA device; absent dependencies skip cleanly:

```sh
python3 tools/zig/check_qwen35_scores.py /opt/tensorfold/fatbins/qwen35.fatbin
```

An earlier prototype had an unresolved external-reference probability tolerance failure. Agreement on greedy
choices does not establish probability or logit equivalence. Those results are not promoted to a public checkpoint
receipt, and the current port has no claim of checkpoint-level numerical parity.
Before acceptance, use a pinned public checkpoint and public token fixtures to run teacher-forced fidelity,
per-projection FP64 error, prompt/decode seams, resumed versus fresh, batched versus solo and lifecycle checks.
Register the new kernel and post the device contract SUMMARY lines. Run native server decode and cold-prefill
receipts at the required contexts, with cache reuse disclosed separately. Metal, other CUDA SMs, multi-device
execution and checkpoint/device tests remain not run for this port.

The contributor-audit revision removes the separate prefix cache and adds a native BF16 refusal regression.
Its full required host baseline plus scorer/targeted checks exited 0: 237/237 steps succeeded;
304 executed tests passed, three skipped, and remaining unchanged tests were cached. The targeted host suite
passed 4/4 tests. This is an incremental receipt, not 304 newly added tests. All 15 changed native source
files match the isolated compiled tree by SHA-256. Formatting and `git diff --check` passed; lean remains
444 inherited findings with zero new.

## Contributor-skill audit and acceptance work

Audited against [the contributor skill](SKILL.md) at the port base above. This is a limited experiment,
not a fully qualified engine contribution. Removing the development runner's separate prefix store and
state-copy routine keeps prompt-cache ownership with the native lane/cache architecture. The runner is still
an operator bring-up tool; its batches do not qualify native server scheduling, drafting or lifecycle behavior.
No prior warm-prefix performance result applies to this fresh-state runner.

| Requirement | State | Evidence or smallest next action |
| --- | --- | --- |
| Target, ownership and publication hygiene | passed | One branch/PR against `main`; original changes; public noreply identity; no private fixtures or generated assets. |
| Acceptance gate | blocked | Issue #1136 proposes gates; no maintainer agreement is recorded. Maintainer must agree workload, tolerances and performance minimum. |
| Public checkpoint identity | not-run | Select a public checkpoint/tokenizer revision and public token fixtures after gate agreement; no public checkpoint run is claimed. |
| Host baseline and plain build | passed | Commands and counts above; exact Zig 0.17.0. Host limit checks cover the existing registry only. |
| Incremental style | passed | 444 inherited lean findings, zero new; preserve the nonzero absolute checker result. |
| Windows CI | failed | At `db919c6`, run 39530409157, job 118994749182: 165/211 steps succeeded; 1,423 tests passed, 15 skipped, 22 build steps failed. |
| Windows baseline reproduction | not-run | All 29 repository files with primary compiler errors are byte-identical between base and candidate. This is not a baseline Windows reproduction; toolchain/Windows compatibility still needs a separate diagnosis. |
| Kernel contracts | not-run | Register score reduction and changed norm/head-norm operators in the CUDA runner and launch registry; declare non-finite/alias/depth semantics and derive real compiler records. Run all five gates and planted controls, retaining complete SUMMARY and exit status. |
| Native model integration | not-run | Scoring has no native server API integration. An explicit admission guard refuses experimental BF16 before weight loading; a host regression covers the refusal and its explanation. Current Qwen CUDA format metadata remains `mlx-q4g64`. Image/vision refusal and model/device lifecycle checks still need qualification. |
| Native binary introspection | passed | `zig build native` with real embedded catalog; `tensorfold-native 1.1.1`, `--help` and `capabilities --json` exit 0; Qwen CUDA advertises only `mlx-q4g64`. This is not a model run. |
| Independent arithmetic | not-run | FP64 references for all changed projection classes and prompt/decode rows; worst columns, teacher-forced margins and resolution of the prior probability tolerance failure. |
| Engine exactness and lifecycle | not-run | Native drafted/plain, concurrent/solo, resumed/fresh, seam, rejection, cancellation/reset and image refusal receipts on a pinned public checkpoint. |
| Served performance and memory | not-run | Existing native benchmark clients; decode, concurrency, cold 2k/8k/32k/64k with cache disabled, full tokens, physical process/device/pinned allocation receipts. |
| GPU/backend matrix | not-run | Changed image compiled for sm_120 only. Device execution, full multi-SM compile/limits, Metal, HIP/XPU and multi-device support are not qualified. |
| Packaging | not applicable | No binary, archive or generated kernel asset is contributed; distribution manifests and release versions are unchanged. |

The unresolved acceptance gate precedes additional arithmetic work. Keep this PR a draft. After agreement,
finish contract-runner registration and native adapter ownership before spending a model/device qualification run.
Use the existing admission/lock procedures and an isolated available test machine; an unrelated running service
must not be stopped to obtain a receipt. No missing, skipped or failed cell above is a pass or a waiver.
