# GLM full-prefix repetition penalty

The native GLM CUDA engine applies `repetition_penalty` to target logits for greedy and sampled requests.
This addresses [#3020](https://github.com/ashhart/TensorFold/issues/3020).
The request value overrides `generation_config.json`; the default is `1`.
The factor must be positive, finite and representable as a positive finite float32 value.
Engines without the capability refuse a nonidentity factor before generation.

```json
{"model":"MODEL","prompt":"Example prompt","max_tokens":16,"temperature":0,"repetition_penalty":1.1}
```

Chat, completions and Responses share the same validation and engine request field.
An explicit factor of `1` overrides a model default. A greedy request retains the model's repetition default.
Presence and frequency penalties keep their separate capability admission.

For a token ID present anywhere in the complete prompt and authoritative committed prefix, a negative logit
is multiplied by the factor and a nonnegative logit is divided by it. Each distinct ID is processed once.
The BF16 input logit is converted to float32, transformed in float32, and consumed by selection before
temperature, min-p, top-k and top-p. The processed value is never rounded back to BF16.

Each target verification row additionally includes its pending token and earlier draft tokens in that window.
Future and rejected drafts never enter committed membership. Request-owned history is rebuilt from actual
token IDs after prefix replacement or shortening, including reuse of an existing KV prefix.
Grammar redraws use the same processor. MTP proposal draws retain their existing behavior.

The identity-only path keeps the original sampler ABI, random keys and lazy history/membership allocation.
Active operations carry a bounded table of at most 32 rows, with independent slot, position, membership and factor.
TP ranks validate and prepare the table before voting readiness and transferring masks.
A refusal clears the table and leaves followers receiving the next command.
This change adds no precision policy, checkpoint conversion or model-specific prompt text.

## Synthetic reference

The ten explicit numeric cases and their CPU regeneration command are documented in
[the fixture README](../../tools/zig/fixtures/README.md#synthetic-repetition-penalty-logits).
They call the actual Transformers 5.20.1 `RepetitionPenaltyLogitsProcessor` with float32 CPU tensors.
They contain no model weights, tokenizer output or conversation data.
The header SHA256 is `9388ddb013232071f6290e43a2d91fa1772f539cf7c0d5830e63115f7c5a9dd3`.

## CPU checks

The initial port was checked against public main `6c5cef8941b27d86b06a986834e13deb9b515e1e`.
Zig 0.17.0 on Linux x86_64 ran these focused checks:

```sh
zig build test-glm-repetition-host test-repetition-api-host compile-glm-native-open
zig build test-cuda-prefill-widths-host
```

The first command passed 48 tests: 22 GLM state/transport tests and 26 API/default/parsing tests.
It also compiled the actual native GLM open path with inert CUDA images.
The existing prefill-width target passed all 10 tests after its recording fixture gained identity-processor hooks.
The standalone generation-default module passed all five tests.
Its greedy-default regression first failed on the base because `repetition_penalty` was omitted, then passed.
The default `zig build` completed all 33 build steps.
The host kernel-contract tests passed 240/240, and Cortheon fuzz checks passed 23/23.
Server goldens passed with 322/335 equal, the same 13 inherited known differences as the clean base,
and zero unexpected differences. All 55 security goldens passed.
The 17 model-metadata snapshots add only the new false capability and the corresponding Content-Length.
The combined `zig build test test-kernel-contract test-golden fuzz-cortheon` command completed all 728 build steps:
9,388 tests passed and 69 skipped, with zero failures.

The review branch is stacked on [#3019](https://github.com/ashhart/TensorFold/pull/3019),
at `0029d0e822f73363cc9e0f774a3e52837bb7a4de`.
Its model-metadata goldens retain the parent's `input_types` fields alongside this repair's capability.
The focused repetition/API, prefill and native-open checks were repeated on that stacked tree:
58/58 tests and all 14 build steps passed.
Strict goldens also passed again: 322/335 equal with 13 inherited known differences, 55/55 security and zero unexpected.
The broader 728-step result above covers the initial public-main port; a full stacked-tree host run is unrun.

The same lean checker reports 345 inherited findings on the base and on this port, with zero new findings.
Inherited findings are retained in that count.

## Device qualification

Device execution of this current-main port is unrun. No current-port GPU, multi-rank, model-fidelity,
performance, Metal or HIP qualification is claimed by the CPU results.
The nvcc 13.3 launch records were measured before this port; nvcc 13.0 and 13.2 records remain pending.
The port's `sample.cu` and `sample_repetition.cuh` are byte-identical to those previously measured kernel sources:

| Source | SHA256 |
| --- | --- |
| `sample.cu` | `746a8bd38d2bb702aaee83b7d06d4ebdeed4a5644e93c743ea6675e11aa052be` |
| `sample_repetition.cuh` | `f755e90362f4ddb1b70f85e9252539a3a82270279454a492c6e6fb4b69953691` |

The bounded contract runner builds independently of a checkpoint. On an available, exclusively owned GPU,
the following commands build the images and run the synthetic gates and planted controls:

```sh
zig build fatbins kernel-contract-fatbins -Dnvcc=NVCC -Dsm=80,86,89,120,121
zig build sample-repetition-contract -Dfatbins=zig-out/fatbin
```

Record the complete `SUMMARY repetition:` line, exit status, GPU, compiler and source revision for a new run.
The complete CUDA contract also registers these cases through `tf-cuda-test kernel-contract`.
Same-engine serial/speculative token equality and independent model fidelity require their own device checks.
