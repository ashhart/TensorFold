# GLM repetition: source-specific qualification update

This receipt supersedes the historical unrun-check statements in
[the implementation recipe](glm-repetition-penalty.md). Each result belongs to the named source.

## Stacked runtime implementation

Measured implementation: `c9d06d2b1ffdc1d4323bc62232df9a3afd2cf0e6`, on public parent
`0029d0e822f73363cc9e0f774a3e52837bb7a4de` ([#3019](https://github.com/ashhart/TensorFold/pull/3019)).
The current review revision changes documentation only.

`zig build test test-kernel-contract test-golden fuzz-cortheon` passed 724/724 steps,
9,425 tests passed, 69 skipped and zero failures. Strict HTTP goldens were 322/335 equal plus
13 inherited known differences; security was 55/55 equal, with zero unexpected differences.
Focused repetition/API/native-open/prefill checks passed 58/58 tests and 14/14 steps.
The same lean checker reported 345 inherited findings on the stacked base and implementation, zero new.

Registered repetition contracts executed on sm120 and sm121 with Zig 0.17.0 and nvcc 13.3.73.
Each device exited 0 with:

```text
SUMMARY repetition: 3 kernels, 3 pass gates 1-4, 0 fail, 0 known; 10 controls, 10 red, 0 missed; 0 limit problems
```

Those results cover the registered greedy-token, sampled-token and FP32-probability operations.
The parent/candidate whole-model attempt stopped before inference at the parent's explicit `Fp8OnPages`
refusal. Zero parent, candidate or shared-audit model requests ran in that attempt.
The separate cache compatibility change is tracked in [#3062](https://github.com/ashhart/TensorFold/issues/3062).

Current-main full-model fidelity, comparative decode/cold-prefill, Metal, HIP and other CUDA device executions
remain unqualified by these results. Nvcc 13.0 and 13.2 resource records remain pending.
The original recipe retains the synthetic reference, source hashes and reproduction commands.

## Qualified release 1.2.7 integration

The public [qualified source snapshot](https://github.com/gilp456/TensorFold/tree/cd3cf2eaa300fdea74736ebc310f6c9b21e0f275)
is based on release 1.2.7 and combines the repetition repair with paged-FP8 compatibility.
Its source is `cd3cf2eaa300fdea74736ebc310f6c9b21e0f275`, distinct from the main-based runtime contribution.

Three GB10 GPUs passed the FP8 shuffled-page numerical reference, ten semantic acceptance cases,
four natural reasoning tasks and eighteen automatic-tool serial/draft/concurrent comparisons.
The longest replay used 335,469 prompt tokens and completed 1,267 tokens with a valid structured tool call
and no repetition stop. Configured 1,048,576-token capacity passed admission; full-window inference was unrun.
Omitted and explicit 1.1 factors produced identical 256-token prefixes. That bounded check does not establish
completed-answer quality or traced effective-factor identity. Generated tools remained inert.

The corrected model's public [1.2.7 instructions](https://huggingface.co/gilp456/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit-Repaired/blob/v0.1.0-rc3/NATIVE-1.2.7.md)
and [sanitized results](https://huggingface.co/gilp456/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit-Repaired/blob/v0.1.0-rc3/native127-results.json)
pin the source, weights, drafter, precision and existing automatic-tool grammar setting.
Private prompts, outputs, token arrays, hashes and infrastructure identities are excluded.
Broader capability, explicit shared-round instrumentation, comparative performance and universal loop freedom
remain outside these bounded regression results.
