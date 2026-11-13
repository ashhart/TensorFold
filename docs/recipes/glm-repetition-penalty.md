# GLM full-prefix repetition penalty

The native GLM CUDA engine accepts `repetition_penalty` on chat, completions, Messages and Responses requests.
The field is request-only: omission and `null` use identity factor `1`; checkpoint and server sampling defaults never activate it.
The factor must be positive, finite and representable as a positive finite float32 value.
An engine without this capability refuses an effective nonidentity float32 factor before generation.

```json
{"model":"MODEL","prompt":"Example prompt","max_tokens":16,"temperature":0,"repetition_penalty":1.1}
```

For each target row, membership includes every distinct token ID in the authoritative committed prefix plus that row's accepted ancestry.
Rejected and future drafts remain outside the set.
The processor widens each BF16 logit to float32, multiplies negative values by the factor and divides nonnegative values by it.
Temperature, min-p, top-k and top-p consume the transformed float32 value.
Grammar redraws use the same processor state.

The identity sampler entry points keep their existing ABI and source path.
Active requests use separate `tf_draw_repetition` and `tf_greedy_repetition` entry points with request-owned membership and per-row factors.
Tensor-parallel ranks validate the row table, prepare their masks and vote readiness before the leader transfers membership.
A refusal clears the table and leaves each follower ready for the next command.

## Host checks

Use Zig 0.17.0:

```sh
zig build test-repetition-api-host test-glm-repetition-host compile-glm-native-open
zig build test-kernel-contract
```

These checks cover API validation and propagation, request-only defaults, full-prefix state, row ancestry, malformed tables, follower refusal/retry and host contract registration.

## Device qualification

Run the registered kernel contract on an exclusively owned CUDA device, then audit the installed images for every served SASS target:

```sh
zig build kernel-contract -Dnvcc=NVCC
zig build fatbins kernel-contract-fatbins -Dnvcc=NVCC -Dsm=80,86,89,120,121
tf-cuda-test kernel-contract-limits zig-out/fatbin
```

Record the revision, CUDA compiler, device, commands, exit statuses, complete `SUMMARY` line and every repetition `RECORD` line.
Compare the common legacy sampler symbols against the parent with `cuobjdump -sass` for every served target; intentional repetition-only symbols have their own numerical and resource contracts.
Full-model serial/speculative, fresh/resumed prompt, grammar/tool, concurrency and refusal/recovery gates require the final review revision on its target topology.
