# Native configuration profiles

`tensorfold serve MODEL --profile stable` explicitly selects today's defaults. Omitting `--profile` selects
the same profile. It enables no additional optimization and supplies no new environment values.

This is the foundation of [#1660](https://github.com/ashhart/TensorFold/issues/1660), beginning with the CUDA KV
switches in its first table and their `_ALL` companions (seven entries). It does not migrate every `TF_*` or
`TENSORFOLD_*` variable yet. Other settings continue to work through their existing readers. `fast-exact` and `fast`
are refused by name; they are not aliases
for `stable` and cannot enable unqualified optimizations.

## Explicit environment values win

Registered readers use the shared [core registry and resolver](../zig/src/core/profile.zig). An explicitly set
environment value wins byte for byte, including `0`, an empty string, or a value the family will reject. The
resolver does not coerce values, mutate the environment, or replace the family's parser and validation errors.
With a variable absent, `stable` leaves it absent so the reader keeps its existing default.

The following command has exactly the same engine options as the command without `--profile stable`:

```sh
TF_QWEN_KV_PAGES=256 TF_QWEN_KV_SHARE=16 tensorfold serve MODEL --profile stable
```

Use these variables only with a checkpoint/backend that supports them, following its recipe and cache admission
requirements. The profile does not waive memory, page-size, tensor-parallel, drafter, or learning restrictions.

## Inspect the registry

```sh
tensorfold capabilities --json
```

The `profiles` object contains `default`, `available`, and `registered_switches`. Every row identifies its switch,
backend, model types, class, effect, evidence link number, syntax, and stable fallback. A null fallback means the
profile supplies no value. Registry metadata comes directly from the same table the readers use.

Initial coverage is:

- Qwen3.5 dense/MoE on CUDA: `TF_QWEN_KV_PAGES`, `TF_QWEN_KV_PAGES_ALL`, `TF_QWEN_KV_SHARE`.
- GLM on CUDA: `TF_GLM_KV_SHARE`.
- Gemma 4, Kolibri-1, and Spark2.5 on the core CUDA page adapter: `TF_KV_SHARE`, where the adapter is served.
- Flash Next on CUDA: `TF_FLASHNEXT_KV_PAGES`, `TF_FLASHNEXT_KV_PAGES_ALL`.

`identical` records the existing switch's token-preserving design and linked evidence; it is not a claim that the
switch is faster for every shape, device, or workload. The `*_ALL` checks stay `debug`; Flash Next paging remains
`experimental`. No class turns a switch on in this increment. Existing default-on paging outside this registry is
unchanged.

## Startup reporting

The serving process prints its profile and applicable registered overrides, for example:

```text
[tensorfold] profile: stable; registered overrides: TF_QWEN_KV_PAGES=256 (identical, env); TF_QWEN_KV_SHARE=16 (identical, env)
```

When no applicable registered setting differs from its default, it prints `registered overrides: none`. This is
explicitly the registry's coverage, not an assertion that the whole environment is at defaults. Settings for another
model type/backend and arbitrary environment variables are omitted. Explicit numeric zero and disabled exact-one
flags do not count as changes. Dependency checks still determine whether a configured switch can be used.

Values are bounded and terminal controls are escaped. Credentials, paths, server addresses, and unrelated environment
variables are never enumerated by this report. Each process's environment remains authoritative on restart; idle
reloads use the same existing family readers.

## Further profile work

Migrate additional switch groups with their existing parsers and tests intact. A future `fast-exact` profile needs
the issue's same-checkpoint token gates: solo, concurrent 1/2/4/8, and resumed runs for every affected family. An
arithmetic-changing `fast` profile additionally needs each switch's measured quality gate and maintainer agreement.
The host tests for this foundation establish lookup/CLI/reporting behavior, not those GPU or checkpoint gates.
