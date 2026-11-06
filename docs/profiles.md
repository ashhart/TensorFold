# Native configuration profiles

`tensorfold serve MODEL --profile stable` explicitly selects today's defaults. Omitting `--profile` selects
the same profile. It enables no additional optimization and supplies no new environment values.

This is the foundation of [#1660](https://github.com/ashhart/TensorFold/issues/1660), beginning with the CUDA KV
switches in its first table, their `_ALL` companions, and GLM's and Nemotron's paging policies (seven variables).
It does not migrate every `TF_*` or `TENSORFOLD_*` variable yet. Other settings continue through their existing readers. `fast-exact` and `fast`
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

The top-level `env` array lists selected documented variables. It is not a complete inventory of runtime environment
readers. `profiles.registered_switches` covers the profile registry in this build, including variables absent from
`env`; it is also not an inventory of every family-specific runtime setting. Consult each family's recipe for those.

The `profiles` object contains `default`, `available`, and `registered_switches`. Every row identifies its switch,
backend, model types, class, effect, evidence link number, syntax, and stable fallback. A null fallback means the
profile supplies no value. Registry metadata comes directly from the same table the readers use.

Initial coverage is:

- Qwen3.5 dense/MoE on CUDA: `TF_QWEN_KV_PAGES`, `TF_QWEN_KV_PAGES_ALL`, `TF_QWEN_KV_SHARE`.
- GLM on CUDA: `TF_KV_PAGES`, `TF_GLM_KV_HOST`, `TF_GLM_KV_HOST_SHARED`, `TF_GLM_PROMPT_TAIL`.
- Nemotron and Nemotron Omni on CUDA: `TF_KV_PAGES`, with their existing default-on parser.
- Gemma 4, Kolibri-1, Spark2.5, Dory and GLM on the core CUDA page adapter: `TF_KV_SHARE`, where the adapter is served.
- Flash Next on CUDA: `TF_FLASHNEXT_KV_PAGES`, `TF_FLASHNEXT_KV_PAGES_ALL`.

`identical` records the existing switch's token-preserving design and linked evidence; it is not a claim that the
switch is faster for every shape, device, or workload. The `*_ALL` checks stay `debug`; Flash Next paging remains
`experimental`. No class turns a switch on. GLM's existing automatic paging default is preserved; other default-on
paging outside this registry is unchanged. The two GLM host-memory switches carry [#1795](https://github.com/ashhart/TensorFold/pull/1795)'s
identical tokens at TP=2 on two GB10s, and `TF_GLM_PROMPT_TAIL` carries [#2226](https://github.com/ashhart/TensorFold/pull/2226)'s.
`TF_GLM_KV_HOST_GIB` stays outside the registry: it sizes those pages rather
than choosing a policy, and its reader keeps its own default.

A reader that resolves its switch by name uses `core.profile.lookup`: a registered name comes from this registry, and
every other name keeps the libc lookup and default it had. GLM's rank settings read that way, and their fingerprint
still carries each resolved value, so ranks that disagree still stop before weights load.

## Startup reporting

The serving process prints its profile and applicable registered overrides, for example:

```text
[tensorfold] profile: stable; registered overrides: TF_QWEN_KV_PAGES=256 (identical, env); TF_QWEN_KV_SHARE=16 (identical, env)
```

When no applicable registered setting differs from its default, it prints `registered overrides: none`. This is
explicitly the registry's coverage, not an assertion that the whole environment is at defaults. Settings for another
model type/backend and arbitrary environment variables are omitted. Numeric zero for count switches and disabled exact-one
flags do not count as changes. Dependency checks still determine whether a configured switch can be used.

For GLM CUDA, `TF_KV_PAGES` uses the registry's `auto` syntax: unset or exactly `auto` leaves the family's automatic choice,
while every other explicit value is a policy override, including `off` and `0`. An explicit policy may select the
same storage as automatic mode at a particular rank count. The GLM selection line below reports that distinction;
the generic registry line does not infer rank count or precision from the environment.
The registry has separate CUDA rows for GLM and Nemotron's same-named setting, with disjoint family scopes.
`capabilities --json` lists both rows under `registered_switches`; select a row by `id`, family and backend rather
than assuming the environment name alone identifies its policy. Lookup remains shared and returns the raw value.
An optional `report_values` list takes precedence over `syntax` for reporting exact overrides; null retains the
existing syntax rule. Nemotron's row lists `0` and `all`, while GLM's row keeps its `auto` rule.

For Nemotron (`nemotron_h` and `nemotron_h_omni`), paging is on by default. Exactly `TF_KV_PAGES=0` disables it;
exactly `TF_KV_PAGES=all` puts every stream on pages for checks. Those two values appear as registered overrides.
Every other value, including empty text, `on`, `off`, `auto`, `00`, uppercase spellings and whitespace, keeps
Nemotron's existing `on` policy and adds no override to the line. In particular, use `0`, not `off`, to disable
Nemotron paging. These are its existing parser rules, not new aliases or defaults. The later load/admission path
can still select windows for learning or a Triton-backed engine; this line reports the environment policy.

Values are bounded and terminal controls are escaped. Credentials, paths, server addresses, and unrelated environment
variables are never enumerated by this report. Each process's environment remains authoritative on restart; idle
reloads use the same existing family readers.

## GLM CUDA paging selection

GLM also logs its resolved KV storage selection on each rank, after connecting rank liveness and resolving settings,
before engine construction. For example, with `TF_KV_PAGES` unset and two BF16 ranks, rank zero reports:

```text
glm kv selection: pool pages; TF_KV_PAGES=auto (default); rank 0 of 2; cache bf16
```

The `default` or `env` label distinguishes an absent variable from an explicit policy. The selected boolean is the
same one passed to engine construction. This is a configuration report; the later load and memory-admission checks
can still refuse startup. The line is printed again when an idle engine reloads.

| `TF_KV_PAGES` | BF16, one rank | BF16, two or more ranks | `--kv-cache fp8`, any rank count |
| --- | --- | --- | --- |
| Unset or exactly `auto` | Window per stream | Pool pages | Window per stream |
| Exactly `off` or `0` | Window per stream | Window per stream | Window per stream |
| Every other value | Pool pages | Pool pages | Refused with `Fp8OnPages` |

These are the existing parser rules, including empty text, `all`, `false`, uppercase spellings and whitespace counting
as `on`. Prefer `auto`, `off` or `on` explicitly. The family line normalizes these choices and never echoes arbitrary
input; the registry line shows bounded, escaped raw overrides. FP8 refusals keep their existing error and peer
notification behavior, and do not print a resolved selection. No paging or FP8 qualification is added by this report;
the existing BF16 paging evidence is [#1322](https://github.com/ashhart/TensorFold/pull/1322).

## Further profile work

Migrate additional switch groups with their existing parsers and tests intact. A future `fast-exact` profile needs
the issue's same-checkpoint token gates: solo, concurrent 1/2/4/8, and resumed runs for every affected family. An
arithmetic-changing `fast` profile additionally needs each switch's measured quality gate and maintainer agreement.
The host tests for this foundation establish lookup/CLI/reporting behavior, not those GPU or checkpoint gates.
