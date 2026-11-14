# Native configuration profiles

`tensorfold serve MODEL --profile stable` explicitly selects today's defaults. Omitting `--profile` selects
the same profile. It enables no additional optimization and supplies no new environment values.

This is the foundation of [#1660](https://github.com/ashhart/TensorFold/issues/1660), beginning with the CUDA KV
switches in its first table, their `_ALL` companions, the shared prompt store, GLM's and Nemotron's paging policies,
the Dory, Gemma 4 and Spark2.5 pool-page switches, and Nemotron's paging on Metal. The registry also covers grouped learning modes.
It does not migrate every `TF_*` or `TENSORFOLD_*` variable yet. Other settings keep their existing readers. `fast-exact` is
refused by name pending its token gates; `fast` is not planned: a speed mode that changes output tokens is not shipped.

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
profile supplies no value. Registry metadata comes directly from the table. Registered readers can use its raw-value resolver; adding metadata does not migrate an existing reader.

Coverage includes:

- Qwen3.5 dense/MoE on CUDA: `TF_QWEN_KV_PAGES`, `TF_QWEN_KV_PAGES_ALL`, `TF_QWEN_KV_SHARE`.
- GLM on CUDA: `TF_KV_PAGES`, `TF_GLM_KV_HOST`, `TF_GLM_KV_HOST_SHARED`, `TF_GLM_PROMPT_TAIL`, and the exact 0/1
  switches `TF_GLM_HOST_CACHE`, `TF_GLM_VISION_HOST` (off unless `1`) and `TF_GLM_PDL`, `TF_GLM_PROMPT_STEPS`,
  `TF_GLM_SHARED` (on unless `0`).
- Nemotron and Nemotron Omni on CUDA: `TF_KV_PAGES`, with their existing default-on parser.
- Gemma 4, Kolibri-1, Spark2.5, Dory and GLM on the core CUDA page adapter: `TF_KV_SHARE`, where the adapter is served.
  Dory's `TF_KV_SHARE` also puts its full-attention caches in pool pages, as `TENSORFOLD_DORY_KV_PAGES=1` does;
  with `TENSORFOLD_DORY_KV_PAGES=0` it refuses to load, by both names. With the switch unset, the startup line
  shows `TENSORFOLD_DORY_KV_PAGES=1 (arithmetic, TF_KV_SHARE)` and kept states carry the same identity as `=1`.
- Every CUDA family whose host prompt cache the CUDA host builds (Nemotron, Nemotron Omni, GLM, MiniMax M3, Qwen3.5
  dense/MoE, Flash Next, Kolibri-1, Gemma 4, Spark2.5, Dory): `TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS`, exact `1` only.
- Flash Next on CUDA: `TF_FLASHNEXT_KV_PAGES`, `TF_FLASHNEXT_KV_PAGES_ALL`.
- Dory on CUDA: `TENSORFOLD_DORY_KV_PAGES`, full-attention caches in pool pages unless it is exactly `0`; unset,
  every stream's caches are contiguous and made at open.
- Gemma 4 on CUDA: `TENSORFOLD_GEMMA4_KV_PAGES`, global caches in pool pages by default, where `0` keeps them linear.
- Spark2.5 on CUDA: `TENSORFOLD_SPARK25_KV_PAGES`, full caches in pool pages by default, where `0` keeps them linear.
- Nemotron-H on Metal: `TF_NEMOTRON_METAL_KV_PAGES`, where exactly `1` puts keys
  and values in the core pool's pages.
- DeepSeek V4.1 on CUDA: the settings below, with no stable fallback.

`identical` records the existing switch's token-preserving design and linked evidence; it is not a claim that the
switch is faster for every shape, device, or workload. The `*_ALL` checks stay `debug`; Flash Next paging remains
`experimental`. No class turns a switch on. GLM's existing automatic paging default is preserved; Gemma 4's and
Spark2.5's default-on paging is registered here, where `0` keeps the linear caches; other default-on paging outside
this registry is unchanged. The two GLM host-memory switches carry
[#1795](https://github.com/ashhart/TensorFold/pull/1795)'s identical tokens at TP=2 on two GB10s, and
`TF_GLM_PROMPT_TAIL` carries [#2226](https://github.com/ashhart/TensorFold/pull/2226)'s. `TF_GLM_KV_HOST_GIB` stays
outside the registry: it sizes those pages rather than choosing a policy, and its reader keeps its own default.

`TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS=1` ([#2346](https://github.com/ashhart/TensorFold/issues/2346)) changes which
prompt states are kept, never a token. A new state may then evict only its own conversation's states: kept states
whose tokens begin the new prompt, that no lookup borrows and that aren't a shared cut or a branch point. A state
that fits only by evicting another conversation's isn't kept: it counts as refused, one line names its size and the
budget no kept state holds, and every kept state stays. The lookup that makes room before a pass, and learned and
disk states read back, follow the same rule. Memory-pressure shedding and a new stream's memory still take any
state. A conversation is known only by its tokens, so one whose history is rewritten (compacted, trimmed to fit
`--context`, or edited and sent again) no longer owns its older states, and with the switch on those can keep its
new states out until memory pressure sheds them. Unset, empty or `0` keeps today's eviction order. Only the CUDA host's prompt cache reads the switch; the
shared paged store (`TF_KV_SHARE`) and the Metal, HIP and XPU caches don't, and a Metal server ignores it.

A reader that resolves its switch by name uses `core.profile.lookup`: a registered name comes from this registry, and
every other name keeps the libc lookup and default it had. GLM's rank settings read that way, and their fingerprint
still carries each resolved value, so ranks that disagree still stop before weights load.

## Additional arithmetic and learning settings

[#2370](https://github.com/ashhart/TensorFold/issues/2370) adds metadata without adding defaults:

| Registry id | Backend and family | Class | Explicit values reported as overrides |
| --- | --- | --- | --- |
| `TENSORFOLD_KOLIBRI_KV_PAGES` | CUDA, Kolibri-1 | arithmetic | Exact `0`, which disables otherwise requested pages |
| `TENSORFOLD_DORY_KV_PAGES` | CUDA, Dory | arithmetic | Every present value except exact `0`, including empty text |
| `GEMMA_DRAFT_BF16` | Metal, Gemma 4 | debug | `all`, `fc`, `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.o_proj`, `mlp.gate_proj`, `mlp.down_proj` |
| `TF_ANE_COLUMNS` | Metal, Qwen3.5 dense adapter | experimental | Nonzero columns; admission still decides whether the share is supported |
| `TF_PROMPT_ROWS` | Metal, Qwen3.5 dense/MoE adapter | arithmetic | Every explicit native prompt width |
| `slide_learn_mode` | CUDA Nemotron/Qwen3.5 and Metal Nemotron, as its `scopes` list states | experimental | The real member names described below |

The grouped slide row has `backend: "multiple"` and exact family/backend pairs in `scopes`.
Ordinary rows retain their single `backend`/`families` scope and have null `scopes` and `members`.
Group `members` lists `TF_SLIDE_POCKET_KEY` with `auto` reporting syntax and `TF_SLIDE_EDIT`,
`TF_SLIDE_PROBE`, `TF_SLIDE_ROUTED` with `present` syntax. Presence counts even for empty text or `0`.
The startup line prints each real member name, never a `slide_learn_mode` environment assignment.
Lookup of a member resolves its own raw bytes; there is no synthetic group value and no group default.
The CUDA Qwen learner uses the pocket key but has no routed sites or edit/probe modes.
Those three members therefore carry a Nemotron-only `families` restriction and are omitted from Qwen startup output.
The CUDA Qwen MoE adapter does not expose a learner, so it is outside the group; the ANE share is dense-only.

`present` reports any explicit value; `not_zero` reports every value except exact `0`.
Gemma's listed values identify the first member of each drafter stack; `1` is not a BF16 selector.
These entries describe potentially changed arithmetic or learning, not new fidelity qualifications.
See [environment validation](environment.md) for strict numeric forms and deliberately lenient switches.

Kept-state identities include each applicable non-identical row. A grouped row hashes its real member names
and values in their family/backend scope, rather than looking up the synthetic group name. Explicit prompt
widths remain distinct from the default and from each other. Dory's unset and exact `0` share the disabled
identity; every other value shares the enabled identity. Existing count, exact-one and auto encodings are unchanged.

## Startup reporting

The serving process prints its profile and applicable registered overrides, for example:

```text
[tensorfold] profile: stable; registered overrides: TF_QWEN_KV_PAGES=256 (identical, env); TF_QWEN_KV_SHARE=16 (identical, env)
```

When no applicable registered setting meets its reporting rule, it prints `registered overrides: none`. This is
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
existing syntax rule. Nemotron's row lists `0` and `all`, while GLM's row keeps its `auto` rule. Dory's row keeps
the `not_zero` syntax: every value except exactly `0`, empty text included, adds an override.

For Nemotron (`nemotron_h` and `nemotron_h_omni`), paging is on by default. Exactly `TF_KV_PAGES=0` disables it;
exactly `TF_KV_PAGES=all` puts every stream on pages for checks. Those two values appear as registered overrides.
Every other value, including empty text, `on`, `off`, `auto`, `00`, uppercase spellings and whitespace, keeps
Nemotron's existing `on` policy and adds no override to the line. In particular, use `0`, not `off`, to disable
Nemotron paging. These are its existing parser rules, not new aliases or defaults. The later load/admission path
can still select windows for learning or a Triton-backed engine; this line reports the environment policy.

Values are bounded and terminal controls are escaped. Credentials, paths, server addresses, and unrelated environment
variables are never enumerated by this report. Each process's environment remains authoritative on restart; idle
reloads use the same existing family readers.

## DeepSeek CUDA settings

[Issue #2370](https://github.com/ashhart/TensorFold/issues/2370) adds these rows for `deepseek_v41` on CUDA.
Their `present` syntax reports every configured value, including `0` and values equal to a current default.
Defaults can depend on the caller or other settings, so this is a record of explicit configuration, not proof
that each setting changed a reply. Each stable fallback is null. The release backend has separate readers.

| Class | Variables |
| --- | --- |
| `arithmetic` | `TF_DSV41_NUCLEUS`, `TF_DSV41_CONTEXT`, `TF_DSV41_ROWS_CAP`, `TF_DSV41_POOL_TOKENS`, `TF_DSV41_KV_SPLIT`, `TF_DSV41_KV_SPLIT_UNION_MIB` |
| `debug` | `TF_DSV41_LAYERS`, `TF_DSV41_NEEDS_POOL` |
| `experimental` | `TF_DSV41_EXPERT_TOPP`, `TF_DSV41_OWN_PREFILL`, `TF_DSV41_PREFILL_PIECES`, `TF_DSV41_SLOTS`, `TF_DSV41_MHC_DEFER`, `TF_DSV41_DRAFTS`, `TF_DSV41_SAMPLING`, `TF_DSV41_IMAGES`, `TF_DSV41_BIAS_VL`, `TF_DSV41_ENGRAM_DIR`, `TF_DSV41_SCHEMA_PROMPT`, `TF_DSV41_TOOL_GRAMMAR` |

The pool settings have separate rows because each can be set independently. `NEEDS_POOL` forces pool geometry
in the AOT asset planner; it does not select the served engine's pool. Its class is therefore `debug`.
The other pool classes conservatively retain the audit's `arithmetic` classification, without claiming exactness.
Images and vision bias also have independent rows. `ENGRAM_DIR` is reported as `<set>`; its path is never printed
by the registry line. The legacy `TF_DSV41_TOOL_GRAMMAR` alias has a server-wide reader, although this row covers
DeepSeek CUDA. A present `TENSORFOLD_TOOL_GRAMMAR` takes precedence and suppresses the masked legacy row.

All 81 `TF_DSV41_*` settings that DeepSeek validates are rows of the shared startup table in
`zig/src/platform/environment.zig` (see [environment settings](environment.md)). Their readers read through that
table; the served-knob readers of `PREFILL_ROWS` and `INDEX_BUDGET_MIB` keep their own stricter refusals after it.
`Model.open` and the AOT planner check those rows before any allocation, device or checkpoint work, and a later
direct read refuses the same text. The refusal names the variable, supplied value and accepted forms.

Empty or whitespace-only values mean unset, as for every other row: the reader's default applies. The one exception
is `TF_DSV41_LOOKUP`, whose blank value has always turned lookup drafts off while unset leaves them on; it keeps that.
Blank `MHC_DEFER`, `COEF_LATE`, `LOOKUP_PLAN` and `BIAS_VL` read as off, which is also their unset default.
Unset defaults and canonical `0`/`1` values are unchanged, and default-on switches stay on when unset or blank.
Only the four switches that already read words accept case-insensitive word forms: `LOOKUP` and `LOOKUP_PLAN`
take `true`/`on`/`yes` and `false`/`off`/`no`, and `MHC_DEFER` and `COEF_LATE` take the same except `no`, which
their old readers read as on and which is now refused. The others read any text but `0` as on before, so a word
form there is refused by name rather than read the other way.
Other switches take exactly `0` or `1`, with their existing whitespace trims.

The served-knob accessor also treats empty and whitespace-only values as absent, so knobs outside the table,
such as `PF_TBO`, `PREFILL` and `PROMPT_TAIL`, keep their defaults. The table's intentional refusals include
generation streams 0, an extra layer-range tail, and image modes other than `native` or `0`.

`ROWS_CAP` accepts the existing reader buckets: 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48 and 64.
The table row, the batch reader and the planner use that list, and a host test keeps it equal to the batch's buckets.
Every floor reader shares its row's space trim and named refusal. The hard floor has one reader in the KV budget,
used by the pool, graph room, scheduler and prefill path; its unset value remains 4 GiB.

`BIAS_VL` retains its accepted presence forms: unset/blank off, while both `0` and `1` select vision bias.
Other text refuses. Engram queue depth remains 64 by default and permits 0 for serial reads. Reader threads
remain 16 by default and accept 0 through 32; larger values refuse instead of clamping. Generation streams
remain 1 by default.

The server validates the legacy tool-grammar alias once at startup for every family. The canonical variable
still masks it when present, so an invalid masked alias is ignored as before. A bad unmasked alias refuses at
startup, rather than failing each later auto-tools request. Both `deepseek_v41` and `deepseek_v4` model-type
names show the same CUDA registry rows on the startup line.

Configured values are length-delimited raw bytes in the shared identity helper, scoped to `deepseek_v41`/CUDA.
Other family/backend identities keep their existing serialization. These registry rows do not claim to bind
a current DeepSeek disk cache. This change adds no DeepSeek prompt-disk integration.
No parser change qualifies a new kernel, arithmetic mode or profile.

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
| Every other value | Pool pages | Pool pages | Pool pages |

These are the existing parser rules, including empty text, `all`, `false`, uppercase spellings and whitespace counting
as `on`. Prefer `auto`, `off` or `on` explicitly. The family line normalizes these choices and never echoes arbitrary
input; the registry line shows bounded, escaped raw overrides. `TF_GLM_KV_HOST=1` with FP8 is refused before the
selection report or engine construction; peer notification carries `GlmFp8HostKvUnsupported`. The existing BF16
paging evidence is [#1322](https://github.com/ashhart/TensorFold/pull/1322); paged-FP8 qualification is tracked in
the [paged-FP8 recipe](recipes/glm-fp8-pages.md).

## Further profile work

Migrate additional switch groups with their existing parsers and tests intact. A future `fast-exact` profile needs
the issue's same-checkpoint token gates: solo, concurrent 1/2/4/8, and resumed runs for every affected family, plus a
measurement where no served cell for that family gets slower. `fast` is not planned: a speed mode that changes output
tokens is not shipped, so the arithmetic class has no profile to join.
The host tests for this foundation establish lookup/CLI/reporting behavior, not those GPU or checkpoint gates.

## Four-rank RoCE relay ceiling

`TF_TP_ROCE_RELAY_MAX_KB` sets the largest all-gather shard per rank sent over an enabled four-rank relay.
It defaults to 64 KiB and accepts integers from 4 to 1024 KiB. Larger gathers use NCCL, including variable-length
gathers whose stride exceeds the ceiling. `TF_TP_MAILBOX_MAX_KB` still sizes the slots; the effective relay ceiling
is the smaller of the two limits. Changing the relay ceiling does not resize those slots or change gathered bytes.

This setting is read directly by the TP parser, outside the profile registry. Invalid values refuse by name, and
relay ranks must agree on the value. GLM uses its actual `TF_GLM_WORLD` and rank for relay validation, so a four-rank
GLM run does not also need `TF_TP_WORLD=4`. With `TF_GLM_ROCE_MULTI=1`, a relay outside four ranks is refused before
weights load.
The relay remains opt-in through `TF_TP_ROCE_RELAY=1`; mesh, two-rank paths and all-reduce routing keep their defaults.

On a verified four-rank verbs relay with no opposite-rank link, `tf-tp-test suite` omits its NCCL send/recv and paired
exchange references, including that exchange in graph replays. NCCL selects those receive ports by channel on this
topology. The suite prints the reason, the two reference types and the number of comparisons left out; all sums,
gathers, remaining graph nodes and timing sweeps still run. Other topologies keep the complete suite.
