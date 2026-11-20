# Control vectors

A control vector changes activations at inference time, without changing checkpoint
weights. Steering is disabled unless the server starts with `--glp`. No trained
vector file is distributed here, and useful steering or preserved capability is
not established by a successful load.

Two additive-only sites are distinct: Qwen3.8-27B on Metal uses the legacy
post-block residual hook; the experimental standalone DeepSeek-V4.1 EXL3 CUDA
pair uses the GLP v3 paired fold/writer hook described below. Native Zig reads
the container and serves requests; Metal/CUDA kernels apply the edits.

## Qwen3.8-27B on Metal

```
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --glp vector.gguf --glp-alpha 1.0
```

`--glp` names the file and `--glp-alpha` the strength. Without `--glp-alpha`
the file's own `glp.alpha_default` applies. The applied fp32 strength must be
finite and above 0.

At startup the server names the file it loaded, the SHA-256 of that file's
exact bytes, and says that replies are steered.

The file name is written escaped, so a name with a newline or a quote cannot
add a line of its own. The file hash is taken over the bytes the server read
once, before the model loads; it is the hash to compare against the file you
meant to load. The tensor digest is the separate SHA-256 over the direction
tensors that a file may claim in `glp.content_sha256`. The engine repeats that
replies are steered, with the file hash, when the vector is in force.

Saved conversation states also include that file hash and the applied strength
in their identity. Changing either starts those states cold; without `--glp`
the existing policy key is unchanged.

A file the server cannot apply is refused and the server does not start.

## Bringing a vector

The repository carries none, and this page names none to download. A vector is
derived by the operator, for the style or topic they want a model to lean
toward, and the prerequisites are theirs to meet:

- Two sets of prompts that differ in the one thing the vector should carry,
  and activations collected from Qwen3.8-27B (the 5120-wide model) at the
  output of its blocks on both sets.
- One direction per layer from those activations, for example the difference
  of the two sets' mean activations, in fp32. llama.cpp's `cvector-generator`
  writes this container; any tool that writes the format below works.
- Layer ids of 1 to 63 for the 64-block model, zero-based (see below), in a
  file whose `glp.base_model`, if present, names the checkpoint served.

The strength is the operator's to find on their own prompts: a direction added
too strongly degrades the text, and one added too weakly changes nothing. The
reader does not judge either.

## Add only

A vector is applied by adding the direction to the residual stream. A file
whose mode is missing is an `add` file, as the format specifies, so a llama.cpp
control vector works as it stands. A file that names another mode is refused,
`project` included, and is never read as an `add` file.

## What the Qwen Metal path refuses

The reader fails closed rather than guessing.

| Refused | Why |
| --- | --- |
| `glp.mode` (or `dspark.mode`) is anything but `add` | Only add vectors are applied. A file that names `project` is refused by name, not coerced to `add`. |
| `glp.hook_point` is not `residual_stream_post_layer` | The vector was derived at another site. |
| `glp.rank` is not the integer 1 | The reader implements rank 1. |
| `glp.spec_version` is not 1 | The reader implements version 1. |
| `glp.alpha_default` is not a floating-point value | A malformed strength is not silently replaced with 1. |
| A direction holds a NaN or an infinity | The direction is added as it stands, so every value has to be finite. |
| The digest does not match the bytes | The file says one thing and holds another. |
| One layer is named twice | `direction.5` and `direction.05` both parse as layer 5; a file has one direction per layer. |
| A layer id is past the checkpoint's depth | The checkpoint's `num_hidden_layers` is read at startup, so the file is refused before the model loads. |
| The width does not match the checkpoint | The vector is for another model. |
| The base model does not match | The vector was derived from another checkpoint. The names are compared without their owners, because a checkpoint is often a quantization of another repo's model under a new name. |
| A Gemma checkpoint | A Gemma is never steered. |
| A CUDA or ROCm backend | Qwen steering is Metal-only. The DeepSeek CUDA exception below is a different site and engine. |
| Any other model family | This residual hook applies only to Qwen3.8-27B. |

A file whose `glp.derived_at` differs from its `glp.hook_point` is a warning,
not a refusal: the vector is applied at the residual stream either way.

## The layer a direction names

`direction.N` is the layer index `N`, and the layers are zero-based: `direction.N`
is the output of the N-th block, the residual stream as that block leaves it.
Layer 0 is not steerable, so a file for a 64-block model names layers 1 to 63.

This is worth stating because the alternative is silently wrong. The container
follows llama.cpp, and llama.cpp's own generator writes `direction.{il+1}` while
its applier reads `direction.il`: the two cancel, so the applier's numbering is
the one a file means. An off-by-one here does not fail loudly either, because
adjacent layers' directions are highly correlated and a stack shifted by one
layer still produces plausible output.

The file says so itself. A vector for this model can carry its layer list in
its metadata:

```
glp.layer_ids_zero_based = 10,11,12,...,58
```

and this reader refuses a file whose tensor names and that list disagree.

The direction lands on the residual stream after a block, and the block that
follows reads the result: the direction is added before the next block's
normalisation sees the stream.

## How the bias is applied

A steered layer takes one dispatch, the same one an unsteered layer takes: the
norm that adds the block's output to the residual stream also adds the
direction, scaled by the strength, to the row, and the normalisation that
follows reads the result. There is no separate pass over the residual row.

The arithmetic is pinned by the kernel contract
(`zig/src/core/contract/steer.zig`): a float64 reference and the bound the
kernel's own fp32 arithmetic is checked against. The GPU half of that contract
(`zig/tests/contract/metal_qwen27_steer.zig`) needs a Mac and has to be run
there; this page carries no timing or output measurement for it.

## The Qwen legacy file

The container is GGUF v3, read by `zig/src/core/glp_container.zig`, the GLP
reader for the subset a vector needs. The metadata follows the GLP standard:

| Key | Meaning |
| --- | --- |
| `glp.spec_version` | 1. |
| `glp.mode` | `add`, or absent. |
| `glp.rank` | 1. |
| `glp.hook_point` | `residual_stream_post_layer`. |
| `glp.derived_at` | The site the direction was captured at. It is a site, not a time: the format's own validator refuses a value that is not a hook name, and a file whose `derived_at` differs from its `hook_point` is a transferred vector, which is warned about. |
| `glp.created` | The day the file was written. |
| `glp.alpha_default` | The strength the file applies with by default. |
| `glp.method` | How the numbers were made, for example `difference_of_means_last_token_per_layer`. |
| `glp.contrast` | What the two prompt sets contrast. |
| `glp.base_model`, `glp.base_revision` | The checkpoint the vector came from. |
| `glp.layer_ids_zero_based` | The layers the file holds. |
| `glp.content_sha256` | SHA-256 over the tensor bytes, which the reader checks. |

The tensors are `direction.N`, fp32, one per layer, each the width of the
residual stream.

The file also says what it is, for a consumer that reads the header before it
reads the directions: `general.name`, `general.description`, `general.author`,
`general.license`, `general.base_model.*` (the repo it came from), and
`controlvector.model_hint` and `controlvector.layer_count`. The reader keeps
these when the file carries them; `tf-glp-check rewrite` writes a file back out
with the fields it read.

### The checkpoint the file names

`glp.base_model` is the checkpoint the vector came from, and the server refuses
a vector whose checkpoint is not the one it is serving. The check compares
names: the served checkpoint's model name has to be the vector's model name or
start with it, so a conversion published under a longer name takes the vector
and a different model does not.

The same weights under another name are refused with it. A file can be
re-stamped for that name:

```
tf-glp-check rewrite vector.gguf vector-mlx.gguf \
    --repo mlx-community/Qwen3.8-27B-4bit --revision REV
```

The directions stay the same bytes and the digest over them does not move; only
the name the file carries changes. That is a claim the operator makes, not one
the reader verifies: the check is a name comparison, so only re-stamp a file
for a checkpoint that holds the same weights.

## DeepSeek-V4.1 EXL3: experimental CUDA pair

This extends the existing `dsv41-serve` experimental pair, not the ordinary
release CUDA server. The release's DeepSeek refusal and whole-model landing
gate remain in place. No new DeepSeek Metal backend, project/ablation mode,
ROCm path or mixed-device consumer is introduced.

Follow the [DeepSeek CUDA setup](recipes/deepseek-v41-cuda.md), including its
checkpoint/assets, experimental guard, TP configuration and normal memory
admission. Use whole prefill on every rank:

```sh
TF_DSV41_PREFILL=whole tensorfold-dsv41 serve "$PACK" --backend cuda \
    --glp "$VECTOR" --glp-alpha 1
TF_DSV41_PREFILL=whole tf-dsv41 follow "$PACK" "$ASSETS" \
    --glp "$VECTOR" --glp-alpha 1
```

These are the rank-0 and follower commands, respectively; the existing TP
environment still places each rank. Every rank loads the same file and
effective strength. The follower reports its escaped filename, full file hash
and active/inactive status. There is no per-request switch or live reload.

The required file is **GLP v3, rank 1, add**, with
`glp.hook_point = hc_fold_pre_engram`. A Qwen residual file is not relabelled
as compatible. For the Flash EXL3 geometry targeted here, directions are H=5120,
`direction.N` names zero-based backbone block N, and valid ids are 1–39.
Layer 0 and MTP/drafter weights are not edited.

The same direction and strength apply at both sites:

1. The completed FFN post's four-copy stream is edited before the next
   Engram/site reads it. This lagged fold of block N runs at block N+1;
   the final backbone fold runs before the head.
2. Block N's complete TP-reduced attention writer is edited before its mHC post,
   not once per partial rank sum.

Each residual copy receives `alpha*d` in FP32, with BF16 round-to-nearest-even
storage. Directions upload once, remain owned through queued work and graphs,
and add no per-token allocation or CPU readback. Strength is finite,
nonnegative and either zero or normal FP32. Zero creates no GPU steering plan;
a loaded zero-strength file nevertheless remains a distinct state policy.

Full file SHA-256, site/version and effective strength bind rank agreement and
disk-session identity before execution. An absent file preserves the old
unsteered path and policy key. Mismatched ranks, geometry, hooks, versions,
scales or content digests fail closed. Known checkpoint repo/revision metadata
must match; a local conversion without retained provenance is not proof of
vector compatibility.

Active steering refuses replay prefill, TBO and traced unsteered folds. Do not
lower memory safety floors to obtain a receipt. Qualification needs the actual
source-built image's arithmetic/reference, planted-control and resource gates,
then real TP2 default-off/zero/add, within-policy sampling/drafting/concurrency
and restored-versus-fresh state checks. Prior fork experiments do not qualify
this upstream source.

The native session store snapshots the last complete 16-row group before the
prompt end: an 8192-token prompt therefore reuses 8176 saved rows, not 8191.
Whole-prefill usage reports that actual restored count. With multiple live
slots, saved-state reuse additionally needs `TF_DSV41_SESSION_SLOTS=1`; qualify
that explicit profile rather than treating the default-off option as a hit.

The standalone served NVMe tier is disposable: startup purges the selected
compatibility/rank directory before tier workers start. This contribution
preserves that behavior; it adds no cross-restart restore or reconciliation.
Qualify RAM and NVMe round trips within a process. Qwen's saved-conversation
persistence is a separate caller and contract.

### Recorded device mechanism check

Kernel source `4da7c1439291b64996409da42ebd8393fe671f0d`, GB10 sm_121,
nvcc 13.0.88 and driver 580.173.02: the family runner passed 195 cases.
The public contract passed gates 1–4 for all six variants, all ten planted
controls were red as intended, and the live driver reported no limit problems.
Other SMs and later compiler versions remain unmeasured.

| Add-only kernel | Registers | Driver static shared bytes | Dynamic shared bytes | Threads |
| --- | ---: | ---: | ---: | ---: |
| Stream | 42 | 0 | 0 | 320 |
| Writer | 44 | 0 | 0 | 320 |
| Final norm | 48 | 176 | 10240 | 320 |

Image SHA-256: `fa3e9d61b0c8facb4503f9dca64f1b9f1848a9667854d1223cb24fe66648f766`.
Control image: `470e9b6fe4e1a4db6d1e90a8cb9b2dfc52b28b90e97a804719b6aae39d76ca1f`.
Memcheck, initcheck and synccheck reported zero device errors on both runners;
racecheck exited zero. Full leak checking reported a shared-runtime pinned
staging allocation at process exit: 4 MiB for the family runner and 256 KiB for
the public runner. This is not a zero-leak receipt.

These operator and graph-replay results do not alone qualify full-model token
identity, served RAM/NVMe reuse, performance or a trained vector's effect.

### Recorded single-slot TP2 served check

Model source `8da263a4743086b70cb6c3c462d9d95ae995cbc4` was compared with
upstream `41969beed8` plus only the same usage-counter correction. Two GB10s
ran the pinned Flash EXL3 checkpoint above, RoCE TP2, one physical slot, whole
prefill, graphs and sampling, with the normal 4/5-GiB hard/target floors.
Each policy passed 13 long requests: cold/warm, greedy and seeded plain/draft,
and four concurrent clients queued through that slot. Four additional short
prose/code/arithmetic prompts passed plain/draft token equality.

No-file and loaded-zero outputs equal baseline. A synthetic unit-direction
control at strength 4 changes all four short prompts and the long output;
this proves a mechanism, not useful style, model quality or semantic efficacy.
Changing only file metadata while overriding both strengths to 4 preserves
the active output tokens but creates a distinct disk namespace on both ranks.
No-file, loaded zero, active strength and metadata-only file identity occupy
separate namespaces as applicable.

RAM and forced-eviction NVMe restores reuse exactly 8176 rows of the 8192-token
input and equal the cold output within each policy. Real mismatched ranks
(loaded zero versus no file; same directions/strength but different file hash)
both stop with `GlpMismatch` before weights open.

Warm-cache client decode-window medians, 64 output tokens, three
plain/draft/draft/plain repetitions, six samples per mode:

| Policy | Plain tokens/s | Drafted tokens/s |
|---|---:|---:|
| Baseline | 38.418 | 181.404 |
| Candidate, no file | 38.337 | 180.362 |
| Candidate, loaded zero | 38.316 | 181.143 |
| Candidate, synthetic strength 4 | 37.758 | 113.236 |

These are single-stream, short-output observations, not a cold-prefill or
pure-kernel speed claim. Active steering changes the target and draft
acceptance. In the seeded active functional cell, drafting is slower:
22.465 versus 30.415 tokens/s, with 17 of 112 proposals accepted. The
no-slowdown gate is therefore not qualified for that control.

The first four-slot baseline request failed with `MissingTritonVariant`
(`_scores`): rank zero had 189 AOT variants, the follower 194. The consumed
RoPE table prefix was bit-identical despite different file capacities.
Private qualification assets now align the 194-variant bundle and the
133120-row RoPE file on both ranks; original serving assets are unchanged.
The wider-stream comparison is not yet qualified by that preparation.


Derive later style/topic vectors for the exact DeepSeek checkpoint, quantization
and paired site. Publish their authors/licence, checkpoint and conversion pins,
derivation and prompt provenance, tensor/file hashes, layer ids, strength ladder,
norm-matched/random controls and held-out capability results separately.
Synthetic arithmetic directions are mechanism fixtures, not semantic vectors.

## The integration test

A file that one implementation accepts has to mean the same thing to another.
The reference implementation for this format is the `weightless` plugin's
container reader, and this repository does not vendor it: the test takes a
checkout and reads it as a package.

```
zig build tf-glp-check
python3 tools/zig/glp_parity.py --reference PATH/TO/weightless \
    --vector vector.gguf --vector PATH/TO/another-vector.gguf
```

Four legs, over the same files:

| Leg | What it checks |
| --- | --- |
| read | The reader in Zig and the reference reader on one file: the layer ids, the metadata and every direction's bytes are equal. |
| rewrite | The file read in Zig and written out again, read by the reference: the numbers, the hook point and the strength survive the round trip. |
| digest | The file's own `glp.content_sha256` over the tensor payloads, hashed the same way on both sides. |
| file hash | The SHA-256 the reader reports for the file's bytes is Python's SHA-256 of the same file. |

The reader that does the Zig half is `tf-glp-check`, which prints the reader's
view of a file and writes its directions out as bytes, so the two sides can be
compared exactly rather than approximately.

`tf-glp-check` checks its command line before it opens a file: a missing or extra
argument, an unknown option or an unknown command prints a usage line and exits
with status 2, and `zig build test-glp` runs those cases against the built tool.

The test needs `numpy` for the reference reader, and a `weightless` checkout.
Nothing of the reference is committed here, and no vector is either: the file
is supplied when the test runs.

## Credits

**Matt Suiche (@msuiche), weightless** originated the GLP concept/format and
DeepSeek paired-site principles. This reader implements the format defined by
that project; see the [pinned specification
`b181a07d3980fc48abe39d95ddb044459dd0b868`](https://github.com/msuiche/weightless/blob/b181a07d3980fc48abe39d95ddb044459dd0b868/docs/spec/GLP.md).
The additive GGUF `direction.N` convention follows **llama.cpp's authors and
contributors**.

**Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery,
Wes Gurnee and Neel Nanda** authored the original directional research,
[*Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717).
This is research attribution, not an implementation of project mode or a claim
to their experimental results.

The TensorFold-native reader and Qwen foundation reuse
[#1835](https://github.com/ashhart/TensorFold/pull/1835), source
`c5947a8d2e9543e6c6a11bd30dc4c43c32f3b0fb`, by @BlivionIaG, adapted to current
upstream. External serving plugins, engine kernels, checkpoint tensors and
trained vector assets are not vendored. TensorFold code remains Apache-2.0;
future vectors require their own publication provenance and licence.

No Aevonix/FlashAlign code, data or method is incorporated. The sources are the
GLP specification and native foundation cited above; this does not claim
priority over related work. See the [original attribution clarification](https://github.com/ashhart/TensorFold/issues/1834#issuecomment-6693354132).
