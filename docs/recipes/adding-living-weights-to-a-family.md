# Adding Living Weights to a family

This is how a family on the Zig engine learns facts with `--slide`. [Living Weights](../living-weights.md) is one
learner for every family (`zig/src/living`): it writes lessons, reads the model, fits a pocket of new neurons and saves
them into the checkpoint's own shards. A family describes itself as data, and its host and trainer give the learner a
few hooks. The steps are in order. It comes from three families:

- Nemotron 3.5 Lightning on Metal and CUDA, the first family and the only one listed in `living/families.zig` today.
  Its relu^2 shared experts carry the pocket at the MoE layers.
- Qwen3.8 Flash Next on CUDA (#895 and #896). A SwiGLU shared expert behind its own gate at every layer, with
  hyper-connections, Gated DeltaNet and gated attention below it.
- Qwen3.8-27B on CUDA, a port on #726's branch that is not a pull request yet. A dense SwiGLU MLP at every layer, with
  no experts at all.

Only Nemotron learns in this tree. Flash Next's and the 27B's learning have not landed, so their file names and
numbers below come from those branches.

Bring the family up on the Zig engine first, exact and at speed ([adding a Zig family](adding-a-zig-family.md)). The
learner trains through the engine's own forward, so everything below assumes that forward is done.

## How to use it

1. Do the steps in order. Each ends in a check that has to pass before the next step means anything.
2. Never teach the original folder. Teach a `cp -RL` copy (`cp -cRL` on a Mac), and check the original's sha256
   after each run.
3. Put the receipts at the end of this page in the PR: an untaught folder serves as before, the trainer's forward is
   the serving forward, lessons are kept, and a taught folder serves.

## 1. Describe the family as data

Add a `Family` to `living/families.zig`. The learner, the shard writer and the server's Living Weights view all read
it, by the checkpoint's `model_type`.

| Field | Nemotron (`nemotron_h`) | Flash Next (`qwen4_exp`) | Qwen3.8-27B (`qwen3_5`) |
| --- | --- | --- | --- |
| `names.layers`, `names.expert` | `backbone.layers.`, `mixer.shared_experts.` | `language_model.model.layers.`, `mlp.shared_expert.` | `language_model.model.layers.`, `mlp.` |
| `names.inputs` | `up_proj` | `gate_proj`, `up_proj` | `gate_proj`, `up_proj` |
| `names.mtp` | its own file, `mtp-4bit.safetensors` | in the index, `language_model.mtp.layers.0.mlp.shared_expert.` | none |
| `neuron` | relu^2, margin 0.25 | SwiGLU, gated, margin 0.5, sharp 16 | SwiGLU, margin 0.5, sharp 16 |
| `quant` | 4-bit, groups of 64 | 4-bit, groups of 32 | 4-bit, groups of 64 |
| `keys` | `moe_shared_expert_intermediate_size` at the top level | `shared_expert_intermediate_size` in `text_config` | `intermediate_size` in `text_config` |
| `experts` | `.pattern`: the MoE layers of `hybrid_override_pattern` | `.every` | `.every` |

**Where the pocket goes.** A pocket is 64 neurons a layer (`Family.neurons`, a whole number of quantization groups),
zero until a fact uses one. Each is one more row of every input projection and one more column of the output
projection, so the first save makes the projection 64 neurons wider and any MLX loader reads a slightly wider model.

- **An MoE layer with a shared expert:** beside the shared expert, as Nemotron and Flash Next do. The routed experts
  are left alone: a pocket there would fire only when its expert is picked.
- **A shared expert behind a gate:** set `neuron.gated`. Flash Next scales its shared expert by
  `sigmoid(shared_expert_gate x)`, so a pocket neuron adds `gate act(x) v` and its output is fitted through the gate.
  The trainer reports the gate at each row (`gateAt`).
- **A dense MLP:** the MLP is the expert (`expert = "mlp."`, `experts = .every`), and the width key is the MLP's
  (the 27B's 17,408 neurons become 17,472).

**The neuron's kind.** A relu^2 neuron is exactly zero below its threshold. A SwiGLU neuron is not: `silu` leaks
below zero. A SwiGLU neuron fitted as `u` is written as gate row `sharp u` and up row `u / sharp`
(`living/neuron.zig`), so at z = u x it fires `z^2 sigmoid(sharp z)`, which is relu(z)^2 only where z is well above
zero: `neuron.zig`'s test checks the two agree within 1e-6, relative, at z = 1, 1.5 and 3 with `sharp` 16, and at
z = 0.1 they differ by about 0.0017. The output fit reads each neuron through its own activation (`Pocket.act`), not
as relu^2. Rows a neuron stays silent on are fitted to z at or below `-margin`, and with margin 0.5 what leaks there
is at most `0.25 sigmoid(-8)`, about 8.4e-5 (the test checks z from -0.5 down to -20). Keep `sharp` a power of two, so
the up row gives `u` back bit for bit in fp32 while `u / sharp` stays a normal number and `sharp u` finite. A save
still rounds every row to the family's affine format, so the checks read the rows as saved (#877).

`Family.check()` refuses a description whose inputs do not match the neuron, whose code width is 0 or does not divide
a 32-bit word, whose group or pocket is empty or not a whole number of groups, or, for SwiGLU, whose `sharp` is not a
whole power of two from 1 up (`living/family.zig`). Add your family to the tests in `families.zig` and `family.zig`:
its model type is found, and a tensor name of its expert gives the layer back while a similar one (a gate, the MTP
head's) does not.

## 2. The pocket in every pass

The engine adds the pocket wherever the model runs: prompt chunks, verify windows and decode rounds. Load it from the
shards as fp32 rows beside the packed weights (Flash Next: `cuda_pocket.zig`; the 27B: its `pocket.zig`), and take
the shard's projection apart at the old width.

- **An untaught folder's bits do not change.** A layer with no neuron in use launches nothing. Flash Next sums a
  pocket layer's MoE output in fp32, adds the pocket and rounds once; every other layer writes back exactly as before.
  Check the untaught folder against the last release: the same replies, drafted and `"draft": false`, and the same
  speed.
- **Row by row.** The pocket's kernels read each row alone, so drafted still equals `"draft": false` and a concurrent
  stream equals its solo run. Flash Next uses the two-kernel form of #787 (`tf_fn_pocket_in`, then `_out`).
- **Untaught streams.** A request with `untaught` set (the lesson writer's and the judge's, in
  `server/slide_ask.zig`) reads the model as shipped. Zero the pocket's output on its rows. The 27B missed this at
  first: the judge then read the lesson's own neurons, called a reworded "I don't know" a changed answer, and took a
  good lesson out.
- **The config check accepts a taught folder.** If the engine checks the checkpoint's widths at load, subtract
  `tensorfold_learned_neurons` from the expert's width first. Flash Next's check refused its own taught folder at the
  first restart.
- **The MTP head.** The first save widens the MTP head's shared expert with zero neurons, so its tensors stay one
  width. Drafts are made without the pocket; the verify rows run it, so drafted replies still equal plain ones.

## 3. The host

The host is a type the learner is generic over (`Learner(H)` in `living/learner.zig`). A family's slide file
(Nemotron's `cuda_slide.zig`, Flash Next's in #896) declares it.

- **Always:** `Trainer`, `max_rows`, `Ctx`, `family`, `layout(ctx)`, `trainer(gpa, io, ctx)`, `free(t, gpa)` and
  `seconds(io)`.
- **The pocket, relu^2:** `pocket(ctx, layer, up, down, used)` hands the forward a layer's neurons;
  `pocketRead(ctx, layer, up, down) usize` gives back the neurons loaded from the shards; and
  `pocketSave(ctx, gpa, io, layers, neurons, up, down) usize` writes every layer and hands the rows back as written.
- **The pocket, SwiGLU:** `pocketSwiglu`, `pocketReadSwiglu` and `pocketSaveSwiglu`, the same three with a `gate`
  beside `up`.
- `up` and `gate` are each neuron's input rows as the forward multiplies them (`[neurons, D]` f32), and `down` is its
  output. A save hands the rows back rounded as the shards hold them (#877 checks every lesson on those).
- **Optional:** `attach`, `write` and `layers` (Nemotron's routed blocks), and `codes`, `commit` and `fold4`
  (weight edits). The learner offers them only when the trainer has `sites`.

`layout(ctx)` is a `Layout`: the hidden size, the expert's own width with any saved pocket left out, the layer count,
the layers whose expert carries the pocket (`widen`) and the ones lessons put neurons in (`learn`).
`Layout.fromExperts` widens every layer that has an expert and learns at all but the last, whose nudge has no
gradient to read. Read the widths from `config.json` at load, as Flash Next's `layoutOf` does.

## 4. The trainer

The trainer runs the learner's passes on the engine: a keys pass (the rows a pocket reads), a loss pass, and a target
pass that runs the loss's gradient back to the lowest nudged layer.

- **What it gives:** `normed(layer, row)` (a row's input to the expert, as the forward normed it), `pass(seqs, mode,
  turn, out)` and `step(ids, start, mode, turn)`, `spans`, `packs()`, `pack`, `count`, `prefixes`,
  `prefixSlots(n)`, `pushAt(i, key, layer)` (the gradient's push at a nudged row), `deltaAt(i, key, layer)`,
  `residual(layer, row)`, `nudge` and `from`. A gated family also gives `gateAt(layer, row)`.
- **Optional:** `tapRows` (#891) reads a question asked after a conversation through the engine's own prompt chunks,
  so a pocket can read it in context. Nemotron's routed blocks, edits and probes also use `sites`, `sketched`,
  `score()`, `projected()`, `key()` and `config()`.
- **Sequences side by side.** A pass packs several sequences into one forward (Flash Next 1,024 rows, the 27B 1,024
  rows over up to 16 sequences). Row-wise launches run over every row; a recurrent mixer or attention runs each
  sequence from position 0. A sequence's loss in a pack equals its loss alone, exactly.
- **What the forward keeps.** The backward reads what the forward computed: each layer's expert input and its routing,
  and, from the lowest layer that learns, every mixer's projections, the norms' inputs, the gates. Keep them in the
  engine's forward rather than in a second forward. Flash Next's batched pass keeps them on request (`Keep`): every
  layer's expert input, and the rest from `layers / 5` up, as the 27B's does.
- **The backward kernels.** One input-gradient kernel set for each layer type above the lowest learned layer, checked
  against float64 with central differences. What another family could use goes in the core, a family's own in its
  folder (#895):
  - core: a Gated DeltaNet layer (`cuda.gdn_train`), 4-bit products at the checkpoint's group, routed SwiGLU experts
    and top-k routing (`cuda.train_q4`), gated attention (`cuda.train_attn`), Nemotron's Mamba (`train.cu`);
  - family: Flash Next's hyper-connections and its gated pocket (`flashnext_train.cu`); the 27B's dense SwiGLU MLP and
    its ungated pocket.
- **Memory.** On a GB10 the trainer keeps MLX copies of the small linears its backward reads, and its keep lists. It
  took about 26 GiB for Nemotron, 10.6 GiB for Flash Next and 23 GiB for the 27B. `open` reports it as
  `learn_bytes`, and the native host keeps it out of the streams' room.

**Checks.** Add a `tf-cuda-test` case for each backward kernel (Flash Next: `gdn-back`, `q4-back`, `moe-back`,
`attn-back`, `fn-hc-back`, `fn-moe-back`, `fn-pocket`), and a tool that runs the trainer on the real checkpoint
(`flashnext-learn`):

- the trainer's keys and loss passes equal the serving forward byte for byte, untaught and with a pocket neuron;
- a row's scores never depend on later rows;
- a pack of sequences gives each one's loss exactly as alone;
- a pocket neuron moves the forward by exactly `gate act v` at its row.

A finite difference against the gradient is noisy in bf16. Below about one ulp of the residual the rounding decides,
and a large step leaves the linear range. Flash Next's direct path agreed within a median of 6% in three windows of
four, and its cross-row paths stayed noisy. The 27B's direct rows gave 2-36%. Treat it as a smoke check: the kernels'
float64 tests and the identity checks above are the real ones, and a lesson that is kept is the end-to-end one.

## 5. The shard writer

`living/shards.zig` writes any family's pocket: `Io(core).write(gpa, io, dir, family, layout, layers, neurons, inputs,
down)` rounds the rows to the family's affine format (`living/affine.zig`), widens every expert on the first save
(the MTP head's with zeros), records `tensorfold_learned_neurons` in the family's key scope, and writes a later save in
place. Wrap it in the family's folder (Flash Next's `living.zig`, `Shards(core)`).

Test it on a small fake checkpoint (Nemotron's and Flash Next's `pocket_shards_test.zig`): the first save widens every
expert and tells `config.json`; a later save lands in place; every other tensor's bytes are kept; and the rows read
back are the rows written.

## 6. `--slide` on the native host

The native host (`native/cuda.zig`) serves `--slide` for a family whose module declares:

- `slide`, a type whose `Learner` is `living.learner.Learner(Host)`, with `Model`, `Example`, `Lesson`, `Report` and
  `Step`;
- `learnModel(ctx, dir)`, the learner's `Model` for a loaded engine;
- `linked(arena, io, dir)`, the first file a lesson would rewrite (config, index, a shard holding an expert) that is a
  link, so a folder of links into the Hugging Face cache is refused at startup.

`open` with `.slide = true` reserves the trainer's rows in the pass's scratch and reads the layout from `config.json`.
The host already refuses `--slide` off a GB10 (sm_121) and with `--segments`. A family adds its own refusals: Flash
Next refuses a captured kernel set, whose read-out kernels keep nothing a backward reads. Add each refusal's message
to the table in [Living Weights](../living-weights.md), and a line for the family under "What you need".

## Receipts to produce

On a fresh copy for each run, greedy, with the checkpoint's revision, the GPU, driver and CUDA version:

- **Untaught, as before.** Against the last release on the same machine: the same replies drafted and with
  `"draft": false`, decode and cold-prompt speed, and the model's memory.
- **Trainer.** The kernel checks and the identity checks of step 4, and the finite-difference numbers as measured.
- **Lessons.** A handful of facts about the user, one learn request each (Flash Next taught five, Nemotron ten). For
  each: kept or not, rounds and steps, seconds, and its held-out wordings right after its lesson and after the last.
  Ask wordings no lesson trained on. Then a fixed set of unrelated prompts against the untaught replies: #891 used
  twelve (capitals, arithmetic, a haiku, a Python one-liner, a translation and "What is my cat called?").
- **A taught folder serves.** Restart without `--slide` and with it: the same replies as after the last lesson.
  Drafted equals `"draft": false`, concurrent equals solo, a resumed prompt equals a fresh one, and the decode cost of
  the pocket (Flash Next: 5% drafted and 1% plain with three facts).
- **Tests.** A plain `zig build` (every step), `zig build test-cuda-host` and `zig build test-living` from a fresh
  cache, `zig fmt --check build.zig zig`, and `python3 tools/zig/lean_check.py` with no new problems.

## Traps

- **A finished lesson can still hurt later ones.** A pocket neuron's output reaches higher layers, so a lower pocket
  neuron can push a higher one past its threshold. On the 27B, after 16 facts, seven of 40 wide unrelated prompts gave
  garbage tokens: one neuron leaking +0.4 on LaTeX digits set off about 100 others, layer by layer. The rows that set it
  off were in no prompt a lesson read. Read unrelated prompts in several languages, code, math and tables while you
  teach, and check them after every lesson, not only at the end.
- **The checks only see what they ask.** Lessons ask single-turn questions. A fact can come back cold and miss after a
  long conversation (#891).
- **A captured kernel set keeps nothing.** If a family still has a captured comparison set, `--slide` must refuse it.
