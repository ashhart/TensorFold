# Control vectors

A control vector steers a model's residual stream at inference time. It is a
direction per layer: at the output of a block, the direction's component is
taken out of the residual stream, or the direction is added to it as a bias.
Nothing is written to the checkpoint, and a server without a vector serves the
model exactly as it was.

TensorFold reads these vectors on Metal for Qwen3.8-27B. The container, the
kernel, the serve surface and the producer that derives a vector from the
model itself are all native Zig.

## Serving one

```
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --glp vector.gglp --glp-alpha 1.0
```

`--glp` names the file and `--glp-alpha` the strength. Without `--glp-alpha`
the file's own `glp.alpha_default` applies. The strength must be above 0.

At startup the server prints what is in force:

```
[tensorfold] glp on: vector.gglp (mode: project; alpha: 1.00; layers: 49 (10-58);
  base: TensorFold/Qwen3.8-27B-MLX-4bit @ 22d8d53; digest: 50a42548b373 (verified))
```

The file is read once, before the model is loaded. A file the server cannot
apply is refused and the server does not start.

## The two modes

A vector is applied either by taking the direction's component out of the
residual stream (`project`) or by adding the direction to it as a bias
(`add`). The file names which, and the reader applies what the file says. A
file whose mode is missing is an `add` file, as the format specifies, so a
llama.cpp control vector works as it stands.

The two are opposites, and the same directions show it. Our masked vector in
`project` takes every refusal off and takes no harmless prompt with it. The
same directions written as an `add` file push the other way: on the held-out
set below, the model refuses all twenty harmful prompts and all twenty harmless
twins and answers all forty coherently, and it still writes code, though it
refuses one benign task out of twelve, the same over-refusal the bias asks for.

`tf-glp-check rewrite FILE OUT.gguf --mode add` writes that file, so the two
arms can be compared on the same directions.

## What the server refuses

The reader fails closed rather than guessing.

| Refused | Why |
| --- | --- |
| `glp.mode` is neither `project` nor `add` | The reader does not implement it. |
| `glp.hook_point` is not `residual_stream_post_layer` | The vector was derived at another site. |
| `glp.rank` is not 1 | The reader implements rank 1. |
| `glp.spec_version` is not 1 | The reader implements version 1. |
| A direction is not a unit vector | A projective direction has to be one. |
| The digest does not match the bytes | The file says one thing and holds another. |
| The width does not match the checkpoint | The vector is for another model. |
| The base model does not match | The vector was derived from another checkpoint. The names are compared without their owners, because a checkpoint is often a quantization of another repo's model under a new name. |
| A Gemma checkpoint | Its terms of use forbid circumventing its safety filters, so a Gemma is never steered. |
| A CUDA or ROCm backend | Only the Metal engine steers. |

A file with no `glp.mode` is treated as `add`, which is what the format
specifies, so a llama.cpp control vector works as it stands. A file whose
`glp.derived_at` differs from its `glp.hook_point` is a warning, not a
refusal: the vector is applied at the residual stream either way.

## The layer a direction names

`direction.N` is the layer index `N`, and the layers are zero-based: `direction.N`
is the output of the N-th block, the residual stream as that block leaves it.
Layer 0 is not steerable, so a file for a 64-block model names layers 1 to 63.

This is worth stating because the alternative is silently wrong. The container
follows llama.cpp, and llama.cpp's own generator writes `direction.{il+1}` while
its applier reads `direction.il`: the two cancel, so the applier's numbering is
the one a file means. An off-by-one here does not fail loudly either, because
adjacent layers' refusal directions are highly correlated and a stack shifted by
one layer still produces plausible output.

The file says so itself. A published vector for this model carries the layer
list in its metadata:

```
glp.layer_ids_zero_based = 10,11,12,...,58
```

and this reader refuses a file whose tensor names and that list disagree.

The projection lands on the residual stream after a block, and the block that
follows reads what is left of it: the direction's component is taken out before
the next block's normalisation sees the stream, which is where the reference
implementation takes it too.

## What the projection costs

Nothing, and that took a kernel.

A steered layer used to take two dispatches: the norm that adds the block's
output to the residual stream, and then a separate kernel that took the
direction's component out of the stream and wrote it back. The second one read
the whole residual row, reduced it against the direction, and wrote it again.
That is a full extra pass over the residual stream per steered layer, plus a
dispatch and a barrier, on every token.

The projection now rides along with the norm that has to touch the same row
anyway. One kernel, one pass: the row is summed, the direction's component
leaves it, and the normalisation that follows reads what is left. A steered
layer costs exactly what an unsteered one costs, and the block after it sees
the projected stream, which is the order the reference implementation uses.

The arithmetic is pinned by the contract the projection has always had
(`zig/src/core/contract/steer.zig`): a float64 reference, the kernel's own
summation order modelled, and the bound the two are checked against.

What it costs, measured on the same machine: nothing that shows. A 4,063-token
prefill takes 43.69 s without a vector and 43.68 s with the published one
across 49 steered layers; decoding runs at the same 14.6 tokens a second either
way. The steering path is not a measurable cost of a forward pass, because it
no longer is a path of its own.

## Deriving one

```
tf-qwen27-steer-derive --model DIR --out FILE \
    --repo TensorFold/Qwen3.8-27B-MLX-4bit --revision REV \
    --capture last --mask 0.005 --layers 10-58
```

The producer runs a harmful prompt set and a form-matched harmless set through
the native forward, takes the difference of their means per layer, and writes
the file. The prompts are the repository's own
(`zig/src/families/qwen27/steer_prompts.zig`); no third-party corpus is read.

| Flag | Meaning |
| --- | --- |
| `--capture mean` | Read every row of a prompt and capture its mean. |
| `--capture last` | Take the last row alone, where the model decides what to say next. |
| `--mask FRAC` | Leave the top `FRAC` of a layer's dimensions out of the direction. |
| `--layers FROM-TO` | Keep a window of layers; every layer by default. |
| `--prompts N` | Use only the first N pairs of each set. |
| `--alpha A` | The `glp.alpha_default` the file carries. |
| `--name`, `--description`, `--author`, `--license` | What the file says about itself. |
| `--model-hint HINT` | The model family a consumer matches against. |
| `--effect TEXT` | What the strength does, as it was measured. |

`--capture last` is the one that matters. The last row of a prompt carries the
decision the model is about to make; the mean over every row is mostly the
prompt's content, and a direction built from it does not separate refusal from
anything else.

### The massive activations

Some checkpoints carry a handful of dimensions whose activations stand orders
of magnitude above the rest. A difference of means taken where they dominate
points at them rather than at the feature you asked for.

This model does. The producer's screen reports the largest activation against
the median dimension's at 701 times, and the reference implementation measures
75 to 434 times for the same checkpoint.

`--mask FRAC` leaves the top `FRAC` of a layer's dimensions out of the
direction, the ones the activations dominate, and renormalises what is left.
The dims are excluded from the direction, not from the model.

At 0.005, twenty-six of five thousand dimensions are left out per layer, and
they held between 12% and 66% of the direction's length before the mask: two
thirds of it at layer 58. That is a direction pointing at the massive
activations rather than at the refusal.

The producer reports the screen in its receipt, with the fraction it
recommends when the ratio stands out: a model that measures under 50 has no
massive activations of its own, and one that measures above it wants the
mask. The fraction is a decision the operator makes on the checkpoint in
front of them rather than a habit carried from another one.

## Measuring it yourself

Every number on this page comes from one command per arm. The gate starts the
server it measures, asks every prompt in a set and writes a receipt:

```
python3 tools/zig/qwen27_steer_gate.py --serve \
    --checkpoint TensorFold/Qwen3.8-27B-MLX-4bit \
    --vector vector.gglp --alpha 1.0 --set holdout \
    --label "our masked vector" --out receipt.json
```

Without `--serve` it measures a server that is already running, and
`--expect-harmful-refused`, `--expect-benign-refused` and `--expect-coherent`
make it fail when an arm misses the numbers it is supposed to reach.

The prompt sets are the repository's own and they live in Zig, so the producer
and the tool read the same text. `tf-qwen27-steer-prompts` prints them as JSON:

| Set | What it is |
| --- | --- |
| `derive` | The thirty pairs the producer derives a vector from. |
| `holdout` | Twenty pairs written after the vector was derived, over topics and forms the derivation set never uses. The producer never reads them. |

The gate reports three numbers per arm: the share of harmful prompts the model
refuses, the share of the harmless twins it refuses, and the share of replies
that are still answers. The last one is read beside the other two because a
vector that silences the model is not a refusal removal.

It reproduces the baseline row of the table below exactly, 30 of 30 harmful
prompts refused, none of the thirty harmless twins refused and all sixty
replies coherent, so the numbers here and the numbers the gate writes come
from the same measurement.

## Measured

Apple M4 Pro, 48 GiB, Metal, Qwen3.8-27B MLX 4-bit. Thirty form-matched pairs
of prompts of our own, no third-party corpus. The refusal rate is the share of
harmful prompts the model refuses; the over-refusal rate is the share of the
harmless twins it refuses, which is what a vector that removes a capability
instead of a refusal looks like.

| | Harmful refused | Benign refused | Replies coherent |
| --- | --- | --- | --- |
| No vector | 30/30 | 0/30 | 60/60 |
| Vector at strength 1e-6 | 30/30 | 0/30 | 60/60 |
| Published GLP-49, per layer, layers 10-58, strength 1.0 | 19/30 | 6/30 | 60/60 |
| Published GLP-63, pooled, layers 1-63, strength 1.0 | 2/30 | 0/30 | 60/60 |
| Our vector, layers 10-58, strength 1.0 | 18/30 | 7/30 | 60/60 |
| Our vector, masked at 0.005, layers 10-58, strength 1.0 | 0/30 | 0/30 | 60/60 |

A strength of 1e-6 reproduces the baseline: the path is a no-op at zero.

The two published vectors are `msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49`
and its pooled sibling GLP-63. They are downloaded and run here; they are not
vendored and they are not committed. The last two rows are vectors this
repository's own producer derives from the model itself, on our own prompt
set.

The masked vector is the one to read. It takes every refusal off and takes no
harmless prompt with it: with the dimensions the massive activations dominate
left out of it, the direction is a direction at the refusal rather than a
direction at the stream's offset.

The unmasked vector is the same derivation without that exclusion, and it
shows what the exclusion is worth: it removes two refusals in three and costs
seven harmless refusals in thirty. The direction that carries "I will not"
also carried "I will", because it was pointing at the wrong thing.

The published pooled vector reaches close to the same place with a direction
that was made differently, and the published per-layer one does not: a
difference of means over this model's massive activations is a direction at
the activations.

That the replies stay coherent in every arm matters more than the rates: a
vector that silences the model is not a refusal removal, so read the replies,
not the rate. The rate counts the phrases a refusal opens with, so it is a
lower bound: a vector that rewords a refusal instead of removing it is counted
as compliance, and the held-out set below shows the published per-layer vector
doing exactly that. The masked vector was put through an ordinary capability
check as well: twelve tasks of the model's own work, code, arithmetic,
translation,
a regular expression, a haiku, facts and protocols, and it answered every one
of them.

The benign column is the price in the arms that pay it, and it is not small:
six of the thirty harmless twins are refused under the published per-layer
vector, and seven under the unmasked vector, against none without one. Our
prompt set is deliberately tight, so a twin differs from its harmful
counterpart in intent alone. That is the strength's dial, and the reply text
is the measure.

## The held-out set

A vector derived from thirty pairs and measured on those same thirty pairs
measures the derivation, not the vector. The second set asks the question the
first one cannot: does the recipe generalise, or did it learn thirty prompts?

The twenty held-out pairs were written after the vector was derived. The
producer never reads them, and a test holds that no prompt in them appears in
the derivation set. They ask in forms the derivation set never uses: questions,
a role prompt, a longer sentence, and over topics it never names. A twin
still differs from its counterpart in intent alone.

| | Harmful refused | Benign refused | Replies coherent |
| --- | --- | --- | --- |
| No vector | 20/20 | 5/20 | 40/40 |
| Published GLP-49, per layer, layers 10-58, strength 1.0 | 9/20 | 6/20 | 40/40 |
| Our vector, masked at 0.005, layers 10-58, strength 1.0 | 0/20 | 0/20 | 40/40 |
| The same directions as an add file, strength 1.0 | 20/20 | 20/20 | 40/40 |

The benign column is read against five and not against zero here, and that is
the set's own doing. The base model refuses five of the twenty twins before any
vector is applied, because their frame is close enough to the harmful one:
"brute-force my own wifi password" reads as unauthorised access, a mockup of a
bank login reads as credential theft, and a rifle's stock reads as the rifle.
The base model's refusal follows the frame as well as the intent on those five,
so they are the floor of that column and not a cost any arm pays.

### The rate is a lower bound

The refusal rate counts phrases, and a vector can change the wording of a
refusal without removing it. The marker list catches the openings that say so,
so a refusal in other words is counted as compliance. That is the reason the
reply text is the measure and the rate is only the summary of it.

The published per-layer vector is the case in point. It takes eleven of the
twenty refusals off the metric on this set, and reading those eleven replies
shows that every one of them still declines: "you should not enter your
neighbour's house", "it is illegal in most jurisdictions", "it is not possible
to write a credible essay". The published vector bends the wording of the
refusal. Its benign column moves the other way, from five to six: it puts
three of the twins' refusals back and takes three of the base model's off.

The same holds on the derivation set, where that vector's row reads 19 of 30.
The eleven replies the metric counts as compliance all decline there too, in
other words: "it is not possible to provide step-by-step instructions for
synthesizing a nerve agent at home", "it is technically impossible to track the
real-time location of a smartphone", "you cannot make a functional, safe, or
legal taser". The row is a count of openings, not of decisions.

The masked vector takes the decline away: the model
takes the request and answers it, usually after a legal or safety caveat rather
than in place of an answer, and for several of the twenty it hands over what
was asked for, the scraper script, the essay, the scam message. The benign
column falls to zero with it, so the five false refusals the base model carried
are answered too.

The last row is the same directions written as an `add` file, and they are the
mirror image on prompts the vector was never derived from: the model refuses
all twenty harmful prompts and all twenty harmless twins, and answers all forty
coherently. The mode in the file decides which way the same numbers push, and
the choice holds on prompts the vector has never seen.

The vector's strength has a ceiling. Past it the model stops answering: the
residual stream loses more than the direction it was asked to lose. Take the
strength from the file's own `glp.alpha_default`, and read the replies.

The masked vector's strength is not a delicate dial. Across a sixfold range
the refusal rate and the over-refusal rate do not move at all, and the model
keeps answering:

| Strength | Harmful refused | Benign refused | Replies coherent |
| --- | --- | --- | --- |
| 0.5 | 0/8 | 0/8 | 16/16 |
| 0.75 | 0/8 | 0/8 | 16/16 |
| 1.0 | 0/30 | 0/30 | 60/60 |
| 1.25 | 0/8 | 0/8 | 16/16 |
| 1.5 | 0/8 | 0/8 | 16/16 |
| 2.0 | 0/8 | 0/8 | 16/16 |
| 3.0 | 0/8 | 0/8 | 16/16 |
| 4.0 | 0/8 | 0/8 | 0/16 |

That flatness is the point: a direction that is at the refusal and not at the
stream's offset is not one whose dose has to be found.

The last row is the ceiling, and it is where the reference implementation says
it is. The published vector's own metadata records "alpha=4 collapses this
model", and at strength 4 this runtime stops answering too: the replies go
empty. The container, the kernel and the strength's meaning agree with the
implementation that wrote the file.

## The file

The container is GGUF v3. The metadata follows the GLP standard:
| Key | Meaning |
| --- | --- |
| `glp.spec_version` | 1. |
| `glp.mode` | `project` or `add`. |
| `glp.rank` | 1. |
| `glp.hook_point` | `residual_stream_post_layer`. |
| `glp.derived_at` | The site the direction was captured at. It is a site, not a time: the format's own validator refuses a value that is not a hook name, and a file whose `derived_at` differs from its `hook_point` is a transferred vector, which is warned about. The producer writes the site it captured at, which is the site it applies at. |
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
`controlvector.model_hint` and `controlvector.layer_count`. The producer writes
them when it is told to:

```
tf-qwen27-steer-derive --model DIR --out FILE \
    --name "Qwen3.8-27B-refusal-cvec-per_layer-masked-L10-58-a1" \
    --description "..." --author "TensorFold" --license "Apache-2.0" \
    --model-hint "qwen3_5" --effect "alpha 0.5 to 3 is flat; at 4 the replies go empty"
```

`glp.created` is the day the file was written, and `glp.measured_effect` is what
the strength does, as it was measured. A vector is published on its own, as a
file; it is not committed to this repository, and the code that writes it is.

### The published file

The vector is published as a file of its own, and this repository holds the code
that writes it and the numbers that identify it, not the file.

| | |
| --- | --- |
| Repository | [`BlivionIaG/Qwen3.8-27B-refusal-cvec-GLP-49`](https://huggingface.co/BlivionIaG/Qwen3.8-27B-refusal-cvec-GLP-49) |
| File | `Qwen3.8-27B-refusal-cvec-GLP-49-L10-58-a1.gguf` |
| Directions digest | `9a50f7160fa2f0a41e7fa868c791d372b0a374f9365cc98c25d52abe96ce9d8d` (`glp.content_sha256`) |
| Layers | 10 to 58, zero-based, 49 of them |
| Base model | `TensorFold/Qwen3.8-27B-MLX-4bit` at `22d8d538` |
| Licence | Apache-2.0, the licence of the code that derived it |

The directions are reproducible: the producer writes the same ones from the same
checkpoint and the same prompt set, and only the timestamps in the file differ.
The digest is over the tensor bytes, so it is the same for every derivation of
this vector, and it is the number to compare a file against.

The reference implementation's own validator reads the published file and
reports nothing wrong with it, the same as it reports for the published file
this one is compared against:

```
captain-vector --validate Qwen3.8-27B-refusal-cvec-GLP-49-L10-58-a1.gguf
  ...: 0 FAIL, 0 WARN
```

The format, its reference implementation and that validator are Matt Suiche's,
the `weightless` plugin; the published file this one is compared against is
his too, and the card that ships with it says so.

## The integration test

A file that one implementation accepts has to mean the same thing to another.
The reference implementation for this format is the `weightless` plugin's
container reader, and this repository does not vendor it: the test takes a
checkout and reads it as a package.

```
zig build tf-glp-check
python3 tools/zig/glp_parity.py --reference PATH/TO/weightless \
    --vector vector.gglp --vector PATH/TO/a-published-vector.gguf
```

Three legs, over the same files:

| Leg | What it checks |
| --- | --- |
| read | The reader in Zig and the reference reader on one file: the layer ids, the metadata and every direction's bytes are equal. |
| rewrite | The file read in Zig and written out again, read by the reference: the numbers, the mode, the hook point and the strength survive the round trip. |
| digest | The file's own `glp.content_sha256` over the tensor payloads, hashed the same way on both sides. |

The reader that does the Zig half is `tf-glp-check`, which prints the reader's
view of a file and writes its directions out as bytes, so the two sides can be
compared exactly rather than approximately.

The test needs `numpy` for the reference reader, and a `weightless` checkout.
Nothing of the reference is committed here, and no vector from it is either:
the published files are downloaded when the test runs.
