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
| The base model does not match | The vector was derived from another checkpoint. |
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

The producer reports the screen in its receipt, so the fraction is a decision
the operator makes on the checkpoint in front of them rather than a habit
carried from another one.
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
| Published GLP-49, layers 10-58, strength 1.0 | 19/30 | 6/30 | 60/60 |
| Our vector, last-row capture, layers 10-58, strength 1.0 | 18/30 | 7/30 | 60/60 |
| Our vector, masked at 0.005, strength 1.0 | 0/30 | 0/30 | 60/60 |

A strength of 1e-6 reproduces the baseline: the path is a no-op at zero.

The published vector is `msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49`. It is
downloaded and run here; it is not vendored and it is not committed. The last
two rows are vectors this repository's own producer derives from the model
itself, on our own prompt set.

The masked vector is the one to read. It takes every refusal off and takes no
harmless prompt with it: the refusal direction, with the dimensions the
massive activations dominate left out of it, is a direction at the refusal
rather than a direction at the stream's offset.

The unmasked vector is the same derivation without that exclusion, and it
shows what the exclusion is worth: it removes two refusals in three and costs
seven harmless refusals in thirty. The direction that carries "I will not"
also carried "I will", because it was pointing at the wrong thing.

That the replies stay coherent in every arm matters more than the rates: a
vector that silences the model is not a refusal removal, so read the replies,
not the rate.

The benign column is the price in the arms that pay it, and it is not small:
six of the thirty harmless twins are refused under the published vector, and
seven under the unmasked vector, against none without one. Our prompt set is
deliberately tight, so a twin differs from its harmful counterpart in intent
alone. That is the strength's dial, and the reply text is the measure.

The vector's strength has a ceiling. Past it the model stops answering: the
residual stream loses more than the direction it was asked to lose. Take the
strength from the file's own `glp.alpha_default`, and read the replies.

## The file

The container is GGUF v3. The metadata follows the GLP standard:
| Key | Meaning |
| --- | --- |
| `glp.spec_version` | 1. |
| `glp.mode` | `project` or `add`. |
| `glp.rank` | 1. |
| `glp.hook_point` | `residual_stream_post_layer`. |
| `glp.alpha_default` | The strength the file applies with by default. |
| `glp.method` | How the numbers were made, for example `difference_of_means_last_token_per_layer`. |
| `glp.contrast` | What the two prompt sets contrast. |
| `glp.base_model`, `glp.base_revision` | The checkpoint the vector came from. |
| `glp.layer_ids_zero_based` | The layers the file holds. |
| `glp.content_sha256` | SHA-256 over the tensor bytes, which the reader checks. |

The tensors are `direction.N`, fp32, one per layer, each the width of the
residual stream.

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

