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

`direction.N` applies at layer `N`, and layer 0 is not steerable. The blocks
are one to the last one, so `direction.N` is the output of the N-th block: the
residual that enters block N. A file for a 64-block model may name layers 1
to 64.

This is worth stating because the alternative is silently wrong. The published
GLP-49 vector for this model removed no refusal at all through an
implementation that landed a direction one block late, and most of it through
one that did not:

| The vector's content applied at | Harmful prompts still refused (8 prompts) |
| --- | --- |
| One block late | 6 |
| **The block it names** | **2** |
| One block early | 5 |
| Two blocks early | 3 |

The ladder was measured at a fixed strength against the published vector, and
the corrected implementation reaches the same result with the vector
unshifted.

## Deriving one

```
tf-qwen27-steer-derive --model DIR --out FILE \
    --repo TensorFold/Qwen3.8-27B-MLX-4bit --revision REV \
    --capture last --layers 10-58
```

The producer runs a harmful prompt set and a form-matched harmless set through
the native forward, takes the difference of their means per layer, and writes
the file. The prompts are the repository's own
(`zig/src/families/qwen27/steer_prompts.zig`); no third-party corpus is read.

| Flag | Meaning |
| --- | --- |
| `--capture mean` | Read every row of a prompt and capture its mean. |
| `--capture last` | Take the last row alone, where the model decides what to say next. |
| `--layers FROM-TO` | Keep a window of layers; every layer by default. |
| `--prompts N` | Use only the first N pairs of each set. |
| `--alpha A` | The `glp.alpha_default` the file carries. |

`--capture last` is the one that matters. The last row of a prompt carries the
decision the model is about to make; the mean over every row is mostly the
prompt's content, and a direction built from it does not separate refusal from
anything else.

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
| Our vector, last-row capture, layers 10-58, strength 1.0 | 15/30 | 1/30 | 60/60 |
| Our vector, last-row capture, layers 10-58, strength 1.25 | 21/30 | 6/30 | 60/60 |

A strength of 1e-6 reproduces the baseline: the path is a no-op at zero.

At strength 1.0 the vector halves the refusal rate and costs one benign refusal
in thirty. The refusals it leaves are the ones a refusal direction does not
carry: the question asked for a weapon and the model answered anyway.

Past 1.0 the strength stops paying. At 1.25 the refusal rate climbs again and
the benign refusals with it: the stream is losing more than the direction it
was asked to lose, and the model starts refusing things it had agreed to. Read
the replies, not the rate.

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
