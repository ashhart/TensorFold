# Control vectors

A control vector steers a model's residual stream at inference time. It is a
direction per layer: at the output of a block, the direction's component is
taken out of the residual stream, or the direction is added to it as a bias.
Nothing is written to the checkpoint, and a server without a vector serves the
model exactly as it was.

TensorFold reads these vectors on Metal for Qwen3.8-27B. The container, the
kernel and the serve surface are all native Zig. The engine ships the mechanism
only: a vector file is supplied by the operator, and the repository carries no
derived direction files.

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

`tf-glp-check rewrite FILE OUT.gguf --mode add` writes the same directions as
an `add` file, so the two modes can be compared on one vector.

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
| One layer is named twice | `direction.5` and `direction.05` both parse as layer 5; a file has one direction per layer. |
| A layer id is past the checkpoint's depth | The checkpoint's `num_hidden_layers` is read at startup, so the file is refused before the model loads. |
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
adjacent layers' directions are highly correlated and a stack shifted by one
layer still produces plausible output.

The file says so itself. A vector for this model can carry its layer list in
its metadata:

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
prefill takes 43.69 s without a vector and 43.68 s with one across 49 steered
layers; decoding runs at the same 14.6 tokens a second either way. The steering
path is not a measurable cost of a forward pass, because it no longer is a path
of its own.

## The file

The container is GGUF v3, read by the engine's own GGUF reader. The metadata
follows the GLP standard:

| Key | Meaning |
| --- | --- |
| `glp.spec_version` | 1. |
| `glp.mode` | `project` or `add`. |
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
Nothing of the reference is committed here, and no vector is either: the file
is supplied when the test runs.
