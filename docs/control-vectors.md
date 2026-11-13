# Control vectors

A control vector steers a model's residual stream at inference time. It is a
direction per layer: at the output of a block, the direction, scaled by the
strength, is added to the residual stream as a bias. Nothing is written to the
checkpoint. Steering is disabled unless the server starts with `--glp`.

TensorFold reads these vectors on Metal for Qwen3.8-27B. The container, the
kernel and the serve surface are all native Zig. The engine ships the mechanism
only: a vector file is supplied by the operator, and the repository carries no
derived direction files.

## Serving one

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

## What the server refuses

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
| A CUDA or ROCm backend | Only the Metal engine steers. |
| Any other model family | Only Qwen3.8-27B takes a control vector. |

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

## The file

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

The test needs `numpy` for the reference reader, and a `weightless` checkout.
Nothing of the reference is committed here, and no vector is either: the file
is supplied when the test runs.
