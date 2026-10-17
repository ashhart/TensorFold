# Drafter releases

A drafter release is a folder that a server can check against the target it loaded before it drafts a single token.
It says which target the drafter was trained against, how that target turns tokens into rows and rows into logits,
which of the target's rows the drafter reads, and the drafter's own shape. The format is version 1. It describes one
architecture, the block drafter, and is read by `zig/src/core/drafter_release.zig` and `drafter_layout.zig`.

A release holds three files:

| File | Contents |
| --- | --- |
| `config.json` | the manifest below |
| `model.safetensors` | the drafter's weights, plus the target's embedding and head rows it uses |
| `draft_vocab.json` | the target token ids the drafter can propose, in the order of the head's rows |

The embedding and the head rows ship in the release, so the drafter never reads the target's weights in their stored
format. For Kolibri-1 in bf16 that is the full embedding (128,000 x 2,560, 655 MB) and a 60,614-row head slice (310 MB).

## config.json

Kolibri-1's block drafter in this format:

```json
{
 "format": "tensorfold-drafter",
 "version": 1,
 "architecture": "block-drafter",
 "target": {
  "repo": "Aleph-Alpha/Kolibri-1",
  "revision": "e52eb4627d11516b0c01de49210ab5a4e4061444",
  "layers": 50,
  "hidden": 2560,
  "vocab": 128000,
  "embed_scale": 1.0,
  "tied": false,
  "softcap": null,
  "head_bias": false
 },
 "taps": {"ids": [45, 48, 50], "stream": "residual", "normed": true, "dtype": "bf16", "row_bytes": 15360},
 "drafter": {
  "hidden": 2560, "heads": 20, "head_dim": 128, "ffn": 6144, "layers": 4, "block": 4, "window": 2048,
  "eps": 1e-06, "theta": 10000.0, "pred_rank": 256, "spine_rank": 256
 },
 "draft_vocab": {"file": "draft_vocab.json", "size": 60614},
 "weights": "model.safetensors",
 "serving": {"depth": 2, "body": "fp8", "head": "nvfp4"}
}
```

Every field is required. An unknown field, a missing one, or a newer `version` is refused, never read in part.

### taps

The rows the drafter reads. They are what the target's `features` gives (`lanes.Features`) and what `--record`
writes to a stream's `.st` file, so a recording and a release can be compared field for field.

| Field | Meaning |
| --- | --- |
| `ids` | tap k is the stream after k of the target's layers, rising. k = `layers` is past the last layer. |
| `stream` | which residual stream; `residual` for a family with one. A family with several streams adds its merge here. |
| `normed` | true: the row as the next norm reads it (layer k's input norm, or the final norm at k = `layers`) |
| `dtype` | `bf16`, `f16` or `f32` |
| `row_bytes` | the taps in `ids` order, each `target.hidden` wide: `len(ids) x hidden x size(dtype)` |

A recording's `.json` carries `taps`, `row_bytes` and `dtype`. `checkRecording` refuses one whose values differ from
the release's. The `.json` does not say `stream` or `normed`: the family defines both once for its tap ids, and the
release states them.

### target

How the target turns tokens into rows and rows into logits. A drafter that uses the target's embedding and head
does the same, so these values change what it must predict.

| Field | Meaning |
| --- | --- |
| `repo`, `revision` | the checkpoint the drafter was trained against |
| `layers`, `hidden`, `vocab` | the target's layer count, row width and vocabulary |
| `embed_scale` | rows = embed[token] x `embed_scale` (1.0 when the target does not scale) |
| `tied` | the head is the embedding. The release then ships no `target.head`. |
| `softcap` | logits = softcap x tanh(logits / softcap); `null` when the target has none |
| `head_bias` | the head adds a bias. The release ships it as `target.head_bias`, one value per draft id. |

`check` compares `layers`, `hidden`, `vocab`, `embed_scale`, `tied`, `softcap` and `head_bias` with the values the
loaded target reports, and refuses the release on the first one that differs, naming it.

### drafter and serving

The block drafter's shape. Row 0 of a block is a chain head on the anchor's own row; rows 1 to `block - 1` are drafted
in the same pass. `pred_rank` and `spine_rank` are 0 when the predecessor head or the spine is absent. `hidden` must
equal `target.hidden`, because the target's head reads the drafter's rows. `serving` holds defaults a server may
override: `depth` (1 to `block`), and the formats the body and the head slice are converted to at load (`bf16`, `fp8`
or `nvfp4`).

## model.safetensors

Every tensor is bf16, f16 or f32. D is `drafter.hidden`, T is `target.hidden`, A is `heads x head_dim`, V is
`draft_vocab.size`. A tensor missing, misshaped, or not in this table is refused.

| Tensor | Shape | When |
| --- | --- | --- |
| `target.embed` | [target.vocab, T] | always |
| `target.head` | [V, T] | `tied` is false |
| `target.head_bias` | [V] | `head_bias` is true |
| `fuse.weight` | [D, len(taps) x T] | always |
| `fc.weight` | [D, T + D] | always: cat(embedding of the token after the row, fused row) |
| `mask` | [D] | always |
| `layers.i.{q,k,v,ck,cv}.weight` | [A, D] | each layer |
| `layers.i.o.weight` | [D, A] | each layer |
| `layers.i.{gate,up}.weight` | [ffn, D] | each layer |
| `layers.i.down.weight` | [D, ffn] | each layer |
| `layers.i.{n1,n2,nc}.w` | [D] | each layer |
| `out_norm.w` | [D] | always |
| `pred_h.weight`, `pred_e.weight`, `pred_out.weight` | [R, D], [R, T], [D, R] | `pred_rank` R > 0 |
| `spine_n.i.w`, `spine_a.i.weight`, `spine_u.i.weight` | [D], [S, D], [D, S] | `spine_rank` S > 0, i < layers - 1 |

## draft_vocab.json

A JSON array of `draft_vocab.size` distinct ids below `target.vocab`. Row r of `target.head` is the head row of id r.

## Two targets

| | Kolibri-1 | Nemotron 3.5 Lightning |
| --- | --- | --- |
| checkpoint | `Aleph-Alpha/Kolibri-1` @ `e52eb462` | `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` @ `d9d758fb` |
| layers, hidden, vocab | 50, 2,560, 128,000 | 52, 2,688, 131,072 |
| taps | 45, 48, 50; normed; bf16; 15,360 bytes a row | 52 (the final norm's rows); normed; bf16; 5,376 bytes a row |
| `embed_scale` | 1.0: the embedding is gathered as stored | 1.0: the embedding is gathered as stored |
| `tied` | false: `tie_word_embeddings: false`, its own `lm_head.weight` | false: `tie_word_embeddings: false`, its own `lm_head.weight` |
| `softcap` | null: the config names none, the head is one matmul | null: the config names none, the head is one matmul |
| `head_bias` | false: no `lm_head.bias` | false: no `lm_head.bias` (`lm_head.biases` is the 4-bit format's offsets) |

The values come from each checkpoint's `config.json` and tensor index, and from the forward passes that read them.
Nemotron's tap is the one `--record` writes on CUDA, the rows its head and MTP head read. Kolibri's ids count layers
the same way: its drafter reads the outputs of layers 44, 47 and 49 (counted from 0), each normed by the norm after it.

Our Kolibri drafter checkpoint (`josch15366/Kolibri-1-block-drafter`, research only, never shipped) predates this
format. Its `config.json` counts taps from 0 (44, 47, 49), has no adapter, and its weights hold an unused `inp.weight`
and neither target tensor. Re-exported with the embedding and its 60,614 head rows, and without `inp.weight`, its 66
tensors pass the layout check above.

## Not in version 1

- Weights stored as FP8 or NVFP4. The release stores floats; `serving` converts them at load.
- A tap that merges several residual streams. `stream` has one value until a family needs another.
- Drafter architectures other than the block drafter.
