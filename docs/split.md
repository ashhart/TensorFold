# Split: one model over a Mac and a CUDA machine (experimental)

A split serves one model whose layers are divided between two machines: a Mac runs the embedding, layers
`[0, K)`, the head, sampling and the drafter; a CUDA machine (a DGX Spark, say) runs layers `[K, N)` and the
final norm. It is for checkpoints that do not fit either machine alone. Every prompt chunk and decode window
crosses the link once each way as activation rows; the weights cross it once, on first use.

```bash
# on the CUDA machine
tensorfold stage --listen 0.0.0.0:18640 --cache ~/.cache/tensorfold/stage

# on the Mac
tensorfold serve MODEL --split SPARK_HOST --split-layers K
```

The Mac is the server: it holds the HTTP endpoint, the tokenizer, the scheduler and the only copy of the
checkpoint. The stage needs no checkpoint of its own.

## How a session works

1. The Mac groups the stage's tensors into **units**, one per layer plus one for the final norm. A unit's key
   is a hash over its tensors' names, dtypes, shapes, byte ranges and their shards' sha256 (a Hugging Face
   blob's name, or computed once and kept beside the path, size and mtime). Nothing is hashed twice.
2. `hello`: the Mac sends the config, the layer range and the units. The stage answers with the keys it lacks.
3. The Mac streams those units' bytes, read straight from its safetensors shards. The stage writes each unit
   as one safetensors file named by its key (data page-aligned) and builds a stage directory of links to them,
   which the family's CUDA loader reads with `layers=(K, N)`: no embedding, no head.
4. The stage loads the layers (or keeps the ones it already holds for that directory) and says `ready`.
5. Rows flow until the Mac disconnects. A new connection always replaces the session in place, so a Mac that
   crashed and restarted is never refused; requests carry the session's epoch and a stale one is refused.

Changing `K` re-sends only the layers the stage has not seen; restarting the Mac re-sends nothing and reloads
nothing.

## The rows path

Each request carries the Mac's residual and pending output projection at layer `K` (bf16 `[R, H]` each), the
commit path of the previous decode window, and the window's parents. The reply carries the final normed rows
(the Mac applies the head, so a row costs `H` values back rather than the vocabulary) and, when a DFlash2
drafter reads layers past `K`, those tap layers' rows.

- **Prompt chunks** run as pieces of `TF_SPLIT_PIECE` rows (512) in a two-stage pipeline: while the stage
  computes piece `j`, the Mac computes piece `j + 1`.
- **Decode windows** are serial (each needs the last one's accepted path). The accepted path rides on the next
  request, so a round costs one round trip.

Two transports: the control TCP connection (`--split-transport tcp`, the default), or a shared-memory mailbox
of an RDMA daemon (`--split-transport mailbox`; the stage takes `--mailbox NAME`, and both ends find the
mailbox's Python module through `TF_SPLIT_MAILBOX_PY`).

## Choosing K

Prefill favours the CUDA machine and decode favours the Mac's memory bandwidth. Measured on an M2 Ultra and
a GB10 with Qwen3.8-27B MLX 4-bit, DFlash2 drafts, RDMA mailbox:

| K (Mac layers) | Prefill 8k | Decode round (8 rows) |
| --- | ---: | ---: |
| 48 | 352 tok/s | 62 ms (Mac 43, stage 18) |
| 32 | 510 tok/s | 68.5 ms (Mac 32, stage 36) |
| 64 (Mac alone) | 280 tok/s | 53 ms |

A round costs about 13.7 ms + 0.62 ms per Mac layer + 1.1 ms per GB10 layer, so for a single stream the split
is a capacity feature, not a speed-up: the two halves run one after the other. Pipelining two streams through
the halves is the next step.

## Limits

- Qwen3.8 dense (`qwen3_5`) on the Mac's row decoder (GPUs without tensor units); the stage runs through the
  last layer.
- One stream at a time (`--parallel 1`); no stored prompt prefixes (`--prompt-cache-gib 0`); no images.
- The stage's arithmetic is the CUDA kernels', so replies equal this split's own serial decoding, not the Mac's
  one-machine replies.
