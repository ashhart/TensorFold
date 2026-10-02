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

- **Prompt chunks** run as pieces of up to `TF_SPLIT_PIECE` rows (512; at least four pieces a chunk, in
  multiples of 128) in a two-stage pipeline: while the stage computes piece `j`, the Mac computes piece `j + 1`.
  A split takes 16k-token chunks (`TF_SPLIT_CHUNK`): a chunk costs no memory beyond its pieces, and each chunk
  ends with the stage computing its last piece alone.
- **Decode windows** are serial (each needs the last one's accepted path). The accepted path rides on the next
  request, so a round costs one round trip.

Three transports:

- `tcp` (the default): the rows ride the control connection.
- `rdma`: a shared-memory mailbox between two `v41rpcd` daemons over RoCE, which TensorFold starts and stops
  itself. The stage, started with `--rdma DEVICE --rdma-dir DIR`, finds its RoCE address by IPv4 (a link flap
  renumbers GID indexes) and starts a fresh daemon for every Mac session, so a Mac that died without goodbye
  never leaves a connection that refuses the next one. The Mac (`--split-rdma-dir DIR --split-rdma-mac MAC`)
  first stops any daemon it left running, then starts its own toward the address and MAC the stage reports,
  taking the other host of the stage's point-to-point subnet unless `--split-rdma-ip` says otherwise; it stops
  its daemon through the daemon's control socket when the server exits, never with a signal.
- `mailbox`: the same mailbox between daemons started by hand (`--split-mailbox NAME`; `TF_SPLIT_MAILBOX_PY`).

```bash
tensorfold stage --rdma rocep1s0f1 --rdma-dir ~/v41
tensorfold serve MODEL --split SPARK_HOST --split-layers 35 --split-prefill-layers 16 \
    --split-transport rdma --split-rdma-dir ~/mcdma-dsv41/src --split-rdma-mac 98:03:9b:80:6a:94
```

Over RDMA the daemons come up before the weights move, and a push rides both links at once: units split by bytes
between the RDMA mailbox (48 MiB pieces; the Mac reads ahead while one crosses, the stage writes to disk on a
thread) and the control connection. Qwen3.8-27B's layers 16..63 (9.57 GiB) arrive in 5.3 s (15.6 Gbit/s) against
7.7 s over 10 GbE alone; the mailbox alone does no better than TCP, since one request at a time serialises the
copies on both ends.

Over RDMA a decode round's crossing costs about 0.6 ms (the stage computing a window takes 30.8 ms of the
Mac's 31.4 ms wait), and bulk replies run at 23 Gbit/s on a 40 GbE link.

## Prompts entering earlier

Prefill is compute-bound and favours the CUDA machine; decode is bandwidth-bound and favours the Mac. With
`--split-prefill-layers KP` (below `--split-layers K`) a fresh prompt enters the stage at layer `KP`: the Mac runs
layers `[0, KP)` of it, the stage `[KP, N)`. After each prompt chunk the Mac fetches the state of layers
`[KP, K)` the stage computed (each recurrent layer's conv tail and state, each attention layer's new keys and
values: 165 MiB for an 8k prompt of Qwen3.8-27B at K=35, KP=16, in about 0.1 s over RDMA), so decode windows
run at `K` as before and cost the same. The stage holds layers `[KP, N)`, so layers `[KP, K)` sit on both
machines. `TF_SPLIT_CHECK=1` makes the Mac compute those layers too and print how far the stage's state is
from its own (bf16 drift between Metal and CUDA, under 6% at the deepest layer).

## Choosing K

Prefill favours the CUDA machine and decode favours the Mac's memory bandwidth. Measured on an M2 Ultra and
a GB10 with Qwen3.8-27B MLX 4-bit, DFlash2 drafts, RDMA mailbox:

| Layers on the Mac | Prefill 8k | Decode round (8 rows) |
| --- | ---: | ---: |
| K=48 | 352 tok/s | 62 ms (Mac 43, stage 18) |
| K=35 | 488 tok/s | 73 ms |
| K=35, KP=16 | 1,019 tok/s | 74 ms |
| 64 (Mac alone) | 280 tok/s | 53 ms |

On the llama.cpp cluster sweep's prompts (cold, 64 greedy tokens), K=35 / KP=16 prefills 2.0-2.6x faster than
llama.cpp's 55/45 layer split of the same model (1,019 against 497 tok/s at 8k, 827 against 323 at 64k) and
decodes 1.4-2.4x faster than it with MTP drafts.

A round costs about 13.7 ms + 0.62 ms per Mac layer + 1.1 ms per GB10 layer, so for a single stream the split
is a capacity feature, not a speed-up: the two halves run one after the other. Pipelining two streams through
the halves is the next step.

### The MoE families

Qwen3.8 Flash Next (`qwen4_exp`, 48 layers) and GLM-5.3-Flash (`glm5_next`, 45 layers) split the same way,
drafting with their MTP heads. What crosses is the residual state the next layer reads: Flash Next's
hyper-connection streams (4 x 2560 values a row) both ways, the Mac mixing the returned streams for its head;
GLM's streams (4 x 4096) to the stage and its final-normed rows (4096) back, which the LM and MTP heads read.
Decode windows are chains. On the same sweep, K=24 for both (RDMA):

| Model | Prefill 8k | Prefill 32k | Decode 8k | Decode 32k | llama.cpp split, prefill / decode 8k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash Next, K=24 | 1,266 tok/s | 1,221 tok/s | 50.6 tok/s | 36.1 tok/s | 547 / 37.5 (MTP) |
| GLM-5.3-Flash, K=24 | 512 tok/s | 516 tok/s | 26.5 tok/s | 22.1 tok/s | 275 / 24.2 (DFlash2) |

GLM's Mac half is its prefill's bottleneck (about 37 ms a layer for a 512-row piece on an M2 Ultra, the stage
waiting on it), so moving layers to the stage helps until its memory runs out: K=30 prefills 415 tok/s, K=24
510, with 102 of the GB10's 121 GiB in use. Prompts entering the stage earlier (`--split-prefill-layers`) are
Qwen3.8 dense only for now.

An M2's Metal pipelines can cap a kernel's threadgroup below 1024 threads (896, 832 or 512 by its registers);
GLM's kernels find the widest group the GPU takes on their first launch (`kernels/glm/flash/v1/launch.py`),
and the fused hyper-connection boundary's prompt variant falls back to MLX ops where it does not launch.

## Limits

- Qwen3.8 dense (`qwen3_5`) on the Mac's row decoder (GPUs without tensor units), Qwen3.8 Flash Next
  (`qwen4_exp`) through its fused Metal kernels, GLM-5.3-Flash (`glm5_next`); the stage runs through the last
  layer. `--split-prefill-layers` is Qwen3.8 dense only.
- One stream at a time (`--parallel 1`); no stored prompt prefixes (`--prompt-cache-gib 0`); no images.
- The stage's arithmetic is the CUDA kernels', so replies equal this split's own serial decoding, not the Mac's
  one-machine replies.
