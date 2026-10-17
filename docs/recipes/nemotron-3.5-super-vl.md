# Nemotron 3.5 Super VL

NVIDIA's Nemotron 3.5 Super VL 120B-A12B (`nemotron_h_omni`) runs on the native CUDA engine from NVIDIA's NVFP4
checkpoint, on one GB10 or split over two. The text model is Nemotron-H with a latent MoE: 88 blocks (40 Mamba-2,
8 attention, 40 MoE), 512 routed experts with 22 picked a token, each reading and writing a 1,024-wide latent between
two bf16 projections, a shared expert on the hidden rows, and one MTP layer. The family is
`zig/src/families/nemotron/` (the latent MoE in `cuda_latent.zig`); images go through the checkpoint's RADIO tower,
described in [vision](../vision.md#nemotron-35-super-vl-on-cuda).

## Checkpoint

`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` at revision `82d93699acd3be9e4617c53360fe5b3f5dbeb1b3`
(75.1 GB). Its ModelOpt export stores the experts in NVFP4 (e2m1 codes, an e4m3 scale every 16 inputs, one fp32 scale
an expert), the Mamba, attention and output-head projections in FP8 with static input scales, and the router, latent
projections, embeddings and the MTP layer's attention in bf16. The engine reads these tensors as stored; nothing is
converted ahead of time.

## Run on one GB10

```bash
tensorfold serve /models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 --name super --vision
```

The weights take 70.8 GiB of the GB10's memory (73.2 GiB with the image tower and its rows). At the default
16,384-token window the server admits 8 streams at 0.35 GiB each and keeps a prompt cache of about 8 to 10 GiB, and
about 24 GB of the GB10's memory stays available. Leave the other ranks' variables below unset.

For CUDA MTP streams, `--mtp-drafts` caps MTP depth (at most 31 for Super) and `--mtp-confidence` sets a
cumulative-confidence threshold from 0 to 1 instead of the cost-based cutoff. A zero draft cap disables
drafting. The same controls apply in lone and shared rounds, including resumes and handoff.
Positive MTP caps leave copied continuations and Recall on their existing row limits.

## Run on two GB10s

Rank 0 runs the whole model and serves; rank 1 holds only its share of every MoE layer's experts (the MTP layer's
too) and computes it for rank 0's jobs. The ranks meet over NCCL, which the engine opens at run time from the system
(`libnccl.so.2` on the library path); start rank 1 first, with the same binary and checkpoint on both hosts:

```bash
# on rank 1
TF_NEMOTRON_RANK=1 TF_NEMOTRON_WORLD=2 NCCL_COMM_ID=192.0.2.1:29741 \
  tensorfold serve /models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
# on rank 0 (192.0.2.1)
TF_NEMOTRON_RANK=0 TF_NEMOTRON_WORLD=2 NCCL_COMM_ID=192.0.2.1:29741 \
  tensorfold serve /models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 --name super --vision --host 0.0.0.0
```

Set `NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA` for the link between the hosts as NCCL documents them. Rank 1 keeps a TCP
line to rank 0 on the port after `NCCL_COMM_ID`'s (29742 here), opened before NCCL. Before any weight loads, both ranks
check that they run the same build, model, cut and precision and the same checkpoint layout: config.json, the index,
and each shard's name, size and header. While the weights load, each rank reads and hashes every byte of every shard
it holds (sha256), and checks each file against its Hugging Face blob name, which is its sha256, when the folder is a
Hugging Face snapshot. The ranks exchange those digests before rank 0 serves, so a copy that differs anywhere, by one
bit, stops both ranks before serving, naming it. A GB10 hashes the 70 GiB checkpoint in about 22 s, at the same time as
it loads its weights. If either rank fails, it sends its error's name and the other stops with it; if either process
dies, the other stops within about a second instead of waiting; rank 1 leaves when rank 0's server stops.
Rank 0 takes 40.1 GiB (42.2 GiB with the tower) and rank 1 31.4 GiB. One prompt pass runs at a time at TP=2
(`--segments` above 1 is refused).

## How the experts split

Every expert's width is cut into two shares. A rank sums the shares it holds in share order and the ranks meet in
rank order, so one GB10 and two give the same bits: on the eight release prompts, all 17 replies (greedy, seeded
sampled and one with thinking) were the same tokens at TP=1 and TP=2. A routed pick carries its share in its pair id,
so one grouped launch serves every share a rank holds.

Passes of up to a verify window's rows read shared-expert plans built when the buffers load, one per row count;
longer prompt passes and the other rank's jobs plan the shared expert at their own row count.
On one GPU those passes run the shared expert on the side stream while the router and routed experts run on the main
stream: its activations, quantized rows and plan are its own, and events order its input and the final sum.

## Precision

On Blackwell the engine runs the checkpoint's own math by default (`--precision checkpoint`): e4m3 rows into the FP8
layers and NVFP4 rows into the NVFP4 experts, on the FP8 and block-scaled FP4 tensor cores. `--precision full` keeps
bf16 rows throughout (W4A16 and W8A16). Each is exact on its own terms; they give different tokens from each other.
On one GB10, `checkpoint` teacher-forced on `full`'s greedy 256 tokens after three prompts:

| Prompt | Mean KL(full \|\| checkpoint), nats | Same top token | NLL of the fed tokens, full to checkpoint |
| --- | --- | --- | --- |
| Python code | 0.0036 | 255 of 256 | 0.0480 to 0.0482 |
| Long prose | 0.0137 | 246 of 256 | 0.1416 to 0.1514 |
| Repetition | 0.0057 | 255 of 256 | 0.0179 to 0.0252 |
| All 768 tokens | 0.0077 | 98.4% | 0.0692 to 0.0749 |

`checkpoint` read a cold 8,192-token prompt in 4.41 s against `full`'s 7.96 s (1.81x; 1.82x at 2,048 tokens) and
decodes at about the same speed. The [quantization guide](../quantization.md#nvidias-nvfp4-checkpoints-on-cuda)
describes both modes. Both ranks of a two-GB10 setup must run the same precision; they check it before loading.

## Exactness and speed

Served on GB10s at a 16,384-token window, the eight release prompts at 256 tokens each, every cell held in both
precisions: drafted replies equal `"draft": false` (greedy, seeded sampled, thinking on), 2 and 4 concurrent streams
equal their solo runs, and three-turn chats with regenerate, edit and delete equal a cache-off server's replies.

| Setup | Precision | MTP drafts (greedy, mean of 8 prompts) | `"draft": false` | Cold 8,192-token prompt |
| --- | --- | --- | --- | --- |
| One GB10 | `checkpoint` (default) | 38.2 tok/s (29.8 to 49.0) | 18.8 tok/s | 4.41 s |
| One GB10 | `full` | 38.7 tok/s (30.2 to 50.8) | 19.4 tok/s | 7.96 s |
| Two GB10s | `checkpoint` (default) | 46.8 tok/s (35.9 to 63.6) | 20.2 tok/s | |
| Two GB10s | `full` | 48.1 tok/s (36.4 to 65.6) | 21.4 tok/s | |

Four concurrent greedy streams on two GB10s took 6.2 s together against 11.8 s one at a time. The MTP layer drafts
with the target's own output head over the whole vocabulary.

## Images

`--vision` loads the checkpoint's RADIO tower and projector (see [vision](../vision.md#nemotron-35-super-vl-on-cuda)).
Super VL's `--vision-image-tokens` defaults to 13,312, so up to four images keep the size NVIDIA's processor gives
them (at most 3,328 tokens each); the flag given explicitly still sets the budget. One request with four full-size
pictures (3000 x 1340, 13,124 prompt tokens) on one GB10 answered its first token in 11.45 s, its drafted reply
equal to `"draft": false`, with device memory at 74.4 GiB and 22.9 GB of the GB10 still available. Four full-size
images take most of the default 16,384-token window, so raise `--context` for long replies after them.

## Not served yet

- `--slide` is refused for this checkpoint, and `--learn-conversations` keeps nothing on disk for it yet (its prompt
  cache stays in memory).
- Video and audio parts are refused; the checkpoint's video embedder and sound encoder stay unloaded.
