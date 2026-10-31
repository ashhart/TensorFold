# MiniMax-M3.1-Flash on CUDA (two DGX Sparks)

The `minimax_m3` family's CUDA backend serves [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
at tensor parallel over hosts from a published EXL3 pack read as published: its trellis projections and routed experts
through the shared EXL3 kernels, the rest as stored. The release's DSpark drafter and vision tower (copied verbatim into
the pack), tools, structured output, thinking on or off, kept prompt states and two streams are served. Rank 0 serves
HTTP; the other ranks follow its steps. It shares the family's config, tokenizer handling, chat template and tool-call
markup with the [Metal backend](minimax-m3.1-flash.md).

The model is MiniMax's, under the [MiniMax Community License](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/main/LICENSE):
free for non-commercial use; commercial use needs a notice to, or above a revenue threshold an authorization from,
MiniMax.

## The checkpoint

Two GB10s (128 GB each) do not hold the release whole (its routed experts are 216.4 GiB NVFP4), so the backend serves
[Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw](https://huggingface.co/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw)
at revision `c33e7a6619f0f12ac99098e07ec5e56acb7eeda9`: exllamav3 1.6.1, the `mul1` codebook, attention q/k/v/o at 6
bits, the dense and shared MLPs at 5, every routed expert at 3, the head at 8; the indexer's projections fp16, the
router fp32, embeddings and norms bf16. This is a lossy third-party quantization of the release, not MiniMax's own
weights: its fidelity against the release is measured below. The release's own MXFP8 and NVFP4 tensors are not served on
CUDA (they need four GB10s; the loader takes EXL3 packs only).

```sh
hf download Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw --revision c33e7a6619f0f12ac99098e07ec5e56acb7eeda9 \
  --local-dir ~/models/mm-exl3   # on every rank
```

The loader checks `quantization_config` (`quant_method` "exl3", `codebook` "mul1", `version` 1.x) and every projection's
trellis, marker and scales as it reads them. The indexer's fp16 weights load as bf16 exactly (each is a dequantized
e4m3 value times a power of two; the loader refuses one that bf16 does not hold). A rank takes whole key heads, and each
MLP's and expert's intermediate and the vocabulary in 128-column blocks (EXL3 tiles and rotations): at TP=2 85.3 GiB a
rank, the vocabulary split 782 and 781 blocks.

## Run

The family is admitted only as it was run: on GB10 (sm_121) at two ranks. Any other GPU is refused
(`MiniMaxM3RequiresSm121`), and so is one rank or any other world (`MiniMaxM3RequiresTwoRanks`). `TF_MINIMAX_RANK`
(0 or 1) of `TF_MINIMAX_WORLD=2` places each rank, and `NCCL_COMM_ID` names rank 0's address and port on the link
between the hosts. With 192.0.2.1 as rank 0's address on their direct link, the same folder on both:

```bash
export NCCL_RUNTIME_CONNECT=0 NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0   # on GB10 pairs
# on rank 1
TF_MINIMAX_RANK=1 TF_MINIMAX_WORLD=2 NCCL_COMM_ID=192.0.2.1:29581 TENSORFOLD_MEMORY_RESERVE_GIB=3 \
  tensorfold serve ~/models/mm-exl3 --context 32768 --parallel 2 --prompt-cache-gib 3 --vision
# on rank 0 (192.0.2.1)
TF_MINIMAX_RANK=0 TF_MINIMAX_WORLD=2 NCCL_COMM_ID=192.0.2.1:29581 TENSORFOLD_MEMORY_RESERVE_GIB=3 \
  tensorfold serve ~/models/mm-exl3 --context 32768 --parallel 2 --prompt-cache-gib 3 --vision
```

Set `NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA` for the link as NCCL documents them; each rank keeps a TCP line to rank 0
on the port after `NCCL_COMM_ID`'s, so a rank that fails stops the others. On two ranks the layers' small gathers go by
RoCE writes (as GLM's); `TF_ROCE=0` leaves them to NCCL and `TF_ROCE_MAX_KB` caps a rank's bytes.

- The key/value cache is bf16, exact; a token takes 75 KiB a rank at TP=2 across the 60 layers (keys and values of the
  rank's two key heads, the shared index key).
- Drafts: the release's DSpark head (`dspark/`, MXFP8 as released), on unless `--no-drafts`; replies are identical
  with drafts off.
- The vision tower runs on the last rank; the others receive its rows. Its patch embedding is fp32 as stored. An image
  takes at most 576 tokens (672 x 672, the release processor's `max_pixels`); no video.
- Kept prompt states start at 2,048-token chunk boundaries (`--prompt-cache-gib`).

## How it computes

- Projections: `cuda.exl3.Dense` (q, k and v by columns, o by rows; gate and up by columns, down by rows; the head by
  columns), each rank's fp32 partials summed in rank order. The indexer's projections are one bf16 lane weight.
- Routed experts: the shared grouped EXL3 kernels (`cuda.exl3_grouped`), with SwiGLU-OAI as their `act_mode` 2:
  (clamp(up) + 1) g sigmoid(1.702 g), g = min(gate, 7), in bf16 steps as transformers and the Metal backend round it.
  A prompt chunk and a decode window take the same path (every pick's down rows in fp32), so a row's bits do not depend
  on the window it arrives in.
- Attention: each index head selects its key head's blocks (the MSA paper's top-k per GQA group): the block's max of
  index scores in fp32, the row's own block always kept, every block while a row sees at most 16. Index scores and
  attention run on the tensor units for prompt chunks and decode windows alike (`tf_mm3_index_scores_tc`,
  `tf_mm3_attn_prompt_tc`: fp32 online softmax, bf16 probabilities into P.V), so drafted replies equal plain ones and a
  resumed prompt equals a fresh one.
- Drafts: DSpark's block over the anchor's 4,096-key window, the Markov walk kept on the device (one host read a
  block). Several streams' blocks share one pass (`Drafter.blocks`, as many as a 16-row decode window holds: two
  blocks of 7): their rows go through the projections, MLPs and head together, so the drafter's weights are read
  once, while each block attends to its own stream's cache and walks its own picks. The lane costs (`cuda_lanes.zig`)
  were timed on two GB10s, every rank at once after the load: verify windows of 1 to 32 rows at positions 1k and 8k
  (their mean, never less than a narrower window's) and a draft block.
- Rounding follows transformers' and the Metal backend's (Gemma norms rounded once, the residual rounded to bf16 after
  every add, cos and sin rounded to bf16, the rotary frequencies' power rounded once to fp32 as torch and Metal's
  `precise::pow`), with one difference: the MoE sums its weighted picks in fp32 and rounds once, where transformers
  rounds after every pick.

## Checks (two DGX Sparks, GB10)

- Fidelity against the release: 2,406 tokens of a three-turn chat (prose, code, math, German, Chinese, JSON),
  teacher-forced. The served engine's logits for every row (`tf-minimax-cuda ... rows` at TP=2) against the release's
  (`tools/zig/minimax_m3_ref.py --logits`: its MXFP8 and NVFP4 decoded exactly, in transformers' order), compared by
  `tools/zig/minimax_m3_kl.py`: KL(release || served) mean 0.0788, median 0.0076, p99 0.97; top-1 agreement 89.90%
  (243 misses, each listed with both sides' top contenders and margins in the tool's report). Rows past 2,048 keys
  (the indexer selecting): KL 0.0674, top-1 89.66%. For scale, the release itself with the MoE's picks summed in fp32
  (this engine's order, `--fp32-picks`) against transformers' order: KL 0.0120, top-1 97.13%.
- Served (recipe harness, TP=2): soak 60/60, concurrent 12/12, streaming 4/4, multi-turn 8/8, stop 3/3, max-tokens 3/3,
  thinking switch 10/10, copy 2/2; DSpark drafted == plain 12/12; structured output 18/18 (drafted == plain 12/12); tool
  calls 5/5; seeded sampling 9/9; a resumed turn equals the same turn cold after a restart. Images: shapes, text and
  counting read right, and 16 image requests at once equal each alone.
- Decode with DSpark drafts, one stream: prose 24.3-24.4, code 38.1-39.4, structured 27.6-29.7, list 29.8-30.4 tokens a
  second, first token 0.60-0.65 s; two streams 31.0, 44.9, 35.7 and 52.0 in total.
- Kernel contract: gates 1-4 for every MiniMax kernel (`tf-cuda-test minimax-contract`): the attention and indexer forms
  and the glue (norms, q/k preparation, SwiGLU-OAI, router and routing, the MoE partial, picks, DSpark's Markov head, the
  tower's RoPE and patch embedding, the copies), with planted controls red. The shared experts' `act_mode` 2 is checked
  against float64 (also with the gate driven past the clamp) and for row invariance (`tf-cuda-test exl3`), with two
  wrong formulas as controls.
- Shared code: only `tf_exl3_plan_finish` changes in the `exl3_experts` image (SASS of the other 11 symbols identical
  on sm_80, sm_86, sm_120 and sm_121). Qwen3.5-MoE's EXL3 checkpoint (Ornith-1.5-122B, the image's other grouped user) on
  one GB10 gives byte-identical greedy replies before and after, at the same speed.

## Not covered

- Only TP=2 on two GB10s was run (TP=1 needs a GPU holding about 160 GiB; TP=4 was not tried).
- The release's MXFP8 and NVFP4 tensors are not read on CUDA; only EXL3 packs.
- Video is not served. With two images in a prompt, a borderline yellow can be named orange or green (single images read
  it right; an earlier build on a mix of the release's NVFP4 and 3-bit experts did the same).
- Quality suites (thinking off and on) were not run on this build.
