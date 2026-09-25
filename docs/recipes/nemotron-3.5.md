# Nemotron 3.5 Lightning 30B-A3B (`nemotron_h`)

Measured on an M5 Max with 128 GB and MLX 0.31.2, with the 4-bit checkpoint. Package:
`src/tensorfold/families/nemotron_h/`.

## What decides the speed

- 52 blocks: 23 Mamba-2, 6 attention, 23 MoE. Hidden size 2,688.
- Attention: 32 query heads over 2 KV heads, head dim 128.
- MoE: 128 routed experts, top 6, expert width 1,856, sigmoid scores with a correction bias, plus a shared
  expert of width 3,712 with a squared-ReLU MLP.
- Mamba-2: 64 heads of dim 64, state size 128, 8 groups, conv kernel 4.
- Vocabulary 131,072. 4-bit affine weights in groups of 64.
- mlx_lm's decode step runs about 900 small kernels a token and takes 7.3 ms, while reading the weights alone
  takes 3.7 ms. The problem here is kernel count, not bandwidth.

## What worked

Through the server with exact sampling (T 1.0, top-p 0.95), thinking on, tok/s:

| Step | Short chat | About 20k context | About 60k context |
| --- | --- | --- | --- |
| mlx_lm model, host sampler | 138 | | 119 |
| fused kernels, shared expert folded into the expert tables, one stacked q/k/v matmul | 159-163 | | |
| plus keyed sampling on the GPU, decoding one step ahead | 185-190 | | |
| plus the Mamba B/C conv computed once per threadgroup | 185-197 | | |
| plus 200-op command buffers and a narrower sampler candidate window | 196-205 | | |
| plus tensor-unit attention (head dim 128) from 10k keys | 200-203 | 175 | 138 |
| plus alternating KV buffers | 200-206 | 175 | 162 |

The pieces (`families/nemotron_h/kernels.py`):

- Fused kernels between MLX's matmuls: the residual add (with the MoE combine) plus the next block's RMSNorm;
  sigmoid routing with the correction bias to the top 6; the whole Mamba step (conv window, conv, SiLU, dt,
  state update, D skip, SiLU(z) gate); the output RMSNorm over groups of 512. About 370 kernels a token.
- The shared expert as two extra half-width experts with weight 1 (ids 128 and 129), so the MoE is one gather.
  With multi-row copy windows the shared expert as its own dense branch, read once however many rows a pass
  has, is better (drafted code rounds 201 to 216 tok/s), so that is the default.
- Each block's work between norms compiled with `mx.compile`, one trace per row count: host time 30 to 4 us a
  layer, same bits. The graph goes to the GPU every 8 layers.
- Tokens drawn on the GPU and kept there. Each round queues the next forward on the new token before reading
  it, so reading, streaming and building the next graph overlap the GPU (`engine/family_engine.py`,
  `gpu_tokens`).
- `MLX_MAX_OPS_PER_BUFFER=200`: MLX's default commits command buffers more often than this model's many small
  kernels need.
- The GPU sampler tries candidate windows of 20, 10 and 5 logit units below the row maximum and takes the
  widest one that holds the nucleus in at most 1,024 tokens, else a radix select. Both give the same
  candidates.
- Tensor-unit attention for head dim 128 (`kernels/lane_attention.py`): the 16 query heads of one KV head form
  one 16-row tile, so each key is read once. At 60k keys 0.323 to 0.194 ms a call, at 32k 0.191 to 0.120, but
  slower below about 10k keys because of its extra launches, hence the switch at 10k. It needs M5 tensor
  units and is gated by GPU generation.
- Alternating KV buffers (`engine/alternating_kv.py`): decoding one step ahead, step s+1 wrote into a cache
  buffer step s was still reading, so MLX copied the whole cache first, about 1 ms a token at 60k keys. Writes
  now alternate between two buffers. Long-context decode went from 138 to 162 tok/s.
- Quality: teacher-forced over 300 tokens against mlx_lm, top-1 agreement 0.967 and NLL 2.3523 against 2.3540.
  mlx_lm's own prefill and decode paths agree 0.953 with each other.

## Tried and rejected

- One kernel for norm, router and expert selection, where a single threadgroup reads the 0.69 MB router: 220
  to 141 tok/s.
- A block attention kernel that shares each key block across the 16 heads of a KV head without tensor ops:
  correct, but 0.73 ms against MLX's 0.51 at 60k keys.
- Heads as query rows through MLX's full attention: slower.
- Tensor-unit attention below 10k keys, and the folded shared expert while drafting.

## Exactness

- The fused kernels take R consecutive rows in order, and a row's value depends only on its own inputs. The
  arithmetic follows mlx_lm's (fp32 math, bf16 where mlx_lm stores bf16) but is TensorFold's own, and serial
  decoding runs through it.
- Projections, including the head, are MLX's quantized matmuls (strategy A in the [recipe book](README.md)).
  That is exact only because on this MLX and GPU a window of 2, 4 or 8 rows gives each row one-row bits. The
  model checks this at load from a 48-token prompt and turns drafting off if it fails. On an M3 Ultra with MLX
  0.32 the check probably fails, which would leave Nemotron correct but undrafted there.
- The router is TensorFold's own matvec: MLX's bf16 matmul summed 2 rows in a different order than 1, a logit
  one bf16 unit off at layer 29 of a real decode. Routing ties go to the lower expert id.
- Attention picks its kernel by each row's own key count. Below the 10k switch a multi-row window runs MLX's
  attention one query at a time over that row's keys.
- The Mamba step returns the conv and SSM states after every row, so dropping a rejected tail keeps the state
  after the last kept row; KV caches are trimmed.

## Next

- MTP drafting. The BF16 release carries an MTP layer (an attention block and a 128-expert MoE block) that the
  MLX conversion drops; `families/nemotron_h/mtp.py` converts it (`convert`) and the engine can draft with it
  (`TF_MTP_ROUNDS=1`). In-engine it measured 217 tok/s on prose and 216-228 on code; it is not on by default
  in the server yet. Chaining several MTP drafts with acceptance-driven depth, as Flash Next does, is untried.
- Tests for the fused kernels (row independence at real dims) and for the GPU sampler against its numpy
  reference (`gpu_sampling.reference`).
- At long context the cost is attention.
