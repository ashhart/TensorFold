# The recipe book

This is how we made four model families decode fast and byte-exact on Apple Silicon, and how to do the same
for yours. The per-family pages record what worked with the numbers we measured, what we tried and threw
away, and what we would try next:

- [Qwen3.8 Flash Next](qwen3.8-flash-next.md): 512-expert MoE, hyper-connections, sparse attention and an
  MTP head, on an M3 Ultra.
- [Nemotron 3.5 Lightning](nemotron-3.5.md): Mamba-2 and MoE, decoded one step ahead, on an M5 Max.
- [Qwen3.8-27B dense](qwen3.8-27b.md): DeltaNet and full attention with a DFlash2 draft model and tensor-unit
  kernels, on an M5 Max.
- [GLM-5.3-Flash](glm-5.3-flash.md#apple-silicon-mlx): Kimi delta attention, sparse MLA and 288-expert MoE with
  an MTP head, on an M3 Ultra.
- [Adding a family](adding-a-family.md): the package interface, the checks and the tests.

On NVIDIA GPUs (DGX Spark), the same contract with CUDA kernels:

- [The CUDA recipe book](cuda.md): the method, the numbers against vLLM, and the traps.
- [Adding a CUDA family](adding-a-cuda-family.md): the engine interface, the exactness tests, measuring.
- [GLM-5.3-Flash](glm-5.3-flash.md): two Sparks (and one Mac, at the end).

## The contract

A multi-row step verifies several consecutive positions in one forward. It is exact when every row gets the
bits a one-row step of the same engine gives that position. Then drafted decoding commits the same tokens as
one-token-a-round decoding, byte for byte. "Serial" always means the same engine and kernels one token at a
time, not stock MLX.

Sampling is keyed. The token at absolute position p is the argmax over the kept candidates of `logit / T` plus
Gumbel noise hashed from the seed, p and the token id. The candidates are the top-k logits (ties broken by id)
cut to the smallest set holding top-p of the probability. That is an exact draw from the model's top-k/top-p
distribution, and it depends only on the row's logits and its position. So a draft is accepted exactly when it
equals the token serial decoding samples there. Why not greedy decoding, which would be simpler: greedy
Qwen3.8 looped in an agent session, one tool call repeated six times and then replies doubling to 32k tokens.

There are two implementations of the rule. `engine/exact_sampling.py` finishes on the host in float64.
`engine/gpu_sampling.py` runs the same rule on the GPU in fp32 and leaves the token on the GPU. They can pick
different tokens at rare near-ties, so a model uses one of them for serial and drafted rounds alike.

## Three ways to make projections row-invariant

All three are in use, and which one fits depends on your MLX version and GPU.

| Strategy | How | Used by | Cost |
| --- | --- | --- | --- |
| A | Use MLX's quantized matmul where it happens to be row-invariant, and check that at load | Nemotron on an M5 Max with MLX 0.31.2 (identical through 9 rows) | free, but breaks when MLX or the GPU changes |
| B | Reproduce MLX's one-row kernel arithmetic for every row and share only the weight reads | Flash Next's `qmv_rows` on an M3 Ultra with MLX 0.32.0 | keeps MLX's one-row bits; each row still pays its own arithmetic |
| C | Write new row-invariant arithmetic and run serial decoding through it too, so it becomes the reference | the 27B lane matmul on M5 tensor units; Flash Next's glue, router, expert and hyper-connection kernels | needs its own quality check, and one row can cost more than MLX's vector kernel |

## The method

1. Serve it serially first. The serial engine serves any family package, one token per forward with keyed
   sampling. That output is the reference every faster path has to reproduce.
2. Get a trusted forward and check it. Nemotron reuses mlx_lm's blocks with the backbone and head split apart.
   Flash Next has a new forward written from the checkpoint layout, matched layer by layer against an existing
   MLX implementation. Detect conversion conventions at load (Flash Next's conversion stores centred norms
   around 1 instead of 0). Check teacher-forced NLL and top-1 agreement against the reference, and compare with
   how far the reference's own prefill and decode paths disagree (0.953 top-1 for mlx_lm on Nemotron).
3. Measure the floor and the gap. The floor is weight bytes read per token over the measured read bandwidth.
   Then count kernels and split host time from GPU time. Metal System Trace (`xctrace`) attached to the server,
   with `MLX_MAX_OPS_PER_BUFFER=1` to get about one GPU interval per kernel, shows where the time goes. What we
   started from: Nemotron 7.3 ms a token against a 3.7 ms floor with about 900 kernels; Flash Next about 33 ms
   against about 5 ms with about 4,500 kernels; the 27B about 33 ms against about 30, already near its floor.
4. Fuse. One kernel per step between weight reads, projections that share an input stacked into one matrix,
   small matvecs folded together, fixed-shape blocks compiled, the graph handed to the GPU every 4 to 8 layers.
   Nemotron went from about 900 to 370 kernels a token and from 138 to about 160 tok/s; Flash Next from about
   4,500 to 690 kernels and from 30 to 79 tok/s serial. On the 27B, fusing about 1,100 glue kernels saved only
   1 to 3 ms of a 45 ms forward: that forward is the matmuls in dependency order. Fusion loses when it removes
   parallelism. One threadgroup reading Nemotron's router cost 220 to 141 tok/s, and folding a norm into a
   split projection made every threadgroup redo the norm.
5. Overlap host and GPU. Draw tokens on the GPU and feed them to the next forward before reading them. When a
   step runs one ahead, alternate its KV writes between two buffers (MLX otherwise copies a buffer the previous
   step still reads). Use bigger command buffers. Cap MLX's buffer cache in a long-running server.
6. Make every kernel row-invariant: rows in a fixed order, no reduction across rows, sparse selection per row,
   attention split by the row's own length, ties broken by id, fp32 where the training framework keeps fp32.
   Pick strategy A, B or C for the projections. Check at load and switch drafting off when the check fails.
7. Test exactness at every level (see [adding a family](adding-a-family.md#tests-to-write)). Regenerate stored
   references after every kernel edit.
8. Add drafting, cheapest first:
   - Copy windows: continue the longest earlier occurrence of the context's suffix, enter only with 8 or more
     matching tokens, verify up to 7 copied tokens a round. Agent traffic has less of this than it seems:
     across 225 recorded sessions, 17 to 35% of output tokens were copyable, which caps a copy-only speedup at
     1.2 to 1.5x. File edits are where copies shine.
   - The model's own MTP head, if the checkpoint has one (convert it from the BF16 release if the MLX
     conversion dropped it). Sample drafts with the target's keyed sampler at their own positions, so a draft
     is the target's token wherever the two distributions agree. Choose the depth from recent acceptance and
     the cost of a verify row.
   - A separate draft model with trees, where extra rows are cheap. On M5 tensor units 16 rows cost about what
     one does.
   - Predict before you build: tokens a round times round time, from what you have already measured.
9. Serve it properly: prefix snapshots keyed by kernels and MLX version, prompts prefilled through the decode
   path (or a documented dependence on what was cached), a thinking budget that depends only on the token
   count, pinned system blocks, and background priority for side requests such as session titles.
10. Measure honestly: a quiet machine, after warm-up, cool and hot, power mode noted. Decide with live A/B runs
    inside one process, blocks of rounds alternating with rests between, identical output hashes.

## Traps we hit

- MLX's quantized matmul row invariance depends on the MLX version and the GPU. On an M5 Max with MLX 0.31.2
  rows match through 9 rows and differ from 10. With MLX 0.32.2 on the same machine, row 0 of a 2-row call
  already differs. On an M3 Ultra with MLX 0.32.0, 2 to 4 rows differ. Check at load on the machine you serve
  from.
- MLX's bf16 (unquantized) matmul can also sum 2 rows differently from 1. Nemotron's router logit came out one
  bf16 unit off at layer 29 of a real decode.
- bf16 router logits over 512 experts tied at the top-10 cut in about a third of Flash Next's layers, and the
  tie decided which expert ran. Keep router logits in fp32 and break ties by id.
- MLX's attention picks its kernel by query count and switches variants with the key count. A multi-row window
  run through it does not match single queries.
- Metal 4 tensor-unit kernels written for the M5 compute wrong values on an M3 Ultra (`applegpu_g15`). Gate
  them by GPU generation (`applegpu_g17` and later).
- Metal's fast math lets the compiler reassociate after any source edit, even one that leaves the arithmetic
  alone. Key snapshots by kernel source and regenerate stored references after every kernel change.
- A kernel templated on a per-token value (key count, row count) compiles a new kernel every token: one forward
  took 280 ms. Pass such values in a small int buffer. MLX caches compiled kernels by name, so put a hash of
  the source in the name, and look kernels up by name rather than re-hashing the source on every call (that
  cost about 2 ms of host time a step on Flash Next).
- `mx.fast.metal_kernel`'s grid counts threads, not threadgroups. Writing threadgroup counts runs singleton
  simdgroups whose `simd_sum` does nothing. MLX may also bind a small input as `constant` rather than `device`,
  so declare derived pointers with `auto`.
- Strided inputs, `mx.take` outputs and column slices can change kernel paths or force a copy. Kernels must
  read the real strides.
- Lazy arrays created on one thread cannot be evaluated on the server's engine thread. Evaluate derived weights
  (stacked matrices and their views) at load.
- Decoding one step ahead makes MLX copy a KV buffer the previous step still reads, about 1 ms a token at 60k
  context on Nemotron. Alternate between two buffers.
- The fused decode path and a reference prefill path differ in their last bits, so prefilling prompts through
  the reference path makes a reply depend on chunk boundaries and on what was cached. The 27B fixed it by
  prefilling through its decode kernels.
- Microbenchmarks of many independent calls pointed the wrong way nine times. They fill the GPU whatever the
  kernel's parallelism. Time dependent chains, then confirm live.
- A laptop slows every kernel 20 to 40% after a long prefill or a few minutes of decoding, and runs about 3x
  slower on battery. A benchmark running while the server warms its snapshots gets half the bandwidth.

## A known limit

For Flash Next and Nemotron, prompts are prefilled through the reference path in 2,048-token chunks, while a
short suffix after a cache hit (16 tokens or fewer) goes through the fused decode kernels. Their last bits can
differ, so the same request can get a different reply depending on what was cached. Drafted and serial
decoding from the same cache state stay byte-identical. The 27B has no such dependence: it prefills through
the lane kernels, which give every prompt row the bits serial decoding gives it.
