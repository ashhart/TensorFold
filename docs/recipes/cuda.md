# The CUDA recipe book

How we made three model families decode fast and byte-exact on NVIDIA GPUs, measured on DGX Spark (GB10), and
how to do the same for yours. The per-family pages have the numbers:

- [Qwen3.8-27B dense](qwen3.8-27b.md#dgx-spark-cuda): one or two Sparks, DFlash2 draft trees.
- [Qwen3.8 Flash Next](qwen3.8-flash-next.md#dgx-spark-cuda): one or two Sparks, MTP drafts, CUDA graphs.
- [GLM-5.3-Flash](glm-5.3-flash.md): two Sparks, MTP and DFlash2 drafts, CUDA graphs.
- [Adding a CUDA family](adding-a-cuda-family.md): the package interface, the tests and how to measure.

| Model | Sparks | vs vLLM with MTP, same client (range over the four columns) |
| --- | --- | --- |
| Qwen3.8-27B | 1 | 2.70-3.05x |
| Qwen3.8-27B | 2 | 1.94-2.49x |
| Qwen3.8 Flash Next | 1 | 1.60-1.79x |
| Qwen3.8 Flash Next | 2 | 1.74-2.24x |
| GLM-5.3-Flash | 2 | 1.78-2.06x |

The columns are code and chat prompts, sampled and greedy, one stream, 64-token replies, medians of seeds 1234
to 1238. Every TensorFold number is byte-identical to serial decoding on the same engine. vLLM's drafted output
is not.

## The contract on CUDA

It is the Mac engine's contract ([the recipe book](README.md#the-contract)) with CUDA's own bits. A verify
window of several rows is exact when every row gets the bits a one-row step of the same engine gives that
position. Serial decoding runs through the same kernels, so it is the reference. The bits differ from the
Mac's, and they differ from a stock PyTorch forward. Sampling is the same keyed rule
(`engine/exact_sampling.py`): a draft is accepted exactly when it is the token serial decoding samples there.

## The method

1. **Serve it serially first.** Write the forward in PyTorch with your own kernels on the verify path and check
   its quality against a plain fp32 reference: teacher-forced NLL over a few thousand tokens, and top-1
   agreement. NVIDIA's container turns on a TF32 override for fp32 matmuls, so set
   `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` when you run the reference, or it is not fp32.
2. **Make every kernel on the verify path row-invariant.** cuBLAS and `torch.matmul` choose algorithms and K
   splits by the number of rows, so a row's bits change with its window. Our 4-bit lane matmul
   (`families/qwen3_5/cuda/qmm.py`) does, for each 64-input group, a tensor-core dot of the bf16 rows with the
   integer-valued weights into fp32, then `acc + p * scale + xs * bias`, groups in order. The K split depends
   only on the weight's shape, and split slices are added in slice order. Rows run as one block of 16, 32, 64
   or 128, and tile sizes, warps and stages change speed, never bits. The same rules apply everywhere else:
   attention chunked by the row's own key range and merged in key order, recurrences updated along each node's
   path in path order, router logits in fp32 with ties broken by id, no reduction across rows.
3. **Lay the weights out for the read.** MLX's packed layout spreads a program's group over many rows. Regrouping
   the words once at load into contiguous blocks per program and group took the 27B's matmuls from 107-130 to
   200-220 GB/s of GB10's measured 240, with the same bits.
4. **Find out whether the host or the GPU is the limit.** Count kernels and compare host time with GPU time
   per forward. The 27B's 918 kernels take 12-14 ms of host time, hidden behind 50-85 ms of GPU work, so CUDA
   graphs would buy it little. Flash Next and GLM run small forwards where launches dominate, so their verify
   forwards and draft chains run as CUDA graphs (with the NCCL all-gathers captured inside on two Sparks).
5. **Add drafting.** The model's own MTP head (Flash Next, GLM), a draft model with trees (DFlash2 for the 27B
   and GLM), and copies from the context. Sample drafts with the target's keyed sampler at their own positions.
   Stop a chain when the drafts' confidence drops (Flash Next stops under 30%). When a model has two drafters,
   let each request pick by measurement: GLM times both at load, and for each greedy request keeps whichever
   commits more tokens per millisecond, which took its greedy code from 52.9 to 66.3 tok/s. Read only part of the
   vocabulary in the draft head: the 27B's drafter reads the first 98,304 ids (99.6% of committed tokens),
   Flash Next's reads 79,591 ids: every id below 65,536, the added tokens, and ids common in public source code and docs.
6. **Choose the window by measurement.** A width pays only when committed tokens divided by the whole round
   time goes up. On one Spark the 27B's 12-row windows accept the same drafts as 16 rows at 3 ms less a round,
   and 32-64 rows accept about one more token for 30-40 ms more verifying. The decoders can trace every round
   (`draft_decode(trace=...)` in the 27B engine): how deep the tree went, where the first wrong guess was, and
   whether the right token was among the drafter's candidates.
7. **Split over two Sparks.** Tensor parallel, one rank per Spark over NCCL. Column-parallel projections keep
   whole output rows. Row-parallel projections return fp32 partial sums, which both ranks all-gather and add
   rank 0 then rank 1, rounding once. That order is fixed. NCCL's all-reduce order is an implementation detail,
   so it is not used. Rank 0 decides each window and sends it to rank 1, and both run the same forwards. Split
   the head by vocabulary: each rank returns its half's top-k candidates and rank 0 draws from their union,
   which is exactly the draw over the whole vocabulary. Split the draft model the same way. The 27B's second
   Spark added 1.6x on code, where vLLM's added 1.9x: the 128 all-gathers of a forward (two per layer) do not
   shrink with a second GPU.
8. **Measure the same way for both engines.** The same OpenAI client (`tools/bench_openai.py`), prompts,
   seeds and reply length for TensorFold and the baseline; medians over seeds, because single seeds vary by
   up to 2x; nothing else running on either GPU; the drafted token hash compared with serial on every run.

## Traps we hit

- A kernel change can change bits even when the arithmetic reads the same: Triton versions differ in how they
  fuse multiply-adds. Compare kernels against themselves at 1 and N rows, never against a plain PyTorch
  reference for equality.
- A buffer sized for 128 rows met a prefill of 34 prompt tokens times 4 hyper-connection streams in Flash Next,
  and the server raised on any prompt of 34 tokens or more. The benchmark prompts were 14 and 31 tokens. Test
  long prompts.
- Two ranks must key their prefix caches the same way. Rank 1 once stored the prompt's ids with the reply's
  final state, so a second chat turn would have run a different number of forwards on the two ranks. The
  server now checks at start that both ranks were given the same settings.
- NCCL needs the InfiniBand devices and locked memory inside the container (`--device /dev/infiniband
  --ulimit memlock=-1 --cap-add IPC_LOCK`) and the right adapters in `NCCL_IB_HCA`. `NCCL_PROTO=LL` made the
  27B's 16-row forwards 56% slower; channel and protocol changes moved Flash Next by 2% or less. The defaults
  were best.
- GB10's memory is unified, and the kernel migrates pages under a running model. Dropping the page cache right
  after loading made it move about 25 GB while GLM decoded. Even without a drop, some runs ran at half speed,
  each during a burst of 45,000 to 265,000 migrated pages, with the clocks steady. Take medians over seeds and
  repeat a slow run before believing it. vLLM across two Sparks, on the other hand, only started after a
  page-cache drop on both.
- PyTorch's profiler records a GPU-side annotation for each `nccl:*` op that spans the same time as the NCCL
  kernel, so summing GPU events (or the `key_averages()` table) counts every all-gather twice. Skip events
  with `is_user_annotation` set.
- A per-round host sync drains the GPU's queue. Draw tokens on the GPU where you can, and keep the host work of
  a round (tree building, sharing a window with rank 1) small.
