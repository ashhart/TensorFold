# MiMo-V2.6-Flash

The `mimo_v2` family serves an EXL3 pack of MiMo-V2.6-Flash (exllamav3's MiMo-V2 module, mul1 codebook) on the
native CUDA server, on one GPU or two DGX Sparks as tensor-parallel ranks. The pack's DFlash drafter sits in its
`dflash/` folder (`config.json` and `model.safetensors`); the server drafts with it when it is there.
Package: `zig/src/families/mimo_v2/`, kernels in `zig/kernels/cuda/mimo_v2/`.

The ranks exchange through the tensor-parallel session and RoCE mailbox of Jay Leaton's two-Spark DeepSeek-V4.1
work, whose port method (launch for launch against the Python engine, then byte checks) the family followed.

## The model

- 48 layers of hidden size 4,096: 9 full-attention layers and 39 sliding-window layers (128 tokens, sink logits).
- Attention: query/key head dim 192 (RoPE on the first 64), value dim 128; the full and sliding layers have their
  own key/value head counts and RoPE bases.
- MoE: 256 routed experts, sigmoid routing with a bias, top 8; the first layer is dense.
- The drafter (DFlash) reads the target's residuals after its tap layers and drafts its whole block in one pass.

## Kernels

Everything is built in; nothing is captured or loaded at run time.

- EXL3 projections and decode experts on the shared operators (`cuda.exl3.Linear`, `cuda.exl3_experts`); prompt
  experts on the common prompt kernels (`exl3_prompt`), whose 3-bit mul1 instance this family uses.
- The family's own kernels: RMS norms, SwiGLU, the fp32 router and its top 8, the prompt experts' slot sum
  (`ops.cu`); queries and the caches, chunked decode attention with sinks, and a tensor-core prompt kernel for the
  full layers (`attention.cu`); embed, residual adds, RoPE tables and the vocabulary argmax (`glue.cu`); the
  drafter's kernels (`dflash.cu`). Every launch is registered with the kernel contract (`contract_mimo*.zig`).

Every prompt token runs on the prompt path, in chunks that end at the prompt cache's marks; a row's bits never
depend on its chunk, so a resumed prompt reads as a fresh one. Decode rows, verify windows and drafts run the
decode arithmetic.

## Prompt cache

A kept state is the slot's full-layer rows, pinned in place, plus copies of the sliding layers' and the drafter's
windows (`kept.zig`, about 33 MiB a rank). A resume binds the stream to the state's slot, or copies the full-layer
rows into a free one. With the pack filling most of a Spark, give the cache room with
`TENSORFOLD_MEMORY_RESERVE_GIB=3`.

## Two Sparks

Rank 1 follows rank 0 over the plan link; start it first, then the server. Both need the same build and checkpoint
path. With rank 0's fabric address 192.0.2.1:

```bash
# rank 1
TF_TP_WORLD=2 TF_TP_RANK=1 TF_TP_MASTER=192.0.2.1 TF_TP_PORT=29661 TF_COMM_BACKEND=roce \
  tf-mimo follow /models/mimo-v2.6-flash-exl3
# rank 0
TF_TP_WORLD=2 TF_TP_RANK=0 TF_TP_MASTER=192.0.2.1 TF_TP_PORT=29661 TF_COMM_BACKEND=roce \
TENSORFOLD_MEMORY_RESERVE_GIB=3 \
  tensorfold-native serve /models/mimo-v2.6-flash-exl3 --host 0.0.0.0 --parallel 4 --context 608256 \
  --top-k 20 --top-p 0.95 --prompt-cache-gib 4
```

`--parallel` is the number of streams; each owns a cache slot of the whole `--context`. A lone stream's draft depth
comes from its own acceptance rate, priced with a round cost that grows with its context.

## Checks

- `tf-cuda-test kernel-contract`: the family's kernels against float64 (gates 1-4) and their launch records.
- `tf-mimo m1 MODEL ORACLE [CONTEXT [DUMP_OUT [DUMP_BASE]]]` on each rank (`tf-mimo follow MODEL` beside it):
  teacher-forced steps against an answer key's logits (KL and argmax), optionally against an earlier build's dump,
  and the prompt path read in one chunk against two.
