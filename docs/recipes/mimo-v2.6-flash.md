# MiMo-V2.6-Flash

The `mimo_v2` family serves an EXL3 pack of MiMo-V2.6-Flash (exllamav3's MiMo-V2 module, mul1 codebook) on the
native CUDA server, on two DGX Sparks as tensor-parallel ranks. The pack's DFlash drafter sits in its `dflash/`
folder (`config.json` and `model.safetensors`); the server drafts with it when it is there.

Only what was validated is admitted: GB10 (sm_121) at two ranks, on the 3.0 bpw pack below. Any other GPU is
refused (`MimoV2RequiresSm121`), and so is one rank or any other world (`MimoV2RequiresTwoRanks`).
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
path. The checkpoint, at the revision validated:

```bash
hf download ajensenwaud/MiMo-V2.6-Flash-RL-exl3-3.0bpw --revision 139d9a4e1eb32193890036fdddd521fe37033a30 \
  --local-dir /models/mimo-v2.6-flash-exl3
```

With rank 0's fabric address 192.0.2.1:

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

## Licences and sources

- Model: Xiaomi's `XiaomiMiMo/MiMo-V2.6-Flash-RL` on Hugging Face, MIT (its model card), at revision
  `5711b268169967567844e1e560e8a3966da959b1`. The packs below are converted from it. The repository's later
  revision `dcd3fd83fb181622fbc606fbb1709901982a62c2` changes only `preprocessor_config.json` (vision).
- Pack: `ajensenwaud/MiMo-V2.6-Flash-RL-exl3-3.0bpw` at revision `139d9a4e1eb32193890036fdddd521fe37033a30`
  (the weights as uploaded in `f2d8d100101b5d8ed09761a33ddd28f277407b41`, plus the model card), MIT as its source,
  quantised from the model's revision above with exllamav3 v1.6.0 (`turboderp-org/exllamav3`, MIT):
  `-b 3.0 -hb 8 -mb 4 -hq -cr 250 -cc 2048 -cb mul1`. Every receipt is on this pack.
- benthecarman's 2.27 bpw pack, `benthecarman/MiMo-V2.6-Flash-RL-exl3` at revision
  `5ae87b830eb950c705967aea6130601a8e1f74a5` (MIT), is not yet validated with this family. It fits one GB10,
  which is refused until one rank has its own receipts.
- Drafter: Xiaomi's DFlash drafter, published in the model repository's `dflash/` folder under its MIT licence.
  The packs' `dflash/` is that drafter as benthecarman repaired it to load and quantised it to 4 bpw EXL3
  (`benthecarman/MiMo-V2.6-Flash-RL-exl3` at `5ae87b830eb950c705967aea6130601a8e1f74a5`); the 3.0 bpw pack
  carries the same file unchanged.
  DFlash, the drafting method and its reference code, is z-lab's (`z-lab/dflash`, MIT).
