# Ling-3.1-flash on CUDA

The native CUDA build serves inclusionAI's Ling-3.1-flash (`bailing_moe_v3`: 63 KDA and 9 MLA layers, 512 routed
experts, an MTP layer) from bf16, FP8 block-128 and EXL3 checkpoints on one GPU, and from EXL3 packs on two ranks over
hosts. Rank 0 serves HTTP and rank 1 follows its steps, as [GLM-5.3-Flash on CUDA](glm-5.3-flash-cuda.md) does. The
family started from Jürgen Schmied's TP=1 port (#874) and keeps his loader, references and checks.

## Context past 262,144: YaRN

The checkpoint's native window is 262,144 positions (`max_position_embeddings`, rope theta 6,000,000). For 1,048,576,
give the config SGLang's override, as transformers reads it:

```json
"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144,
                 "rope_theta": 6000000, "partial_rotary_factor": 0.5},
"max_position_embeddings": 1048576
```

Only `yarn` is read; other rope types are refused by name. The MLA layers' 64 rope columns (and the MTP layer's) take
transformers' `_compute_yarn_parameters` frequencies (beta_fast 32, beta_slow 1, the ramp bounds truncated) and its
attention factor, 0.1 ln 4 + 1, on cos and sin; the softmax scale changes only when the config gives
`mscale_all_dim`, as Ling's modeling applies it. YaRN applies when `--context` is past the native window; below it
the plain rope runs, unchanged bit for bit.

## Two ranks

`TF_LING_RANK` of `TF_LING_WORLD` (1 or 2) places each rank and `NCCL_COMM_ID` names rank 0's address and port; each
rank keeps a TCP line to rank 0 on the next port, and a rank that fails stops the other. Each rank loads half the KDA
and MLA heads, half of every MLP and expert width (EXL3's 128-wide blocks) and half the vocabulary of the head. The
latent cache is whole on both ranks up to the native window; when `--context` is past it, each rank keeps every other
512-position chunk, so at 1M a rank holds half the cache (5.6 GiB instead of 11.25 on the full model). Each rank then
runs all 64 heads' attention over its own chunks (the absorbed queries gathered first), merges its chunks per head in
fp32, swaps the other rank's heads' partials, and merges the two as GLM's chunk merge does; the result differs from the
whole cache only in the order the chunks' softmax sums are added. Give both ranks the same arguments:

```bash
# on rank 1
TF_LING_RANK=1 TF_LING_WORLD=2 NCCL_COMM_ID=192.0.2.1:29651 NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0 \
  tensorfold-native serve /models/ling-3.1-flash-exl3 --context 262144
# on rank 0 (192.0.2.1)
TF_LING_RANK=0 TF_LING_WORLD=2 NCCL_COMM_ID=192.0.2.1:29651 NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0 \
  tensorfold-native serve /models/ling-3.1-flash-exl3 --context 262144
```

Other formats and worlds other than 1 and 2 are refused at TP.

## Prompt states

The family keeps prompt states for the server's prompt cache: the conv windows, each KDA layer's committed state and
the MLA rows below the state's position. Past the native window a kept state leaves its rows in place in the slot's
caches. With `--learn-conversations` or `--spill-gib`, states go to main's rows store on rank 0, holding every rank's
share in one image.

## Checks

`tf-cuda-test ling-contract` runs the family's kernels through kernel-contract gates 1-4. `ling-route`, `ling-mla`
and `ling-clamp` check routing, MLA and the SwiGLU order against float64; `ling-synth` writes small checkpoints with the
real schema for `ling-forward-synth`, `ling-weights`, `ling-lanes` (chunked == whole, drafted == plain, resumed ==
fresh, concurrent == solo) and `ling-init-faults`. `ling-tp` runs a prompt and greedy rows at TP=1 or TP=2 and writes
every row's logits; `--split` splits the latent caches at any capacity.
