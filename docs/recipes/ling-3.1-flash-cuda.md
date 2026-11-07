# Ling-3.1-flash on CUDA

The native CUDA build serves inclusionAI's Ling-3.1-flash (`bailing_moe_v3`: 63 KDA and 9 MLA layers, 512 routed
experts) from EXL3 packs on one GPU or on two ranks over hosts, within the checkpoint's native window of 262,144
positions. Rank 0 serves HTTP and rank 1 follows its steps, as [GLM-5.3-Flash on CUDA](glm-5.3-flash-cuda.md) does.
The family started from Jürgen Schmied's TP=1 port (#874) and keeps his loader, references and checks.

## What it serves

- EXL3 packs (exllamav3's trellises, on main's shared EXL3 kernels). bf16 checkpoints load too, for the synthetic
  checks; FP8 and NVFP4 checkpoints are refused by name.
- `--context` up to 262,144 (`max_position_embeddings`, rope theta 6,000,000, the plain rope). A longer context is
  refused by name, and so is a config with `rope_scaling`.
- Drafts from the prompt's own text (copies). The checkpoint's MTP layer is not loaded.
- Prompt states for the server's prompt cache, in memory: the conv windows, each KDA layer's committed state and the
  MLA rows below the state's position.

## Two ranks

`TF_LING_RANK` of `TF_LING_WORLD` (1 or 2) places each rank and `NCCL_COMM_ID` names rank 0's address and port; each
rank keeps a TCP line to rank 0 on the next port, and a rank that fails stops the other. Each rank loads half the KDA
and MLA heads, half of every MLP and expert width (EXL3's 128-wide blocks) and half the vocabulary of the head; the
latent cache is whole on both ranks. Give both ranks the same arguments:

```bash
# on rank 1
TF_LING_RANK=1 TF_LING_WORLD=2 NCCL_COMM_ID=192.0.2.1:29651 NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0 \
  tensorfold-native serve /models/ling-3.1-flash-exl3 --context 262144
# on rank 0 (192.0.2.1)
TF_LING_RANK=0 TF_LING_WORLD=2 NCCL_COMM_ID=192.0.2.1:29651 NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0 \
  tensorfold-native serve /models/ling-3.1-flash-exl3 --context 262144
```

Other formats and worlds other than 1 and 2 are refused at TP.

## Checks

`tf-cuda-test ling-contract` runs the family's own kernels through kernel-contract gates 1-4. `ling-route`, `ling-mla`
and `ling-clamp` check routing, MLA and the SwiGLU order against float64; `ling-synth` writes small checkpoints with the
real schema for `ling-forward-synth`, `ling-weights`, `ling-lanes` (chunked == whole, drafted == plain, resumed ==
fresh, concurrent == solo) and `ling-init-faults`. `ling-tp` runs a prompt and greedy rows at TP=1 or TP=2 and writes
every row's logits.
