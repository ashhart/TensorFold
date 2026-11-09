# Ripple-65B-Preview on two GB10s (CUDA, EXL3, `--tp 2`)

[ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) from an ExLlamaV3 pack, cut over
two NVIDIA GB10s (DGX Spark, sm_121) at `--tp 2`: each rank holds half of every projection and the whole bf16
embedding, the full (global) layers run to a 1,048,576-token context, and rank 0 serves the OpenAI API while rank 1
follows it. One GPU runs Ripple through the one-GPU CUDA path; this path admits two ranks on sm_121 only.

## Packs

One ExLlamaV3 pack of the release: `ajensenwaud/Ripple-65B-Preview-exl3-5.0bpw` at revision
`1d260f752b753d2b17571de237029b5e2251db07`. It averages 5.0 bits a weight with an 8-bit head (`quant_method exl3`), and
it is the pack measured against ExLlamaV3; a pack at another bitrate is refused by name. It takes about 19.8 GiB of
weights a rank. Copy it to the same path on both hosts.

## Serve

Both ranks run the same binary with the same `--context` and `--parallel`; NCCL meets at rank 0's address. Start
rank 1 first:

```sh
# host B (rank 1)
NCCL_COMM_ID=<host A address>:29600 tensorfold-native serve <pack> --tp 2 --rank 1 --context 262144 --parallel 4
# host A (rank 0)
NCCL_COMM_ID=<host A address>:29600 tensorfold-native serve <pack> --tp 2 --rank 0 --context 262144 --parallel 4 --port 8000
```

With RoCE between the hosts set `NCCL_IB_GID_INDEX` (and `TF_ROCE_GID_INDEX`) to the link's RoCE v2 GID; the ranks'
small gathers then go by RoCE writes, the rest by NCCL, with the same bits either way.

- `--context` up to 1,048,576 at `--parallel 1` (about 52.7 GiB of device memory a rank at 1M with a 5.0-bit pack);
  256K at `--parallel 4` holds the same cache bytes.
- Kept prompt states (the prompt cache) are on by default; `--prompt-cache-gib 0` turns them off.
- Thinking: Ripple opens `<think>` itself or answers bare; a reply with no think block is all answer.
- A rank on another GPU, a third rank or a single rank is refused by name.

## Drafter

The publisher's DFlash2 drafter ([ricursiveAI/Ripple-65B-Preview-DFlash2](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2),
BF16) runs cut over both ranks as the target is. Give both ranks the same `--drafter <dir> --drafter-bits 0`: it runs
its BF16 weights as published, and `--drafter-bits 4` is refused. Drafted replies equal `"draft": false` ones token
for token.

## What is checked

`tf-cuda-test ricursive-tp2-contract` runs the family's kernels through the kernel contract. `tf-ricursive-tp2`
writes a pack's teacher-forced logits, greedy tokens and per-layer residuals (prompt chunks and decode rows) for a
comparison against ExLlamaV3's own forward on the same pack.
