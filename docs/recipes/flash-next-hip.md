# Qwen3.8 Flash Next on HIP

This page is the home for Flash Next on HIP: the native build serving `qwen4_exp` checkpoints on AMD GPUs.
The integrated-GPU platform (gfx1151, Strix Halo) has its own section in the MLX prefill recipe,
[Native HIP on a Strix Halo iGPU](qwen3.8-flash-next.md#native-hip-on-a-strix-halo-igpu), which this page
does not repeat.

Discrete gfx12 cards are covered here by two receipts,
[docs/receipts/gfx1201-flashnext-tp2-f8dacd77](../receipts/gfx1201-flashnext-tp2-f8dacd77/README.md) and
[docs/receipts/gfx1201-flashnext-tp4-6fc30cf0b](../receipts/gfx1201-flashnext-tp4-6fc30cf0b/README.md):
two Radeon AI PRO R9700s (gfx1201, 32 GiB each) at `--tp 2` and four of them at `--tp 4`, one host,
with `Qwen3.8-Flash-Next-MLX-oQ2-MTP` at `0a1bf6fa` — decode at `--parallel 4 --context 4096` (121.2 and
122.9 tok/s, `bench_openai` fibonacci-raw) and cold prefill at `--parallel 1 --context 40960`
(1,191/1,121/1,058 and 1,412/1,358/1,263 tok/s at 2k/8k/32k tokens), all with `--prompt-cache-gib 0`, every
`alone` and `serial` comparison equal. The world-4 receipt also carries a world-2 control on the same binary
that isolates the world effect. What follows is those measured paths only.

## Two-rank serve path

```bash
# rank 1 (does not serve HTTP)
HIP_VISIBLE_DEVICES=<card> tensorfold-native serve TensorFold/Qwen3.8-Flash-Next-MLX-oQ2-MTP@0a1bf6faa0a2f16a314da889e8e6c80226455dbf --tp 2 --rank 1 --master 127.0.0.1 --prompt-cache-gib 0 --parallel 4 --context 4096
# rank 0 (serves HTTP)
HIP_VISIBLE_DEVICES=<card> tensorfold-native serve TensorFold/Qwen3.8-Flash-Next-MLX-oQ2-MTP@0a1bf6faa0a2f16a314da889e8e6c80226455dbf --tp 2 --rank 0 --master 127.0.0.1 --port 8092 --prompt-cache-gib 0 --parallel 4 --context 4096
```

- Both ranks run the same binary; `--master` names rank 0's host and `--master-port` (default 29551) its
  control port, and the ranks wait for each other.
- `--prompt-cache-gib 0` was used with drafts at `--tp 2`, per the CUDA recipe's qualification terms;
  prompt-cache retention with MTP was not measured.
- HIP enumeration order differs from `rocm-smi` card numbers; verify the target card by occupancy
  (`rocm-smi --showmeminfo vram --csv`) and pin it with `HIP_VISIBLE_DEVICES`, not by trusting an index.
- The weights take 27.3 GiB per rank at `--tp 2`.

## Four-rank serve path

```bash
# ranks 1-3 (do not serve HTTP)
HIP_VISIBLE_DEVICES=<card> tensorfold-native serve TensorFold/Qwen3.8-Flash-Next-MLX-oQ2-MTP@0a1bf6faa0a2f16a314da889e8e6c80226455dbf --tp 4 --rank R --master 127.0.0.1 --prompt-cache-gib 0 --parallel 4 --context 4096
# rank 0 (serves HTTP)
HIP_VISIBLE_DEVICES=<card> tensorfold-native serve TensorFold/Qwen3.8-Flash-Next-MLX-oQ2-MTP@0a1bf6faa0a2f16a314da889e8e6c80226455dbf --tp 4 --rank 0 --master 127.0.0.1 --port 8092 --prompt-cache-gib 0 --parallel 4 --context 4096
```

- Measured at `v1.2.7-1722-g6fc30cf0b` on four R9700s, one rank per card
  ([gfx1201-flashnext-tp4-6fc30cf0b](../receipts/gfx1201-flashnext-tp4-6fc30cf0b/README.md)): decode 122.9 tok/s,
  concurrent code 123.8/242.0 and chat 103.5/205.2 tok/s at 1/4 streams, cold prefill 1,412/1,358/1,263 tok/s.
- A world-2 control on the same binary, cards and flags isolates the world: +15.8/+16.6/+17.1% on cold prefill,
  +1.4% on single-stream decode, and −3.0% on 4-stream code aggregate. World 4 is the prefill choice; it is not
  a decode-throughput choice.
- HIP admission is geometry-gated: heads, experts and vocabulary must split evenly over the world
  (`flashnext/hip/native.zig`). World 4 admitted cleanly for this checkpoint.
- The device holds 15.74 GiB per rank at the decode window (15.72 GiB at the prefill window).

Not measured by those receipts: other card counts, `--tp 8`, worlds other than 2 and 4, multi-host ranks,
other checkpoints or revisions, other context or parallel windows, fp8 KV, and drafter checkpoints. A host
build does not establish checkpoint fidelity, qualify later `main` commits, or qualify other gfx12 variants.
