# MiniMax Music 3 on Metal (Apple M5)

`music3-generate` on macOS runs MiniMax Music 3 (MiniMax-Music3, Comfy-Org's single-file checkpoints) on TensorFold's
own Metal runtime: the autoregressive LM with its depth decoder, the flow-matching DiT and the DAV decoder, in Zig with
kernels of its own (`zig/kernels/metal/music3/`, one metallib built ahead of time). There is no Python, MLX or PyTorch
at run time. It is the Metal half of [the CUDA recipe](minimax-music3-cuda.md): the same config, prompt, checkpoint
loaders, request settings, WAV writer, `--serve` protocol and parity tools.

No weights are included. MiniMax-Music3 is released under the MiniMax-Music3 Community License: a commercial product
must display "MiniMax-Music3" in its interface, use by a business above $20M in yearly revenue needs MiniMax's
written authorization, and a hosted service must keep the license's safeguards in place.

## Build and run

    zig build music3-generate -Doptimize=ReleaseFast
    ./zig-out/bin/music3-generate --model DIR --caption-file style.txt --lyrics-file lyrics.txt --seconds 60 --seed 7 \
        --out song.wav --stats song.json                      # int8 (default); --weights bf16 for the checkpoint's own
    ./zig-out/bin/music3-generate --serve 1 --model DIR        # the CUDA recipe's protocol, line for line

`--model DIR` (or `$TF_MUSIC3_MODEL`) is the folder holding Comfy-Org's three MiniMax-Music3 files. The DiT, the
prompt rows and the DAV run on the tensor units, so an M5 or later is required; on an earlier GPU the engine stops at
load with a message. `--flash`, `--dav-tf32` and `--graphs` choose CUDA kernels and are accepted and ignored here.

## What was built

| Stage | How |
| --- | --- |
| Prompt | the shared `prompt.zig` and the checkpoint's own tokenizer: the ids equal ComfyUI's |
| AR prompt | both CFG rows through the body in bf16 GEMMs on the tensor units (`m3_linear_b`, MLX's `steel_gemm_fused_nax` blocks through `tfp::gemm_block`); int8 weights are dequantized a projection at a time for it |
| AR frame | lm_head, the CFG top-k draw on the device, the depth decoder's 7 passes and the frame fed back through 36 layers, encoded straight into a serial compute encoder and committed a frame at a time, so the next frame encodes while the GPU runs the last; position, step and count advance on the GPU and the host reads the drawn tokens every 16 frames for the stop code |
| AR GEMV | a simdgroup two weight rows (two outputs, or a gate and up row whose SwiGLU it writes), one 16-byte load a row a lane in flight, fp32 sums; bf16, int8 a scale a row (body, lm_head) or a scale a group of 128 (depth decoder and heads), as on CUDA |
| Attention | split-key decode attention, four keys a step under one `simd_sum`; the depth decoder's 8 positions in one group a head |
| DiT | windows of 689 latent rows every 344, all of a 60 s song's windows and both CFG branches in one forward; fp16 GEMMs on the tensor units with fp32 sums; q and k leave the qkv GEMM roped and ff_in's GEMM writes the GLU (its rows are paired at load), so neither has a pass of its own; attention on the tensor units (heads of 64, 64 queries a group, online softmax, fp16 probabilities for P V as on CUDA) |
| DAV | fp32, every conv as one GEMM a tap over time-major rows with zero halos; each GEMM runs on the tensor units from bf16 halves of both operands (x = xh + xl: xh wh + xh wl + xl wh, fp32 sums), chunks of 512 latent rows with 32 rows of context, ComfyUI's level rule |
| Sampling | the CUDA sampler's radix select and inverse-CDF draw on the same Philox counters; a song is not CUDA's song for the same seed, as the logits differ in the last bits |

Precision is the reference's: bf16 activations in the AR with bf16 roundings where torch rounds, fp32 norms, softmax
and rope; fp16 in the DiT with the reference's fp16 roundings; fp32 in the DAV. `music3.metallib` is built with safe
math and precise fp32 functions. One exception: the fused GLU's sigmoid uses `fast::exp`, since `precise::exp` in that
epilogue made the ff_in GEMM 2.6 times slower (final latent 38.92 dB with `precise::exp`, 38.97 dB with `fast::exp`).

## Parity against ComfyUI

The reference is ComfyUI's native MiniMax Music 3 nodes on the same Mac (MPS): hiphop bench prompt, 30 s, seed 7,
teacher-forced so the rows are comparable (the dumps the CUDA recipe describes, `tools/music3/parity.py`). The MLX
engine (a separate Python port, not part of this tree) is against the same dumps; CUDA's columns are against ComfyUI
on its own GB10, so they show the same model's spread, not a shared reference. JSON: `minimax-music3-metal/`.

| Check | Metal bf16 | Metal int8 (default) | MLX bf16 | MLX int8 | CUDA bf16 / int8 |
| --- | --- | --- | --- | --- | --- |
| prompt ids equal | yes | yes | yes | yes | yes / yes |
| frame states cos, mean / min | 0.999955 / 0.999748 | 0.999838 / 0.997195 | 0.999951 / 0.999821 | 0.999889 / 0.998982 | 0.999956 / 0.999827 |
| code-0 logits cos (cond), mean / min | 0.999975 / 0.999839 | 0.999906 / 0.99749 | 0.999973 / 0.999724 | 0.999938 / 0.99862 | 0.999977 / 0.999897 |
| code-0 greedy agreement, cond / guided | 0.9814 / 0.968 | 0.968 / 0.9694 | 0.9774 / 0.964 | 0.98 / 0.9507 | 0.9747 / 0.9667 (cond) |
| code-0 top-50 overlap | 0.9873 | 0.9821 | 0.9872 | 0.9842 | 0.9854 / 0.9799 |
| ComfyUI's drawn code, median rank in ours | 1 | 1 | 1 | 1 | 1 / 1 |
| depth logits cos, codebook 1..7 | 0.99994 to 0.999955 | 0.999807 to 0.999859 | 0.99993 to 0.99996 | 0.99986 to 0.99991 | 0.99996 / 0.9998 |
| depth greedy agreement, codebook 1..7 | 0.951 to 0.976 | 0.933 to 0.977 | 0.951 to 0.983 | 0.933 to 0.969 | 0.964-0.981 / 0.945-0.967 |
| DiT probe cos (cond / uncond), sigma 1.0 | 0.999995 / 0.999996 | (the DiT is fp16 in both) | 0.99999 / 0.999998 | | 0.999991 / 0.999994 |
| DiT probe cos, sigma 0.5 / 0.1 (cond) | 0.999982 / 0.999993 | | 0.999971 / 0.999992 | | 0.999958 / 0.999987 |
| DiT final latent (ComfyUI's noise and states), cos / SNR | 0.999939 / 38.97 dB | | 0.999838 / 34.9 dB | | 0.999914 / 37.6 dB |
| DAV probe (one pass), cos / SNR | 1.0000000 / 103.9 dB | | 0.9999993 / 52.0 dB | | 1.0000000 / 71.0 dB |
| DAV of ComfyUI's latent (chunked as served), cos / SNR | 1.0000000 / 103.9 dB | | 0.9999994 / 50.5 dB | | 1.0000000 / 71.0 dB |

The tensor units' own fp32 GEMM path gave the DAV 42.3 dB here (it keeps fewer bits of each operand); plain fp32
simdgroup matrices gave 118.5 dB at 8 times the time; the bf16 halves give 103.9 dB at 1.4 s for 30 s of song.

## Speed and memory (one Mac Studio, M5 Ultra, 256 GB)

Hiphop bench prompt, 60 s, seed 7. The MLX engine is the same Mac's Python port with `--quant int8` (8-bit in groups
of 64 everywhere); this port is int8 a scale a row in the body and lm_head, a scale a group of 128 in the depth
decoder and heads. The runs alternated (Metal, MLX, Metal, MLX, ...), three each, one model process at a time under a
lock shared with the machine's other GPU jobs. Peak is the process's own peak footprint (`/usr/bin/time -l`).

| Run | Engine | AR frames/s | prefill s | AR s | DiT s | DAV s | **Total s** | Peak footprint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Metal int8 | 69.09 | 0.27 | 22.0 | 34.7 | 2.80 | **59.5** | 17.9 GB |
| 1 | MLX int8 | 64.66 | 0.18 | 23.4 | 35.3 | 2.06 | **62.3** | 28.1 GB |
| 2 | Metal int8 | 68.87 | 0.28 | 22.1 | 37.9 | 2.81 | **62.8** | 17.9 GB |
| 2 | MLX int8 | 64.66 | 0.18 | 23.4 | 37.2 | 2.02 | **63.9** | 28.1 GB |
| 3 | Metal int8 | 68.90 | 0.27 | 22.1 | 38.1 | 2.80 | **63.0** | 17.7 GB |
| 3 | MLX int8 | 64.76 | 0.19 | 23.3 | 37.3 | 2.02 | **63.8** | 28.1 GB |
| one | Metal bf16 | 41.36 | 0.27 | 36.6 | 37.9 | 2.80 | **77.2** | 25.3 GB |
| one, long-pop 120 s, seed 5 | Metal int8 | 66.95 | 0.43 | 45.3 | 72.5 | 5.90 | **123.7** | 18.4 GB |

- **The AR reads 11.15 GB a frame in int8,** so 69 frames/s is 770 GB/s. The GEMVs alone run at 1,040 GB/s (body)
  and 990 GB/s (depth decoder), measured on the same weights; the rest of a frame is attention, norms, the 8 draws
  and the dispatches between them (about 3.6 ms of a 14.5 ms frame).
- **The DiT is GEMM-bound:** its 3.0 PFLOP of products for 60 s of song in 34.7 to 38.1 s is 79 to 86 TFLOP/s, on
  the same kernel blocks as MLX's (35.3 to 37.3 s). Folding the rope and the GLU into the GEMMs took 4% off it.
- **Load** is 2.8 to 3.4 s from a warm page cache: the weights are copied into shared Metal buffers (14.2 GB in int8),
  int8 quantized on the GPU as they load.

**Other processes on the machine:** an Ollama server and another app were resident during the timed runs; I did not
measure whether they used the GPU.

## Lyric accuracy

Three 45 s songs in each format through `--serve`, transcribed by faster-whisper large-v3-turbo on the CPU and scored
by the Studio's `music.heard` (`tools/music3/lyrics_check.py`, `minimax-music3-metal/lyrics.json`).

| Song | int8 score | int8 words heard | bf16 score | bf16 words heard |
| --- | --- | --- | --- | --- |
| hiphop, seed 7 | 99 | 143 of 145 | 98 | 121 of 124 |
| pop, seed 11 | 100 | 78 of 78 | 100 | 70 of 70 |
| long-pop, seed 5 | 97 | 56 of 58 | 100 | 46 of 46 |

## Determinism

- **Three processes** (pop prompt, 30 s, seed 11) wrote byte-identical WAVs: MD5 `e81021f8…`.
- **Three requests in one `--serve` process** wrote the same bytes as the fresh processes.
- Every kernel computes a row's outputs in a fixed order with no atomics in a sum (the sampler's histogram counts
  are integers), so the bits do not depend on timing.

## Kernel contract and checks

On the M5 Ultra:

    SUMMARY 27 kernels: 27 pass gates 1-4, 0 fail, 2 known findings; 5 controls: 5 red as planted, 0 missed;
    0 limit problems
    PASS limits: 134 registered Metal launches against M1-M5, 0 problems

- **Gates 1-4:** `m3_rmsnorm` (plain gamma) against the rows family's float64 RMS norm, at the served shapes (2, 4 and
  600 rows of 4096): non-finite classes match, five runs byte-identical, the shapes within the bound, fp32 sums.
- **Gate 5:** all 57 music3 Metal launches in `registry.zig`. Each kernel reserves its threads in the source, so none
  needs an M1/M2 compile record, and each passes the on-GPU check of its pipeline and threadgroup bytes. The two known
  findings are affine_mm's, from before this change.
- `python3 tools/zig/lean_check.py`: 517 problems, the same 517 as the CUDA branch. Every file is at most 600 lines.

## Next steps

- **The DiT.** It is now 60% of a song. The GEMMs are at the rate of MLX's blocks; int8 or fp8 products for the DiT
  would be the next lever, a quality risk against the fp16 reference, not tried.
- **The AR's frame overhead.** 3.6 ms a frame is outside the GEMVs: folding the RMS norms into the GEMVs that read
  them and fewer, larger attention dispatches would take most of it.
- **The DAV.** 1.4 s for 30 s, against MLX's 1.0 s at 50 dB; one pass with the hi and lo products sharing loads is
  done, the halves could also be written by the kernels before each GEMM instead of a pass of their own.
- **Before M5.** The tensor-unit kernels have a simdgroup-matrix form (nax.h's `TF_SIMD_FRAGS`) that gate 5 compiles
  for M1-M4; a second metallib for those GPUs was not built or run.
- **Streaming.** The MLX engine can stream audio while the AR draws; this port answers when the song is done.
