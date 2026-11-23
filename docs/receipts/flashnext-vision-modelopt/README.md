# flashnext-vision-modelopt: the ModelOpt tower decode, served

The CUDA vision tower's reader takes every linear's weight as stored bf16, so `--vision` on a checkpoint whose tower ModelOpt quantizes stops at the first quantised linear and names it. This receipt covers that pair on the device: main refuses the checkpoint's tower, and the branch (main plus the commit below) loads it and serves an image, deterministically, with the three exactness contracts holding. The decode itself is proved exact on the host against binary64 references; this window is a load, sanity and determinism check.

## Source and environment

- Branch `ac207000030b969a259143ac32bf2eac6b5e29e7` (head), cut from main `077998f7d9b14adebec3a2acc2caa47ff7ae6ec7` (its tip of 2026-11-16T15:26:10Z). The pull request's current head carries the same change on main `8cbef9777b9a05dfaff99d5aa9c2ce52fe2b00f8`: its code commit `d46832d66` has the five source files of `ac2070000` byte for byte (`zig/src/vision/qwen_cuda_dequant.zig`, `qwen_cuda_weights.zig`, `qwen_cuda_weights_logs.zig`, `qwen_cuda_weights_quant_test.zig`, `qwen_cuda_weights_test.zig`), `zig/src/vision` is unchanged between the two mains, and only the documentation around the change moved. No GPU run is claimed on the current head: every figure below was measured on `ac2070000`.
- Images: `tensorfold-qwen38:vis-m-077998f7` (main) and `tensorfold-qwen38:vis-b-ac207000` (this branch), each built from a `git archive` of its own commit, blob by blob against its own `git ls-tree` manifest, with `zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast fatbins install native`, the kernels embedded (`tf.kernels=embedded`).
- One NVIDIA GB10 (sm_121) machine, one fresh server an arm through the serving recipe: `CHECKPOINT=nvfp4`, `KV_DTYPE=bf16`, `--parallel 8 --context 32768`, `--vision` only where the leg says so.
- Checkpoint nvidia/Qwen3.8-Flash-Next-FP4-Local at b9ce18a6887e18c089fdea8509a33e63429a3300: 499 vision tensors in 1 shard. Of its two-dimensional weights 1 (the position embedding) take the unchanged stored-bf16 path, 54 are MXFP8 and 56 are NVFP4; the snapshot's dtype census is BF16/bias=166; BF16/weight=57; F32/weight_scale_2=56; F8_E4M3/weight=54; F8_E4M3/weight_scale=56; U8/weight=56; U8/weight_scale=54.
- The checkpoint's own `config.json` gives the tower depth 27, hidden 1152, intermediate 4304, out_hidden 2560, patch 16 and spatial merge 2: two attention linears a block and two MLP linears a block plus the merger's two make the census's 54 MXFP8 and 56 NVFP4 weights.
- The floating-point tower leg uses azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound at 1464274120d36a4d8fcaa934552334a7d83ce0fd: 333 vision tensors, every one floating point (BF16/bias=166; BF16/weight=167). Neither image served it: both refused the checkpoint's `config.json` before reading any weight (`exactness.md`), so this leg has no result on this engine.

## The load pair

Main, `--vision` on the FP4 checkpoint, fails its boot:

    2026-11-17T00:30:32.520841803Z error: model.visual.blocks.0.attn.qkv.weight: f8_e4m3 { 3456, 1152 }, expected bf16 { 3456, 1152 }

The branch, same image settings, serves it: the startup line is `[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 74.23 GiB; 8 streams at once, 1.31 GiB each at a 32768-token window, of 26.2 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 15.7 GiB prompt cache; own glue kernels`, and the first image request returns `A close-up portrait captures a tabby cat with striking green eyes, gazing intently at the viewer against a softly blurred background.` (28 tokens, 1.141 s).

## Memory, as the server reports it

| boot | startup line |
|---|---|
| branch, `--vision` off | `[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 72.71 GiB; 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels` |
| branch, `--vision` on | `[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 74.23 GiB; 8 streams at once, 1.31 GiB each at a 32768-token window, of 26.2 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 15.7 GiB prompt cache; own glue kernels` |

- The tower's resident bytes, read as the startup line's own model figure on the same image and settings, `--vision` off against on: 72.71 GiB against 74.23 GiB, so the tower holds 1.52 GiB; the per-stream figure moves from 1.29 to 1.31 GiB.
## Speed, the default path (vision off)

The change is unreachable without `--vision`: `Options.vision` is null unless the serve flag sets it, the tower loads only inside that branch, and no kernel, launch, scheduler, KV or prompt arithmetic is edited. The cells therefore measure the same code on the same bytes, and their verdict is the noise rule's.

| streams | main (two runs) | branch (two runs) | branch vs main | verdict |
|---|---|---|---|---|
| 1 | 155.7, 152.9 | 152.1, 153.1 | -1.11% | level |
| 4 | 65.5, 64.3 | 64.4, 64.3 | -0.82% | level |

End-to-end tokens a second, temperature 0, the eight golden prompts; `cells.csv` has every run with its per-request decode rate and its alone checks. The greedy hashes: 8 of 8 prompts carry the same token ids on all four boots, and every concurrent reply equals its solo run.

Each request's own wall time, from the client's first byte to its last: M1 1 streams: requests [1.64]; B1 1 streams: requests [1.68]; B2 1 streams: requests [1.69]; M2 1 streams: requests [1.67] | M1 4 streams: requests [3.5, 5.24, 15.68, 3.15]; B1 4 streams: requests [3.56, 5.28, 15.91, 3.2]; B2 4 streams: requests [3.57, 5.3, 15.95, 3.21]; M2 4 streams: requests [3.56, 5.3, 15.97, 3.2]. At one stream the cell is one request in flight; at four they all start together, and the aggregate is over the window to the last byte, so a straggler trades aggregate for per-stream rate the same way on both arms.

## Exactness

The three contracts with vision on: drafted equals `"draft": false` on 8 prompts; concurrent equals solo in 52 comparisons (0 unequal, 0 failed); resumed equals fresh on 7 turns, every resumed request served [13, 16, 16, 13, 24, 18, 3406] cached tokens and every fresh one 0. The image request's own pairs: same boot twice, `"draft": false`, streamed twice, and the second boot - all equal. `exactness.md` has the pairs and their basis.

## What this window does not establish

- It does not prove the decode is the format's exact decode: that is the host tests' job, which compare every tensor the reader hands the sink against binary64 references byte for byte.
- It does not compare the FP4 tower's replies with a floating-point twin of the same checkpoint: the snapshot stores the tower quantised, so no such twin exists. The int4 checkpoint's tower is floating point, but both configurations were refused before loading, so no reply from a floating-point tower was compared between main and this branch; that equality is not claimed.
- It is one machine, one boot an arm, one image; it says nothing about Metal, HIP, the Python engine, or two ranks.

## Restore

- prod /health 200 (start rc 0) 2026-11-16T22:58:29+00:00

## Method

Two images from verified trees, the recipe's own argument check on both (vision off and on), then one GPU window holding the box's GPU lock: the four text boots ABBA, main's refusing vision boot, the branch's vision boot and its no-cache-prompt reboot, and the two int4 boots. The raw run records (every command and its output) are kept with the measurement and are available on request.
