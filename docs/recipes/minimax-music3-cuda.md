# MiniMax Music 3 on one CUDA GPU (DGX Spark GB10)

`music3-generate` runs MiniMax Music 3 (MiniMax-Music3, Comfy-Org's single-file checkpoints) on one CUDA GPU: the
autoregressive LM with its depth decoder, the flow-matching DiT and the DAV decoder, in Zig over the CUDA runtime with
kernels of its own (`zig/kernels/cuda/music3.cu`). It speaks the `--serve` protocol of our Mac MLX engine (branch
`music3-mlx` on the drowzeys fork, not part of this tree), so the Wasserman Slate AI Studio's `app/music.py`
([#941](https://github.com/ashhart/TensorFold/issues/941)) can drive either one.

No weights are included. MiniMax-Music3 is released under the MiniMax-Music3 Community License: a commercial product
must display "MiniMax-Music3" in its interface, use by a business above $20M in yearly revenue needs MiniMax's
written authorization, and a hosted service must keep the license's safeguards in place.

## Plan, written before building (ceilings and the odds of each lever)

Baseline (ComfyUI 5a665890's native nodes driven by the Studio, on the same GB10): a 60 s song in 289 s = AR 176 s (about 9
frames/s; a second of song is 25 frames) + DiT 96 s (30 Euler steps, CFG 1.7) + DAV 4 s, 29.3 GB. The Mac MLX engine (M5 Ultra)
does it in 75.8 s (AR 37.8 s at 40 frames/s, DiT 34.6 s).

What a frame reads. The AR body is a Qwen3-8B (36 layers, width 4096, GQA 32/8, MLP 12288): 6.95 G params, 13.9 GB
in bf16, read once a frame for both CFG rows together (batched). The depth decoder (4 layers, width 4096, 16 heads of
256, MLP 6144, a 4096x4096 input projection, 7 heads of 1024) is 0.59 G params, 1.17 GB, read 7 times a frame (the
first pass holds two positions, the next six one each): 8.2 GB. lm_head (16385 x 4096) 0.13 GB. About 22.2 GB a
frame in bf16; the body's KV cache adds 0.3 MB a context position (both rows, 36 layers), 0.6 GB a frame at 2,000
positions.

Ceilings at GB10's ~273 GB/s (to be replaced by the measured GEMV bandwidth below):

| AR weights | bytes a frame | ceiling frames/s | 60 s song (1,500 frames) | P(reach >= 80% of ceiling) |
| --- | --- | --- | --- | --- |
| bf16 | 22.2 GB | 12.3 | 122 s | 0.8 |
| int8 weight-only, per-output-channel scales | 11.1 GB | 24.6 | 61 s | 0.7 |
| fp8 e4m3 weight-only | 11.1 GB | 24.6 | 61 s | 0.5 (same bytes as int8, more quality risk, no gain over int8) |
| int4 / NVFP4 weight-only | 5.6 GB | 49 | 31 s | 0.3 for quality equal to bf16 |

ComfyUI's 9 frames/s is already ~70% of the bf16 ceiling, so a bf16 port alone wins at most ~1.3x on AR. The lever is
bytes: int8 weight-only halves them, and per-channel symmetric int8 of an 8B transformer is close to lossless (P(quality
equal to bf16 by the checks) ~0.8). Plan: int8 GEMV with the scale applied once per output row (no dequant in the
inner loop: the sum of int8 x bf16 products in fp32, times the row's scale), CFG rows batched, then CUDA graphs.

Launch overhead. A frame is 36 body layers + 7 x 4 depth layers + norms, attention, samplers: ~550 launches. At
~4 us each eager that is ~2 ms a frame (5% of a 40 ms frame); the frame step is captured once as a CUDA graph
(P(success) 0.9; it worked for Drama 3). Codes are drawn on the device (top-k CFG sampler), the stop code checked
every few frames on the host, so the GPU never waits for the host.

DiT. 2.42 G params, 36 blocks of width 2048 (32 heads of 64, MLP 8192), windows of 689 latent frames (200 AR frames)
every 344, CFG cond + uncond batched. A 60 s song is 5,168 latent frames = 14 windows, so a step is 14 x 2 x 690 =
19,320 tokens: 2 x 2.42 G x 19,320 = 94 TFLOP of GEMMs + ~4 TFLOP of attention a step, 2.9 PFLOP for 30 steps.
GB10's dense fp16 tensor rate with fp32 accumulation is ~100 TFLOP/s nominal; at 70% of it the DiT is ~41 s, at the
Mac's pace 35 s. ComfyUI's 96 s is ~30 TFLOP/s achieved, so there is ~2x of headroom from cuBLASLt fp16 GEMMs (fp32
accumulation) with fused epilogues and a proper attention (P(DiT <= 45 s) 0.6). flex_attention is wrong on sm_121 and
not used: attention here is cuBLAS batched GEMMs (QK^T in fp32, a softmax kernel, PV) or our own kernel. int8 tensor
cores (the H3 products) would double the GEMM rate again but are a quality risk; tried only if fp16 lands short.

DAV. ~4 s in ComfyUI; fp32 convolutions as im2col + cuBLAS SGEMM (TF32 off), in chunks with halo. Not a lever.

Target. 60 s song <= ~120 s on one GB10: AR int8 ~65-80 s + DiT ~40 s + DAV ~4 s + load. Reachable only with int8
AR and a DiT near 70% of the fp16 tensor rate; with bf16 AR the AR alone is >= 122 s, so the target is out of reach
in bf16 by construction.

Measured on an idle GB10 before building (`gemv_bench`, weights rotated through a 2 GB pool so L2 never holds
them): a plain read reaches 230 GB/s; a warp-a-row GEMV with two activation rows reaches 220 to 240 GB/s in bf16 and
in int8 alike; cuBLAS fp16 GEMMs with fp32 accumulation at the DiT's shapes (19,320 tokens) reach 96 to 104
TFLOP/s. So the real ceilings are 10.4 frames/s in bf16 and 20.7 frames/s in int8 (60 s of song: 145 s and 73 s of
AR), and the DiT's GEMMs alone are about 29 s for 60 s of song.

## What was built

| Stage | How | Notes |
| --- | --- | --- |
| Prompt | `prompt.zig`: ComfyUI's `prompt.py` regular expressions by hand, the checkpoint's own `tokenizer_json` through the core tokenizer | 10 of 10 cases equal to ComfyUI's ids, markdown, `<\|tag\|>` markers, `^`, CRLF and non-ASCII among them |
| AR prompt | both CFG rows (conditional, unconditional) through the body in cuBLAS bf16 GEMMs; int8 weights are dequantized a layer at a time for it | split-key attention kernel |
| AR frame | one CUDA graph: lm_head GEMV, the CFG top-k draw on the device, the depth decoder's 7 passes (first pass M = 4), the frame's 8 x 4096 states written for the DiT, the frame fed back through 36 layers (M = 2) | GEMVs are warp-a-row with the weights' first loads in flight before `griddepcontrol.wait` (PDL); gate/up GEMVs write the SwiGLU directly (rows n and N/2 + n in one warp); the host reads the drawn tokens every 16 frames for the stop code |
| int8 | weight-only, symmetric, quantized on the GPU at load; body and lm_head one fp32 scale a row, depth decoder and its heads one scale a group of 128 inputs (`--int8-group 0` for a row) | no cache file needed: quantizing is part of the load |
| DiT | windows of 689 latent rows every 344 (as ComfyUI), up to 16 windows x both CFG branches in one forward; cuBLASLt fp16 GEMMs (fp32 compute) with each shape's algorithm timed once; our flash attention (mma.sync m16n8k16, rope applied as q and k load); biases folded into the GLU and the next LayerNorm, so no GEMM carries an epilogue (the bias + accumulate epilogue picked 14 TFLOP/s kernels at some shapes) | rope angles rounded to fp16 before cos / sin, as ComfyUI does (`--comfy-rope 0` for fp32 angles) |
| DAV | fp32, every conv as one SGEMM a tap over time-major rows with zero halos (no TF32), transposed convs as one SGEMM and an overlap-add; chunks of 512 latent rows with 32 rows of context; ComfyUI's level rule (divide by max(1, 5 std)) | `--dav-tf32 1`: 66.8 dB against ComfyUI, 14% faster |

## Build and run

    zig build music3-generate -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc
    ./zig-out/bin/music3-generate --caption-file style.txt --lyrics-file lyrics.txt --seconds 60 --seed 7 \
        --out song.wav --stats song.json                      # int8 (default); --weights bf16 for the checkpoint's own
    ./zig-out/bin/music3-generate --serve 1                    # the MLX port's protocol, below
    ./zig-out/bin/music3-generate --ids --caption-file style.txt --lyrics-file lyrics.txt     # prompt ids, no GPU

`--model DIR` (or `$TF_MUSIC3_MODEL`) is required: the folder holding Comfy-Org's three MiniMax-Music3 files, read as
they are. Run it under a MemAvailable kill-guard on a Spark (`tools/music3` has none of its own; the Studio's
`voice.Served` client has one).

### --serve

The Mac engine's protocol, line for line: one JSON line on stderr when loaded,
`{"ready":true,"engine":"tensorfold-music3-cuda","load_s":..,"sample_rate":44100,"channels":2,"license_display":"MiniMax-Music3","weights":"int8"}`;
then one request a line on stdin, `{id, caption, lyrics | lyrics_file, wav, seconds, seed, ar_cfg, top_k, steps,
dit_cfg, stats, progress}`, and one answer a line on stderr, `{ok, id, wav, audio_s, frames, eos, ar_ms, dit_ms, dav_ms,
total_ms, rtf, first_audio_ms, stream, seed, peak_memory_gb}` (`{ok:false, error, id}` on a failure, `{"ok":false,
"error":"bad json"}` for a bad line). `"progress": true` adds `{"id":..,"progress":{"stage":"ar"|"dit"|"dav","done":n,"total":m}}`
lines, at most one a second a stage. `"stats"` names a JSON file of the request's timings. Log lines never start with
`{`. The engine ends at the end of stdin.

Checked with the Studio's own client (`tools/music3/served_test.py`): app/music.py's `Mlx` route (Studio main
86ae680) with `argv()` pointed at `music3-generate --serve 1` loaded the engine (37.3 s), answered two songs in a row
(20 s and 15 s), passed progress lines through, wrote the stats files, and unloaded on `stop`.

## Parity against ComfyUI (same GB10)

The reference is ComfyUI 5a665890's native MiniMax Music 3 nodes on the same GB10 (`--disable-cuda-graphs` so
forward hooks see every frame): hiphop bench prompt, 30 s, seed 7. ComfyUI ran the text encoder in bf16, the DiT in
fp16 and the DAV in fp32 on cuda:0. The dumping harness is a ComfyUI custom node and is not part of this tree (ComfyUI
is GPL-3); see [Reproduce the receipts](#reproduce-the-receipts) for what it records. Then `music3-generate --parity-ar / --parity-probe /
--parity-dit / --parity-dav REF --out MINE` and `python tools/music3/parity.py REF MINE` (numpy only). The tables'
JSON is in [minimax-music3-cuda/](minimax-music3-cuda/), beside the Mac MLX engine's own (`parity_mlx_mac.json`, against
ComfyUI on MPS).

| Check | CUDA bf16 | CUDA int8 (default) | int8, a scale a row everywhere | Mac MLX engine (bf16) |
| --- | --- | --- | --- | --- |
| prompt ids equal | yes (and 10/10 ids cases) | yes | yes | yes |
| frame states cos, mean / min | 0.999956 / 0.999556 | 0.999827 / 0.998345 | 0.999519 / 0.99816 | 0.999951 / 0.999821 |
| code-0 logits cos (cond), mean / min | 0.999977 / 0.999807 | 0.999897 / 0.997585 | 0.999897 | 0.999973 / 0.999724 |
| code-0 greedy agreement, cond / guided | 0.9747 / 0.9614 | 0.9667 / 0.9561 | 0.9667 / 0.9561 | 0.9774 / 0.964 |
| code-0 top-50 overlap | 0.9854 | 0.9799 | 0.9799 | 0.9872 |
| ComfyUI's drawn code, median rank in ours | 1 | 1 | 1 | 1 |
| depth logits cos, codebook 1..7 | 0.999957 to 0.999966 | 0.99978 to 0.99984 | 0.99913 to 0.99947 | 0.99993 to 0.99996 |
| depth greedy agreement, codebook 1..7 | 0.964 to 0.981 | 0.945 to 0.967 | 0.908 to 0.937 | 0.951 to 0.983 |
| DiT probe cos (cond / uncond), sigma 1.0 | 0.999991 / 0.999994 | (the DiT is fp16 in both) | | 0.99999 / 0.999998 |
| DiT probe cos, sigma 0.5 | 0.999958 / 0.999961 | | | 0.999971 / 0.999977 |
| DiT probe cos, sigma 0.1 | 0.999987 / 0.999987 | | | 0.999992 / 0.999993 |
| DiT final latent (ComfyUI's noise and states), cos / SNR | 0.999914 / 37.6 dB | | | 0.999838 / 34.9 dB |
| DAV probe (one pass), cos / SNR | 1.0000000 / 71.0 dB | | | 0.9999993 / 52.0 dB |
| DAV of ComfyUI's latent (chunked as served), cos / SNR | 1.0000000 / 71.0 dB | | | 0.9999994 / 50.5 dB |

Teacher forcing makes the AR rows comparable; the draws themselves cannot match (torch's multinomial vs our Philox
inverse-CDF draw), so a song is not ComfyUI's song for the same seed, only the same model.

## Lyric accuracy

Three songs, the same lyrics in bf16 and int8 (45 s each: hiphop seed 7, pop seed 11, long-pop seed 5), plus
ComfyUI's own 60 s hiphop song from the baseline run. Whisper is faster-whisper large-v3-turbo on the CPU through the
Studio's `qa.transcribe`; `tools/music3/lyrics_check.py` runs the engine first and Whisper after it ends.
"Studio score" is app/music.py's `heard` (words heard up to the last lyric word heard); "span" counts only matched
runs of 3 or more words, so a stray word far down the lyrics does not stretch the span (the int8 long song's "down"
matched the bridge's "go down": score 31, span 100).

| Song | bf16 Studio score / span | int8 Studio score / span | words heard bf16 / int8 |
| --- | --- | --- | --- |
| hiphop, seed 7 | 96 / 96 | 98 / 98 | 130 of 135 / 127 of 130 |
| pop, seed 11 | 100 / 100 | 100 / 100 | 78 of 78 / 59 of 59 |
| long-pop, seed 5 | 90 / 89 | 31 / 100 | 50 of 56 / 52 of 52 |
| ComfyUI hiphop 60 s (reference) | 96 / 96 | | 159 of 166 |

## Speed and memory (an idle GB10)

Hiphop bench prompt seed 7 (the 120 s request ended itself at 63.3 s: the model drew its stop code; the long-pop
prompt seed 5 is the 120 s row). AR GB/s is the weight bytes a frame reads (int8 11.15 GB, bf16 22.3 GB) times
frames a second. Peak is MemTotal - MemAvailable sampled twice a second, minus the idle machine's (about 5.1 GB);
device is the engine's own allocations. Load (37 s, cold page cache) is outside "total".

| Song | Weights | AR s (frames/s, GB/s) | DiT s | DAV s | Total s | RTF | Peak machine GB | Device GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 s | int8 | 36.4 (20.9, 233) | 22.0 | 1.9 | 60.3 | 2.01 | 16.6 | 16.5 |
| 60 s | int8 | 73.9 (20.4, 228) | 46.4 | 3.6 | 123.9 | 2.07 | 17.9 | 18.1 |
| 63 s (asked 120, ended itself) | int8 | 78.0 (20.4, 228) | 46.3 | 3.9 | 128.2 | 2.03 | 18.9 | 18.5 |
| 120 s (long-pop) | int8 | 151.7 (19.9, 222) | 93.0 | 7.3 | 252.0 | 2.10 | 19.3 | 18.9 |
| 30 s | bf16 | 72.1 (10.5, 233) | 21.9 | 1.9 | 96.0 | 3.20 | 24.0 | 24.1 |
| 60 s | bf16 | 144.4 (10.4, 232) | 46.3 | 3.6 | 194.3 | 3.24 | 25.5 | 25.8 |
| 63 s (asked 120, ended itself) | bf16 | 147.9 (10.4, 232) | 46.3 | 3.7 | 198.0 | 3.23 | 25.7 | 26.1 |
| 120 s (long-pop) | bf16 | 292.7 (10.3, 229) | 93.1 | 7.3 | 393.1 | 3.28 | 26.6 | 26.5 |

Against the baseline (ComfyUI through the Studio, 60 s: 289 s, AR 176 s at ~9 frames/s, DiT 96 s, DAV 4 s, 29.3 GB):
int8 is 2.3x faster in total (AR 2.4x, DiT 2.1x) in 18 GB; bf16 is 1.5x. The Mac M5 Ultra does it in 75.8 s.

Where the 60 s target of ~120 s stands: 123.9 s. The AR is at the measured read ceiling (228 of 230 GB/s), so
only fewer bytes move it (int4 / NVFP4, not tried: the depth decoder is already the most sensitive part in int8).
The DiT is 46 s: its GEMMs run at 80 to 98 TFLOP/s (about 31 s of it, the hardware's fp16 rate); GLU, LayerNorm
and attention are the other 15 s. What is left is fp8 GEMMs for the DiT (GB10's fp8 rate is twice fp16; not tried,
a quality risk against the fp16 reference) or fusing the GLU into a hand-written ff_out GEMM.

## Default: int8, and why

int8 (body and lm_head a scale a row, depth decoder and heads a scale a group of 128) is the default: it halves the
bytes a frame reads (2x AR, 1.6x total) and 7.6 GB less memory, its parity numbers are within a few 1e-4 of bf16, the
ComfyUI-drawn code is rank 1 in our logits in both, and its lyric accuracy is not lower than bf16's on three songs.
Group scales for the depth decoder cost <1% of AR speed and lift its greedy agreement from 0.91-0.94 to 0.95-0.97.
`--weights bf16` reads the checkpoint's own numbers.

## Kernel contract

`zig/tests/cuda/contract_music3.zig`: m3_rmsnorm, m3_gemv_bf16_2, m3_gemv_i8_2 and m3_gemv_g8_2 through gates 1-4;
all 55 music3 launches in `registry.zig` with their records for sm_80, sm_86, sm_89, sm_120 and sm_121 (from
`tf-cuda-test kernel-contract-limits` over `-Dsm=80,86,89,120,121` fatbins). On a GB10 (sm_121), on main 6ee828c2:

    SUMMARY 16 kernels: 16 pass gates 1-4, 0 fail, 0 known findings; 5 controls: 5 red as planted, 0 missed;
    0 limit problems
    PASS kernel-contract-limits: every registered launch fits sm_80, sm_86, sm_89, sm_120 and sm_121, every control red

`python3 tools/zig/lean_check.py`: 517 problems, the same 517 as main (no new findings); `zig build test` (450/456,
6 JPEG cases skipped without libjpeg-turbo) and `zig build test-cuda-host` (856/856, main 852/852) pass;
`zig fmt --check zig build.zig` clean.

## Reproduce the receipts

The reference run hooks ComfyUI's own loaders, nodes and sampler and only records tensors (numpy files in `REF`):

| Stage | ComfyUI run | Files |
| --- | --- | --- |
| ids | the MiniMax text encoder's tokenizer on a list of [caption, lyrics] cases | `ids_cases.json` |
| ar | the AR on the caption and lyrics, teacher state recorded each frame | `hidden.npy` [T, 32768], `codes.npy` [F, 8], `tokens.npy`, `ids.json`, `c0_logits.npy` [F, 2, 16385] fp16, `depth_logits.npy` [F, 7, 2, 1024] |
| dit | KSampler (euler, simple) over `hidden.npy`, the zeroed condition as negative | `latent.npy`, `noise.npy` |
| dav | VAE decode of `latent.npy` with the level rule | `audio.npy` [2, N] |
| probe | the first 200 frame states, seeded noise and the latent's first 689 rows: the DiT's output at sigma 1.0 / 0.5 / 0.1 for both branches, the DAV of `probe_latent.npy` | `probe_dit.npz`, `probe_dav.npy` |

Then, on the same machine:

    music3-generate --model DIR --weights bf16 --parity-ar REF --parity-probe REF --parity-dit REF --parity-dav REF \
        --out MINE --caption-file C --lyrics-file L
    python tools/music3/parity.py REF MINE --report parity.json

## Not tested

- GPUs other than GB10 (sm_121): the kernels build and pass gate 5 for sm_80/86/89/120, nothing ran there. PDL is used
  only on sm_90 and up.
- fp8 or int4 / NVFP4 weights, fp8 DiT GEMMs, an int8 weight cache file (quantizing at load costs nothing extra).
- Songs over 120 s (the cap is 360 s / 9000 frames; the caches and windows are sized for it, not run), prompts near
  5000 tokens, an instrumental-only song's lyric check, non-ASCII section tags (lower-cased as ASCII only).
- The Studio app itself driving this engine (only its `Served` client was run, from a test script).
- Two requests at once (the server answers one at a time, as the Mac engine does).
