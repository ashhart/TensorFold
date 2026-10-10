# MiniMax H3 / FastH3 transformer on Metal (M5 tensor units)

This is a video model's denoising transformer, not a token model. It has no lane round and no drafted tokens, and
it runs its projections and attention in int8, which trades precision for speed. It is a test tool beside the
engine; nothing in the lane core or the server changes.

## What it is

`zig build tf-h3-dit -Dcpu=apple_m1 -Doptimize=ReleaseFast` builds `zig-out/bin/tf-h3-dit`. It reads the FastH3
checkpoint's shards and a case file, runs every denoising forward for the packed
[text | keyframe | audio | video] rows, and writes the final video and audio rows as raw float32.

```sh
zig-out/bin/tf-h3-dit <transformer_dir> <shards> case.safetensors out_prefix [dense | swiglu64 | row-scales | half | w7 | profile]
```

It needs the M5's tensor units and refuses to load without them.

## The case file

The prompt's rows, the schedule and the tile map come in one safetensors file. The exporter that writes it is not
in this change: it uses the Python 0.6 H3 family, which lives on a fork (`drowzeys/TensorFold`, `tools/zig/h3_case.py`
on branch `h3-1.0.4`).

| Tensor | Holds |
|---|---|
| `text` | The prompt's rows after the condition projection and the token refiner, bf16 |
| `video`, `audio`, `condition` | Starting noise rows; a first frame's rows when the clip starts from an image |
| `cos`, `sin` | Rotary tables for every packed row |
| `adaln`, `times`, `tables.N`, `final` | Each row's line in a block's six modulation tables per step, and the tables |
| `video_step`, `audio_step` | Per step: the sigma the model sees and the Euler ratio |
| `tile_slot`, `tile_sizes`, `geometry` | FastVideo's tile map: each row's padded slot, each tile's rows, the counts |
| `video_in.*`, `audio_in.*`, `video_out.*`, `audio_out.*`, `final_norm.weight` | The small float projections |
| `video_velocity`, `audio_velocity` | The reference's first forward, for the parity line the tool prints |

## Arithmetic

- Block projections are read from the checkpoint and rounded to int8 at load with a scale per output channel;
  activations are int8 with a scale per row (per row and 1024 columns for the MLP's wide rows).
- Attention is FastH3's video sparse attention over 64-row tiles: a video tile attends the prefix tiles and its top
  20% of video tiles by pooled score, and the pooled branch is gated in. Scores and values are int8; softmax weights
  are 8-bit per key tile with an online softmax across tiles.
- Rounding to bf16 where the reference's bf16 arithmetic rounds.

## Receipt

Measured 2026-10-10 on a Mac Studio (Apple M5 Ultra, 256 GB, macOS 27.0.1, Zig 0.17.0), checkpoint
`FastVideo/FastVideo-FastH3-8-Step-V2`, 864x480, 124 frames, 8 steps.

| Case | Rows | A forward | First forward against the Python float reference |
|---|---|---|---|
| Text to video | about 15,400 | 5.33 s | video cosine 0.9606, relative error 0.280; audio cosine 0.9981, relative error 0.062 |
| From a first frame | 16,501 | 6.15 to 6.44 s | not recorded for this case |

The reference is the Python 0.6 family in float with FastVideo's reference routing, so the difference includes int8
rounding of weights, activations and attention. Against the engine this code was split out of (the fork's branch
above), the final video and audio rows are bit-identical on both cases.

## Not done

- No exporter in the tree, so the tool cannot be exercised from this repository alone.
- No FP64 reference per projection class.
- One machine tested (M5 Ultra). Chips without tensor units are refused.
- The CUDA family for the same model is a separate change.
