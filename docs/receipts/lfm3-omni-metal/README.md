# LFM3-Omni-2B on Metal: receipt

The checks behind [the recipe](../../recipes/lfm3-omni-metal.md), run for the review list of
[#1635](https://github.com/ashhart/TensorFold/pull/1635) against the gate agreed in
[#1627](https://github.com/ashhart/TensorFold/issues/1627). The margins were written into the recipe before these
runs. The audio margins and the text-miss bound hold; the fp32 bound holds on some rows and not on others, and
the table says which.

## Setup

- M5 Ultra (256 GB), macOS 27.0.1, Zig 0.17.0, Xcode 27.2 with the Metal compiler 32023.921.
- Weights: `LiquidAI/LFM3-Omni-2B` revision `5797d07c125503bd4f4c0e7f915ae2d3d703e8de`, bf16 as stored.
- Reference: `transformers` 5.21.0 with the checkpoint's own code, PyTorch 2.14.1, MPS, on the same Mac, in bf16
  and in fp32. Its codec is the bundled one as stored (bf16 weights); `reference_codec.py` casts it to fp32.
- The 16 synthetic turns of `tools/lfm3_omni/utterances.txt` (`make_utterances.sh`, macOS voices), a voice-assistant
  system text, greedy text and codes, identity codebook order.
- Other services were resident on the Mac; the runs waited while its interactive voice service spoke
  (`LFM3_OMNI_BUSY`). The reference's dumps are byte-identical to those of the first receipt (384 files).

```sh
T=tools/lfm3_omni; B=zig-out/bin/lfm3-omni-generate   # $MODEL, $WAVS (u00.wav ... u15.wav), $SYS as in the recipe
python $T/reference_dump.py $MODEL ref_bf16 mps bf16 $SYS $WAVS/u*.wav
python $T/reference_dump.py $MODEL ref_f32 mps f32 $SYS $WAVS/u*.wav
sh $T/parity.sh $B $MODEL ref_bf16 tf_bf16 $SYS $WAVS mps bf16
sh $T/parity.sh $B $MODEL ref_f32 tf_f32 $SYS $WAVS mps f32
python $T/reference_forced.py $MODEL ref_bf16 forced_bf16.json mps bf16 $SYS $WAVS/u*.wav   # itself: 100%
python $T/reference_forced.py $MODEL ref_f32 forced_f32.json mps f32 $SYS $WAVS/u*.wav      # itself: 100%
python $T/reference_forced.py $MODEL ref_f32 forced_bf16.json mps bf16 $SYS $WAVS/u*.wav    # bf16 on fp32's replies
for d in ref_f32/u*; do
  python $T/reference_depth.py $MODEL $d prep mps bf16 ref_depth_bf16.npy
  python $T/reference_depth.py $MODEL $d score ref_depth.npy ref_depth_bf16.npy depth_noise.json
done
python $T/reference_codec.py $MODEL mps f32 ref_bf16/u*
for d in bf16 f32; do for w in $WAVS/u*.wav; do u=$(basename $w .wav)   # step rows of the teacher-forced runs
  $B --model $MODEL --wav $w --system-file $SYS --audio-temperature 0 --identity-order 1 \
    --force ref_$d/$u --dump tf_$d/$u/forced
done; done
python $T/precision_table.py ref_bf16 ref_f32 tf_bf16 tf_f32 precision.json
python $T/text_misses.py ref_bf16 tf_bf16 text-misses.json
python $T/text_misses.py ref_f32 tf_f32 text-misses-fp32.json forced_f32.json
```

## Every stage against bf16 and against fp32

`precision.json`. The third column is the reference's own bf16 run against its fp32 run: the gate's bound for the
second. Teacher-forced rows follow each reference's own reply (524 text slots and 1,752 frames in bf16, 468 and 1,545
in fp32); a frame's later passes see the scorer's own earlier picks, for the engine and for the reference alike.

| Stage | Metal vs MPS bf16 | Metal vs MPS fp32 | MPS bf16 vs MPS fp32 | Metal no further from fp32 |
| --- | --- | --- | --- | --- |
| Log-mel, cosine (min), max abs | 1.000000, 3.1e-4 | 1.000000, 3.1e-4 | identical (fp32 on the host in both) | n/a |
| Speech encoder, cosine (min) | 0.999991 | 0.999993 | 0.999994 | no, by 1e-6 |
| Projector, cosine (min) | 0.999898 | 0.999891 | 0.999920 | no, by 2.9e-5 |
| Prompt token ids | 16/16 identical | 16/16 identical | 16/16 identical | n/a |
| Backbone prefill rows, cosine: median of medians (min of medians) | 0.999822 (0.999769) | 0.999881 (0.999837) | 0.999882 (0.999844) | no, by 1e-6 (7e-6) |
| Backbone step rows, teacher-forced, cosine: median of medians (min of medians) | 0.999914 (0.999900) | 0.999935 (0.999912) | 0.999931 (0.999906) | yes |
| Text top-1, teacher-forced | 520/524 (99.2%) | 460/468 (98.3%) | 464/468 (99.1%) | no: 8 misses against 4 |
| Depth decoder on the reference's states, greedy agreement | 20084/21024 (95.5%) | 17126/18540 (92.4%) | 17143/18540 (92.5%) | no, by 17 picks |
| Depth decoder logits, cosine: median (min) | 0.999994 (0.999889) | 0.999994 (0.999890) | 0.999994 (0.999869) | yes |
| Codes, teacher-forced through the whole loop | 18753/21024 (89.2%) | 16094/18540 (86.8%) | 16019/18540 (86.4%) | yes |
| Codec on the bf16 reference's codes, waveform SNR: median (min) | 37.24 dB (25.09 dB) | 36.16 dB (20.20 dB) | 36.41 dB (21.15 dB) | no, by 0.25 dB (0.95 dB) |
| Codec spectrum rows, cosine (min) | 0.999938 | 0.999218 | 0.999523 | no |

The depth decoder's states are each reference's own, rounded to bf16 so the three runs read the same rows.
"No further from an fp32 run than the reference's own bf16 run" holds for the step rows, the depth logits and the
codes through the loop. It does not hold for the encoder, the projector, the prefill rows, text top-1, the depth
decoder's greedy agreement and the codec; the differences are the ones in the last column.

## Teacher-forced text misses

`text-misses.json`: the four misses against the bf16 reference, with the reference's own logits of the two tokens
(`reference_forced.py`, its top four a slot) and the engine's (`--force` prints them). A bf16 ulp at these logits is
0.0625. All four are within 2 bf16 ulps of the reference's pick by the reference's logits.

| Turn, slot | Reference token | Engine token | Reference logits (its pick, the engine's) | Reference margin | Engine logits (reference's, its pick) | Engine margin |
| --- | --- | --- | --- | --- | --- | --- |
| u05, 58 | 278 | 671 | 15.625, 15.625 | 0 ulps | 15.5, 15.6875 | 3 ulps |
| u06, 39 | 6134 | 3246 | 10.6875, 10.625 | 1 ulp | 10.75, 10.75 | 0 ulps |
| u13, 38 | 2222 | 1125 | 16.375, 16.25 | 1 ulp | 16.375, 16.375 | 0 ulps |
| u15, 1 | 6963 | 47 | 15.5, 15.375 | 2 ulps | 15.4375, 15.4375 | 0 ulps |

`text-misses-fp32.json` lists the eight misses against the fp32 reference the same way. By the fp32 logits their
margins are 0.19, 0.57, 0.79, 2.15, 2.17, 2.40, 5.96 and 8.82 bf16 ulps. The reference's own bf16 run misses four
slots of the fp32 replies, at 0.02, 0.02, 0.19 and 5.96 ulps; the 5.96 is the same slot (u15, 3) and the same token.

## Replies: audio margins

Replies with the card's settings (temperature 0.4, penalty 1.2, seeds 1000 + i) through the server and
`speech_chat_client.py`; the reference through `reference_speech.py`. Two rounds, this head, the rebased head before
the codec and pick changes, and the reference in turn. `score_replies.py`: UTMOS22 strong (SpeechMOS v1.2.0) and
the checkpoint's own ASR on each reply against its text. `quality.json`, `latency.json`.

| | TensorFold Metal | Reference (MPS bf16) | Margin held to | Met |
| --- | --- | --- | --- | --- |
| UTMOS22 strong, median (min) | 4.427 (4.352) | 4.438 (4.149) | median no more than 0.1 below | yes: 0.011 below |
| Speech vs its own text, WER | 1.47% | 1.20% | no more than 1.0 point above | yes: 0.27 points above |

Both rounds give the same replies, byte for byte, on each side (16/16).

| Median of 16 turns, rounds 1 / 2 | This head | Before the two changes | Reference (MPS bf16) |
| --- | --- | --- | --- |
| First text | 0.034 / 0.034 s | 0.034 / 0.034 s | n/a |
| First audio | 0.064 / 0.064 s | 0.063 / 0.063 s | 0.175 / 0.171 s |
| Whole reply (6.5 s of speech) | 0.655 / 0.655 s | 0.643 / 0.645 s | 2.314 / 2.323 s |

## Streamed and whole decodes

`chunking.txt`. The bf16 reference's codes of each turn through the codec in five chunkings (`--first-chunk` /
`--chunk`: 1 / 8 as served, 64 / 64, 3 / 5, 1 / 1, 2 / 7), the spectrum rows and the waveform compared byte for byte
against the served chunking:

- this head, tensor units: 128 comparisons, 0 differ;
- this head, `LFM3_OMNI_NAX=0` (the simdgroup products): 128 comparisons, 0 differ;
- before the change: 128 of 128 differ, the waveforms 23.0 to 51.6 dB apart (median 41.2 dB).

Two things made a chunk's rows depend on the chunk: the few-row product for a one-frame chunk, and the tiled
attention's key sums starting at its tile's first window. The codec now takes the tiled product at every row count
and attention one row a launch. The other stacks launch as before: the greedy dumps of every stage before the codec
are byte-identical to the previous head's (160 of 160 files).

## The code pick's draw

`om_depth_pick` now draws u = float(h >> 40) / 2^24, in [0, 1). The contract's third registration for the kernel runs
a seed whose draws reach both ends of the 24 bits; its control restores the old line and goes red. Of the 16 replies
above, three change their frame count against the head before the change; the text of all 16 is the same.

## No text family changed

`text-family-tokens.json`. `tensorfold-native` built from main at `b4f1bf263` (this branch's base) and from this
branch, each serving Qwen3.8-27B MLX 4-bit (revision `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c`), main and branch in
turn, two rounds:

```sh
tensorfold-native serve $QWEN --name text --host 127.0.0.1 --port 8473 --no-update-check
python3 tools/bench_openai.py http://127.0.0.1:8473 text --tokens 256 --reps 3 --temperatures 0 --output tokens.json
```

| Prompt | main | branch |
| --- | --- | --- |
| `fibonacci-raw`, 256 tokens | `a10c9b848ed9`, 6 of 6 | `a10c9b848ed9`, 6 of 6 |
| `gpu-chat-no-think`, 256 tokens | `a06fab348fbe`, 6 of 6 | `a06fab348fbe`, 6 of 6 |

The stream's SHA-256 is equal too. Decode medians were 47.9 to 48.1 tokens a second on main and 44.1 to 47.9 on the
branch, with other services resident on the Mac: not a speed measurement.
