# LFM3-Omni-2B speech to speech, native Metal

`lfm3_omni` runs LiquidAI's [`LiquidAI/LFM3-Omni-2B`](https://huggingface.co/LiquidAI/LFM3-Omni-2B) (revision
`5797d07c125503bd4f4c0e7f915ae2d3d703e8de`) on Apple silicon: a spoken turn in, interleaved text and 24 kHz speech out,
streamed. The weights are the bf16 release as stored, the bundled Liquid codec (`audio_tokenizer/`) included. The
engine is `zig/src/families/lfm3_omni/metal/` over one metallib (`zig/kernels/metal/lfm3_omni/`). Checked on an M5
Ultra; chips without tensor units (M1 to M4) are untested. The vision tower is not loaded.

```bash
hf download LiquidAI/LFM3-Omni-2B --revision 5797d07c125503bd4f4c0e7f915ae2d3d703e8de --local-dir "$HOME/models/lfm3-omni-2b"
# the Metal compiler: xcodebuild -downloadComponent MetalToolchain (macOS 26 and later), with Xcode selected
zig build lfm3-omni-generate -Doptimize=ReleaseFast -Dcpu=apple_m1
# one turn: a WAV in (any rate, resampled to 16 kHz), the reply as a WAV and its text
zig-out/bin/lfm3-omni-generate --model "$HOME/models/lfm3-omni-2b" --wav question.wav --out reply.wav \
  --system "Respond with interleaved text and audio. You are a helpful voice assistant."
# a server: POST /v1/audio/chat, GET /health
zig-out/bin/lfm3-omni-generate --model "$HOME/models/lfm3-omni-2b" --serve 127.0.0.1:8441 --system-file system.txt
```

About 4.4 GB of unified memory.

## The endpoint

`POST /v1/audio/chat` takes JSON: `audio` (a base64 WAV, required), `system`, `stream` (default true),
`audio_temperature` (0.4), `audio_repetition_penalty` (1.2), `seed`, `max_new` (generated slots, a text token or a
speech frame each). With `stream` the answer is server-sent events as the reply is made:

```text
data: {"type":"text","delta":"I'm on it,"}
data: {"type":"audio","format":"pcm16","sample_rate":24000,"data":"<base64 s16le>"}
...
data: {"type":"done","text":"...","frames":87,"speech_s":6.96,"first_text_ms":36.0,"first_audio_ms":61.2,"total_ms":640.4}
data: [DONE]
```

Without it, one JSON object with the same fields and `audio`, the whole reply as a base64 WAV. The server takes one
request at a time; it is the tool's own (not a route of `tensorfold-native serve`, whose speech route is text in).
Text is greedy, as the reference's `do_sample=False`; speech frames are sampled with the card's settings.

## What runs

- Log-mel (16 kHz, 400-point periodic Hann, 128 Slaney bins) as fp32 products on the device; the speech encoder's
  three stride-2 convolutions, its 17 LFM2 layers and the projector.
- The 24-layer LFM3 backbone (gated short conv, full and sliding attention), the prompt in chunks of 512 rows.
- The depth decoder's three mask passes over a frame's 12 codebook slots, each codebook's pick on the GPU (repetition
  penalty, greedy or a Gumbel-max draw, exactly a softmax sample).
- The Liquid codec's decoder and its spectrum head, the inverse STFT as an fp32 product; overlap-add on the host.
- Products of many rows (prompt chunks, the encoder, codec chunks) on the tensor unit through the prefill GEMM's
  blocks (`zig/kernels/metal/prefill/gemm_nax.metal`), fp32 sums; a GPU without tensor units takes a
  simdgroup-matrix product (`LFM3_OMNI_NAX=0` forces it).
- Up to 16 rows (a backbone row, the depth decoder's 12 slots): a weight-streaming product, the heads' norm, rotary
  and cache write in one dispatch, one-row attention with LFM3's output gate inside, SwiGLU as the product's
  epilogue; the greedy text pick in two stages.
- One serial command buffer from one copy back to the next: a backbone row runs while the host does the overlap-add;
  the first frame is decoded alone, then every 8.

Every op rounds to bf16 where the reference's op rounds. `LFM3_OMNI_PROFILE=1` logs each phase's GPU time per reply.

## Checks

`tools/lfm3_omni/` holds the checks' clients (Python dev tools, not part of the engine): `make_utterances.sh` makes
the 16 synthetic spoken turns of `utterances.txt` with macOS voices; `reference_dump.py` and `reference_depth.py` dump
the reference's stages (transformers with the checkpoint's own code); `parity.sh` runs `lfm3-omni-generate --dump`,
`--force` and `--depth-check` against them and `summarize_parity.py` prints the table below;
`speech_chat_client.py` streams turns through the server and times them; `reference_speech.py` does the same with the
reference runtime; `score_replies.py` scores replies (UTMOS22 strong through SpeechMOS, and the checkpoint's own ASR
against each reply's text).

## Receipt

M5 Ultra (256 GB), macOS 27.0.1, Xcode 27.1; the reference is `transformers` 5.21.0 with the checkpoint's own code on
PyTorch 2.14.1 (MPS, bf16), on the same Mac. The 16 turns of `tools/lfm3_omni/utterances.txt` (macOS voices), a
voice-assistant system text, identity codebook order, greedy text and codes:

| stage | check | Metal vs MPS bf16 |
| --- | --- | --- |
| log-mel | cosine (min), max abs | 1.000000, 3.1e-4 |
| speech encoder | cosine (min of 16) | 0.99999 |
| projector | cosine (min) | 0.99990 |
| prompt | token ids | 16/16 identical |
| backbone prefill | row cosine (median of medians, min of medians) | 0.99982, 0.99977 |
| text | teacher-forced greedy top-1 | 520/524 (99.2%) |
| depth decoder | on the reference's own states: greedy agreement, logits cosine median / min | 95.5%, 0.999994 / 0.99989 |
| codes | teacher-forced through the whole loop | 89.2% |
| codec | the reference's codes to audio: SNR median (min); spectrum rows cosine (min) | 36.6 dB (25.1 dB); 0.99994 |

Text and code misses are near ties (within 2 bf16 ulps of the reference's pick); 15% of the reference's own greedy
codebook picks are exact bf16 ties, so free-running codes part after a frame or two. Against MPS fp32 the port sits
where the reference's own bf16 run sits (encoder 0.999995, prefill rows 0.99988 median, for both).

Replies with the card's audio settings (temperature 0.4, penalty 1.2, seeds 1000 + i, the server and
`speech_chat_client.py`; the reference through `reference_speech.py`, its first audio being the first sampled frame
plus that frame's codec decode). Two rounds, the servers and the reference alternated; other services were resident
on the Mac but idle (runs paused while its interactive voice service was in use):

| | TensorFold Metal | reference runtime (MPS bf16) |
| --- | --- | --- |
| first text, median | 0.034 s | |
| first audio, median | 0.063 s | 0.172 to 0.191 s |
| whole reply, median (6.5 s of speech) | 0.67 to 0.68 s | 2.41 to 2.54 s |
| UTMOS22 strong, median (min) | 4.43 (4.35) | 4.44 (4.15) |
| speech vs its own text, WER | 1.5% | 1.2% |

WER is the checkpoint's own ASR on each reply's audio against the reply's streamed text.


## Licence

The weights are under Liquid AI's LFM Open License v1.0 (`license: other`, `lfm1.0` on the model card; the text is the
`LICENSE` file in the checkpoint). Use, changes and redistribution are allowed; commercial use is licensed only to an
organisation under US$10 million in annual revenue, and a qualified non-profit may use it for research or other
non-commercial purposes regardless. TensorFold's code stays Apache-2.0.
