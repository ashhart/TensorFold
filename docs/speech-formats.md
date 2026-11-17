# Speech reply formats

`POST /v1/audio/speech` answers in the format `response_format` names. A request without one gets WAV, as before.

| `response_format` | Reply | Needs |
|---|---|---|
| `wav` (default) | 16-bit mono PCM in a RIFF/WAVE header, `audio/wav` | nothing |
| `pcm` | the same samples raw, little-endian, `audio/pcm` | nothing |
| `flac` | FLAC, lossless: the decoded samples equal the `pcm` reply's, `audio/flac` | nothing |
| `mp3` | MPEG-1 Layer III, 128 kbit/s constant, mono, `audio/mpeg` | LAME (`libmp3lame`) |
| `opus` | Opus in an Ogg container (RFC 7845), 64 kbit/s, mono, 48 kHz, `audio/ogg` | `libopus` |
| `aac` | AAC-LC in ADTS frames, 96 kbit/s, mono, `audio/aac` | FDK AAC (`libfdk-aac`) |

Any other name is refused with `response_format must be one of: aac, flac, mp3, opus, pcm, wav`. `"stream": true` takes
`pcm` only.

- **Sample rate:** the model's own (44,100 Hz for Drama 3 and ZONOS3). Opus has no 44.1 kHz mode, so an Opus reply is
  the samples resampled to 48 kHz (a Kaiser-windowed sinc, 32 taps a sample); its header records the original rate.
- **Raw PCM names its rate:** a `pcm` reply, whole or streamed, carries `X-Audio-Sample-Rate`. Clients written for
  OpenAI's 24 kHz PCM read it there.
- **Replies repeat:** the encoders are given the same samples in the same way every time and the Ogg stream's
  serial number is fixed, so a request that repeats as WAV repeats in the other formats too, on one machine with one
  library release. Encoded bytes are not promised equal across releases of the three libraries.
- **FLAC is written by the server:** 4,096-sample blocks, the best fixed predictor of order 0 to 4 a block, one Rice
  parameter. It favors a small encoder over the smallest file.

## The three libraries

TensorFold does not ship LAME, libopus or FDK AAC. Each is looked for when the first request asks for its format, and the
answer is kept until the server stops:

1. the file `TENSORFOLD_LIBMP3LAME`, `TENSORFOLD_LIBOPUS` or `TENSORFOLD_LIBFDKAAC` names; while the variable is set, even to an empty value,
   nothing else is tried;
2. `lib/` beside the server's `bin/`, then the server's own folder;
3. on macOS, Homebrew's `lame`, `opus` and `fdk-aac` packages;
4. the system loader's path.

| | LAME | libopus | FDK AAC |
|---|---|---|---|
| Linux | `libmp3lame.so.0` | `libopus.so.0` | `libfdk-aac.so.2` |
| macOS | `libmp3lame.0.dylib` | `libopus.0.dylib` | `libfdk-aac.2.dylib` |
| Windows | `libmp3lame-0.dll` | `libopus-0.dll` | `libfdk-aac-2.dll` |

Run with LAME 3.100, libopus 1.4 and FDK AAC 2.0.2 on Linux; the macOS and Windows names were not exercised. When a library is missing, a request for its format is refused with
400 and the line to act on, for example `libmp3lame.so.0 is required for MP3 speech replies (response_format mp3);
install LAME 3.100 (libmp3lame) and set TENSORFOLD_LIBMP3LAME to its file, or put it on the loader path`. The other
formats are unaffected.

## OpenAI's clients

A program written for OpenAI's text to speech works against this route when it is given the server's address and
model name, with these differences:

- OpenAI answers MP3 when no format is named; this route answers WAV. Clients that name a format are unaffected.
- `speed` other than null and `instructions` are refused: the speech models served have no such controls.
- ZONOS3 takes OpenAI's stock voice names (`alloy`, `ash`, `ballad`, `cedar`, `coral`, `echo`, `fable`, `marin`,
  `nova`, `onyx`, `sage`, `shimmer`, `verse`) as its default voice, since such clients send one unless told
  otherwise; `/v1/voices` lists the model's own. Drama 3's voices are unchanged.
- A server speaks one request at a time unless its model serves more; a request that arrives meanwhile gets 503.
