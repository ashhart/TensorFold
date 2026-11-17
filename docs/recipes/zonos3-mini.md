# ZONOS3 Mini on CUDA

`zig/src/families/zonos3` runs Zyphra's ZONOS3 Mini text-to-speech model on the CUDA runtime, from the files the
publisher released, and serves it on `POST /v1/audio/speech`. This page records what is ported, where every source
came from, and how each part was checked, so the checks can be rerun.

Status: a work in progress on one branch, proposed in #3615. The kernels are not registered with the kernel contract
yet, memory admission is not written, and the quality suite has not been run on this build.

## Sources

| What | Where | Pin |
| --- | --- | --- |
| Checkpoint | `huggingface.co/Zyphra/ZONOS3-Mini` (Apache-2.0) | revision `d2d142a86b1e6aeb4252679673c86ecd86177bf8` |
| Reference code | `github.com/Zyphra/ZONOS3-Mini` (MIT) | commit `96d9e0ec1e39083ed336ac78f24e3dbc93721c0f` |
| Reference environment | Python 3.12.3, torch 2.9.1+cu128, torchaudio 2.9.1+cu128, transformers 4.57.3 | the repo's `constraints.txt` |

Nothing is converted: the server reads the release folder as published.

| File in the release | Read by | sha256 |
| --- | --- | --- |
| `lm/zonos3_mini/model.pth` (bf16) | the backbone | `3b4ac0a6e3002bd575ad9a0761ae7524806a908f553042171dead8f3a2146caa` |
| `lm/zonos3_mini/params.json` | the geometry check | `28dc488d7142831d739f427e048c699b7ed12852d88da0a9123a137f15aa5a5b` |
| `lm/zonos3_mini/tokenizer/superbpe8192.json` | the text tokenizer | `90322f910600027dee7fe63d400d1d9dc1035c167d560aadf0871e3661d08bf3` |
| `autoencoder/generator.pt` (fp32) | the codec's quantizer, semantic head, resamplers, encoder | `75a587a3eda9ca3d345f0c305756a7ff8619b01685c261b08ae243c69c2caf9a` |
| `autoencoder/heads/keep2_plain/step_0050000.pt` (fp32) | the codec decoder's stages and iSTFT head | `7432e335e717417961267f2242547ce389a796cdc296a8b8d5289658e3c28936` |
| `autoencoder/teacher/model.safetensors` (bf16) | the encoder's SSL teacher | `a15182741d6ee3cd780b8b830f49fa6b7fcce383655d451970841b24114767a5` |
| `registries/codec_registry_keep2_plain.json`, `registries/silence_frame_*.json` | the decoder's files, the silence frame | `5437b4159b2bdbc33d3c1a27a316df7ae361b06f5ce3c9822f585402bcb4351a` (the registry) |
| `gguf/voices/<language>/*.codes.npy`, `*.embedding.npy` | the default voices | |

The `.pth` files are torch.save archives; `zig/src/core/torch_pth.zig` reads them without Python (stored zip
members, a pickle VM limited to the tensor allow-list, nested state dicts named by their keys).

## What runs

- **Backbone** (`cuda/engine.zig`, `zonos3.cu`): 11 layers at width 1,024, nine Gated DeltaNet and two global
  attention layers, Full AttnRes residuals, a dense feed-forward in layer 0 and a 64-expert top-2 latent mixture of
  experts in the rest, one linear head for 10 codebooks of 4,098 logits a frame.
- **Sampler and generation** (`sampler.zig`, `generation.zig`, on the host): the release's masks, repetition
  penalty, temperature, top-k, top-p and min-p, its end-of-audio countdown and cold-start padding. Draws come from a
  counter-based generator keyed by the request's seed, the frame and the codebook.
- **Codec decoder** (`cuda/codec.zig`, `zonos3_codec.cu`): the `keep2_plain` decoder, the reference servers'
  default: code tables, the semantic head, five causal transformer stages from 15 Hz to 240 Hz, the ConvNeXt head,
  two centered iSTFTs, the FIR mixer and the 44,160 to 44,100 Hz resampler. It keeps its state between calls and
  hands out the samples no later frame can change.
- **Prompt and output** (`prompt.zig`, `speech.zig`, `release.zig`, `voices.zig`): the training layout
  `[reference prefix | speaker row] [condition rows] text [silence]`, the undelayed frames to decode, the decoder's
  -3 dB headroom and the soft limiter above -1 dBFS.
- **Codec encoder** (`cuda/clip.zig` over `cuda/encoder.zig`, `cuda/teacher.zig`, `cuda/quantizer.zig`,
  `features.zig`): a reference clip to its codes. The acoustic branch (the 44,100 to 44,160 Hz resampler, a
  convolutional front end, five causal transformer stages down to 15 Hz), the SSL teacher (Wav2Vec2-BERT's first 16
  conformer layers over log-mel features computed on the host), then the quantizer: the teacher's rows pooled onto
  the frames give the semantic code, and nine residual codebooks take what its decoded prediction leaves. A clip
  runs in the publisher's lanes: 20 s each, a later lane carrying 10 s of what came before.
- **Served route** (`native/zonos_host.zig`, `server/zonos_speech.zig`): one request at a time; the release's 27
  default voices by their reference codes; a caller's own voice as `ref_audio` (a WAV file in base64, encoded by the
  engine) or as `speaker_reference_codes`; WAV or PCM, whole or streamed in chunks of 1, 2, 4, 8 and then 15 frames.
  The clip encoder's weights load on the first request that brings a clip.

Refused or absent: clips in formats other than WAV, text normalization (text is spoken as typed), speaker and control
guidance, LACI, steering, speech editing, accent adapters, phoneme, start and pause tokens, short-text prompts,
condition values (every condition row is its "unknown" bucket), the `full_mr` decoder, concurrency, Metal.

## Precision

- Weights stay as published: the backbone's and the teacher's in bf16, the codec's in fp32.
- Every activation, cache and state is fp32, and every sum is fp32. No int8, fp8 or bf16 activations.
- Each row kernel takes prompt rows and decode rows alike, with one fixed reduction order a row, so a row's bits do
  not depend on how the rows before it were chunked. The codec's kernels are written the same way per row and per
  sample.
- All four kernel images build with the glue flags (`--fmad=false`, IEEE division and square root).

The reference's oracle (`zonos3.reference`, `tools/dump_golden.py lm`) runs the backbone in fp32 on the CPU, which
is what the port is compared with. The publisher's CUDA servers run parts in bf16; see "The reference's bf16 paths".

## How it was checked

All figures below were measured on one RTX 5090 (driver 615.71, nvcc 13.3, Zig 0.17.0, `-Dsm=120`) at branch commit
`39a2864ef`, with every reference dump regenerated for that run.

### The reference side

`tools/zig/zonos3_reference.py` writes the dumps from the publisher's PyTorch code; `--help` lists the
subcommands. With `RELEASE` a snapshot of the checkpoint, `ZONOS` a checkout of the reference repo with its `model`
package installed, and `LM=RELEASE/lm/zonos3_mini`:

```sh
python tools/zig/zonos3_reference.py requests RELEASE D/requests
python tools/zig/zonos3_reference.py rows LM D/requests/greedy_embedding.json D/gen_greedy_embedding --device cpu
python tools/zig/zonos3_reference.py rows LM D/requests/greedy_reference.json D/gen_greedy_reference --device cpu
python ZONOS/tools/dump_golden.py lm LM --input-ids D/gen_greedy_embedding/input_ids.npy --dtype f32 --mode full -o D/lm_embedding
python ZONOS/tools/dump_golden.py lm LM --input-ids D/gen_greedy_reference/input_ids.npy --dtype f32 --mode full -o D/lm_reference
python tools/zig/zonos3_reference.py codec RELEASE/gguf/voices/English/Savannah_en-American_F_1.codes.npy D/codec_savannah
python ZONOS/tools/dump_golden.py codec-encoder --generator RELEASE/autoencoder/generator.pt \
    --audio RELEASE/gguf/voices/English/Cillian_en-Irish_M_1.wav -o D/enc_cillian
python tools/zig/zonos3_reference.py encoder RELEASE/autoencoder/generator.pt D/enc_cillian/audio.npy D/enc_cillian_fp32
python tools/zig/zonos3_reference.py teacher RELEASE/autoencoder/teacher D/enc_cillian/lane.0.ssl.features.npy D/teacher_cillian_fp32
python tools/zig/zonos3_reference.py clip RELEASE/autoencoder/generator.pt RELEASE/autoencoder/teacher D/enc_cillian/audio.npy D/clip_cillian
python tools/zig/zonos3_reference.py clip RELEASE/autoencoder/generator.pt RELEASE/autoencoder/teacher D/enc_cillian/audio.npy D/clip_cillian_47s --seconds 47.3
```

The encoder's clip is the release's own `Cillian_en-Irish_M_1.wav` (9.67 s, one lane); the 47.3 s clip is that
audio repeated, which takes four lanes. `clip` follows the publisher's lane layout; its production tool reports the
same four pieces for the 47.3 s clip (300 frames, then 150, 150 and 110 after 150 of context each).

The test request is "The quick brown fox jumps over the lazy dog, and then it rests by the river." in the release's
Savannah voice, once by her speaker embedding (a 30-row prompt) and once by her reference codes (a 209-row prompt).
"Greedy" is the release's sampler at temperature 0: its masks and repetition penalty still apply.

### The port's side

`zig build zonos3-generate -Dnvcc=... -Dsm=120 -Doptimize=ReleaseFast` builds the tool. Each mode ends with one
`RESULT zonos3 ...` line; the table quotes them.

| Check | Command (after `zonos3-generate`) | Compared with | Result |
| --- | --- | --- | --- |
| Backbone, embedding prompt | `--model LM --check D/lm_embedding` | the oracle's 109 sublayer captures, 30 rows | every row's cosine 1.0000000; worst relative difference 3.3e-6 |
| Backbone, reference prompt | `--model LM --check D/lm_reference --max-rows 256` | 109 captures, 209 rows | every row's cosine 1.0000000; worst relative difference 3.7e-6 |
| Greedy frames, embedding voice | `--model LM --frames-check D/gen_greedy_embedding --max-rows N` | the reference's 78 frames | 78 of 78 identical at N = 1, 7, 64 and 256 |
| Greedy frames, reference voice | `--model LM --frames-check D/gen_greedy_reference --max-rows N` | the reference's 80 frames | 80 of 80 identical at N = 1, 7, 64 and 256 |
| Codec decoder | `--codec-check D/codec_savannah --generator G --head H` | 13 captures and the audio of 171 frames, CPU fp32 | cosine 1.0000000; audio SNR 115.9 dB |
| Codec in chunks | the same run | frames fed 1, 3, 7, 15, ... at a time, the settled samples taken after each | bit-identical to one pass |
| Text to audio, embedding voice | `--checkpoint RELEASE --text "..." --voice-embedding E --temperature 0 --reference D/gen_greedy_embedding` | the reference's prompt rows, frames and CPU audio | rows equal, 78 of 78 frames, 205,800 of 205,800 samples, audio SNR 111.7 dB |
| Text to audio, reference voice | the same with `--voice-codes C` and `D/gen_greedy_reference` | as above | rows equal, 80 of 80 frames, 211,680 of 211,680 samples, audio SNR 117.5 dB |
| Teacher's features | `--features-check D/enc_cillian` | the reference's 999 feature rows of a 20 s lane | cosine 1.0000000; largest difference 2.4e-7 |
| Encoder, acoustic branch | `--encoder-check D/enc_cillian_fp32 --generator G` | 101 captures of the fp32 run, 145 frames | every row's cosine 1.000000 |
| SSL teacher | `--teacher-check D/teacher_cillian_fp32 --teacher RELEASE/autoencoder/teacher` | 18 captures of the fp32 run, 999 rows | worst row's cosine 0.9999252 |
| Clip to codes, one lane | `--encode-check D/clip_cillian --checkpoint RELEASE` | the fp32 run's codes, 145 frames | 1,450 of 1,450 codes equal |
| Clip to codes, four lanes | `--encode-check D/clip_cillian_47s --checkpoint RELEASE` | the fp32 run's codes, 710 frames | 7,100 of 7,100 codes equal |

`G` is `RELEASE/autoencoder/generator.pt` and `H` the `keep2_plain` head checkpoint. A capture passes when its worst
row's cosine against the reference is at least 0.9999. The backbone's "equal prompt rows" also covers the tokenizer:
the port's SuperBPE ids for the test sentence equal the reference's.

The teacher is the loosest match (its largest element difference is 0.09 against values up to 24). Its rows attend
to the whole 20 s lane, most of it the zero padding after this 9.7 s clip, and the difference grows from layer 8 on.
I have not yet found whether a reduction order or the padded rows explain it.

### Sampled speech

The reference draws with PyTorch's generator, so sampled frames cannot match it. Two indirect checks:

- **Repeatability:** the served route returns the same bytes for the same seed, and a streamed reply's bytes equal
  the whole reply's.
- **Cloned voices:** two clips sent as `ref_audio` (the release's Cillian clip, and its Thaddeus clip as 48 kHz
  stereo) each gave speech that Whisper transcribed with no errors. By the release's speaker encoder, each output's
  cosine to its own source clip was 0.78 and 0.83, against 0.21 and 0.17 to the other clip.
- **Transcription:** Whisper large-v3-turbo (resampled to 16 kHz) transcribed the greedy test sentence with no
  errors, from the port and from the reference alike, and a 62-word sampled passage with word error rates of 0.033
  and 0.016 in two voices; each error was the word "TensorFold".

### The served route

`tensorfold-native serve RELEASE --parallel 1`, default context 4,096 and 2,000 frames a request:

| Measure | Value |
| --- | --- |
| A 19.4 s sampled reply, whole | 1.28 s (15 times real time) |
| The same reply streamed | first response byte at 0.06 s, done at 1.29 s, bytes equal to the whole reply |
| Device memory in use after it | 3.57 GiB (2.36 GiB of it weights) |
| Device memory once a clip has been cloned | 8.82 GiB (the clip encoder adds 4.43 GiB of weights) |
| The first request with a clip (9.7 s), which loads the encoder | 9.0 s |
| Later requests with a clip: a 9.7 s mono clip, a 15.4 s stereo 48 kHz clip | 1.07 s, 1.53 s |
| A second request while one runs | 503 |
| An unknown voice, `top_k` above 4,096, malformed `speaker_reference_codes`, a `ref_audio` that is no WAV or is under half a second | 400 with the reason |

The standalone tool measured the parts: 22.0 s of sampled audio took 1.15 s to generate and 0.27 s to decode; the
decoder alone turns 11.4 s of codes into audio in 0.13 s; a 9.7 s clip encodes in 0.75 s and a 47.3 s clip in 5.1 s
(the acoustic branch 0.48 s and the teacher 0.26 s a lane). There are no CUDA graphs and the sampler runs on the host; none of this is tuned.

### A test page

`tools/zig/zonos3_test_page.py` serves a small browser page (`zonos3_test_page.html`) and passes its three routes
(`/v1/voices`, `/v1/models`, `/v1/audio/speech`) through to the model's port, so the page and the API share an
origin. It uses the standard library only.

```sh
tensorfold-native serve RELEASE --parallel 1 --port 8080
python3 tools/zig/zonos3_test_page.py --upstream 127.0.0.1:8080 --port 8081
```

The page lists the server's voices, speaks text with the seed and sampling fields, keeps each take with its player
and timing, and clones a voice from an audio file: the browser decodes the file, makes a 16-bit mono WAV of its
first 60 s and sends it as `ref_audio` with each request. It adds no authentication; it binds 127.0.0.1 unless
`--bind` says otherwise. Checked with curl through the pass-through (the page, the voice list, a spoken reply, a
cloned reply, a refusal's message passed back, other paths 404); the page's script was parsed but not driven in a
browser by an automated test.

### Repo checks

`zig fmt --check build.zig zig`, `tools/zig/lean_check.py` (0 new over the base), `tools/zig/cuda_common_check.py`
(0 violations) and `zig build test-cuda-host` pass; the family's host tests run in it and alone as
`zig build test-zonos-host`. On a machine whose umask is 0002 one unrelated test in that suite
(`serve --config`, which refuses a group-writable file) fails; it passes under umask 022.

Not run: `zig build test`, `test-kernel-contract`, `test-golden`, and any quality suite.

## The reference's bf16 paths

The publisher's CUDA code runs some parts under bf16 autocast, and those runs sit visibly away from the same code in
fp32. Measured with the reference alone, on the dumps above:

| Part | bf16 run against fp32 run |
| --- | --- |
| Codec decoder audio (the CUDA server's decode against the CPU decode, same frames) | 35.4 dB SNR |
| Encoder latent, 145 frames | worst row's cosine 0.966, mean 0.998 |
| Teacher output, 999 rows | worst row's cosine 0.085, mean 0.984 |
| Codes re-encoded from a release clip against the codes the release ships for it | 97% equal in the semantic codebook, 56% to 87% in the residual ones |
| Codes of the 47.3 s clip, the fp32 run against the bf16 production run | 96% equal in the semantic codebook, 40% to 75% in the residual ones |

The port computes in fp32 and matches the fp32 runs. Two consequences: its audio is closer to the publisher's CPU
decode than the publisher's own CUDA server is, and the native encoder's codes are not the publisher's production
codes. Residual quantization amplifies small differences, so the codes diverge while describing the same audio.
Decoding the 9.67 s clip's three sets of codes with the reference decoder and comparing each with the clip:

| Codes | Log-mel distance to the clip (L1) | Speaker cosine to the clip |
| --- | --- | --- |
| Shipped with the release | 0.574 | 0.946 |
| The reference's bf16 run | 0.576 | 0.944 |
| The fp32 run (what the port produces) | 0.620 | 0.947 |

That is one clip: the fp32 codes keep the speaker as well and sit about 8% further in log-mel distance. Whether
that gap is real across clips, and whether the backbone (trained on bf16-encoded references) clones as well from
fp32 codes, needs a proper listening or scoring pass before the encoder's gate is agreed.

## Shared code this touches

- `drama3/codec_pth.zig` moved to `core/torch_pth.zig` (its zip half to `core/torch_zip.zig`) and gained nested
  checkpoints and a raw reader; Drama 3's codec loads through the same code and its host tests pass.
- `drama3/wav.zig` moved to `core/wav.zig`: both families read WAV files and resample with it.
- `core/speech_api.zig`: `SpeechRequest.speaker`, an optional speaker vector, and `reference_audio`, an optional clip.
- `server/`: voices load through `speech_voices.zig`; the speech route, the sampling parser, the voices listing,
  model detection and the tokenizer loader each gained a branch for a release folder, which has no `config.json`
  and is recognised by `lm/zonos3_mini/params.json`.
- `native/`: `zonos3` in the CUDA open dispatch, the capabilities list and the rank refusals.
