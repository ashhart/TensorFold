# ZONOS3 Mini on CUDA

`zig/src/families/zonos3` runs Zyphra's ZONOS3 Mini text-to-speech model on the CUDA runtime, from the files the
publisher released, and serves it on `POST /v1/audio/speech`. This page records what is ported, where every source
came from, and how each part was checked, so the checks can be rerun.

Status: one branch, proposed in #3615 and open as draft pull request #3720. All 40 kernels have their limits
records, numeric contract tests and a planted control each; the quality suite has not been run on this build.

## Sources

| What | Where | Pin |
| --- | --- | --- |
| Checkpoint | `huggingface.co/Zyphra/ZONOS3-Mini` (Apache-2.0) | revision `d2d142a86b1e6aeb4252679673c86ecd86177bf8` |
| Reference code | `github.com/Zyphra/ZONOS3-Mini` (MIT) | commit `96d9e0ec1e39083ed336ac78f24e3dbc93721c0f` |
| Steering bundle | the reference repo's `assets/steering/studio_bundle_mect60k_x10.json` and the 14 set folders it names (MIT) | the same commit |
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
  penalty, its end-of-audio countdown and cold-start padding, on fp32 logits. Each codebook's code is then drawn
  by TensorFold's keyed sampler (`core/lanes/sampling.zig`: temperature, top-k, top-p and min-p in float64) among
  the ids left unmasked, at position `frame * 10 + codebook` under the request's seed. Temperature 0 is its greedy
  rule: the first maximum, ties to the lower id.
- **Codec decoder** (`cuda/codec.zig`, `zonos3_codec.cu`): the `keep2_plain` decoder, the reference servers'
  default: code tables, the semantic head, five causal transformer stages from 15 Hz to 240 Hz, the ConvNeXt head,
  two centered iSTFTs, the FIR mixer and the 44,160 to 44,100 Hz resampler. It keeps its state between calls and
  hands out the samples no later frame can change. Its linear layers are cuBLAS's fp32 GEMM (CUDA 13's
  `libcublas`, opened when the decoder loads; without it the load fails and says so), a bias added by a second,
  rank-one GEMM. See "The decoder's GEMM" under "Precision".
- **Prompt and output** (`prompt.zig`, `speech.zig`, `release.zig`, `voices.zig`): the training layout
  `[reference prefix | speaker row] [condition rows] text [silence]`, the undelayed frames to decode, the decoder's
  -3 dB headroom and the soft limiter above -1 dBFS.
- **Codec encoder** (`cuda/clip.zig` over `cuda/encoder.zig`, `cuda/teacher.zig`, `cuda/quantizer.zig`,
  `features.zig`): a reference clip to its codes. The acoustic branch (the 44,100 to 44,160 Hz resampler, a
  convolutional front end, five causal transformer stages down to 15 Hz), the SSL teacher (Wav2Vec2-BERT's first 16
  conformer layers over log-mel features computed on the host), then the quantizer: the teacher's rows pooled onto
  the frames give the semantic code, and nine residual codebooks take what its decoded prediction leaves. A clip
  runs in the publisher's lanes: 20 s each, a later lane carrying 10 s of what came before.
- **Served route** (`native/zonos_host.zig`, `server/zonos_speech.zig`): `--parallel` requests at once; the release's 27
  default voices by their reference codes; a caller's own voice as `ref_audio` (a WAV file in base64, encoded by the
  engine) or as `speaker_reference_codes`; WAV or PCM, whole or streamed in chunks of 1, 2, 4, 8 and then 15 frames.
  The clip encoder's weights load on the first request that brings a clip.
- **Steering** (`steering.zig`, `steer_bundle.zig`, `server/zonos_steer.zig`): with `--steer-bundle`, a request's
  `controls` and `control_spans`. See "Steering controls".

Refused or absent: clips in formats other than WAV, spans placed by the reading position
(`control_span_follow: "reading"`), an embedding made from a clip (the release's speaker encoder), LACI and its
extended speaking-rate buckets, speech editing, accent adapters, phoneme, start and pause tokens, short-text prompts, the `full_mr` decoder,
Metal.

## Text normalization

The reference servers rewrite written text into its spoken form before the tokenizer sees it ("$5" becomes "five
dollars") with Zyphra's [zyphra-textnorm](https://github.com/Zyphra/zyphra-textnorm), a Rust library with a C
interface. The server loads that same library when a request first needs it (`server/zonos_textnorm.zig`), so the
rewriting is the publisher's own code, not a port of it.

- **`language`** names the grammar: one of the reference's 44 request codes (`en_us`, the default, `en_gb`, `fr_fr`,
  `de`, `pt_br`, `ja`, `cmn`, ...), read whatever its case and with `-` for `_`. Another code is refused with the list.
  It selects the grammar only; nothing else in the prompt depends on it.
- **`text_normalization`** is `true` or `false`. Only `true` normalizes. Left out or `false`, the text is spoken as
  typed, whether or not the library loads, so installing it changes no reply. `true` without the library is refused
  with the line below.
- Line breaks are collapsed either way, and start and pause markers pass through the grammar untouched, as in
  `zonos3.tokenizer.textnorm`.

The library is not built by `zig build`. Build release 0.2.0 (commit `170c571e`, the reference repository's
`third_party/zyphra-textnorm` submodule; Rust 1.88 or later) as a shared library and name it in
`TENSORFOLD_LIBZYPHRA_TEXTNORM`, or put it in `lib/` beside the installation:

```
cargo rustc --release --locked --lib -p zyphra-textnorm --features capi --crate-type cdylib
```

`server/zonos_textnorm_test.zig` checks the text step against every case of the reference's own tests
(`model/tests/test_textnorm_wrapper.py`): with `true`, each text in each language it covers must equal the expected
spoken form byte for byte, and left out or `false`, the text as typed with its line breaks collapsed. The cases that
need the library skip where it does not load; run them with `TENSORFOLD_LIBZYPHRA_TEXTNORM` set.

## Condition values and speaker guidance

Both are the reference's (`zonos3.tts.plan`, `zonos3.speaker_cfg`), under its field names.

- **`condition_values`** and **`condition_buckets`** set the three condition rows: `loudness` (LUFS), `fmax` (Hz)
  and `speaking_rate` (UTF-8 bytes of text a second), as an object by name or a list in that order. A raw value goes
  to its bucket through the edges in `params.json` (the count of edges at or below it); a bucket id is taken as
  given, from 0 to the table's count, the count itself meaning unknown. A bucket wins over a value for the same row,
  and a row the request leaves out is unknown, as every row was before.
- **`speaker_cfg_scale`** is speaker guidance: 1 (the default) is off, larger follows the voice harder, 0 is the
  prompt without its voice. The host runs a twin beside the request: the same prompt without its reference prefix
  (or speaker row) and with every condition unknown, on a second set of caches, fed the same generated rows. Each
  frame is sampled from `unconditional + scale * (conditional - unconditional)`, mixed in fp32 on the host before
  the sampler, as the reference does. There is no new kernel: each branch is the single-stream pass already
  checked.
- **Memory:** the second set of caches and recurrent state is 10,913,792 bytes at the default context, allocated by
  the first guided request after asking the pool for that room, and kept.
- **Speed:** a guided request runs the backbone twice a frame.

`controls_cfg_scale`, the guidance on steering controls, is under "Steering controls".

Checked on the RTX 5090, served, with the Savannah voice sent as `speaker_reference_codes` at temperature 0: the
replies to the four requests above have the reference audio's sample counts and match it at 64.8 dB (unguided, the
earlier baseline), 64.9 dB (condition values), 61.9 dB (guidance 1.5) and 63.6 dB (guidance 2 with buckets) signal
to noise, the reference decoding on the CPU. Sampled: a guided request repeats byte for byte and streams the same
bytes, also with a cloned clip; scale 1 gives the unguided bytes; `condition_values` give the bytes of the buckets
they resolve to, and all-unknown buckets the bytes of a request that names none. One 13-word sentence in the
`rupert` voice lasted 6.53 s at `speaking_rate` 8, 4.40 s with none and 2.47 s at 25.

## Steering controls

The reference steers a voice with direction vectors (`zonos3.tts.steering`, `zonos3.tts.directions`): emotion axes,
speaking styles, accents, voice qualities, word emphasis. They come as a steering bundle, a JSON file naming
direction sets, which the publisher keeps in its code repository, not in the checkpoint. `tensorfold-native serve
RELEASE --steer-bundle FILE` loads the bundle and its sets (a `manifest.json` and one float32 `.npy` vector a
direction, beside the file); `GET /v1/audio/controls` lists the directions, and without the flag it is 404 and the
fields below are refused. The pinned bundle has 65 directions in 14 sets.

- **`controls`**: `{"<set>.<direction>": dose}`. A direction is one of two kinds:
  - **Hidden** (1,024 values; `mood.valence`, `delivery.whisper`, `emphasis_nv_nodur.emphasis`): dose times the
    vector is added to one source of the Full AttnRes bank on every generated frame, before the later sublayers
    and the head read it. The pinned bundle's hidden sets steer sources 15 and 19 (the outputs of layer 7's attention and layer 9's
    GDN, counting layers from 0).
  - **Speaker embedding** (192 values; `identity.age`, `texture.deep`, `accents_en.british`): the doses' vectors
    are summed onto the speaker embedding, which is then scaled back to its norm. These need a voice by its
    embedding (below) and are refused otherwise.
- **`speaker_mode: "embedding"`** speaks a release voice by its speaker embedding (the prompt's speaker row) in
  place of its reference codes; `speaker_embedding` (192 numbers) gives a caller's own. An embedding from a clip is
  not served: that needs the release's speaker encoder.
- **Steer then hold** (`steer_hold`, `steer_onset_seconds`, `steer_ramp_seconds`; defaults from the bundle's
  `steer_release`: 4 s, 1 s, 0.35): the request's hidden controls are added at full dose for the onset, ramp
  linearly to `steer_hold` times the dose, and stay there. The bundle exempts `delivery.tempo` and
  `delivery.whisper`, which stay constant; `steer_hold: 1` keeps every control constant.
- **`control_spans`**: `[{"start": c0, "end": c1, "controls": {...}}]` adds hidden controls while characters
  [c0, c1) of the text as sent are spoken. Frames are placed by character position at `control_span_rate` frames
  a character (1 when left out), stretched by the bundle's `timing` tables of the controls that change the pace,
  with ramps of 3 frames at both edges. Spans may not overlap. `control_span_follow: "reading"`, which places
  spans by the model's reading position, needs the alignment heads and is refused.
- **`controls_cfg_scale`**: guidance on the hidden controls through a twin stream that runs the same prompt
  without them, mixed as speaker guidance is. It takes effect with an embedding voice and a request-level hidden
  control, as in the reference, and cannot be combined with `speaker_cfg_scale`: a request has one twin. A
  speaker-guidance twin does take the request's deltas.

How it runs: the server resolves a request's controls on the host into at most three deltas (constant, released,
and a row a span), each a row of 1,024 for every source the bundle steers. Each frame's delta is computed on the
host in the reference's own order of fp32 steps and added on the device to that row of the bank source by
`z3_combine` (the row and the delta weighed 1 and 1: one fp32 add), which the kernel contract now also checks at
that shape. Prompt rows are never steered, and neither is the first generated frame's forward, as in the reference.

Checked against the publisher's code with its bundle, greedy, five requests (`requests` writes them; `rows
--steer-bundle` dumps each frame's delta beside the frames):

- **The host's arithmetic** (`zonos3-generate --steer-check D/gen_steer_X --steer-bundle BUNDLE`, no GPU): every
  frame's delta and the moved speaker embedding equal the reference's bit for bit in all five. The embedding's
  norm follows torch's CPU sum for 192 values (eight running sums of squares, added left to right), found by
  matching 44 vectors; a different torch build may sum otherwise.
- **The frames** (`--frames-check`, the dumped deltas added): identical to the reference's at `--max-rows` 1, 7, 64
  and 256 in all five (the table under "The port's side").
- **Served**, at `--parallel 4`, the same five requests through `/v1/audio/speech`: each reply has the reference
  audio's sample count and matches it at 69.6 dB (released hidden controls), 72.9 dB (speaker-embedding controls),
  71.8 dB (spans on two sources with a pace control), 69.8 dB (controls guidance 1.5) and 62.9 dB (a reference
  voice with speaker guidance 1.5), the reference decoding on the CPU.
- **Sampled, served:** a steered request repeats byte for byte and streams the same bytes, with a reference voice,
  an embedding voice and controls guidance alike; a dose of 0 and `controls_cfg_scale: 1` give the bytes of a
  request without them; four requests at once (steered, controls-guided, plain, and speaker-guided with a
  speaker-embedding control) each gave the bytes they give alone.
- **Speed:** steering costs about 1%: the long passage ran at 37.3 times real time plain and 36.9 steered on a
  reference voice, and at 38.9 and 38.4 on an embedding voice (a steered frame adds a row copy, an upload and one
  small launch for each steered source). With controls guidance 1.5 it ran at 33.1, the cost of the second stream.
- **Memory:** 524,544 bytes more at load, the pairs of rows `z3_combine` reads.

Not measured: what the controls do to the voice. No listening test or classifier was run; the checks above say the
port adds what the reference adds.

## Requests at once

`--parallel N` (1 to 16; 1 without the flag) serves N requests at once. Each has its own lane: a set of the
backbone's caches and recurrent state, and a set of the decoder's. One worker gives every request in progress a turn,
a frame each: the frame is sampled and decoded when its audio is due. The rows the turns leave then run in one pass
(`Engine.runLanes`), a row a lane, a guided request's twin among them: the launches that read no lane's state take
all the rows at once, and the four that do (the GDN layers' convolution and recurrence, the attention layers' cache
write and attention) run a lane at a time on that lane's state. The kernels are the ones already checked, and each
computes a row from that row alone, so a request's arithmetic does not depend on what runs beside it.

- **Memory:** each request at once past the first adds 1,236,862,976 bytes at the default context and frame budget
  (10,913,792 of them the backbone's lane, the rest the decoder's), admitted with the load: 3,862,487,552 bytes at
  `--parallel 1`, 7,573,076,480 at 4. A guided request's twin lane is still made by its slot's first guided request.
- **Speed:** a frame of one request is about 226 small launches, 24 of them a lane's own, so the backbone's cost
  grows slowly with the lanes in a pass (1.1 ms for one row, 1.6 ms for four). Sampling on the host and the decoder
  do not share. The decoder's linear layers took 78% of its 0.19 s for a 17.8 s reply while a kernel of ours
  (`zc_linear`) ran them a row at a time; with cuBLAS's GEMM that reply's whole time fell from 0.61 s to 0.48 s.
- **A clip is encoded in its request's first turn**, about half a second in which the other requests wait.
- A request past N is refused with 503, as the second one was before.

Checked on the RTX 5090 at `--parallel 4`: eight kinds of request (a long passage, a short one, guided, streamed,
a condition value with normalized text, a cloned clip, a guided clone, a guided stream) were each asked alone, then
four at once in three mixes; all twelve replies equalled the same request's reply alone, byte for byte. Four copies
of the 17.8 s passage at once took 1.17 s, 60.9 times real time in total against 37.3 alone, each reply the solo
bytes. Alone, a guided request of 3.87 s took 0.14 s. Of six at once, four were spoken and two got 503.

## Precision

- Weights stay as published: the backbone's and the teacher's in bf16, the codec's in fp32.
- Every activation, cache and state is fp32, and every sum is fp32. No int8, fp8 or bf16 activations.
- Each row kernel takes prompt rows and decode rows alike, with one fixed reduction order a row, so a row's bits do
  not depend on how the rows before it were chunked. The codec's kernels are written the same way per row and per
  sample; the decoder's linear layers are the exception, below.
- All four kernel images build with the glue flags (`--fmad=false`, IEEE division and square root).

### The decoder's GEMM

The decoder's linear layers run through cuBLAS (`cublasGemmStridedBatchedEx`, fp32 in, fp32 compute, fp32 out), as
Drama's decoder does. The clip encoder and the quantizer keep `zc_linear`, so a clip's codes are unchanged. What
this costs in exactness, measured on the RTX 5090 with CUDA 13.3's library:

- **cuBLAS chooses its order of sums by the shape of a call**, so a row's bits depend on how many rows are decoded
  with it. The served route therefore decodes every request in one schedule, streamed or whole: chunks of 1, 2, 4, 8
  and then 15 frames. A streamed reply and a whole reply make the same calls and give the same bytes, and so does a
  request beside others, since each decodes on its own lane.
- **Against the earlier kernel** the 17.8 s sampled reply differs in 338 of 784,980 16-bit samples, each by one
  step (97 dB). Against the publisher's decoder the decoder check gives 115.3 dB where the kernel gave 115.9
  ("The port's side").
- **Another cuBLAS release may round differently.** Replies are byte-stable on one install, not promised across
  installs.

### The arithmetic reference

One reference, fixed for every part of the model: the publisher's PyTorch code (the pinned reference repo) run in
fp32 on the CPU over the published weights, with no autocast. The backbone's oracle (`zonos3.reference`,
`tools/dump_golden.py lm`) already runs that way; for the codec decoder, the encoder and the teacher,
`tools/zig/zonos3_reference.py` runs the publisher's modules in fp32 where its servers would use bf16 autocast.

The gates against it are exact: greedy frames equal frame for frame, a clip's codes equal code for code, streamed
bytes equal whole ones. Cosines, SNR figures and voice scores on this page describe how close intermediate values
are; none of them is a gate.

What this choice costs: the publisher's CUDA servers run the codec and the teacher in bf16, so the codes this port
encodes from a clip are not the codes the publisher's server encodes from it. "The reference's bf16 paths" measures
the difference. The port has no bf16 path and none is planned as an option.

## How it was checked

All figures below were measured on one RTX 5090 (driver 615.71, nvcc 13.3, Zig 0.17.0, `-Dsm=120`). The table of
port-side checks was last run in full on the branch rebased onto main `6d0a128b0`, with the contract tests and the
two kernel changes they brought in; the reference dumps are one set, regenerated before the first full run.

### The reference side

`tools/zig/zonos3_reference.py` writes the dumps from the publisher's PyTorch code; `--help` lists the
subcommands. With `RELEASE` a snapshot of the checkpoint, `ZONOS` a checkout of the reference repo with its `model`
package installed, and `LM=RELEASE/lm/zonos3_mini`:

```sh
python tools/zig/zonos3_reference.py requests RELEASE D/requests
python tools/zig/zonos3_reference.py rows LM D/requests/greedy_embedding.json D/gen_greedy_embedding --device cpu
python tools/zig/zonos3_reference.py rows LM D/requests/greedy_reference.json D/gen_greedy_reference --device cpu
python tools/zig/zonos3_reference.py rows LM D/requests/conditions.json D/gen_conditions --device cpu
python tools/zig/zonos3_reference.py rows LM D/requests/guided_reference.json D/gen_guided_reference --device cpu
python tools/zig/zonos3_reference.py rows LM D/requests/guided_embedding.json D/gen_guided_embedding --device cpu
python tools/zig/zonos3_reference.py rows LM D/requests/guided_conditions.json D/gen_guided_conditions --device cpu
for X in hidden raw spans guided reference; do
  python tools/zig/zonos3_reference.py rows LM D/requests/steer_$X.json D/gen_steer_$X --device cpu \
      --steer-bundle ZONOS/assets/steering/studio_bundle_mect60k_x10.json
done
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
| Greedy frames, condition values (loudness -20, speaking rate 18) | `--model LM --frames-check D/gen_conditions --max-rows N` | the reference's 68 frames | 68 of 68 identical at N = 1, 7, 64 and 256 |
| Greedy frames, speaker guidance 1.5, reference voice | `--model LM --frames-check D/gen_guided_reference --max-rows N` | the reference's 77 frames | 77 of 77 identical at N = 1, 7, 64 and 256 |
| Greedy frames, speaker guidance 1.5, embedding voice | `--model LM --frames-check D/gen_guided_embedding --max-rows N` | the reference's 81 frames | 81 of 81 identical at N = 1, 7, 64 and 256 |
| Greedy frames, speaker guidance 2 with condition buckets 3, unknown, 25 | `--model LM --frames-check D/gen_guided_conditions --max-rows N` | the reference's 62 frames | 62 of 62 identical at N = 1, 7, 64 and 256 |
| Greedy frames, steered: hidden controls that release (`mood.valence` 2, `delivery.whisper` 1, onset 1 s) | `--model LM --frames-check D/gen_steer_hidden --max-rows N` | the reference's 75 frames, its deltas added | 75 of 75 identical at N = 1, 7, 64 and 256 |
| Greedy frames, steered: speaker-embedding controls (`identity.age` 1.5, `texture.deep` 1) | `--model LM --frames-check D/gen_steer_raw --max-rows N` | the reference's 81 frames | 81 of 81 identical at N = 1, 7, 64 and 256 |
| Greedy frames, steered: two spans, `delivery.tempo` 1 and `mandarin_hidden.cmn` 0.5 | `--model LM --frames-check D/gen_steer_spans --max-rows N` | the reference's 70 frames, its deltas added | 70 of 70 identical at N = 1, 7, 64 and 256 |
| Greedy frames, steered: controls guidance 1.5 (`mood.arousal` 1, `identity.age` -1) | `--model LM --frames-check D/gen_steer_guided --max-rows N` | the reference's 80 frames, its deltas added | 80 of 80 identical at N = 1, 7, 64 and 256 |
| Greedy frames, steered: a reference voice, `mood.dominance` 1.5, speaker guidance 1.5, hold 1 | `--model LM --frames-check D/gen_steer_reference --max-rows N` | the reference's 83 frames, its deltas added | 83 of 83 identical at N = 1, 7, 64 and 256 |
| Steering resolved on the host | `--steer-check D/gen_steer_X --steer-bundle BUNDLE`, the five dumps above | each frame's delta and the moved speaker embedding | bit-identical in all five |
| Codec decoder | `--codec-check D/codec_savannah --generator G --head H` | 13 captures and the audio of 171 frames, CPU fp32 | cosine 1.0000000; audio SNR 115.3 dB |
| Codec in chunks | the same run | frames fed 1, 3, 7, 15, ... at a time, the settled samples taken after each or all at the end | the two bit-identical; 116.2 dB SNR to one pass of all the frames |
| Text to audio, embedding voice | `--checkpoint RELEASE --text "..." --voice-embedding E --temperature 0 --reference D/gen_greedy_embedding` | the reference's prompt rows, frames and CPU audio | rows equal, 78 of 78 frames, 205,800 of 205,800 samples, audio SNR 114.0 dB |
| Text to audio, reference voice | the same with `--voice-codes C` and `D/gen_greedy_reference` | as above | rows equal, 80 of 80 frames, 211,680 of 211,680 samples, audio SNR 113.3 dB |
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

The reference draws with PyTorch's generator, so sampled frames cannot match it; a seeded draw is defined by
TensorFold's keyed sampler instead. The checks of it are on the served build, at the default sampling settings:

- **Repeatability:** one seeded request of 1,569,960 PCM bytes (784,980 samples, 17.8 s at 44.1 kHz) returned the
  same bytes sent whole three times, streamed three times, and once while two other requests arrived.
- **At `--parallel 1`** the two requests that arrived while it spoke were refused with 503 `engine_busy`; "Requests
  at once" compares solo replies with concurrent ones.
- **Transcription:** Whisper large-v3-turbo (resampled to 16 kHz) transcribed that 17.8 s reply, 49 words in the
  Rupert voice, with no errors.
- **Cloned voices:** two clips sent as `ref_audio` (the release's Cillian clip, and its Thaddeus clip as 48 kHz
  stereo) each gave speech that Whisper transcribed with no errors.

Measured earlier, with the family's own seeded generator before the sampler moved to the keyed one, and not
repeated since: by the release's speaker encoder each cloned output's cosine to its own source clip was 0.78 and
0.83, against 0.21 and 0.17 to the other clip; and Whisper's word error rates on a 62-word sampled passage were
0.033 and 0.016 in two voices, each error the word "TensorFold". It also transcribed the greedy test sentence with
no errors, from the port and from the reference alike. None of these is a gate.

### The served route

`tensorfold-native serve RELEASE --parallel 1`, default context 4,096 and 2,000 frames a request, after one short
warm-up request:

| Measure | Value |
| --- | --- |
| The 17.8 s sampled reply, whole (three runs) | 0.47 to 0.48 s (37 times real time) |
| The same reply streamed (three runs) | first response byte at 0.04 s, done at 0.47 to 0.48 s, bytes equal to the whole reply |
| Device memory in use after it | 3.60 GiB (2.36 GiB of it weights) |
| Device memory once a clip has been cloned | 8.85 GiB (the clip encoder adds 4.43 GiB of weights) |
| The first request with a clip (9.7 s), which loads the encoder | 9.2 s |
| Later requests with a clip: a 9.7 s mono clip, a 15.4 s stereo 48 kHz clip | 0.89 s, 1.27 s |
| A second and a third request while one runs | 503 each |
| An unknown voice, `top_k` above 4,096, malformed `speaker_reference_codes`, a `ref_audio` that is no WAV or is under half a second | 400 with the reason |

The standalone tool measured the parts: the decoder alone turns 11.4 s of codes into audio in 0.13 s; a 9.7 s clip
encodes in 0.75 s and a 47.3 s clip in 5.1 s (the acoustic branch 0.48 s and the teacher 0.26 s a lane). There are
no CUDA graphs and the sampler runs on the host; none of this is tuned.

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

On the branch rebased onto main `6d0a128b0`, the no-GPU workflow's steps were run on one Linux machine as the
workflow runs them (`-Dfixed-cpu`, its test timeouts) and passed:

- `zig fmt --check build.zig zig`; `tools/zig/lean_check.py` against the base with `lean_new.py` (221 problems on
  both, 0 new); `tools/zig/cuda_common_check.py` (0 violations), `cuda_fragment_check.py` (0 misses) and their unit
  tests; `host_tests_test.py`.
- `zig build test-host-a`, `test-host-b`, `test-kernel-contract`, `test-golden`,
  `test-gradient-serving test-qwen-serving-host`, `test-server-security test-pull-security`, `fuzz-cortheon`, `test`
  and `check-release-paths`.
- The family's host tests alone (`zig build test-zonos-host`), with Drama's and the CUDA host suite
  (`test-drama-host test-cuda-host`).

They ran under umask 022: with 0002 one unrelated test (`serve --config`, which refuses a group-writable file)
fails. In the first run of `test-host-a` one real-clock test of the rank bring-up (`two ranks settle on the real
clock and agree on one plan and view`) failed while other jobs shared the CPU; the step passed when run again alone.

On macOS (an M5 Ultra) the default build, `zig build native` and `zig build test` pass. The family has no Metal
path; the Metal contract jobs were not run, and neither was any quality suite.

The branch changes the speech route Drama 3 is served through and moves Drama's `torch.save` reader and WAV code to
core, so Drama 3 (`fishaudio/drama-3`) was run on that M5 Ultra from main `6d0a128b0` and from the branch: six
`drama3-generate` runs (greedy, seeded, a reference clip encoded, a cloned voice, streamed PCM, frames) three times
each, and four `/v1/audio/speech` requests (seeded whole, seeded streamed PCM, default, sampled) twice from each of
two server starts. Every output is byte-identical between the two, and the speeds agree (the cloned run 73.21 to 73.55
frames a second on main, 73.31 to 73.49 on the branch). Drama on CUDA was not run: the checkpoint does not fit the
5090.

The family registers four kernel images, so `zig/tests/fixtures/gradient-serving.tsv` re-pins
`zig/src/cuda/kernels.zig`, as earlier image registrations did.

### Memory admission

The host works out what a load takes before it allocates anything and asks the pool every CUDA family reads
(`native/cuda_family_memory.zig`: the card's free memory less the reserve, and `TENSORFOLD_CUDA_MEMORY_LIMIT_GB`).

- **The load:** the pinned release's weights as the device holds them (2,531,483,136 bytes: the backbone's bf16 and
  the `keep2_plain` decoder's fp32), plus the backbone's and the decoder's state and workspaces, which their own
  allocation code counts without allocating (`Engine.stateBytes`, `Codec.stateBytes`). At the default
  `--context 4096` and 2,000 frames that is 3,862,487,552 bytes (3.60 GiB), 33,554,432 of them the workspace
  cuBLAS is given. What cuBLAS allocates for itself is outside this count: `nvidia-smi` showed the process about
  64 MiB larger than with the earlier kernel, and the server's log shows the same
  number admitted and held after the load. A load that holds more than it was admitted for is an error.
- **The clip encoder** loads with the first request that brings a clip and takes 5,643,510,688 bytes (5.26 GiB) of
  weights, state and workspaces, all of fixed size; the log again shows that number admitted and held.

Checked on the served build with `TENSORFOLD_CUDA_MEMORY_LIMIT_GB`: at 3 the server refuses to start ("ZONOS3 Mini
needs 3.60 GiB of device memory at --context 4096; 3.00 GiB is free for it after the reserve"); at 5 it starts, a
cloning request is refused (HTTP 500, "cloning from a clip needs 5.26 GiB of device memory for its encoder, and less
than that is free"), and the same seeded request before and after that refusal returns the same bytes. The two
weight sizes are constants for this release and are not derived from the files; another checkpoint would need its
own. These counts are the engine's buffers: `nvidia-smi` showed the process at 4,360 MiB after the load, the CUDA
context, the kernel images and what cuBLAS allocates for itself included.

### Kernel contract

**Launch limits (gate 5).** The four images hold 40 kernels, each registered as one launch in
`zig/src/core/contract/zonos3_cuda_records.zig` with the thread count and the largest dynamic shared memory the
engine asks for. The records were measured, not written by hand: every image was built for sm_80, sm_86, sm_89,
sm_120 and sm_121 and audited with

    zig build fatbins kernel-contract-fatbins compile-cuda-tests -Dnvcc=NVCC -Dsm=80,86,89,120,121 --prefix OUT
    tf-cuda-test kernel-contract-limits OUT/fatbin

and the `RECORD` lines it prints were copied into the file. With nvcc 13.0.88 all 200 launch and chip pairs pass
with their record current, and the same records are current for images built by nvcc 13.0.48. nvcc 13.2.86 compiles
seven of the launches with other register counts (`z3_attention`, `z3_conv_silu`, `z3_embed`, `zc_dwconv`,
`zc_resample`, `ze_resample_in`, `zt_attention`); those are the file's `cuda_132` set, joined into the registry's
13.2 set, and nvcc 13.3.73 compiles all 40 as 13.2 does. With the set in place all 200 pairs are current under each
of the four compilers. The audit also reports 18 stale records for launches of other families
(`exl3_experts tf_exl3_prompt_down` and `tf_exl3_prompt_gu`, `flashnext_mtp fn_row_topk`, `qwen35 tf_q35_gdn_slots_1`,
`_2` and `_4`); this branch does not change those kernels, and the same audit was not run on the base to compare.

**Numeric gates (1 to 4).** `zig/tests/cuda/contract_zonos3.zig` and the `contract_zonos3_*.zig` files beside it
register all 40 kernels, 41 registrations in all because `zc_add_scaled` is tested with and without its layer scale:

- the two norms and the two linears through the contract's row and dense input families, with the bf16 inputs
  widened to the fp32 rows these kernels read;
- the 15 pointwise and moving kernels (`contract_zonos3_point.zig`), which have no sums: each output is checked
  against the range fp32 allows around its float64 value;
- the router and the quantizer's nearest code (`contract_zonos3_pick.zig`), exact by their own stated rules;
- the other 19 kernels (`_backbone`, `_gdn`, `_attn`, `_windows`, `_conv`), each against a float64 reference that
  follows the kernel's own order of sums; 18 of them sum, and for those gate 4's inputs are built so fp32 sums are
  exact in any order.

Every registration has a planted control, 41 in all: the same kernel launched wrong in one stated way (a row, a
value, a key or a block short, an input read one place late, a wrong epsilon), which gate 3 must turn red.

Two kernels changed to pass gate 1: `zc_spec` clamped its log-magnitude with `fminf`, which drops a NaN, and now
compares; `ze_nearest` wrote an out-of-range code for a row whose every dot is NaN and now writes code 0. Neither
changes a finite input's result: every check in the table above was rerun after the change and gave the same
frames, codes and audio figures.

On an RTX 5090 (driver 615.71.09, images from nvcc 13.0.88, the runner built for sm_80, 86, 89, 120 and 121),
`tf-cuda-test kernel-contract zonos3` ended with

    (filtered) SUMMARY 41 kernels: 41 pass gates 1-4, 0 fail, 0 known findings; 41 controls: 41 red as planted, 0 missed; 0 limit problems; gates 6395 ms, limits 8 ms

The whole run on the same card and images (`tf-cuda-test kernel-contract`, 83 minutes of gates and 15 of limits,
while other builds shared the machine's CPU), made on the earlier base `9489c3817` and not repeated since, ended with

    SUMMARY 684 kernels: 684 pass gates 1-4, 0 fail, 8 known findings; 459 controls: 459 red as planted, 0 missed; 186 limit problems; gates 4966609 ms, limits 899084 ms

and exit code 1 for the limit problems. None of the 186 names a ZONOS3 launch: all 40 of the family's live launch
checks pass, and the three live failures are `flashnext_mtp fn_row_topk` and `exl3_experts tf_exl3_prompt_gu` and
`tf_exl3_prompt_down`, whose records read 0. The same run was not made on the base, so this page does not claim the
186 are the base's.

The layer norm's test adds a term to its bound: the kernel scales in fp32 and then adds the bias, so the roundings
before the bias are a fraction of the scaled value and not of the sum, which gate 4's rows make cancel.

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

That is one clip: the fp32 codes keep the speaker as well and sit about 8% further in log-mel distance. These
scores are context, not a gate: the encoder's gate is code equality with the fp32 reference ("The arithmetic
reference"). Not measured: whether that gap holds across clips, and whether the backbone, which was trained on
bf16-encoded references, clones as well from fp32 codes.

## Shared code this touches

- `drama3/codec_pth.zig` moved to `core/torch_pth.zig` (its zip half to `core/torch_zip.zig`) and gained nested
  checkpoints and a raw reader; Drama 3's codec loads through the same code and its host tests pass.
- `drama3/wav.zig` moved to `core/wav.zig`: both families read WAV files and resample with it.
- `core/speech_api.zig`: `SpeechRequest.speaker`, an optional speaker vector, and `reference_audio`, an optional clip.
- `server/`: voices load through `speech_voices.zig`; the speech route, the sampling parser, the voices listing,
  model detection and the tokenizer loader each gained a branch for a release folder, which has no `config.json`
  and is recognised by `lm/zonos3_mini/params.json`.
- `core/speech_api.zig` also gained `SpeechRequest.conditions` and `steering` (`SpeechSteering`: deltas on a
  family's hidden sources, re-exported by `core/engine_api.zig`).
- `server/`: the `--steer-bundle` flag (`cli.zig`, `serve.zig`, `server.zig`, `speech_voices.zig`), the
  `GET /v1/audio/controls` route (`routes.zig`) and its place among the listing routes a read key may open
  (`auth_scope.zig`); `build/server.zig` passes the family's steering modules to the server.
- `native/`: `zonos3` in the CUDA open dispatch, the capabilities list and the rank refusals.
- `platform/optional_libraries.zig`: the cuBLAS entry's note names the ZONOS3 decoder beside Drama's GEMMs. The
  cuBLAS wrapper itself is a copy in the family's folder (`zonos3/cuda/blas.zig`); Drama's is untouched.
