# ZONOS3 Mini on CUDA

`zig/src/families/zonos3` runs Zyphra's ZONOS3 Mini text-to-speech model on the CUDA runtime, from the files the
publisher released, and serves it on `POST /v1/audio/speech` (speech edits on `POST /v1/audio/edit`). This page
records what is ported, where every source came from, and how each part was checked, so the checks can be rerun.

Status: one branch, proposed in #3615 and open as draft pull request #3720. All 40 kernels have their limits
records, numeric contract tests and a planted control each; the quality suite has not been run on this build.

## Sources

| What | Where | Pin |
| --- | --- | --- |
| Checkpoint | `huggingface.co/Zyphra/ZONOS3-Mini` (Apache-2.0) | revision `2921b8821772b0bd10a04456863d21abf9d2f283` |
| Reference code | `github.com/Zyphra/ZONOS3-Mini` (MIT) | commit `96d9e0ec1e39083ed336ac78f24e3dbc93721c0f` |
| Short-text registry | the reference repo's `model/src/zonos3/registries/short_text.json` (MIT) | the same commit |
| LACI alignment heads | the reference repo's `assets/laci_heads/*.json` (MIT) | the same commit |
| Steering bundle | the reference repo's `assets/steering/studio_bundle_mect60k_x10.json` and the 14 set folders it names (MIT) | the same commit |
| Reference environment | Python 3.12.3, torch 2.9.1+cu128, torchaudio 2.9.1+cu128, transformers 4.57.3 | the repo's `constraints.txt` |

Nothing is converted: the server reads the release folder as published. The revision moved from
`d2d142a86b1e6aeb4252679673c86ecd86177bf8` when the publisher added the speech-edit adapter (its one change to the
files read here).

| File in the release | Read by | sha256 |
| --- | --- | --- |
| `lm/zonos3_mini/model.pth` (bf16) | the backbone | `3b4ac0a6e3002bd575ad9a0761ae7524806a908f553042171dead8f3a2146caa` |
| `lm/zonos3_mini/params.json` | the geometry check | `28dc488d7142831d739f427e048c699b7ed12852d88da0a9123a137f15aa5a5b` |
| `lm/zonos3_mini/tokenizer/superbpe8192.json` | the text tokenizer | `90322f910600027dee7fe63d400d1d9dc1035c167d560aadf0871e3661d08bf3` |
| `autoencoder/generator.pt` (fp32) | the codec's quantizer, semantic head, resamplers, encoder | `75a587a3eda9ca3d345f0c305756a7ff8619b01685c261b08ae243c69c2caf9a` |
| `autoencoder/heads/keep2_plain/step_0050000.pt` (fp32) | the codec decoder's stages and iSTFT head | `7432e335e717417961267f2242547ce389a796cdc296a8b8d5289658e3c28936` |
| `autoencoder/teacher/model.safetensors` (bf16) | the encoder's SSL teacher | `a15182741d6ee3cd780b8b830f49fa6b7fcce383655d451970841b24114767a5` |
| `registries/codec_registry_keep2_plain.json`, `registries/silence_frame_*.json` | the decoder's files, the silence frame | `5437b4159b2bdbc33d3c1a27a316df7ae361b06f5ce3c9822f585402bcb4351a` (the registry) |
| `lm/zonos3_mini/text_rows/pause_tokens_s5000.pt` (fp32), `.json` | the phoneme, start and pause tokens' text rows and their ids | `35868b0ff855081f3340a1c96644cfe9bf21b42935d9ed991ddd59ea0375e621` |
| `lm/zonos3_mini/edit_lora/z3m_omni3b_edit_lora.pt` (fp32 holding bf16 and f16 values) | the speech-edit adapter | `60c1f2c452ed3895a820f035433ab113226e11a34e11da52bb48cb42a86ae978` |
| `speaker_encoders/ownmect_s2/ckpt_0037.pth` (fp32), `registries/speaker_encoder_registry.json` | the speaker encoder | `3cef65631f3161cbd8eacc6ea40e59a237c79656d85c7232dcbe0c8d07cae88f` (the checkpoint) |
| `web/aligner/manifest.json` and `shard-000.bin` to `shard-008.bin` (8-bit matrices, fp32 heads) | the forced aligner | `fe68218a5a785cd64a08d0a1fa367685f6749a6e71fe3cf9a1dd7f39303ea2b7` (the manifest), `efa75cb1b8f8a0a2699282536e1dc3b5348e9554100ec82aac011baba37367cc` (the nine shards in order) |
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
- **LACI** (`laci.zig`, `laci_detect.zig`, `laci_probe.zig`, `laci_run.zig`, `server/zonos_laci.zig`): with
  `--laci-heads`, a request's `laci`. See "LACI".
- **Phoneme, start and pause tokens** (`cuda/weights.zig`, `server/zonos_tokens.zig`) and **short texts**
  (`short_text.zig`, `server/zonos_short.zig`, with `--short-text`). See the sections of those names.

- **Speech editing** (`edit.zig`, `edit_plan.zig`, `cuda/adapter.zig`, `server/speech_edit.zig`,
  `native/zonos_edit.zig`): `POST /v1/audio/edit` and `/v1/audio/edit/align`, when the release folder holds the
  edit adapter. See "Speech editing".
- **Speaker encoder** (`speaker.zig`, `speaker_blocks.zig`): a clip's 192-d embedding, on the host's cores. See
  "The speaker encoder".
- **Forced aligner** (`cuda/aligner.zig`, `cuda/aligner_weights.zig`, `align_words.zig`): where each word of a
  transcript is spoken in a clip. See "The aligner".

Refused or absent: clips in formats other than WAV, LACI's duration targets and extended speaking-rate buckets,
accent adapters (the publisher has published none: its guide's adapters were trained on an earlier backbone in its
private repository), the `full_mr` decoder, Metal. Of speech editing: speech recognition (`/v2/edit/transcribe`, `verify`, an edit without
the clip's transcript), Japanese and Mandarin, clips over 60 s, and the aligner's long-form routing (a clip over
30 s needs its word timings from the caller).

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
  with ramps of 3 frames at both edges. Spans may not overlap. With `control_span_follow: "reading"` they follow
  the model's reading position instead: see "Spans that follow the reading".
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

## LACI

LACI (localized attention-constrained inference, `zonos3.tts.laci`) watches where a request is in its text while
it decodes and regenerates the part where it went wrong. `tensorfold-native serve RELEASE --laci-heads DIR` loads
the publisher's alignment-head files (its code repository's `assets/laci_heads`: one default file and 42 by speech
code), and `"laci": true` on a request turns it on. Without the flag the field is refused.

- **The reading position.** A head file names a few attention heads (four in the default file, all in layer 7)
  whose attention over the text follows the speech. After every generated row the host takes those heads' queries
  from the pass and their softmax over the text rows' keys (read once from the cache), sums them, and keeps the
  argmax and the centroid (`laci_probe.zig`).
- **Failures** (`laci.zig`, the reference's `Detector`): a skip (the reading jumps past a bin of eight text tokens
  that got far fewer frames than its share), a hallucination (the reading stops advancing), a truncation (the
  model ends with two or more bins unread). The text is done once the last bin has had its share and the reading
  has reached the last token; generation stops 15 frames later, and six rows after the text is done the sampler
  adds 3 to the stop id's logit.
- **A rollback** keeps the rows up to eight before the failure's onset, resets the request's lane, runs the prompt
  and the kept rows again as a prompt, and generates on under the seed plus 1,000 times the attempt, at most eight
  times. A streamed request only hands on rows no rollback can still reach; a whole reply holds every row until
  the result is final. That is why a streamed LACI reply may differ from the whole one when a rollback is cut
  short at rows already sent.
- **ACI masks.** A regeneration starts on probation: each alignment head is recomputed over the keys it may see,
  every key outside the text and the text keys within its radius of a centre that follows the reading forward,
  until the reading passes the point of failure. The engine closes those keys and values up into one scratch cache
  and runs the same attention kernel over it, so no kernel changed; the masked head's 32 values replace the
  unmasked ones before the output projection.
- **No guidance.** A LACI request runs one stream: `speaker_cfg_scale` and `controls_cfg_scale` other than 1 are
  refused. Steering controls work; a regeneration's kept rows are prompt rows, which are not steered, as in the
  reference.

Not served: duration targets (the reference's studio plug-in that fits a take to a length by retrying in another
speaking-rate bucket), and with them the extended speaking-rate buckets.

Checked against the publisher's code:

- **The session logic** (`zonos3-generate --laci-replay TRACE --expected EXPECTED`): the reference repository's
  own replay fixtures (`servers/ggml/tests/fixtures/laci-replay`), which pin its three implementations. The seven
  that need nothing unserved pass (`clean`, `early_eos`, `skip`, `slow_two_bins` and three short-text ones):
  every drained row, mask change, continuation and the final record. With the span follower ported (see "Spans
  that follow the reading"), three more pass: `stall` and `stall_held` with the follower's weights at every step,
  and `span_follow_scheduler`, 278 scripted readings, decisions and rollbacks. One is skipped: `pace_miss` needs a
  duration target.
- **On the model** (`zonos3-generate --model LM --laci-check D/gen_laci_X --laci-heads FILE`), greedy, eleven
  requests dumped with `rows --laci-heads` (each attempt's readings, masks and rows): two voice paths, three texts
  that might invite a failure, and six with strong steering doses, four of which the reference retried eight
  times each (stalls, a skip, truncations). In all eleven every attempt's reading positions, masks and rows equal
  the reference's, and so does the record (attempts, failures with their rows, the end, the final rows); 2,398
  readings in all. The centroids differ by at most 5.5e-4 text positions (the reference's softmax and sums are
  torch's).
- **Served**, at `--parallel 4`, ten of those requests through `/v1/audio/speech`: each reply has the reference
  audio's sample count. Eight match it at 64.8 to 79.2 dB, two of them after eight retries. One retried request
  ends after a single frame of near silence (peak 0.0008), where 16-bit rounding leaves 12.6 dB, and one ends
  before its first frame: no samples, as in the reference.
- **Sampled, served:** a LACI request repeats byte for byte and streams the same bytes; on two requests that need
  no retry it gives the bytes of the request without LACI, in the same time (the 17.8 s passage in 0.48 s both
  ways). Four at once (two under LACI, one plain, one steered under LACI) each gave the bytes they give alone.
- **Memory:** 4,458,496 bytes more at load: the probed queries on their way out and the scratch cache of a masked
  head.

Not measured: how often LACI rescues a sampled reply, or what its retries cost on real text. The checks say the
port does what the reference does.

## Phoneme, start and pause tokens

The release holds 143 text rows beyond the tokenizer's (`lm/zonos3_mini/text_rows/pause_tokens_s5000.pt`), which
the reference's own entry point appends to the text table: 138 for pronunciation groups, one start row and four
pause rows. The engine loads them when the folder has them, and the text may then hold their markers, private-use
characters as in the reference (`zonos3.tts.superbpe`):

- **A pause** after a word: U+E003, U+E004, U+E005 or U+E006, for about 0.25, 0.5, 1 and 2 seconds.
- **The start marker** U+E002 as the text's first character, or `"start_immediately": true`, which puts it
  there: speech from the first generated frame, and no silence frames after the prompt.
- **A pronunciation group**: U+E000, IPA code points of the release's 136, U+E001. Each phone is a row.

The text between markers is tokenized on its own, as the reference does it. A marker the release has no row for,
a start marker that is not first, an unclosed or empty group and a phone outside the inventory are refused.

- **Exact rows.** The rows are fp32 and were trained after the table's padding row. The reference in fp32 appends
  them as they are, so the engine does not round them into its bf16 tables: a row with such an id embeds as the
  padding row on the device, and the host replaces that row's sum with the ten audio padding rows' sum plus the
  fp32 row, in the reference's order of adds.
- **Checked:** greedy frames identical to the reference's for a pause (84 frames), the start marker (74, its
  prompt without silence rows) and a pronunciation group on a reference voice (55), at `--max-rows` 1 and 64;
  served, the three replies have the reference audio's sample counts and match it at 68.9, 76.2 and 67.0 dB.
  `start_immediately` gives the bytes of the typed marker.

## Spans that follow the reading

`control_span_follow: "reading"` places a request's `control_spans` by where the model is in its text rather than
by the character rate (`zonos3.tts.span_follow`). It needs the server's LACI heads (`--laci-heads`); without them,
as in the reference, the spans keep the character rate. `control_span_lead` (0 to 8, default 1) and
`control_span_lag` (-4 to 8, default 0) are in text tokens.

- **The tokens of a span** (`server/zonos_reading.zig`): a span's characters of the text as sent are mapped into
  the text the prompt spells (after normalization and the start marker: exact in unchanged stretches, proportional
  in rewritten ones, through the same `difflib` diff the reference takes), then to the text positions whose
  characters, outer whitespace trimmed, overlap it. A token's characters are those its bytes fall in, which is
  what the reference's tokenizer offsets give for this byte-level BPE; a marker or a phone is its own position.
- **The reading:** after every generated row the host reads the row's queries at the two attention layers against
  the text rows' keys (read once after the prompt) with the language's heads, as LACI does, and takes the argmax
  position. Under LACI the session's own readings are used, masks included, and a rollback rolls the follower back.
- **The gate** (`span_follow.zig`): the running median of the last three readings, never moving back and at most a
  token a reading. A span ramps in over 3 frames on the first frame decided with the reading at or past its first
  token less the lead, and ramps out on the first frame decided at or past the token after it plus the lag. Each
  frame's weights are decided once, so a speaker-guidance twin takes the same ones; they are kept in float64 and
  scale the span's delta in fp32, as the reference's do.

Checked against the publisher's code, greedy, six requests (`requests` writes them; `rows` dumps the follower's
readings, rollbacks and every frame's weights in `follow.json`): two spans on the test sentence alone, under LACI,
with speaker guidance on a reference voice, with a lead of 2, a lag of 1 and a constant control, on a text the
normalizer rewrites ("It costs $5 at 3 pm today"), and under LACI with `delivery.tempo` at -6, which the reference
retried eight times.

- **Each check** (`zonos3-generate --model LM --follow-check D/gen_follow_X --steer-bundle BUNDLE --laci-heads
  FILE`) resolves the request's steering with the bundle (the span deltas bit-identical to the reference's), feeds
  the follower the port's own readings, and compares the readings after every rollback, every frame's weights
  (exact in float64) and the frames (`rows.npy`, or the LACI check's readings, masks, rows and record). All six
  pass at prompt passes of 64 and 7 rows; the retried one over nine attempts and 970 readings.
- **Served**, `/v1/audio/speech` at `--parallel 4`: all six have the reference audio's sample count and match it at
  64.4 to 73.5 dB; each repeats byte for byte and streams the same bytes. The token ranges the server found are
  the reference's, or the frames would differ. The same request with `"rate"` gives other bytes, and equals the
  request without the field.

## Short texts

Alone, a text of a word or two often comes out wrong or runs on: the model was trained on sentences. The
reference (`zonos3.tts.short_text`) therefore generates such a text with a pause marker and a filler sentence
after it, under LACI, and cuts the take where the reading reaches the filler. `tensorfold-native serve RELEASE
--laci-heads DIR --short-text FILE` does the same, `FILE` being the reference package's
`registries/short_text.json`; both flags are needed, and the release's pause tokens.

- **What is short:** at most two whitespace-separated words whose tokens, times five, are at most twice the
  filler's ("Seven birds flew over the quiet river." in English; a filler for each of the 44 languages), with no
  marker of its own, in a request without guidance or control spans.
- **The cut:** the first frame a codes-only silence classifier calls silent (ten int8 weights a frame, summed past
  a threshold), searched from two frames before the reading reached the filler, after at least three voiced
  frames. A take with none within 15 frames ran into the filler: it is regenerated whole, twice at most, then cut
  where the reading reached the filler.
- **It is on by default** once the server has the registry and the heads, as in the reference: a request of one
  or two words is then spoken this way, and `"short_text": "off"` speaks it as before. The request is held until
  the cut, so a streamed short text arrives at once.

Checked against the reference: the three short-text replay fixtures (under "LACI"); on the model, "Hello." on an
embedding voice and "Good morning." on a reference voice, where every reading, every row and the cut (13 and 17
frames) equal the reference's; served, those two and "Hello." with `short_text` off have the reference audio's
sample counts (70.7, 63.8 and 64.2 dB). Sampled and served, a short text repeats byte for byte, streams the same
bytes and gives other bytes than with `short_text` off; four requests at once, two of them short, each gave the
bytes they give alone.

## Speech editing

`POST /v1/audio/edit` re-says part of a recorded clip: the regions that change are generated again by the backbone,
with the rest of the clip as context, and spliced into the clip's own codec frames. It is the release's `/v2/edit`
(`zonos3.tts.edit_runtime`, `zonos3.tts.edit`, `zonos3.tts.edit_align`), served when the release folder holds
`lm/zonos3_mini/edit_lora/z3m_omni3b_edit_lora.pt`; without that file the route answers 400 and nothing else
changes.

- **The adapter** is a LoRA on the release backbone (`zonos-accent-lora/3`): rank-64 factors on 59 linear layers,
  rank-16 factors on each expert's two matrices in ten layers, a shift of ten routers' balancing biases, and the
  rows of a vocabulary one id larger: a span-end audio id (4098) in every codebook's table and head block, and three
  edit-marker text rows. Every fill runs on it at strength 1 and every other request never reads it.
- **On the device** the factors are the bf16 the file's values are (all 158 factor tensors are bf16 values stored
  in fp32; a value that is not is refused, not rounded). A shifted layer is three launches of kernels the backbone
  already has: `z3_linear` for the down product, `z3_linear` for the up product, `z3_combine` to add the shift to
  the layer's own rows. An expert's factors are picked by the router's picks as its weights are. The span-end rows
  are f16 values, which bf16 does not hold, so they stay in fp32 on the host: a prompt row that holds the span-end
  code is summed there in `z3_embed`'s order and replaces the device's row, and the ten span-end logits of a frame
  are the host's dot products with the row the head read.
- **A fill** is one request: the prompt is `[speaker row] [condition rows] text` and then the clip around its
  masked spans, a marker row a span, the unmasked context ended by end-of-audio, each fill made so far ended by
  span-end, and the marker of the span to fill. It is sampled over codes and span-end only, the span-end masked in
  codebook `c` until row `min_frames + c`, and it ends ten rows after the first span-end row. Its frames before that
  row are the fill.
- **Other requests beside it:** rows on the adapter run in a pass of their own, so a spoken reply generated while
  an edit fills is the reply it is alone (checked below).

The request takes the release's fields: `audio_base64` (a WAV file), `target_text`, and either `spans` (up to
three `[start, end]` in seconds) or `original_text`, from which the regions are the words that differ. With
`original_text` the server places the words with the aligner (clips up to 30 s), or takes `original_words` from the
caller. `margin_seconds`, `max_span_seconds`, `min_fill_ratio`, `max_context_seconds`, `context_seconds`,
`num_takes` (1 to 4), `seed`, the sampling fields, `condition_buckets` and `condition_values`, `controls`,
`region_controls` and the steer-then-hold fields, `language` and `text_normalization` mean what they mean there. A
clip longer than `max_context_seconds` (16) is edited on windows: each region with `context_seconds` (4) of words
on both sides, cut halfway between words. The reply is the release's JSON: `spans`, `contexts`, `words`,
`expected_seconds` and `takes`, each take a 16-bit 44.1 kHz WAV in base64 with its `edited_regions`,
`fill_seconds` and `stopped`.

What differs from the release:

- **`speaker_embedding`** may also be 192 numbers, which fill the speaker row in place of the clip's own embedding.
- **No speech recognition:** `/v2/edit/transcribe` is not served, `verify` is refused, and an edit that reads the
  clip's words needs `original_text`.
- **Languages:** en_us, en_gb, fr_fr, de, es, it, pt_br and ko. The release also edits ja and cmn, cutting their
  text into words with ICU, which this port does not have; both are refused.
- **Lengths:** a clip of at most 60 s (the clip encoder's limit; the release takes 150 s), and an edited clip of at
  most the server's frame budget. The aligner places words in clips of at most 30 s and transcripts of at most
  1,022 bytes; past that the release routes by voice activity and recognized anchors, and this port asks for
  `original_words`.
- **Sampling:** `top_k` is the route's name for the release's `topk`, and the draws are TensorFold's keyed sampler,
  as for speech, so a sampled edit is not the reference's sample. Without `seed`, takes follow from the clip's
  length and the target text, so the same request gives the same edit.
- **Clips at other rates** are resampled to 44.1 kHz first, and the speaker encoder and the aligner hear that
  resampled to 16 kHz; the release resamples the upload to 16 kHz directly.

`POST /v1/audio/edit/align` (`{audio_base64, text}`) answers the text's words with their place in the clip, in the
form `original_words` takes.

### The speaker encoder

The release's speaker encoder (`zonos3.models.mect_speaker`: MECT-B2, 11.2 million weights, the registry's
`ownmect_s2`) gives a clip's 192-d embedding. An edit puts it in the speaker row unless `speaker_embedding` is
`false`, and `/v1/audio/speech` with `speaker_mode: "embedding"` and `ref_audio` now speaks in the clip's voice by
its embedding (it was refused before).

It runs in fp32 on the host's cores, not on the device (`speaker.zig`, `speaker_blocks.zig`): 16 kHz audio (the
teacher's resampler), 80 Kaldi log-mel bins with a Hamming window, each bin less its mean, a 3 x 3 convolution,
fourteen blocks of grouped 3 x 3 and 1 x 1 convolutions with a transformer layer over the frames (a soft mixture of
four experts), the four stages' rows weighed and pooled by attentive statistics, and a linear layer. The 1 x 1
convolutions are most of the work and run sixteen positions at a time over up to sixteen threads. It reads at
most the clip's first 60 s, as the release does.

### The aligner

The release's aligner is FuseAlign: a 12-layer transformer of width 1,024 over the clip's audio patches (16 ms
each) and the transcript's bytes, behind a frozen 12-layer text encoder (Dia's), with two heads that score every
millisecond as each word's start and end and a third that gives the word's presence. A word under 0.1 presence is
not placed and takes an equal share of the gap around it.

- **The weights** are the release's `web/aligner` folder, the publisher's export for its in-browser engine: the
  two transformers' matrices in 8 bits with a scale for every 32 values (GGUF's `q8_0`), the heads in fp32. The
  publisher's fp32 checkpoint is in a repository that is not public, so the 8-bit export is what is published and
  what is served, dequantized as the export defines it (the int8 value times its block's scale) into fp32 on the
  device: 1,846,591,492 bytes.
- **The arithmetic** follows the publisher's engines for this export (`servers/web/src/aligner`, which mirrors
  their ggml code): fp32 rows throughout, the text encoder's RoPE tables and its output rounded to bf16. The
  publisher's Python package runs the text encoder in bf16 instead; the check below is against that package.
- **No new kernel:** the linears are cuBLAS's GEMM (the decoder's handle) or, where a row's inputs overlap the
  next row's (the audio patches, the transposed convolutions as two phases, the kernel-5 convolution), `zc_linear`;
  the norms are `z3_rmsnorm` without a weight, each weight folded into the matrices that read its rows; the gated
  feed-forward is `z3_mul_silu`; the joint attention is the teacher's `zt_attention` with a zero relative table.
  RoPE is applied on the host (the queries and keys make a round trip a layer), and so is the text encoder's
  attention, whose heads are 128 wide where the kernel's are 64.
- **It loads** with the first edit that needs it and is admitted for its weights and 512 MiB of a forward's rows.

## Requests at once

`--parallel N` (1 to 16; 1 without the flag) serves N requests at once. Each has its own lane: a set of the
backbone's caches and recurrent state, and a set of the decoder's. One worker gives every request in progress a turn,
a frame each: the frame is sampled and decoded when its audio is due. The rows the turns leave then run in one pass
(`Engine.runLanes`), a row a lane, a guided request's twin among them: the launches that read no lane's state take
all the rows at once, and the four that do (the GDN layers' convolution and recurrence, the attention layers' cache
write and attention) run a lane at a time on that lane's state. The kernels are the ones already checked, and each
computes a row from that row alone, so a request's arithmetic does not depend on what runs beside it.

- **Memory:** each request at once past the first adds 1,236,862,976 bytes at the default context and frame budget
  (10,913,792 of them the backbone's lane, the rest the decoder's), admitted with the load: 3,866,946,048 bytes at
  `--parallel 1`, 7,577,534,976 at 4. A guided request's twin lane is still made by its slot's first guided request.
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
for X in tokens_pause tokens_start tokens_phonemes short_hello short_reference short_off; do
  python tools/zig/zonos3_reference.py rows LM D/requests/$X.json D/gen_$X --device cpu --laci-heads ZONOS/assets/laci_heads \
      --text-rows RELEASE/lm/zonos3_mini/text_rows/pause_tokens_s5000.pt
done
for X in embedding reference repeats digits long; do
  python tools/zig/zonos3_reference.py rows LM D/requests/laci_$X.json D/gen_laci_$X --device cpu --laci-heads ZONOS/assets/laci_heads
done
for X in plain laci guided gate normalized retry; do
  python tools/zig/zonos3_reference.py rows LM D/requests/follow_$X.json D/gen_follow_$X --device cpu \
      --steer-bundle ZONOS/assets/steering/studio_bundle_mect60k_x10.json --laci-heads ZONOS/assets/laci_heads \
      --text-rows RELEASE/lm/zonos3_mini/text_rows/pause_tokens_s5000.pt
done
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

Speech edits, the speaker encoder and the aligner have their own tool, `tools/zig/zonos3_reference_edit.py`, run in
the same environment (the reference repo's `third_party/fusealign-multilingual` is the aligner's package). With
`A=LM/edit_lora/z3m_omni3b_edit_lora.pt`, `S` the steering bundle, `H` the LACI heads folder and
`P=LM/text_rows/pause_tokens_s5000.pt`:

```sh
python tools/zig/zonos3_reference_edit.py edit-cases RELEASE D/gen_greedy_embedding D/gen_laci_long D/edit-cases
for X in edit_span edit_sampled edit_plain_row edit_three edit_words edit_windows edit_window_spans; do
  python tools/zig/zonos3_reference_edit.py edit LM A D/edit-cases/$X.json D/$X --steer-bundle S --laci-heads H --text-rows P
done
python tools/zig/zonos3_reference_edit.py clips D/gen_greedy_embedding D/gen_laci_long D/served
for N in short long; do
  python tools/zig/zonos3_reference.py clip RELEASE/autoencoder/generator.pt RELEASE/autoencoder/teacher D/served/${N}_audio.npy D/served/${N}_clip
  python tools/zig/zonos3_reference_edit.py speaker RELEASE/speaker_encoders/ownmect_s2/ckpt_0037.pth D/served/${N}_audio.npy D/served/speaker_$N
  python tools/zig/zonos3_reference_edit.py aligner RELEASE/web/aligner D/served/${N}_audio.npy D/served/${N}_tokens.json D/served/align_$N
done
python tools/zig/zonos3_reference_edit.py served-cases RELEASE D/served
for X in s_span s_plain s_three s_words s_windows s_window_spans v_span v_windows a_span a_words a_windows; do
  python tools/zig/zonos3_reference_edit.py edit LM A D/served/$X.json D/served/ref_$X --steer-bundle S --laci-heads H --text-rows P
done
```

`edit` runs the publisher's own `EditRuntime` on its reference backbone, grown to the adapter's vocabulary, with
the adapter on every fill; the clip's codes and embedding are handed to it, so it needs no encoder. The seven
`edit_*` cases edit the frames the reference generated (4.47 s and 17.33 s of them), with word timings spread by
letter count: they exercise the plan, not the aligner. The eleven served cases edit the two clips as a server
receives them: the reference's audio of the same two requests as 16-bit WAVs (4.67 s and 17.53 s), their codes
from `clip`. `aligner` loads the release's `web/aligner` weights into the publisher's `fusealign_hub` modules (the
8-bit matrices dequantized) and runs the package in fp32 on the CPU, its text encoder in bf16 as the package runs
it there.

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
| Spans that follow the reading, six requests | `--model LM --follow-check D/gen_follow_X --steer-bundle BUNDLE --laci-heads FILE --max-rows N` | the span deltas, every reading after each rollback, every frame's weights, the frames | all equal at N = 64 and 7 in all six |
| Speech edit, one span (1.0 to 1.6 s) | `--model LM --edit-check D/edit_span --edit-adapter A --max-rows N` | the reference's region, minimum length, prompt and 20 rows; the spliced codes | all equal at N = 64 and 7 |
| Speech edit without the speaker row, no minimum length | the same with `D/edit_plain_row` | 20 rows | all equal at N = 64 and 7 |
| Speech edit, three spans with `controls`, `region_controls` (a hidden and a speaker-embedding control), an onset and a speaking rate | the same with `D/edit_three` | three fills of 18, 18 and 16 rows, the later prompts holding the earlier fills | all equal at N = 64 and 7 |
| Speech edit by word diff ("brown fox" to "small grey cat", "rests" to "sleeps") | the same with `D/edit_words` | two regions from the diff, two fills of 71 and 20 rows | all equal at N = 64 and 7 |
| Speech edit on windows, a 17.3 s clip, by word diff | the same with `D/edit_windows` | two windows, their texts, two fills of 18 rows | all equal at N = 64 and 7 |
| Speech edit on windows, explicit spans, 2 s of context | the same with `D/edit_window_spans` | two windows, two fills of 21 and 22 rows | all equal at N = 64 and 7 |
| Speech edit, sampled | the same with `D/edit_sampled` | the region, minimum length, prompt and splice (the reference's rows carry the replay on) | the plan and the prompt equal; the rows are not compared |
| Speaker encoder, 4.67 s | `--speaker-check D/served/speaker_short --speaker-encoder CKPT` | the filterbank, the first convolution, the four stages, the pooled rows and the embedding | every row's cosine 1.0000000; the embedding within 4.0e-6; 0.43 s |
| Speaker encoder, 17.53 s | the same with `D/served/speaker_long` | as above | every row's cosine 1.0000000; the embedding within 4.1e-6; 1.23 s |
| Aligner, 4.67 s and 16 words | `--aligner-check D/served/align_short --aligner RELEASE/web/aligner` | each word's start and end from `fusealign_hub` on the same weights | 15 of 16 words identical, the other 1 ms off; the joint rows' cosine 0.999997, the text rows' 0.9989; 0.06 s |
| Aligner, 17.53 s and 64 words | the same with `D/served/align_long` | as above | 61 of 64 words identical, the others 1 ms off; the joint rows' cosine 0.999994, the text rows' 0.9958; 0.36 s |

`G` is `RELEASE/autoencoder/generator.pt` and `H` the `keep2_plain` head checkpoint; `A` is the edit adapter and
`CKPT` is `RELEASE/speaker_encoders/ownmect_s2/ckpt_0037.pth`. An edit check plans the edit from the dump's request
(regions, windows, expected and minimum lengths), lays out each fill's prompt after the reference's text rows,
generates the fill greedily on the adapter with the earlier fills it made itself, and splices; "all equal" is the
plan, every prompt row, every generated row and every spliced frame. The aligner's check passes when no word is
more than 20 ms from the reference's; the publisher's package runs its text encoder in bf16 and the port in fp32,
which is what the text rows' cosine shows. A capture passes when its worst
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

**Speech edits.** `tensorfold-native serve RELEASE --parallel 4` with the steering bundle; each of the eleven
served cases was sent to `/v1/audio/edit` three times and compared with the reference's run of it
(`D/served/ref_X`): the regions, the windows, the expected lengths, each fill's length and whether it stopped, and
the take's audio against the reference's CPU audio.

| Case | What it edits | Fills (frames) | Audio | Time |
| --- | --- | --- | --- | --- |
| `s_span` | one span of the 4.67 s clip, a given embedding | 9 | 205,800 of 205,800 samples, 63.6 dB | 0.56 s |
| `s_plain` | the same without the speaker row | 10 | 208,740 of 208,740, 62.6 dB | 0.56 s |
| `s_three` | three spans, `controls` and `region_controls`, a speaking rate | 7, 6, 5 | 191,100 of 191,100, 65.4 dB | 0.63 s |
| `s_words` | a word diff over the caller's `original_words` | 41, 7 | 285,180 of 285,180, 59.9 dB | 0.66 s |
| `s_windows` | the 17.53 s clip on two windows, by word diff | 8, 7 | 773,220 of 773,220, 63.5 dB | 1.36 s |
| `s_window_spans` | the same clip, two spans, 2 s of context | 14, 13 | 790,860 of 790,860, 63.4 dB | 1.37 s |
| `v_span` | one span, the clip's own embedding from the speaker encoder | 9 | 205,800 of 205,800, 64.3 dB | 0.96 to 1.01 s |
| `v_windows` | two windows, the clip's own embedding | 7, 8 | 773,220 of 773,220, 63.8 dB | 2.53 to 2.54 s |
| `a_span` | one span, `original_text` alone: the server aligns it | 7 | 205,800 of 205,800, 65.0 dB | 1.03 to 1.05 s |
| `a_words` | a word diff from `original_text` alone | 19, 9 | 220,500 of 220,500, 66.0 dB | 1.10 to 1.19 s |
| `a_windows` | the 17.53 s clip on two windows from `original_text` alone | 9, 8 | 776,160 of 776,160, 64.2 dB | 3.03 to 3.08 s |

In all eleven the regions, windows, expected lengths, fill lengths and stops equal the reference's. In the three
`a_` cases the reference was given the word timings its own aligner run found, and the server's words are within
1.0 ms of them. The times are the second and third request of each case; the first edit after a start also loads
the clip encoder (8.7 s) and the first to align loads the aligner (about 1.3 s more). An edit's time is mostly
outside the backbone: the clip's encoding, its speaker embedding (0.43 s for 4.67 s of audio, 1.23 s for 17.53 s,
4.6 s for 60 s) and the decode of the whole edited clip; in the standalone tool the fills of a whole edit take
0.04 to 0.16 s.

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

The same steps were run again with speech editing, the speaker encoder and the aligner in the tree, and all passed
(the lean check again 221 problems on both sides, 0 new). That change adds a field to the speech request Drama 3 is
served through and a reader to `core/torch_pth.zig`; Drama's host tests pass, and its A/B on the M5 Ultra below was
not repeated for it.

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
  `--context 4096` and 2,000 frames that is 3,866,946,048 bytes (3.60 GiB), 33,554,432 of them the workspace
  cuBLAS is given. What cuBLAS allocates for itself is outside this count: `nvidia-smi` showed the process about
  64 MiB larger than with the earlier kernel, and the server's log shows the same
  number admitted and held after the load. A load that holds more than it was admitted for is an error.
- **The clip encoder** loads with the first request that brings a clip and takes 5,643,510,688 bytes (5.26 GiB) of
  weights, state and workspaces, all of fixed size; the log again shows that number admitted and held.
- **The edit adapter**, when the release folder holds it, is part of the load: its factors and shifted biases as
  the device holds them (counted from the file's index before anything is allocated) and the scratch its passes
  take, 143,684,608 bytes in all at the default context. With it, four requests at once load at 7,721,219,584
  bytes, admitted and held alike.
- **The aligner** loads with the first edit that needs word timings and is admitted for 2,383,462,404 bytes: its
  1,846,591,492 bytes of weights and 512 MiB for one forward's rows, which are allocated for the forward and freed
  after it. Its count is not checked after the load as the clip encoder's is.
- **The speaker encoder** holds no device memory: its 45 MB of weights and a forward's maps, reused from layer to
  layer, are host memory. Its check's process peaked at 1.49 GB resident on a 60.3 s clip, the reference's dumps it
  compares with included.

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

Speech editing, the speaker encoder and the aligner add no kernel. The edit adapter and the aligner launch nine
of the 40 at shapes they were not served at before, and those shapes joined the kernels' served lists: `z3_linear`
at an adapter's down product (1,024 to 64), its up product onto the head (64 to 40,980) and an expert's factors
(256 to 16 to 3,072, picked); `z3_combine` adding a shift at the head's width, the dense feed-forward's and an
expert's; `z3_rmsnorm`, `z3_mul_silu`, `ze_add` and `zt_attention` over the aligner's joint sequence; `zc_linear`
over the aligner's audio patches and its kernel-5 convolution; `zc_gelu` over a refinement head's rows. The
aligner calls `zc_linear` with rows that overlap (a row starts 256 values after the last and reads 512), which the
contract's inputs do not: its test rows do not overlap. The launches' thread counts and largest shared memory are
within the records, so gate 5's records stand. With the shapes added, the same command on the same card ended with

    (filtered) SUMMARY 41 kernels: 41 pass gates 1-4, 0 fail, 0 known findings; 41 controls: 41 red as planted, 0 missed; 0 limit problems; gates 6903 ms, limits 5 ms

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
- `core/speech_api.zig` gained `SpeechRequest.edit` (`SpeechEdit`: one step of a speech edit on the engine: a
  clip's codes, a fill's rows, edited codes' audio, words placed in a clip), re-exported by `core/engine_api.zig`.
  A request without it is as before.
- `core/speech_api.zig`: `SpeechSteering.follow` (each span's text positions, the gate's lead and lag, the heads
  that read). A request without it is as before.
- `core/torch_pth.zig` gained `readF32Strided`, which reads a tensor stored in any stride order (the speaker
  encoder's checkpoint holds 13 channels-last convolution weights). The existing readers are unchanged.
- `server/`: `speech_edit.zig` and its two routes (`routes.zig`), their place among the routes an inference key may
  call (`auth_scope.zig`); `build/server.zig` passes the family's edit, alignment and speaker modules to the server.
- `platform/optional_libraries.zig`: the cuBLAS entry's note names the ZONOS3 decoder beside Drama's GEMMs. The
  cuBLAS wrapper itself is a copy in the family's folder (`zonos3/cuda/blas.zig`); Drama's is untouched.
