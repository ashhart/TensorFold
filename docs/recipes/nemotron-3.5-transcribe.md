# Nemotron 3.5 Transcribe

NVIDIA's Nemotron 3.5 Transcribe 30B-A4B (`nemo_speechlm`) runs on the native CUDA engine from NVIDIA's bf16 release,
on one GB10. It is a speech LLM: an audio encoder turns a recording into rows that stand in the prompt where the
chat template writes `<|audio|>`, and a Nemotron 3.5 Lightning language model (Mamba-2, attention and MoE blocks,
128 routed experts with relu^2, a shared expert, one MTP layer) writes the transcript, translation or answer. The
language model is the Nemotron family (`zig/src/families/nemotron/`, the checkpoint's names in `cuda_salm.zig`); the
encoder is `zig/src/families/nemo_audio/`, with its kernels in `zig/kernels/cuda/nemo_audio.cuh`.

The encoder was written from the release's `config.json`, its tensor names and shapes, the model card, the
Sortformer papers (arXiv 2409.06656, 2507.18446) and textbook definitions, and checked against stage-by-stage dumps
of NVIDIA's NeMo run unmodified as a black box. No NeMo or vLLM source was used to write it.

## Checkpoint

`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` at revision `c421d6acdb436edd7ae00565f6414631fb9bd56d`: one
`model.safetensors` of 63 GiB in NeMo's layout, every tensor bf16 except the Mamba layers' fp32 parameters, the
routers' fp32 biases and the CTC timestamp head. The engine reads these tensors as stored; nothing is converted
ahead of time. The routed experts stay in bf16 on a grouped kernel of their own (`zig/kernels/cuda/experts_bf16.cu`).

Not read: NVIDIA's FP8 and NVFP4 variants, the DFlash2 draft model, and the CTC timestamp head.

## Run on one GB10

```bash
tensorfold serve /models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B --name transcribe
```

The weights and the encoder take 64.89 GiB. The default window is the card's trained 65,536 tokens (config.json's
`llm_config` names the backbone's 262,144, which the speech model was not trained to; `--context` above 65,536 is
refused). At that window the server admits 8 streams, the engine's own at 0.50 GiB and 7 more taking 512-row pages
(3.09 GiB in all) as they need them, and keeps a prompt cache of 64 states (3.0 GiB) and 13.4 GiB of copied rows.

The chat template turns thinking on by default. The card asks for transcription, translation and language
identification with thinking off and temperature 0, and for audio questions with thinking on, temperature 0.6 and
top_p 0.95. Send `chat_template_kwargs: {"enable_thinking": false}` per request, or start the server with
`--no-thinking`.

## Audio in a request

`/v1/chat/completions` (and `/v1/completions` given `messages`) takes audio in user messages as OpenAI's
`input_audio` part or as vLLM's `audio_url` part with a data URL:

```json
{"messages": [{"role": "system", "content": ""},
  {"role": "user", "content": [
   {"type": "text", "text": "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only."},
   {"type": "input_audio", "input_audio": {"data": "<base64 WAV>", "format": "wav"}}]}],
 "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}
```

- A clip is a WAV file: 16 kHz, mono, PCM (16, 24 or 32-bit) or 32-bit float, at least 20 ms. Other rates, channels
  and formats get HTTP 400 naming the clip.
- A request holds up to 8 clips and up to an hour of audio in all (12.5 rows a second, 45,000 rows), and no more
  than its window leaves after the text. Chat bodies may be 160 MiB on this model, enough for an hour of 16-bit
  WAV in base64; other routes keep the 32 MiB limit.
- `/v1/messages` refuses audio: Anthropic's API has no audio block.
- Audio parts in assistant, system or tool messages are refused, as is text that spells `<|audio|>` with no clip for it.
- The card's [prompt guide](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B#prompt-guide)
  lists the instructions for written or verbatim form, context terms, speaker tags (`<spk:0>` ...), translation and
  language identification; they are plain text beside the clip, with the system prompt empty.

The encoder runs on the GPU on each request's prompt pass: log-mel features, the 32-layer ASR encoder and the streaming
speaker encoder, in windows of 40-second cores with 4 seconds of context on each side and the speaker cache carried
between them, fused and projected to the language model's 2,688 columns. The speaker cache scores its frames in f64 and
applies two of the release's keys as the release does. `pred_score_threshold` (0.25) is the least a speaker's own
probability counts for in its frame scores; frames under 0.5 never score, and when fewer than 188 frames score, silence
rows fill the rest. With `min_pos_scores_rate` (0.5), a speaker with at least 11 frames that score above a lone 0.5
keeps only those: 11 is floor(0.5 x 22), and 22 is each speaker's share of the 188 rows after its silence row. A speaker
heard briefly and weakly keeps its boosted frames instead. `sil_threshold` changed nothing on any clip tried, and is not
read. A clip's rows are keyed by a hash of its samples in the prompt cache, so a request that repeats an earlier prompt
up to and past the clip (the same request again, or a later turn of the same chat) resumes after it instead of encoding
it again; the card's prompts put the instruction before the clip, so a new instruction about the same clip encodes it
again. `tf-cuda-test audio-check MODEL WAV OUT.safetensors` writes every encoder stage for one clip.

## Drafts

The checkpoint's MTP layer drafts by default; its head reads the shifted prompt with the clip's audio rows in place,
as the language model does. `--mtp-drafts` and `--mtp-confidence` set it as for the other Nemotron checkpoints;
`"draft": false` decodes serially. Copied continuations stay off for this model. The head adds 2.88 GiB
(64.89 GiB against 62.01 with `--no-drafts`).

## Exactness and speed

Measured on one GB10 at the default 65,536-token window, greedy, thinking off, the card's verbatim prompt with an
empty system prompt:

- **Encoder against NeMo:** on ten clips (3.5 s to 153 s, one to four speakers, one to four windows), every stage is
  within bf16 rounding of NeMo's bf16 and fp32 runs, apart from rows where NeMo's own two runs disagree as much: a
  few rows whose size grows by chance under bf16 inside the ASR encoder, and the speaker cache's choice among
  tied bf16 scores. The prompt has NeMo's length on all ten: the template's 51 tokens around the placeholder and
  the clip's rows.
- **Short clips:** the ten clips' replies have a WER of 1.55% against the LibriSpeech text (17 of 1,095 words, nearly all
  names and spellings).
- **Drafted against `--no-drafts`:** byte-identical replies on the ten clips and on 11, 30 and 60 minutes of audio.
- **Prompt cache:** a server with `--prompt-cache-gib 0` gave the same replies as the default one to every check below: seeded sampling (twice, the
  same reply each time), speaker tags on two speakers, translation to German and French, language identification,
  two audio questions, thinking on, a two-turn chat and the Anthropic route with text.
  The prompt cache resumes a request after the longest prefix it has seen: the second turn of a chat about a clip
  resumed 96 of its 153 prompt tokens, past the clip, and a repeated 11-minute request resumed 8,812 of 8,818 and
  answered its first token in 0.30 s instead of 4.16 s. The card's prompts put the instruction before the clip, so a
  new instruction about the same clip in a new chat encodes the clip again.
- **Concurrency:** four clips at once (11, 11 and 15 minutes and 130 seconds) took 67.0 s together against 26.9, 32.1,
  40.0 and 6.0 s alone, each reply identical to its solo run.

Long recordings, LibriSpeech test-clean read end to end without repeats (WER on the normalized text):

| Audio | Prompt tokens | Reply tokens | WER | First token | MTP drafts | `--no-drafts` |
| --- | --- | --- | --- | --- | --- | --- |
| 11 min | 8,818 | 1,978 | 2.14% | 4.31 s | 31.6 s, 72.4 tok/s | 62.8 s, 33.8 tok/s |
| 30 min | 22,402 | 5,163 | 1.32% | 10.98 s | 80.5 s, 74.3 tok/s | 166.5 s, 33.1 tok/s |
| 60 min | 44,900 | 11,255 | 1.17% | 22.69 s | 184.6 s, 69.5 tok/s | 371.4 s, 32.2 tok/s |

The first token waits for the encoder and the whole prompt pass. The drafts kept 78% and 74% of their proposals on
the 11 and 30-minute recordings (1,680 of 2,162 and 4,555 of 6,185). The harness's quality suite was not run on this
build.

## Not served yet

- Word timestamps and diarization segments (the card's `ctc_timestamps`, from its CTC head): the head stays
  unloaded.
- Audio URLs other than data URLs, and formats other than WAV.
- The DFlash2 draft model, and the FP8 and NVFP4 variants.
- More than one GPU: the model fits one GB10.
