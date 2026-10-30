# Nemotron 3.5 Transcribe

NVIDIA's Nemotron 3.5 Transcribe 30B-A4B (`nemo_speechlm`) runs on the native CUDA engine from NVIDIA's bf16 release,
on one GB10. It is a speech LLM: an audio encoder turns a recording into rows that stand in the prompt where the
chat template writes `<|audio|>`, and a Nemotron 3.5 Lightning language model (Mamba-2, attention and MoE blocks,
128 routed experts with relu^2, a shared expert, one MTP layer) writes the transcript, translation or answer. The
language model is the Nemotron family (`zig/src/families/nemotron/`, the checkpoint's names in `cuda_salm.zig`); the
encoder is `zig/src/families/nemo_audio/`, with its kernels in `zig/kernels/cuda/nemo_audio.cu`.

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

The weights and the encoder take 64.7 GiB. The default window is the card's trained 65,536 tokens (config.json's
`llm_config` names the backbone's 262,144, which the speech model was not trained to; `--context` above 65,536 is
refused). At that window the server admits 8 streams, the engine's own at 0.50 GiB and 7 more taking 512-row pages
(3.09 GiB in all) as they need them, and keeps a prompt cache of 64 states (3.0 GiB) and 13.0 GiB of copied rows.

The chat template turns thinking on by default. The card asks for transcription, translation and language
identification with thinking off and temperature 0, and for audio questions with thinking on, temperature 0.6 and
top_p 0.95. Send `chat_template_kwargs: {"enable_thinking": false}` per request, or start the server with
`--no-thinking`.

## Audio in a request

`/v1/chat/completions` (and `/v1/completions` given `messages`) takes audio in user messages as OpenAI's
`input_audio` part or as vLLM's `audio_url` part with a data URL:

```json
{"messages": [{"role": "user", "content": [
   {"type": "text", "text": "Transcribe the following audio."},
   {"type": "input_audio", "input_audio": {"data": "<base64 WAV>", "format": "wav"}}]}],
 "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}
```

- A clip is a WAV file: 16 kHz, mono, 16-bit PCM or 32-bit float, at least 10 ms. Other rates, channels and formats
  get HTTP 400 naming the clip.
- A request holds up to 8 clips and up to an hour of audio in all (12.5 rows a second, 45,000 rows), and no more
  than its window leaves after the text. Chat bodies may be 160 MiB on this model, enough for an hour of 16-bit
  WAV in base64; other routes keep the 32 MiB limit.
- `/v1/messages` refuses audio: Anthropic's API has no audio block.
- Audio parts in assistant, system or tool messages are refused, as is text that spells `<|audio|>` with no clip for it.
- The card's [prompt guide](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B#prompt-guide)
  lists the instructions for written or verbatim form, context terms, speaker tags (`<spk:0>` ...), translation and
  language identification; they are plain text beside the clip.

The encoder runs on each request's prompt pass: log-mel features on the host, the 32-layer Transformer ASR encoder
and the streaming Sortformer speaker encoder on the GPU, in 30-second windows, fused and projected to the language
model's 2,688 columns. A clip's rows are keyed by a hash of its samples in the prompt cache, so a repeated clip (a
second question about the same recording) resumes after it instead of encoding it again. `tensorfold audio-check
MODEL WAV OUT.safetensors` writes every encoder stage's rows for one clip, for comparison with NeMo's.

## Drafts

The checkpoint's MTP layer drafts by default. Its head reads the shifted prompt with the clip's audio rows in place,
as NeMo trains it. `--mtp-drafts` and `--mtp-confidence` set it as for the other Nemotron checkpoints;
`"draft": false` decodes serially. Copied continuations stay off for this model. The head adds 2.9 GiB (64.73 GiB against 61.85 with
`--no-drafts`).

## Exactness and speed

Measured on one GB10 at the default 65,536-token window, greedy, thinking off, "Transcribe the following audio.":

- **Against NeMo's model in bf16:** on five LibriSpeech clips (3.5, 5.9, 34 and 130 seconds, and a 19.5-second
  clip of two speakers), four replies are the reference's tokens, the 130-second clip's 379 among them. The 34-second
  clip writes "at 'em" for the reference's "at em", a near tie that the engine's bf16 logits settle the other way.
- **Encoder stages:** at every stage the engine's rows are as far from NeMo's fp32 rows as NeMo's own bf16 rows are,
  on eight clips; the speaker features are identical on all of them.
- **Drafted against `--no-drafts`:** byte-identical replies on 11, 30 and 60 minutes of audio.
- **Prompt cache:** a server with `--prompt-cache-gib 0` gave the same replies as the default one to every check
  below: two instructions after the same 11-minute clip, a question after a transcribed clip, seeded sampling,
  speaker tags, translation to German and French, language identification, audio questions, thinking on, a two-turn
  chat and the Anthropic route with text. A repeated clip resumed 261 of 267 prompt tokens and answered its first
  token in 0.067 s instead of 0.381 s.
- **Concurrency:** four long clips at once (11, 11 and 15 minutes and 130 seconds) took 81.8 s together, each reply
  identical to its solo run.

Long recordings, LibriSpeech test-clean read end to end without repeats (WER on the normalized text):

| Audio | Prompt tokens | Reply tokens | WER | First token | MTP drafts | `--no-drafts` |
| --- | --- | --- | --- | --- | --- | --- |
| 11 min | 8,730 | 2,061 | 1.13% | 13.3 s | 40.4 s, 76.1 tok/s | 73.7 s, 34.1 tok/s |
| 30 min | 22,374 | 5,238 | 1.27% | 33.9 s | 104.2 s, 74.8 tok/s | 190.8 s, 33.4 tok/s |
| 60 min | 44,872 | 11,084 | 1.18% | 70.2 s | 228.9 s, 70.0 tok/s | 411.2 s, 32.4 tok/s |

The first token waits for the encoder and the whole prompt pass. The drafts kept 72% to 76% of their proposals
(9,748 of 13,142 on the hour). The 130-second clip answered its first token in 2.32 s. The encoder's attention is a
plain per-window kernel, not yet a tiled one. The harness's quality suite was not run on this build.

## Not served yet

- Word timestamps and diarization segments (the card's `ctc_timestamps`, from its CTC head): the head stays
  unloaded.
- Audio URLs other than data URLs, and formats other than WAV.
- The DFlash2 draft model, and the FP8 and NVFP4 variants.
- More than one GPU: the model fits one GB10, and the two-rank expert split is Super VL's alone.
