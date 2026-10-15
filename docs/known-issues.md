# Known issues

This is the current list. Each release's notes summarize it and say what changed; an issue leaves the list only when a
release fixes it, and the release notes say which one.

## Models

- GLM-5.3 (EXLR8) decodes slowly next to the other models: a one-row round on two M5 Ultras takes about 45 ms. No CUDA
  engine serves it, and only the `home` profile on M5 Ultras has run. Replies past 2,048 tokens, where the sparse indexer
  starts picking keys, have not been compared with a whole-model reference.
- Qwen3.5-2B with a JSON schema: on an integer field it wants to write as a decimal, the 2B can keep adding digits until
  `max_tokens`, because the grammar allows no decimal point there. Nemotron writes the same schema correctly.
- Qwen3.8-27B on an M1 to M4 Mac: the drafter's landing counts last for the server's life, so after four prose replies
  greedy code drafts shallower, about 4% slower on an M3 Ultra. The tokens are the same.
- Qwen3.8-Flash-Next in speed-up mode: a drafted request right after a `"draft": false` one decodes slowly, and the next
  request is back to speed. The tokens are the same.
- Qwen3.8-Flash-Next on a Mac: the first turn after the server has been idle for a few seconds takes about 2 s to its
  first token on an M3 Ultra, even when its prompt is cached.
- On a Mac without tensor units (M1 to M4), Flash Next's uniform 6-bit checkpoint decodes different tokens from 1.0.9
  since 1.0.10's faster products. `FZ_M3_LANES=1` keeps the earlier kernels and their tokens.

## Living Weights (`--slide`)

- Facts about the same subject can overwrite each other, a later fact about the user can disturb an earlier one's
  answers on a GB10, yes-or-no questions about a changed value are unreliable, facts pile up badly past a few dozen, and
  thinking on brings back fewer answers. 1.0.8's release notes have the measurements.
- Learned facts cost decode speed on a Mac. 1.0.9 took back about 8% on an M3 Ultra with two facts; 1.0.8's M5 Ultra
  figures (213 tok/s after ten facts, 248 before any) have not been measured again.
- A lesson changes the checkpoint's own files in its folder, and nothing undoes it. Serve a copy of the checkpoint
  folder when you try it, and set `--api-key` so only your clients can teach it.

## Web research (`--cortheon`)

- The instruction filter that removes orders aimed at the model from web text is a heuristic. It reads each line on its
  own, so an order split across a line break is not seen whole, and a page written to steer the model could still
  make it put conversation text into the path of a URL it fetches. [docs/cortheon.md](cortheon.md) says what leaves the
  machine.
