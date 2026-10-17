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
- Qwen3.8-Flash-Next on a Mac: a turn more than `--keep-warm` seconds after the last request (900 by default) takes
  about 2 s to its first token on an M3 Ultra, even when its prompt is cached, while macOS wires the model again.
- On a Mac without tensor units (M1 to M4), Flash Next's uniform 6-bit checkpoint decodes different tokens from 1.0.9
  since 1.0.10's faster products. `FZ_M3_LANES=1` keeps the earlier kernels and their tokens.
- GLM-5.3-Flash's ModelOpt NVFP4 checkpoint on two CUDA ranks has not been run since 1.0.12: on two GB10s the weights
  leave no room for its streams beside the memory reserve. Code that path shares changed in 1.1.0 (NVFP4 scale planes
  and their tensor-parallel slices, the copy-source owner, NCCL's allocations and the RoCE cap on the ranks), so its
  regression run is owed. The EXL3 checkpoint is the two-rank path we measure.

## Living Weights (`--slide`)

- Facts about the same subject can overwrite each other, a later fact about the user can disturb an earlier one's
  answers on a GB10, yes-or-no questions about a changed value are unreliable, facts pile up badly past a few dozen, and
  thinking on brings back fewer answers. 1.0.8's release notes have the measurements.
- Learned facts cost decode speed on a Mac. 1.0.9 took back about 8% on an M3 Ultra with two facts; 1.0.8's M5 Ultra
  figures (213 tok/s after ten facts, 248 before any) have not been measured again.
- A rollback is all or nothing: `tensorfold slide rollback` returns the folder to its snapshot, undoing every lesson
  and delete since the first. One fact can be deleted on its own (`DELETE /v1/slide/fact/{id}`), which silences its
  neurons rather than erasing them. A key keeps other clients from teaching, but anyone with the key can teach a wrong
  fact. Teach a copy of the checkpoint folder.
- On file systems without clones, such as ext4, the snapshot is a full copy of the files a lesson can change, 18.5 GB
  for Nemotron 3.5 Lightning, and the rollback copies them back.
- The folder hold that keeps a server and a rollback apart works between TensorFold processes only: other tools that
  write files ignore it.

## Web research (`--cortheon`)

- The instruction filter that removes orders aimed at the model from web text is a heuristic. It reads each line on its
  own, so an order split across a line break is not seen whole, and a page written to steer the model could still
  make it put conversation text into the path of a URL it fetches. [docs/cortheon.md](cortheon.md) says what leaves the
  machine.
