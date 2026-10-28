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
- Qwen3.8-27B on Metal allocates DeltaNet rollback scratch lazily for the verification slots that need it. The bank
  can be resized or released between completed rounds; an idle or newly lone host can release it after the previous
  verification has been kept. A revised physical-memory peak comparison with 1.0.12 remains unmeasured.
- Qwen3.8-27B cold prefill at 64K tokens on RTX 50-series GPUs (sm_120) measures about 2% slower than 1.1.0; 32K and shorter prompts, and GB10, are at parity or faster.
- Flash Next on Metal with explicit --parallel 2 or more retains a cache-count/performance issue: on an M3 Ultra with
  the MLX 4-bit checkpoint, 9 of 12 chat steps reused 17 to 312 fewer tokens than the single host, with the same
  output tokens, and a lone request decoded 7% slower. The default --parallel auto uses the single host and matched
  1.0.12's cache counts; the separate shared-session repair is held for a later 1.1.x and is absent from the current
  1.1.1 series.
- The 1.0.12 gfx1151 Qwen receipts leave the 27B cancellation failure open and the 9B/27B drafted/plain comparisons
  unrun. Those receipts do not establish a new 1.1.0 device pass.
- Qwen3.8-Flash-Next in speed-up mode: a drafted request right after a `"draft": false` one decodes slowly, and the next
  request is back to speed. The tokens are the same.
- Qwen3.8-Flash-Next on a Mac: a turn more than `--keep-warm` seconds after the last request (900 by default) takes
  about 2 s to its first token on an M3 Ultra, even when its prompt is cached, while macOS wires the model again.
- Qwen3.8-Flash-Next on one GB10 reads its n-gram embedding tables (29.8 GiB in the MLX 4-bit MTP checkpoint) from
  the checkpoint files through the page cache, so they compete with the server's own memory. With
  `TF_FLASHNEXT_CHUNK=16384` the larger prompt scratch leaves too little room for all of them: in our runs, time to
  first token at 65,536 tokens ranged from 32.9 s to 49.1 s, about 1.5 times, with heavy paging in the slow runs. With
  the default 1,536-token chunks and the prompt cache off, three runs differed by at most 2.2%; saved prompt states
  (up to 16 GiB by default) use the same memory and can push table rows out at that chunk size too.
- On a Mac without tensor units (M1 to M4), Flash Next's uniform 6-bit checkpoint decodes different tokens from 1.0.9
  since 1.0.10's faster products. `FZ_M3_LANES=1` keeps the earlier kernels and their tokens.
- @Bizuayeu's regression run on shipped 1.1.0 loaded the whole GLM-5.3-Flash NVFP4 checkpoint on two GB10s at TP=2 in 302.1 s, kept one token hash per prompt across three runs and found a passphrase in 199,307 tokens; scored prompts and more than two ranks were not covered ([#1040](https://github.com/ashhart/TensorFold/issues/1040)).
- DeepSeek-V4.1 CUDA pieces land in this release, but no release binary serves DeepSeek checkpoints until its whole-model gate ([#827](https://github.com/ashhart/TensorFold/pull/827)) passes.


## HIP builds

- gfx1100 with Ubuntu's clang 21 hipcc fails to compile `affine_dot2_stream_relu2_pair<Dot, 6, 1, 8>` ("branch size
  exceeds simm16"). Build with ROCm 7.13, or pass `-Dhip-relax-branches=true`, which adds
  `-mllvm -amdgpu-s-branch-bits=15` to the gfx1100 code objects. The option is off by default and has not been run on a
  gfx1100 GPU ([#1521](https://github.com/ashhart/TensorFold/issues/1521)).

## Living Weights (`--slide`)

- Living Weights implementation and safety work is in the 1.1.1 tree, including CUDA learners, shared retention
  controls, lesson progress and save/restart work. Model, retention and save/restart qualification remains pending.
  The #1029 lesson-counter repair is excluded and remains held: rejected lessons must not shift the prompts used
  for later training and checks.

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
