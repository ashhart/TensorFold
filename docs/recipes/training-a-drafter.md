# Training a drafter

This is how a learned block drafter is trained from what the native server records, and released in the
[drafter release format](../drafter-release.md). The steps are in order. Each comes with what it gained on the one
target it has been through, so treat the numbers as Kolibri's, not as laws:

- Kolibri-1 (78B MoE, 3.5B active, FP8) on one DGX Spark, 3-10 Oct 2026, released as
  [josch15366/Kolibri-1-block-drafter](https://huggingface.co/josch15366/Kolibri-1-block-drafter). Decode speed
  against copy drafting alone, one stream, prompts no checkpoint trained on: 1.40x on SWE agent turns, 1.31x on chat
  and 1.40x on German greedy; 1.49x, 1.27x and 1.24x sampled as served; 1.10x at four streams.

Kolibri was recorded and served on a fork of the Python engine, now frozen. The native server records with
`--record` (CUDA, Nemotron 3.5 Lightning first). Nemotron is the second target, and the check of whether these steps
carry over.

Training runs outside TensorFold. Ours is in
[jschmied/qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10/tree/c4d69bacb6edab1b23829f98e79c2cfd6205d329/tools/kolibri/drafter):
`probe.py` (the tap probe), `model_block.py` and `train_block.py` (the drafter and its training), `eval_block.py`
(acceptance on recordings) and `export_block.py` (the release). It reads recordings and the target's head, and
prefills text through the target with that fork's Kolibri forward. TensorFold ships no training code.

## How to use it

1. Do the steps in order. Score after each on the same fixed slices, and keep their prompts out of every training set.
2. Score on output the target generated for prompts no checkpoint trained on: per class (code or agent turns, chat,
   each language), accepted drafts per round of 3 and 4, and agreement per position.
3. Acceptance is a speed measure, never a quality one: the target verifies every draft, so a drafter cannot change a
   reply. Its quality check is the contract below.
4. When a step proves out on another target, add its result. When one loses, add it to "Don't bother".

## Record

`tensorfold serve MODEL --record DIR` writes what serving computes. Each server makes its own new folder
`DIR/<pid>-<k>`, and each stream writes `<n>.*` in it:

| File | Contents |
| --- | --- |
| `.tok` | u32: the token at each committed row |
| `.pos` | u32: the row's cache position |
| `.kind` | u8: 1 for a reply's row, 0 for a prompt's |
| `.st` | the tapped rows, `row_bytes` each, as the family's `features` gives them |
| `.top` | with `logprobs`: each reply row's best ids (u32 x k), then their logprobs (f32 x k) |
| `.json` | `rows`, `first_row`, `prompt_len`, `taps`, `row_bytes`, `dtype`, `top_k`, and `gap` if the stream stopped early |

- Record with drafts on. On Nemotron the recorded rows were byte-identical drafted, with `"draft": false` and with
  concurrent streams, and replies were identical with and without `--record`. Check both on your recording build
  before you record for days: the same prompts with and without `--record`, drafted and `"draft": false`, must give
  equal `token_sha`.
- A prompt pass records the rows it computed, from its cached prefix on. Rows a prompt pass recomputes differ from the
  decode rows of the same positions by about 1-5 % (relative L2), since the kernels differ. Both are fine to train on.
- Include the tap past the last layer (`taps` = the layer count, normed). The head reads that row, so the target's
  logits at every recorded row can be computed offline over the draft vocabulary: the head rows times that row, then
  the adapter's bias and softcap. Without it, the target's distribution comes only from `.top`, which exists for
  reply rows of requests that ask for `logprobs`.
- Disk: `row_bytes` per row. Nemotron's one tap is 5,376 bytes, so a million rows is 5.4 GB. Kolibri's three bf16
  taps were 15,360 bytes a row, so its 6.03M recorded rows held about 93 GB of tapped rows. A stream starts no files
  below 25 GiB free.
- Check every `.json` against the release you train for (`checkRecording` in `zig/src/core/drafter_release.zig`):
  the same `taps`, `row_bytes` and `dtype`. A recording with other taps cannot train that release.
- Serve with at least as many `--parallel` slots as clients, or the prompt cache misses and every agent turn records
  its whole prompt again (on Kolibri, 135 of 2.8M prompt tokens hit with 7 clients on 6 slots; 95 % with 8).
- Generation and training can't share a 128 GB machine with a ~77 GB target. Alternate them.

## The steps

1. **Probe the layers.** Fit a linear map from each layer's rows to the target's own final row k ahead (k = 1 to 3),
   on held-out conversations, and score top-1 through the target's head. Tap where the next tokens sit. On Kolibri
   that was the outputs of layers 44-48 (counted from 0) of 50, not the middle; three taps (ids 45, 48 and 50 in the
   release format) added 1.5-2 points over one. Minutes on the GPU. A family defines what its tap ids mean once, in
   `features`.
2. **Pick the draft vocabulary from the target's own output.** Count the target's top-1 tokens over its recorded
   output per class, and take the slice that covers what you serve. A token outside the slice can never be drafted,
   so coverage caps acceptance. On Kolibri a 32k slice covered German at 0.896. Widening it to every token seen
   (60,614) gave +0.075 / +0.014 / +0.037 accepted (code / chat / German) without retraining.
3. **Train on the target's distribution, not on text.** The loss is KL to the target over the draft vocabulary at
   every block position, weighted 0.8^j, with the target's tapped rows as inputs. On the same 10M tokens it gained
   about 2x what L1 state regression with cross-entropy gained, and trained 17 % faster.
4. **Get volume and variety first.** 45M tokens of broad text, regenerated by a strong model and prefilled through the
   target, generalised. About 1M rows of the target's own output, repeated, were memorised and moved nothing on unseen
   prompts.
5. **Then add the target's own served output.** Mix recordings into every batch at a fixed share: 40 % on Kolibri,
   code 0.3 / chat 0.4 / German 0.3, at most one pass over code recordings and three over chat and German. German rose
   14 % and chat 7 % on one chain drafter. When one class runs out, keep the others' shares. When all run out, code
   acceptance falls back toward the text-only level, so end a run with them.
6. **Keep every language and class in every batch.** One 26M-token batch without German cost 18 % of German
   acceptance.
7. **Draft a block, with a chain head at position 1.** A 4-layer block drafter proposing four positions in one pass,
   whose row 0 is an EAGLE-3 chain head, against a 1-layer chain drafter: accepted of 3 went from 1.38 / 1.23 / 0.95
   to 1.91 / 1.29 / 1.11. Start the chain head from a trained chain drafter.
8. **Train position 1 at a low learning rate.** Frozen, it capped the block. At the full rate it forgot its chain
   skill. At 0.1x in its own optimizer group, with loss weight 2, code went from 1.50 to 1.62 in 1,621 steps.
9. **Add a predecessor head.** Each position j >= 2 reads its predecessor token through a low-rank head (rank 256,
   identity at the start), Kolibri's argmax in training. It came with the recordings in one step, so its own share is
   not separated.
10. **Export to the release format and check it.** The release carries the target's embedding, the head rows of the
    draft vocabulary, and the target's adapter values (`embed_scale`, `tied`, `softcap`, `head_bias`) read from its
    config and forward pass. It must open with `Release.open` and pass `check` against the target the server loads.

Kolibri's final hyper-parameters, for reference: hidden 2,560, 20 heads x 128, FFN 6,144, 4 layers, block 4, a
2,048-row context window, 1,024 anchors a window, AdamW from 2e-4 down to 2e-5 (cosine) after 100 warm-up steps, a
fresh optimizer per stage, predecessor and spine rank 256, checkpoints every 2,500 steps. The release took 76,083 steps
over 213.6M tokens. On Kolibri's held-out recordings it accepts 1.937 / 1.321 / 1.130 per round of 3 (code / chat /
German), against 1.38 / 1.23 / 0.95 for the chain drafter it replaced. Those slices chose the checkpoint, so they are
development numbers, not a test set.

## Check it in TensorFold

- **Replies.** Drafted replies equal `"draft": false` ones, greedy and sampled with a seed, solo and concurrent: the
  same `token_sha`. The drafter changes how fast a reply comes, never which reply. Any difference is a bug in the
  serving path, not a weak drafter.
- **Speed.** Decode tok/s against the same build with copy drafting alone, one stream and several, on prompts no
  checkpoint trained on. The reply's `acceptance_rate` and `tokens_per_round` explain a speed change; they don't rank
  drafters on their own, because a round's cost depends on depth, batch and the bytes the drafter reads.
- **Format.** The release opens (`Release.open`), its adapter matches the target the server loaded (`check`), and
  every recording it was trained on matches its taps (`checkRecording`). A loader runs `check` before serving, so a
  release whose `embed_scale`, `tied`, `softcap` or `head_bias` differ from the target is refused, not served with
  wrong logits.

## Don't bother

- **Loss variants past step 3.** KL, a soft-target cross-entropy and AUF tied on Kolibri.
- **A teacher-to-self schedule for the predecessor.** Under greedy acceptance a drafted predecessor survives only if
  it equals the target's argmax, so on every accepted prefix the two are the same token.
- **The data token as predecessor, and a DSpine-style spine.** Both within +-0.002 of the control. Kolibri's release
  kept the spine because it costs little.
- **More recordings of the class you have most of.** On Kolibri the scarcest class (German) bounded every pass.
  Record that one.
- **Depth 4 on a MoE target.** Slower than depth 2 on Kolibri: each extra verify row brings its own experts.

## Serving notes from Kolibri

On the GB10 a draft round costs the bytes it reads, not its launches. The bf16 block drafter was slower than the
1-layer chain and lost 17 % at four streams. With an FP8 body, an NVFP4 head slice (the same drafts, a quarter of the
bytes) and one batched pass for a round's streams, it is 1.10x at four streams. The release's `serving` field holds
those defaults (depth 2, `fp8`, `nvfp4`).

## Nemotron as the second target

Nemotron gives one tap today: the final norm's rows (`taps` = 52), which its head and its own MTP head read. Its
adapter is plain (`embed_scale` 1.0, untied, no softcap, no head bias). Its MTP head is the baseline: a learned
drafter is worth serving only if it beats MTP at the same depth on the same prompts, by decode speed with replies
unchanged. A tap probe across its layers needs `features` to give more tap ids first.
