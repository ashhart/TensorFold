# Nemotron causal pocket placement

This is an experimental Living Weights placement and shrink path for Nemotron's Metal and CUDA trainers (#1968).
Host checks cover causal selection and rejection/rollback. Real-checkpoint recall, lesson time and device exactness
remain unqualified until the device comparison below runs on each backend.

With `--slide`, Nemotron defaults to causal placement. Set `TF_SLIDE_PLACEMENT=legacy` before starting the server to
restore the previous middle-layer selection and disable shrinking. Other families retain their existing selection.
The changes run only in the learner; a server without `--slide` does not construct or call the trainer.

## Locate before placing

The learner reads up to eight of the lesson's own training answers. For each, it runs a baseline forward and one
intervention per learnable MoE layer. At the subject's last token, the intervention replaces the residual after that
layer with the residual entering it. Questions with no marked subject use their decision row. This is a forward
intervention, not a gradient, activation norm or input-fit score. Held-out examples do not enter its ranking.

Each layer's score is the mean positive decrease in correct-answer raw logits, averaged over the answer tokens.
The selected layers have at least a tenth of the largest positive score. The budget has a floor of two and a ceiling
of ten layers; the floor fills from the largest remaining scores, including zero scores when necessary. Full pockets
are excluded. Fewer than two available layers refuses the lesson. The key shares are retained, rounded to this budget,
and each selected layer receives at most one new neuron. Input and output fitting then use the existing learner.

Metal copies the saved entering residual over the one output row. CUDA applies the replacement after the next
block's deferred residual sum, renormalizes only that row and repeats its original projection batch shape.
Both read logits before softmax and return without running a backward or optimizer step. No new shader is needed.
The fit stage reports progress for these forwards; per-layer drops and selections appear in the server log.

## Shrink after the acceptance gate

Only after the complete lesson gate and earlier-fact recheck pass does the server try each new neuron once.
It stages a removal by zeroing that neuron's input row and freeing its slot when it is last in the layer. Its output column and every other neuron's weights stay as
they were, avoiding changes to their shared quantization groups. The server runs the same full verifier, including
thinking checks when enabled, then checks every earlier fact again. A failed or interrupted check restores the
original input bits and count. A passing check keeps the removal and updates the lesson's neuron list.

There is no disk write during a trial. The pre-lesson snapshot remains available throughout shrinking, so taking
out the complete lesson still restores its original pocket state. Upload failures use the existing full rollback and
stale-device refusal path. Deletion and persistence use the existing implementations.

## Device comparison

Use the same candidate binary and a fresh Nemotron 3.5 Lightning MLX 4-bit source checkpoint on a Mac and a GB10.
Run one GPU job under the machine's GPU lock. The runner creates independent writable copies for both arms, refuses
hard links and symbolic links, uses an empty graph, strips inherited `TF_SLIDE_*` settings and hashes the original
checkpoint before and after. Allow room for both writable copies. It retains their files for inspection.

```sh
python3 tools/zig/living_locate_ab.py \
  --binary zig-out/bin/tensorfold-native \
  --checkpoint "$MODEL" --checkpoint-revision "$REVISION" \
  --backend metal --output "$RUN" \
  --command '["zig-out/bin/tensorfold-native","serve","{model}","--slide","--slide-graph","{graph}","--api-key-file","{key}","--host","127.0.0.1","--port","{port}","--parallel","1","--prompt-seed","--prompt-cache-gib","0"]'
```

Use `--backend cuda` with the GB10's native CUDA binary. `--backend` labels the receipt; the supplied binary determines
the serving backend. The runner launches that exact binary directly and records its SHA256 and the checkpoint manifest.
Its `legacy` arm restores today's placement and its `locate` arm includes checked shrinking. Both use the same twenty
invented access-word facts, supplied training questions, separate held questions, greedy replies and prompt-keyed seeds.
The lesson writer can still produce different examples across devices; this comparison does not assume otherwise.

`summary.json` reports neurons per fact, accepted lessons, final recall, earlier facts lost, changed unrelated/untold
replies and each lesson's elapsed time, including its checks and save. Each arm retains all lesson events, graph
snapshots and full held-query replies in `results.json`. `original.json` must say `unchanged: true`.
The client's held-query recall uses case-insensitive access-word presence and is an additional measurement; the server's acceptance checks remain authoritative.
Inspect every failure and compare recall and disturbances before interpreting neuron counts or time.
