# Kimi K3 and capped K3.1 on native Metal

This is a resident adapter bring-up for synthetic components and a cut-down graph.
No real weights, no model-forward reference, no full-model run.
The complete released checkpoint does not run on one Mac through this adapter: admission prices every expert,
prepared projection, stream state, prompt cache and reserve before opening its weights, and refuses by budget.
There is no SSD residency path.

Build on a Mac with the Metal toolchain:

```sh
zig build native tf-k3-serve-check tf-k3-text-golden -Dcpu=apple_m1
zig-out/bin/tf-k3-serve-check
```

The native serving entry for an explicitly supplied local cut-down checkpoint is:

```sh
zig-out/native/bin/tensorfold-native serve LOCAL_CHECKPOINT --parallel 1 --context 2047 --prompt-cache-gib 0
```

This command is not a claim that the released checkpoint fits.
The directory needs its matching config, safetensors index and shards, `tiktoken.model`, and `tokenizer_config.json`.
Stored MXFP4 experts and BF16 projections retain the existing K3 geometry; synthetic fixtures may reduce layer,
expert and vocabulary counts.
K3.1 validates the released DSA config and refuses over 2047 total evaluated tokens, including restored prefix,
reply and speculative rows.
Its all-visible equivalence is inferred from our GLM implementation, not confirmed by a K3.1 model reference.
Prompt-state restore separately checks `max_ctx` before copying cache or logits.

The native text path uses release XTML segments and masked tiktoken encoding.
Thinking efforts are low, medium, high and max; thinking defaults to max effort when enabled.
Reasoning and complete calls to offered tools are read from actual added-token controls.
Ordinary spellings remain text, and raw completions retain generated XTML as text.
Images, external draft heads, TP, learning, disk states, required tool grammar, structured-output enforcement,
thinking budgets and answer-only limits are unsupported.

The K3.1 release license's Model-as-a-Service condition requires a separate Moonshot agreement for commercial use by operators whose aggregate affiliate revenue exceeds US$20 million over any consecutive 12 months, subject to its Section 4 exceptions; see [the supplied license](../../LICENSES/Kimi-K3.1-Pro.txt).

The prompt/value port and golden fixtures are credited in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
For an already available local tokenizer with the pinned release assets, run
`zig-out/bin/tf-k3-text-golden LOCAL_TOKENIZER_DIRECTORY` to compare all 15 release prompt ID arrays.
Synthetic tokenization checks do not replace that real-asset check or real-model qualification.
