# gpt-oss host package

This package reads config and tensor metadata for `openai/gpt-oss-20b` revision
`6cee5e81ee83917806bbde320786a8fb61efebee` (Apache-2.0). Its fixtures preserve all
459 supplied tensor names, dtypes, shapes and byte offsets; `fixtures/SOURCE.json`
records the original and whitespace-normalized header digests.

`config` retains explicit `layer_types` and YaRN fields. `names` checks complete
coverage and the mixed BF16/U8 projection geometry. `core/mxfp4.zig` reads standard
32-value E2M1/UE8M0 blocks without expanding a checkpoint or changing its bytes.
Nibble order is a required argument because shapes alone cannot establish it.

`attention` supplies FP64 causal/sliding attention primitives under a documented
denominator-only sink and current-token-inclusive window convention. Its inputs
are already rotated query/key rows; YaRN is metadata only. `route.select` chooses
the configured count from router logits with a documented low-ID tie convention;
`softmaxSelected` is a separate explicit reference operation, not an inferred
checkpoint normalization policy. These conventions still need model-specific
confirmation before an execution path is added. The gate/up row order and exact
clamped activation formula also remain unresolved by the supplied files.

Harmony parsing is the existing server `reply_text.parseHarmony` function, shared
by alias. No second splitter, tokenizer rewrite, GPU code, CPU generation fallback,
native registration, scheduler or serving capability is included.

```sh
zig build -Dfixed-cpu test-gpt-oss-host test-gpt-oss-fast
```

The tests cover real headers, named refusals, analytic sink/window/routing cases,
all finite MXFP4 code/scale values against independent FP64 arithmetic and explicit
wrong-order input. They do not establish checkpoint output tokens or device speed.
