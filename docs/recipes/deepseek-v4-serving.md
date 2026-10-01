# Embedded DeepSeek serving

The DeepSeek CUDA family lazily selects `DeepSeekApp` through the existing
`CUDA_APP` family hook. The app uses TensorFold's shared HTTP server, request
queue, sampling controls, context errors, stop strings, cancellation, health
counters and tool-call streaming. Family discovery imports neither MLX nor
torch. Generic CUDA families retain their tokenizer/Jinja behavior.

## Adapter boundaries

- `NativeTokenizer` implements only the `encode(...).ids`, `decode(...)` and
  `token_to_id(...)` APIs shared serving uses. Encoding calls the pinned native
  rendered-chat tokenizer; no tokenizer download or replacement BPE exists.
  Decoding joins native token bytes before UTF8 conversion. Bounded caches
  avoid repeated token-text RPCs, including during streamed multibyte text.
- `DeepSeekTemplate` uses the existing `DeepSeekTokenizer` prompt wrapper and
  vendored DeepSeek encoder. It normalizes OpenAI messages and preserves the
  encoder's JSON-string tool arguments and generation-prefix conventions.
  Thinking switches and effort annotations use the existing family mapping.
- DSML replies use the existing shared DSML parser and envelope hiding, for
  both streamed and nonstreamed calls. Required/named tools reuse `CallGate`:
  the first native token starts its envelope, and the remaining DSML/invoke
  prefix is the forced lead. The full envelope is not assumed to be one token.
- Native engine closure is called on HTTP shutdown and bind failure. Client
  cancellation uses the shared callback boundary and leaves normal subsequent
  requests available. Unsupported structured output and media remain refused
  through shared request validation.

These are family adapters, not a second HTTP server or an HTTP proxy to ds4.
The native engine remains responsible for tokenizer and inference arithmetic;
TensorFold remains responsible for serving and shared sampling.

## Verification and remaining qualification

Four focused adapter tests cover golden prompt rendering, byte-safe streamed
decoding, actual shared HTTP routes with a mocked native session, ordinary chat,
thinking, required single-tool streaming, tool-result followup, context errors,
stop/disconnect and native closure. Affected existing prompt/admission/error/
disconnect/health/discovery checks also passed (290 checks before the fourth
cleanup test was added; that test and the three adapter tests passed afterward).

No second real model was loaded. Mocked HTTP behavior is not proof of actual
GGUF token-ID identity, model quality or Hunyuan coexistence. Capacity admission
must be completed before native loading; device and real-model qualification
remain board milestones. S07 must rebuild the final committed wheel to include
this adapter rather than reuse the earlier S05 smoke wheel.

For the final ml-infra recipe, use `make tensorfold-deepseek-3d`. ml-infra owns
binding (`TF_DEEPSEEK_HOST`, default `0.0.0.0` for Tailscale access), while
development qualification uses loopback and local readiness probes.
