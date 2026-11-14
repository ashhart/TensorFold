# Checkpoint thinking fixtures

These are unmodified checkpoint assets supplied for the GLM-5.3-Flash-NVFP4
revision `da920bb0b9f4` (nvidia/GLM-5.3-Flash-NVFP4, MIT) and LiquidAI/LFM2.5-24B-A2B-MLX-4bit at revision
`667408d494bde191c2cb9dd4f4549827816b9275` (LFM Open License v1.0), and the Qwen3.8 files from
TensorFold/Qwen3.8-27B-MLX-4bit at revision `70ae7fac63274ff2eac54152031433374cb80f2f` (Qwen/Qwen3.8-27B's template,
Apache-2.0); `THIRD_PARTY_NOTICES.md` records all three.
They exercise the native template loader, renderer and thinking-off request path.

The test vocabulary is deliberately small. Its marker IDs are fixture IDs,
and its ASCII text round trips exactly. No checkpoint tokenizer or weights are
included, and these tests do not qualify model output on a device.

| File | SHA256 |
| --- | --- |
| `glm53-tokenizer_config.json` | `98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc` |
| `glm53.jinja` | `34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891` |
| `lfm2-tokenizer_config.json` | `11f1de897317b489dd09199284528382eeb17f566398b16e2039bb2266d26b09` |
| `lfm2.jinja` | `aadd425edb1d42fe9288e5de3624c21c1c03099129b9cfea5978f1bc67222c0a` |
| `qwen38-tokenizer_config.json` | `792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b` |
| `qwen38.jinja` | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
