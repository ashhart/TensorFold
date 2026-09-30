# Minimal stored-quant fixtures

Q8_0 and Q2_K use16 fixed blocks and four tests: exact decoding against an
independent C oracle and one nonzero payload corruption per format. The
Q2_K layout is scales[16], qs[64], fp16 d, fp16 dmin (offsets0,16,80,82),
verified against the pinned ds4 donor. The former fixtures put d/dmin at16/18;
they were incorrect and have been regenerated.

Oracle provenance and source digest are in manifest.json. Build oracle.c
with `cc -O2 -ffp-contract=off oracle.c -o /tmp/deepseek-quant-oracle`.
Its input is a format byte (1=Q8_0,2=Q2_K), a reserved zero byte, then one
stored block; output is one decimal float32 bit pattern per line. The C
implementation is a test reference only; future CUDA adapters must not
generate their expected values from candidate kernels. Copyright/license
for adapted ggml/ds4 portions is preserved in LICENSE.

Removed duplicated sign/extreme/zero assertions and exhaustive mutation
thresholds that contradicted these fixtures. This CPU pass does not claim
IQ2_XXS, CUDA, full-model quality or Hunyuan coexistence qualification.
