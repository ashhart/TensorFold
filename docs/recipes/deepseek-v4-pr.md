# Add serial DeepSeek V4 mixed GGUF support on CUDA

## Changes

- Register a lazy DeepSeek CUDA family behind existing TensorFold interfaces.
- Reuse pinned MIT ds4 tokenization/CUDA inference through an isolated native ABI.
- Retain packed mapped IQ2_XXS/Q2_K/Q8_0 weights and actual native KV precision.
- Validate real GGUF wire format, embedded tokenizer/EOS and schema.
- Admit explicit context from allocator geometry before weights/KV loading.
- Serve chat, thinking, streaming and DSML tools through shared TensorFold HTTP.
- Build an offline platform wheel in an isolated candidate environment.

## Evidence

See docs/recipes/deepseek-v4-release.md and docs/evidence/deepseek-v4-*.json.
Real named 0731 GGUF at context 262144 passed chat/code/SSE/tools/followup/thinking,
Hunyuan generation during chat and CUDA moderation on one GB10 Spark.
Admission failure/lifetime/API checks are focused; donor math is reused unchanged.
Full 262K-token prompts, exhaustive GPU oracle comparisons and optional features
remain outside this reduced serial release. No proxy to ds4-server is used.

The separate ml-infra launcher provides the local one-command deployment;
its host binding and companion policy do not change TensorFold CLI defaults.
