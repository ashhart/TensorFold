# Cancelled CUDA prefill retention

Historical contributor observations from Sapphire-Rapids' receipt record
14908db5d9ebe0570c3fda20991b889d02490a2b:

An interrupted prefill can retain its latest complete snapshot in the existing
prompt cache. On the tested Qwen checkpoint, one cancellation retained 6,016
tokens; two cancellations retained 10,752. Every completed retry produced the
same 32 output tokens as the uninterrupted, cache-disabled reference.

This is a draft lifecycle-fix receipt. Device coverage is one checkpoint on
GB10; it does not qualify every changed family or establish a performance gate.

Only this host-free prose is retained. The raw receipt artifacts and their
artifact-dependent instructions are omitted. These historical observations
have not been rerun for this integrated carry. Its GPU gates remain unchecked.
