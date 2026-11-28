# DeepSeek-V4-Pro-Qwen3.5-9B (MLX 4-bit) on one DGX Spark

The native CUDA build serves a locally converted MLX affine 4-bit, group-64 checkpoint of
DeepSeek-V4-Pro-Qwen3.5-9B, standard and Heretic variants, one variant at a time, on a
single DGX Spark (GB10, sm_121). @Ting-Ruei reported this setup in
[#3905](https://github.com/ashhart/TensorFold/issues/3905) on the 1.2.6 release archive:
both variants loaded, answered `/health` and `/v1/models`, and completed short chat
requests with thinking and drafts off. That report covers short text requests at a
configured context of 262,144 tokens and one stream; vision, MTP drafting, throughput and
full-context validation are outside its scope.

Serve a converted checkpoint folder by its path:

```bash
tensorfold-native serve /path/to/DeepSeek-V4-Pro-Qwen3.5-9B-MLX-4bit \
  --parallel 1 --context 262144 --no-thinking --no-drafts
```

Check the serve the way the report did:

```bash
curl -s http://127.0.0.1:8080/health                       # 200
curl -s http://127.0.0.1:8080/v1/models                    # lists the model
curl -s http://127.0.0.1:8080/v1/chat/completions -d '{
  "model": "DeepSeek-V4-Pro-Qwen3.5-9B",
  "messages": [{"role": "user", "content": "Reply with the single word: pong."}]
}'
```

The port and the model id come from the startup line; both variants serve one at a time
with the same flags. A checkpoint whose tensors or config the family cannot read is
refused by name before load, as for every native family.
