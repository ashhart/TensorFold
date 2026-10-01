# Image input

The opt-in `--vision` flag accepts image and text content parts through the existing OpenAI-compatible chat API. It supports GLM-5.3-Flash on MLX and Qwen3.5/3.8 dense checkpoints on MLX and CUDA. Image features enter the existing model's prompt prefill; generated text still uses that family's normal decoder and speculative path.
The checkpoint must contain its vision tower, tokenizer, processor files and vision configuration; text-only conversions cannot recover image support from a flag. GLM-5.3-Flash uses its own GLM5-Next image processor and tower while sharing TensorFold's already-loaded language model and MTP head.
Video, audio and image generation are not supported by this adapter.

## Start a server

Install the optional image dependencies (from a checkout, `python -m pip install '.[vision]'`):

```bash
python -m pip install 'tensorfold[vision] @ git+https://github.com/ashhart/TensorFold.git'
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --vision
tensorfold serve Vontra/GLM-5.3-Flash-MLX-4bit-MTP --vision
```

GLM-5.3-Flash image input is currently MLX-only. CUDA uses the same flag with `--backend cuda` for supported Qwen checkpoints; their vision tower must use floating-point weights.
MLX also reads per-module quantized tower weights when the checkpoint declares their format.
The tower shares the server process and the existing language model's embeddings; it does not load a second language model.
CUDA two-rank mode encodes images on rank zero and sends their features and positions to rank one.
Use the model and drafter prerequisites from the [Qwen recipe](recipes/qwen3.8-27b.md) or [GLM recipe](recipes/glm-5.3-flash.md).
GLM derivatives may retain selected BF16 attention output projections, including the MTP layer; these use the existing dense projection path alongside the quantized weights.

## Send an image

Use OpenAI `image_url` content parts in user messages, interleaved with text in the intended order:

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

image = base64.b64encode(Path('photo.png').read_bytes()).decode()
body = {
    'model': 'Qwen3.8-27B-MLX-4bit',
    'messages': [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'Describe this image.'},
            {'type': 'image_url', 'image_url': {
                'url': 'data:image/png;base64,' + image,
                'detail': 'auto',
            }},
        ],
    }],
    'max_tokens': 128,
    'stream': False,
}
request = Request('http://127.0.0.1:8080/v1/chat/completions',
                  data=json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json'})
with urlopen(request) as response:
    print(json.load(response)['choices'][0]['message']['content'])
```

`GET /v1/models` gives the exact model ID for the running server.
Remote image URLs are off by default, so a server other machines can reach never fetches URLs on a client's behalf.
Start the server with `--vision-urls` to accept public HTTPS URLs on port 443 that serve `image/jpeg`, `image/png` or `image/webp`; plain HTTP, other ports, private, loopback, link-local and metadata addresses, file URLs and redirects to any of them are still refused.
For a local image, send a data URL as above.
`stream: true` uses the usual text completion stream.
Image output is not generated.

## Limits and state

Requests accept up to four JPEG, PNG or WebP images, 10 MiB encoded per image and 20 MiB total, within a 32 MiB HTTP body.
Decoded images are bounded to 8,192 pixels per dimension, 16 million pixels per image and 32 million total.
EXIF orientation is applied and transparency is composited onto white; animated and multipage inputs are refused.
Up to 16 requests decode and process images at once; more wait up to a minute, and past 128 waiting the server answers 503 so the client retries.
The request log (`TENSORFOLD_REQUEST_LOG`) records image parts as `<redacted>`.
The model processor bounds the total expanded image tokens to 4,096, with a smaller budget for `detail: low`.
Those expanded tokens count toward prompt usage and the context window before model execution.
The available memory budget may impose a smaller practical image or context limit.

Image requests currently start with a fresh KV cache and do not write reusable prompt checkpoints.
This prevents identical image-placeholder token IDs from reusing another image's state; ordinary text requests retain their prefix caching.
Multi-turn image conversations work when the request includes the original image content parts, but image-prefix reuse and persisted image KV are not implemented.
For Qwen, each image request carries its own multimodal rotary positions and continuation offset, including during concurrent lane rounds. GLM uses its native KDA/NoPE attention state.

## Verification

Compare image requests with `draft: true` and `draft: false` at identical sampling settings and seed, then compare concurrent requests with their solo results.
Tests cover input validation, bounded fetching, expanded prompt accounting, cache isolation, memory admission, rotary metadata and distributed transport contracts.
Hardware qualification is separate from these tests: each backend and chip needs real image understanding, drafted/serial equality, concurrency, chunked-prefill and memory checks before a release claim.
Checkpoint metadata must describe the decoder separately from MTP: the `mlp_layer_types` list must match `num_hidden_layers` for Transformers validation. Preserve the separate MTP configuration and weights.
Image input stays experimental until the hardware matrix is complete, and makes no throughput claim.
