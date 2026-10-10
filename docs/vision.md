# Image input

The opt-in `--vision` flag accepts image and text content parts through the existing OpenAI-compatible chat API. It supports GLM-5.3-Flash on MLX and Qwen3.5/3.8 dense checkpoints on MLX and CUDA. Image features enter the existing model's prompt prefill; generated text still uses that family's normal decoder and speculative path.
The checkpoint must contain its vision tower, tokenizer, processor files and vision configuration; text-only conversions cannot recover image support from a flag. GLM-5.3-Flash uses its own GLM5-Next image processor and tower while sharing TensorFold's already-loaded language model and MTP head.
Video, audio and image generation are not supported by this adapter.
Qwen3.8 Flash Next supports images on one CUDA GPU with `--parallel` of at least two; see the
[Flash Next image recipe](recipes/flash-next-vision.md), including offline reconstruction of an EXL3 vision sidecar.

## Start a server

Install the optional image dependencies (from a checkout, `python -m pip install '.[vision]'`):

```bash
python -m pip install 'tensorfold[vision] @ git+https://github.com/ashhart/TensorFold.git'
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --vision
tensorfold serve TensorFold/GLM-5.3-Flash-MLX-4bit-MTP --vision
```

GLM-5.3-Flash image input is currently MLX-only. CUDA uses the same flag with `--backend cuda` for supported Qwen checkpoints; their vision tower must use floating-point weights.

On a small CUDA card the resident tower and its 4 GiB workspace reserve take a large part of the startup budget. `--vision-offload` (with `--vision`, CUDA only) keeps the tower in host RAM, copies it to the GPU only while an image is encoded, and reserves 2.25 GiB instead. On one RTX 4090 with the Qwen3.8-27B EXL3 3.50bpw pack and DFlash2, the largest window with `--vision` went from 11,922 to 38,686 tokens; a 4,096-token image peaked about 1.3 GB above idle. Each image request pays the copy of the roughly 0.9 GiB tower to the GPU and back.
MLX also reads per-module quantized tower weights when the checkpoint declares their format.
The tower shares the server process and the existing language model's embeddings; it does not load a second language model.
Dense Qwen CUDA two-rank mode encodes images on rank zero and sends their features and positions to rank one.
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
Tool results (`role: "tool"`) take `image_url` parts the same way, as agents send screenshots; the model's chat
template renders them inside the tool response.
Remote image URLs are off by default, so a server other machines can reach never fetches URLs on a client's behalf.
Start the server with `--vision-urls` to accept public HTTPS URLs on port 443 that serve `image/jpeg`, `image/png` or `image/webp`; plain HTTP, other ports, private, loopback, link-local and metadata addresses, file URLs and redirects to any of them are still refused.
For a local image, send a data URL as above.
`stream: true` uses the usual text completion stream.
Image output is not generated.

## Send a video

Flash Next on CUDA (`--vision`, `--parallel` of at least two) also takes `video_url` parts in user messages, as
MP4, WebM, MOV or MKV data URLs (or public HTTPS URLs with `--vision-urls`); decoding needs PyAV (`pip install av`,
part of the `vision` extra on Linux):

```json
{"role": "user", "content": [
  {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}},
  {"type": "text", "text": "What happens in this clip?"}]}
```

Frames are sampled as Qwen3-VL's processor does: two a second spread over the whole video, at least 4 and at most
256, each pair of frames one block that the prompt prefixes with its time (`<1.5 seconds>`). Frames are sized so a
video takes at most 768 tokens a block and 16,384 in all (`TENSORFOLD_VIDEO_TOKENS`). The tower encodes a run of
blocks of at most 16,384 patches at a time, the scratch one full-size image needs, and blocks never attend to one
another. A request takes up to two videos of 16 MiB each, 20 MiB in all, so the base64 still fits the 32 MiB
request body, and up to an hour of footage. Videos and images can share a message; the image limits count images
only.

## Limits and state

Requests accept up to four JPEG, PNG or WebP images by default. `--vision-max-images N` sets a positive
image-count limit when serving with `--vision`, on both backends:

```bash
tensorfold serve TensorFold/GLM-5.3-Flash-MLX-4bit-MTP --vision --vision-max-images 8
```

The count includes **all images in the submitted message history**, including images from earlier turns
and tool results, in tool messages or sent as user image parts. Reading images one at a time can therefore
reach the limit. Once that history exceeds it, even a text-only follow-up is refused if the client resends
the images. Remove older image content from the submitted history, start a new conversation, or restart
the server with a larger count limit. The server does not discard images automatically.

A request's images share 4,096 visual tokens, so a higher count makes each image smaller: eight images get about
512 tokens each. On CUDA Qwen checkpoints (Qwen3.5/3.8 dense and Flash Next), `--vision-image-tokens N` raises
that shared budget, up to 65,536, while each image keeps at most 4,096, so one image is sized as before. The tower
encodes runs of whole images of at most 16,384 patches, the scratch one full-size image already needs; a request
that fits one run is encoded in one call, as before. The byte and pixel limits still apply, and the longer prompt
counts against the context window.

```bash
tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP --parallel 2 --vision \
  --vision-max-images 50 --vision-image-tokens 16384
```

Changing the count does not change the other limits: 10 MiB encoded per image and 20 MiB total, within a 32 MiB HTTP body.
Decoded images are bounded to 8,192 pixels per dimension, 16 million pixels per image and 32 million total.
EXIF orientation is applied and transparency is composited onto white; animated and multipage inputs are refused.
Up to 16 requests decode and process images at once; more wait up to a minute, and past 128 waiting the server answers 503 so the client retries.
The request log (`TENSORFOLD_REQUEST_LOG`) records image parts as `<redacted>`.
The model processor bounds the total expanded image tokens to 4,096, with a smaller budget for `detail: low`.
That budget is shared across the images; a higher count can reduce the detail available for each image.
Those expanded tokens count toward prompt usage and the context window before model execution.
The available memory budget may impose a smaller practical image or context limit.

Image requests currently start with a fresh KV cache and do not write reusable prompt checkpoints.
This prevents identical image-placeholder token IDs from reusing another image's state; ordinary text requests retain their prefix caching.
Multi-turn image conversations work when the request includes the original image content parts, but image-prefix reuse and persisted image KV are not implemented.
For Qwen, each image request carries its own multimodal rotary positions and continuation offset, including during concurrent lane rounds. GLM uses its native KDA/NoPE attention state.

## The native server

`tensorfold-native serve` takes `--vision`, `--vision-max-images` (default 4) and `--vision-image-tokens` (default
4,096) for engines that read images; GLM-5.3-Flash on one Mac reads them so far, and other native engines refuse the
flag at startup.

- Images arrive as `image_url` data URLs in user messages and tool results. With `--vision-urls` (it needs
  `--vision`), a part may name a public HTTPS URL instead, fetched as 0.6.6 fetched it: port 443 without credentials
  or a fragment; every address the host resolves to public (no private, loopback, link-local, carrier-grade NAT,
  documentation, multicast or metadata address, and for IPv6 global unicast only, without 6to4 or Teredo); TLS
  verified for the host against the system's certificates, on the address that was checked; up to 3 redirects,
  each checked again; a declared `image/jpeg`, `image/png` or `image/webp`, uncompressed; 10 MB an image and 20 MB
  a request across data URLs and downloads; 10 s a download and 30 s a request's downloads. The refusals are
  0.6.6's words; a TLS failure, like any other failed connection, is "image download failed or timed out". Host
  names that aren't ASCII and IPv6 address literals are refused. Without the flag, an HTTPS URL is refused with the
  flag's name.
- PNG and JPEG are decoded by the platform's decoder (ImageIO on macOS). WebP is refused by name until a decoder
  exists; other formats are refused.
- The limits and the processing are 0.6.6's: EXIF orientation, alpha composited over white, the GLM5-Next processor's
  sizing and PIL's bicubic resampler. Six drawn PNG fixtures give patches bit-equal to 0.6.6 with mlx-vlm
  (`tools/zig/glm_vision_fixtures.py`). JPEG patches differ slightly, because ImageIO and libjpeg-turbo decode
  differently.
- Images are counted from the request's image parts. A user or tool message whose text spells the model's image
  markers (`<|image|>`, `<|begin_of_image|>`, `<|end_of_image|>`) is tokenized as ordinary text, so only the chat
  template's own markers, one wrapper for each image part, become image tokens.
- The prompt cache keys each image's placeholder rows by the image's content, so a kept state resumes the same image
  and never another one under the same placeholders. Image conversations reuse their history like text ones.

## Verification

Compare image requests with `draft: true` and `draft: false` at identical sampling settings and seed, then compare concurrent requests with their solo results.
Tests cover input validation, bounded fetching, expanded prompt accounting, cache isolation, memory admission, rotary metadata and distributed transport contracts.
Hardware qualification is separate from these tests: each backend and chip needs real image understanding, drafted/serial equality, concurrency, chunked-prefill and memory checks before a release claim.
Checkpoint metadata must describe the decoder separately from MTP: the `mlp_layer_types` list must match `num_hidden_layers` for Transformers validation. Preserve the separate MTP configuration and weights.
Image input stays experimental until the hardware matrix is complete, and makes no throughput claim.
