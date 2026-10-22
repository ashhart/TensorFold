# Image input

This page describes image input in the Python engine, frozen at 0.6.6, until [The native server](#the-native-server),
which covers the native binary: it reads images for GLM-5.3-Flash, Qwen3.8-27B and Qwen3.8 Flash Next on a Mac and
refuses `--vision` for other models. Install the Python engine from the
[`v0.6.6` tag](https://github.com/ashhart/TensorFold/tree/v0.6.6) for the rest.
Parts of it came from pull requests: GLM-5.3-Flash images by @mgoldwasser
([#101](https://github.com/ashhart/TensorFold/pull/101)), Flash Next images on CUDA by @shantanugoel
([#146](https://github.com/ashhart/TensorFold/pull/146)) from @MiaAI-Lab's patch, video by @MiaAI-Lab
([#240](https://github.com/ashhart/TensorFold/pull/240)), `--vision-max-images` by @edurdias
([#170](https://github.com/ashhart/TensorFold/pull/170)), `--vision-offload` by @barelyworkingcode
([#187](https://github.com/ashhart/TensorFold/pull/187)), `--vision-image-tokens` by @MiaAI-Lab
([#239](https://github.com/ashhart/TensorFold/pull/239)), and HTTPS image URLs and the redacted request log by @di37
([#64](https://github.com/ashhart/TensorFold/pull/64)). The [credits](credits.md) list them all.

The opt-in `--vision` flag accepts image and text content parts through the existing OpenAI-compatible chat API. It supports GLM-5.3-Flash on MLX and Qwen3.5/3.8 dense checkpoints on MLX and CUDA. Image features enter the existing model's prompt prefill; generated text still uses that family's normal decoder and speculative path.
The checkpoint must contain its vision tower, tokenizer, processor files and vision configuration; text-only conversions cannot recover image support from a flag. GLM-5.3-Flash uses its own GLM5-Next image processor and tower while sharing TensorFold's already-loaded language model and MTP head.
Video, audio and image generation are not supported by this adapter.
Qwen3.8 Flash Next supports images on one CUDA GPU with `--parallel` of at least two; see the
[Flash Next image recipe](recipes/flash-next-vision.md), including offline reconstruction of an EXL3 vision sidecar.

## Start a server

Install the Python engine with its optional image dependencies (from a checkout of the `v0.6.6` tag,
`python -m pip install '.[vision]'`):

```bash
python -m pip install 'tensorfold[vision] @ git+https://github.com/ashhart/TensorFold.git@v0.6.6'
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
4,096; 13,312 for Nemotron 3.5 Super VL) for engines that read images: GLM-5.3-Flash, Qwen3.8-27B, Qwen3.8 Flash Next, Gemma 4
[31B](recipes/gemma-4-native.md#images) and [12B](recipes/gemma-4-12b-native.md#images) and Step-5-Preview on one
Mac, and Nemotron 3.5 Super VL on CUDA. Other native engines refuse the flag at startup.
The CLI accepts explicit image budgets from 16 to 262,144 tokens; each engine still applies its own limits and memory admission.

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
- PNG and JPEG are decoded to the pixels 0.6.6 decoded with Pillow 12.3 (`zig/src/core/picture/`, which d1-3B's
  pictures on CUDA use too): PNG from the PNG specification, JPEG by libjpeg-turbo 3.1.4.1's TurboJPEG library
  (`libturbojpeg`, the release Pillow 12.3 bundles) at Pillow's settings. TensorFold does not ship that library; the
  server checks it at startup with `--vision`, or when running `capabilities --json`, from the file
  `TENSORFOLD_LIBTURBOJPEG` names, else `lib/` beside the server's
  folder, beside the server, Homebrew's `jpeg-turbo` on macOS, then the system's library path. It checks the version
  libjpeg-turbo writes into the library and refuses JPEG images by name when the library is missing or another
  release; PNG images still serve. Homebrew is a search location, not a guarantee of the pinned version: a newer
  `jpeg-turbo` package is refused too. Use a separately built libjpeg-turbo 3.1.4.1 library with the override or
  beside the server; updating the package does not satisfy this exact pin.
- WebP is decoded by libwebp 1.6.0's decoder library (`libwebpdecoder`, the release Pillow 13.0.0 bundles), to the
  RGB bytes Pillow 13.0.0 gives the same file (lossy and lossless, with and without alpha, EXIF-oriented:
  `zig/tests/fixtures/picture/webp.json`, from `tools/zig/picture_webp_fixtures.py`). TensorFold does not ship it
  either; the server checks it at startup with `--vision`, or when running `capabilities --json`, from the file
  `TENSORFOLD_LIBWEBP` names, else `lib/` beside the
  server's folder, beside the server, Homebrew's `webp` on macOS, then the system's library path, and refuses WebP
  images by name when it is missing or another release (`WebPGetDecoderVersion`). Animated WebP is refused as
  animated pictures are. Other formats are refused.
- Startup warns about unusable JPEG and WebP decoders without refusing PNG or text requests. In
  `capabilities --json`, optional libraries' `found` field means the loader opened a candidate. JPEG and WebP also
  report `usable`, checked against the decoder's required exports and exact version, and a `problem` when refused.
  For other libraries, `usable` and `problem` are null and further checks remain unchecked. No pixels are decoded
  by these diagnostics. Restart after changing a codec library, because each process caches its decoder check.
- The limits and the processing are 0.6.6's: EXIF and XMP orientation, alpha composited over white, the model's
  processor's sizing (GLM5-Next's, or Gemma 4's PIL processor) and Pillow's bicubic resampler. The decoder gives 0.6.6's pixels for every picture in
  `zig/tests/fixtures/lfm2/images.json` and `zig/tests/fixtures/picture/edges.json`, except where it departs on
  purpose: a CMYK JPEG is refused, and a JPEG holding several pictures (MPO) gives its first, which 0.6.6 refused.
- Images are counted from the request's image parts. A user or tool message whose text spells the model's image
  markers (GLM's `<|image|>`, `<|begin_of_image|>`, `<|end_of_image|>`; Qwen's `<|vision_start|>`, `<|image_pad|>`,
  `<|vision_end|>`) is tokenized as ordinary text, so only the chat template's own markers, one wrapper for each image
  part, become image tokens.
- The prompt cache keys each image's placeholder rows by the image's content, so a kept state resumes the same image
  and never another one under the same placeholders. Image conversations reuse their history like text ones.

### Qwen3.8-27B

With `--vision`, the 27B loads its checkpoint's image tower (the `vision_tower.*` tensors of
`TensorFold/Qwen3.8-27B-MLX-4bit`, kept in bf16) beside the language model. Without the flag nothing of the tower
loads. An image is prepared as transformers' Qwen2-VL processor prepares it: decoded as above, sized by `smart_resize`
to multiples of 32 pixels between the processor's 65,536 pixels and the image's share of `--vision-image-tokens`
(split evenly over the request's images, 1,024 pixels a token, at most 16,777,216 pixels), resized with Pillow's
bicubic, and cut into 16-pixel patches. The tower runs once an image as its request's prompt pass starts, on the GPU
in bf16 with fp32 sums. Its 2 x 2 merged rows replace the `<|image_pad|>` placeholders' embeddings. The text's
attention layers turn each image row by its row and column (M-RoPE), and the rows after an image carry the shift those
positions leave, through decoding, drafts and later turns. Text-only requests get the tokens they got without the
flag. Video parts are refused with HTTP 400.

The tower, its scratch and a request's image rows take 1.79 GiB at the default 4,096 image tokens, and the server
counts them against the 70% load limit and the prompt cache's budget. On an M3 Ultra, six test pictures got the
processor's grids, token counts and patches bit for bit, and the tower's rows came closer to transformers' float32
tower than its bf16 run does (mean row error 1.0 to 2.2% against 1.6 to 4.1%). A 1,120-token photo takes 0.6 s in the
tower there. With an image, drafted replies equal `"draft": false`, four requests at once equal each alone, a later
turn resumes past its image and equals a fresh server's reply, and DFlash2 keeps 2.6 to 3.0 tokens a round where the
same asks without the image get 2.8 to 3.4.

### Qwen3.8 Flash Next

With `--vision`, Flash Next loads its checkpoint's image tower beside the language model, one stream or several
(`--parallel`). It is the 27B's tower (the same 333 bf16 `vision_tower.*` tensors and processor) with 2,560-wide output,
run by the same code (`zig/src/core/qwen_vl/`); images are prepared as above. Its rows replace the `<|image_pad|>`
embeddings in all four of the model's hyper-connection streams, and the prompt pass turns them by interleaved M-RoPE
(sections 11, 11 and 10 over the first 64 dimensions). The positions are transformers' `get_rope_index` shifted so that
the text after the last image keeps its own slot: decoding, drafts and later tokens run as for text. Text-only requests
get the tokens they got without the flag.

- **Prompt cache:** image and video conversations resume past their images. The shift moves every position before the
  request's last image, so a kept state is named by the shift as well as the tokens and images: a later turn with the
  same images resumes, and a turn that adds an image reads its prompt again. The tower skips the images a resumed state
  already holds. Replies equal the cache off's in multi-turn image and video chats, on one stream and shared sessions.
- **Memory:** the tower and its scratch take 1.73 GiB at the default 4,096 image tokens, counted against the load limit;
  a request's rows take 20 KiB a visual token while its prompt pass runs.
- **Checks on an M5 Ultra:** five test pictures got the grids and patches of transformers' Pillow processor bit for bit;
  the tower's rows came closer to transformers' float32 `Qwen4ExpVisionModel` than its bf16 run does (mean row error 1.4
  to 3.1% against 1.8 to 5.7%). Six fixed image prompts gave drafted replies equal to `"draft": false`, the same tokens
  on one stream and on shared sessions, and the expected answers. A 441-token picture takes 53 ms in the tower.
- **Video:** `video_url` parts as described below, in the Qwen3-VL layout: each pair of frames is one image, and its
  block is `<X.Y seconds><|vision_start|>` and its `<|video_pad|>` rows then `<|vision_end|>`, inside the template's own
  `<|vision_start|>…<|vision_end|>`, as transformers 5.17's `Qwen3VLProcessor` writes it. Frames are sampled at 2 a
  second (at least 4; at most 256, where the processor allows 768), the clip's pixels kept between 65,536 and 16,777,216
  and inside the request's `--vision-video-tokens` share, and resized with torch's antialiased uint8 bicubic. On three
  test clips the served prompt was as long as the processor's (239, 387 and 8,215 tokens); a 30-second 720p clip takes
  3.2 s to its first token.

### Video on the native server

With `--vision`, GLM-5.3-Flash also reads `video_url` parts in user messages (Flash Next too, in its own layout above),
as transformers 5.19's `Glm5NextProcessor` prepares them:
- **Sampling:** 2 frames a second, deduplicated and padded to an even count, at most 256 frames.
- **Frame pairs:** each pair of frames is one image to the tower.
- **Prompt:** the template's `<|begin_of_video|><|video|><|end_of_video|>` holds one
  `<|begin_of_image|>…<|end_of_image|>T.T seconds` block a pair.
- **Budget:** a request's videos share `--vision-video-tokens` visual tokens (default 16,384; the processor's
  `max_image_tokens`). The frames are sized under it and resized with torch's antialiased uint8 bicubic.

Video needs `libtfvideo`, an optional library over FFmpeg 9.0.2 (the release PyAV 19.0.1 ships). The server loads it
when the first video arrives and refuses videos by name without it:

```sh
tools/zig/build_ffmpeg.sh ~/ffmpeg-9.0.2        # static FFmpeg: H.264, HEVC, VP8, VP9, MPEG-4; MP4/MOV, MKV/WebM
zig build video -Dffmpeg=$HOME/ffmpeg-9.0.2     # zig-out/native/lib/libtfvideo.dylib, beside the server
```

- **Limits (0.6.6's):** 2 videos a request, 16 MB each and 20 MB together, an hour of footage, 8192 pixels a side.
  Data URLs work, and with `--vision-urls` public HTTPS URLs too (MP4, WebM, QuickTime or Matroska; 20 s a download,
  120 s a request).
- **Too short:** a clip too short for one sampled frame (under half a second) is refused, as the processor samples
  nothing.
- **Frame counts:** a container without one (Matroska, WebM) is counted by its video packets. transformers' PyAV loader
  can't read these at all.
- **Checks:** `tools/zig/glm_video_fixtures.py` checks the patches bit for bit against the processor on frames PyAV
  decodes, over 13 clips. `tools/zig/video_served.py` checks a served model's prompt ids against the processor's.

### Nemotron 3.5 Super VL on CUDA

With `--vision`, Nemotron 3.5 Super VL reads images through the checkpoint's own RADIO tower and projector, run in
TensorFold's CUDA kernels (`zig/src/families/radio/`).

- **Processor:** the checkpoint's. Each image becomes one tile whose patch grid (16 pixels a patch, an even number each
  way) holds up to 13,312 patches and at least 1,024, except that a picture smaller than that sent with
  `"detail": "low"` keeps its own grid, down to 2 x 2 patches and one token. It is resized with torch's antialiased
  bicubic on float samples and normalized with CLIP's mean and deviation. The pixels equal the checkpoint processor's
  bit for bit. An image takes a quarter of its patches as tokens (at most 3,328), written between `<img>` and `</img>`.
- **Budget:** a request's images share `--vision-image-tokens`, 13,312 by default for Super VL, so up to four images
  keep the size NVIDIA's processor gives them; a smaller budget makes them smaller. Four full-size images on one GB10
  took 13,124 prompt tokens and 11.45 s to the first token.
- **Tower:** the checkpoint's bf16 weights as stored, bf16 activations and fp32 sums. Against a float32 run of the
  checkpoint's own modules on the same pixels, its rows' error equals the modules' own bf16 run's (1.2% to 1.9% a row,
  relative, on six pictures). The tower and projector take about 2.1 GiB of GPU memory beside the language model, and
  a 13,000-patch image about 0.9 s on a GB10. On two GB10s the first runs the tower; the second holds only its share
  of the experts.
- **Not read yet:** video and audio. The checkpoint's video embedder and sound encoder stay unloaded, and video and
  audio parts are refused.
- **Checks:** `tf-cuda-test radio-processor` compares the decoded pictures, the processor's pixels, its resizes and
  3,000 patch grids with the checkpoint's processor; `tf-cuda-test radio` compares the tower's and projector's rows
  with a float32 reference.

## Verification

Compare image requests with `draft: true` and `draft: false` at identical sampling settings and seed, then compare concurrent requests with their solo results.
Tests cover input validation, bounded fetching, expanded prompt accounting, cache isolation, memory admission, rotary metadata and distributed transport contracts.
Hardware qualification is separate from these tests: each backend and chip needs real image understanding, drafted/serial equality, concurrency, chunked-prefill and memory checks before a release claim.
Checkpoint metadata must describe the decoder separately from MTP: the `mlp_layer_types` list must match `num_hidden_layers` for Transformers validation. Preserve the separate MTP configuration and weights.
Image input stays experimental until the hardware matrix is complete, and makes no throughput claim.

## Native CUDA: Qwen3.5/3.8

`tensorfold-native serve <checkpoint> --vision` loads the checkpoint's vision tower beside the language model on the
Qwen CUDA engine (`families/qwen3_5/cuda_vision.zig`, kernels in `qwen_vision.cu`, its Linears on cuBLASLt). The
server prepares each image as it does on a Mac: core/picture decodes it and core/qwen_vl resizes and cuts it with the
checkpoint's processor settings, so an image's patches, grid and cache identity are the same on both backends. The
tower and the scratch of one 16,384-patch call are allocated at startup and counted as model memory, so the stream
budget never has to make room for an image. A request's images go through the tower before its prompt; their rows
replace the `<|image_pad|>` placeholders in the prompt chunks, which rotate at M-RoPE's t, h and w positions
(interleaved sections 11, 11 and 10), and the reply's rows rotate at their position plus the layout's shift, as
transformers' `get_rope_index` lays them out. Text-only requests keep their bits. One image holds at most 4,096
tokens (one tower call); `--vision-image-tokens` bounds a request's images together. JPEG needs libturbojpeg on the
machine (PNG decodes everywhere); video parts are refused.
