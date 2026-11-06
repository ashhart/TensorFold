# GLM-OCR on HIP

The native HIP engine serves the BF16 `zai-org/GLM-OCR` checkpoint at
revision `2e85a62840ccac27daa451df36c736c4636b8628`, with no Python or
libtorch in the served engine. It was qualified with Zig 0.17.0 on a
Radeon AI PRO R9700 (gfx1201). An idle card is not required; native
admission checks current free memory.

## Build and start

Use a coherent ROCm tree, including its compiler and device bitcode:

```sh
zig build -Dhip=true -Dhip-arch=gfx1201 \
  -Dhipcc=/opt/rocm/core/bin/hipcc \
  -Dhip-device-lib=/opt/rocm/core/lib/llvm/amdgcn/bitcode \
  native-hip -j4 --summary all
```

The engine loads external native libraries at run time. Keep AOTriton
0.11.1 with its adjacent `aotriton.images` directory; its gfx120x images
are required for this GPU. libjpeg-turbo 3.1.4.1 keeps JPEG decoding
equal to the reference. MIOpen keeps its find database in the user
directory below, so set it to persistent storage.

```sh
export HIP_VISIBLE_DEVICES=<ordinal>
export TENSORFOLD_AOTRITON=<aotriton>/lib/libaotriton_v2.so.0.11.1
export TENSORFOLD_LIBTURBOJPEG=<libjpeg-turbo>/lib64/libturbojpeg.so
export MIOPEN_USER_DB_PATH=<miopen-db>
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"

zig-out/native-hip/bin/tensorfold-native serve <GLM-OCR checkpoint> \
  --host 127.0.0.1 --port 18897 --name glm-ocr --vision \
  --context 8192 --parallel 2 --prompt-cache-gib 1 --mtp-drafts 0 \
  --max-tokens 512 --temperature 0 --no-update-check --idle-unload 0
```

For a cache-off control, use a separate process and port with
`--prompt-cache-gib 0`. Leave the shared chat-template defaults in
place. The model returns OCR in `content`; its template's thinking
variable does not make it a reasoning model.

Structured JSON uses the existing optional grammar runtime. Build it
with `zig build grammar -Dxgrammar=<xgrammar source> --prefix <prefix>`
from the official xgrammar v0.2.8 source, and place
`libtfgrammar.so` in `zig-out/native-hip/lib`, beside the executable's
`bin` directory.

## Requests and checks

Send normal OpenAI chat-completion requests to `/v1/chat/completions`.
Use an `image_url` part containing a PNG, JPEG or WebP data URL and a
text part such as `Text Recognition:`, `Formula Recognition:` or
`Table Recognition:`. Greedy decoding and the original image bytes are
the reference settings. The default GLM image budget is 6,144 tokens,
which includes the original bee photo.

`tools/zig/glm_ocr_reference.py` is the one Python reference script.
Its `examples` subcommand builds the pinned six-case model-card
manifest (`examples.json`) with unchanged images, and `capture` runs
official Transformers 5.5.4 on one case to retain the reference
evidence. `processor` rewrites the committed fixtures in
`zig/src/families/glm_ocr/fixtures` without a GPU, and `attention`
writes the SDPA fixtures that `TF_GLM_OCR_ATTENTION_FIXTURES` names
for `zig/tests/hip/aotriton_test.zig`. See each subcommand's `--help`.

## Supported scope

The qualified path is BF16 on gfx1201, with one rank, up to four
independent shared lanes and exact verification windows through 32
rows. Copy drafts are supported; the checkpoint's separate MTP head,
tensor parallelism, fp8 KV, Sliding Weights and YaRN are not qualified
and their dedicated flags are refused. Custom HIP kernel-policy
profiles have not been qualified; use the settings above.

Prompt caching uses the shared core Store. Default retention is at
most 1 GiB after admitted lane/workspace storage, with the shared
snapshot-slot limit. Zero disables retention. A snapshot keeps one
keyed prepared row and the full prompt row count. A compatible resume
keeps that full row shape and skips vision work only for images wholly
inside its saved prefix. A different full prompt row count runs fresh
and logs `GlmOcrPromptShapeChanged`. Cache hits are visible in the
response's `usage.prompt_tokens_details.cached_tokens`.

Arbitrary shorter prompt chunks are not byte-identical to a full pass.
The implementation retains the full prompt arithmetic to preserve
fidelity. It does not claim arbitrary-cut arithmetic or changed-shape
cache reuse. On the qualified GPU, the original six examples, all
generated logits, positive cache resumes, shared lanes and copy drafts
showed zero output differences from the reference. Other hardware and
runtime versions need their own qualification.
