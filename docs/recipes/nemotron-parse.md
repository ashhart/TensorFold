# Nemotron 3.5 Parse on native HIP

This port runs `nvidia/Nemotron-3.5-Parse` on one HIP rank. The
validated checkpoint revision is
`4035ffb1d8f8bafcdb867dfaf6f7514f101464f9`. The family is
`nemotron_parse`; it is separate from Nemotron-H and Super VL.

The original FP32 safetensors stay unchanged. The loader validates all
510 tensors, then owns BF16 device copies of the 509 floating tensors.
Activations and KV use BF16 with FP32 sums. The image tower, horizontal
neck, causal mBART decoder and trained one-step proposal run in native HIP
kernels. Python is used only for independent development references.

## Build and serve

Use the repository Zig version. On ROCm installations with a different
layout, give the actual compiler and device-library paths:

```sh
zig build native-hip -Dhip=true -Dhip-arch=gfx1201 \
  -Dhipcc=/opt/rocm/core/bin/hipcc \
  -Dhip-device-lib=/opt/rocm/core/lib/llvm/amdgcn/bitcode

zig-out/native-hip/bin/tensorfold-native serve "$PARSE_MODEL" \
  --backend rocm --device 0 --vision --name parse --parallel 2 --context 4096 \
  --mtp-drafts 1 --prompt-cache-gib 0.25 --host 127.0.0.1 --port 8000
```

`PARSE_MODEL` names the already cached snapshot. Select the intended
visible device and verify its PCI identity; management and HIP ordinals
can differ. The engine checks driver free memory at load and admission.
The developer checks require at least six GiB free before a bounded
two-lane run.

Send one user message with one PNG `image_url` data URL to
`/v1/chat/completions`. For example, with an ordinary OpenAI client:

```python
import base64
from pathlib import Path
from openai import OpenAI

image = base64.b64encode(Path("page.png").read_bytes()).decode()
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
reply = client.chat.completions.create(
    model="parse",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}}
    ]}],
    temperature=0,
    max_tokens=1024,
)
print(reply.choices[0].message.content)
```

Use the served model name reported by `/health`, or set `--name` explicitly.
The built-in media template puts the visual prefix before the example's
six task tokens. Optional text parts follow those tokens. Generated OCR,
including its location/class markers, is returned in `content`. A custom
`--chat-template` still overrides the built-in bridge.

The image processor shrinks into 2048-by-1664 without enlargement, pads
bottom/right to 16/64, and preserves the visible content size for RADIO
positions. A full page uses 3329 visual context rows. Visual and text rows
share the 16384-row model context; `--context` may lower this limit. One
image is required. Multi-turn chat, video, tensor parallelism, lossy KV,
Sliding Weights and multiple proposal heads are not part of this port.

Default generation is greedy with repetition penalty 1.1. The shared
sampler widens raw logits to FP32 before applying penalties, and forces
EOS at the configured final output position. `"draft": false` selects
plain target decoding; normal drafted requests use the trained head and
the existing target verifier. Shared rounds batch dense work across all
admitted streams. Prompt snapshots use image content and frame identities.

## Reproduce the checks

Create offline BF16 references from the checkpoint's own example workflow:

```sh
python tools/zig/nemotron_parse_reference.py --model "$PARSE_MODEL" \
  --image page.png --output "$REFERENCE"
python tools/zig/nemotron_parse_reference.py positions --model "$PARSE_MODEL" \
  --output "$POSITION_REFERENCE"

zig build test-nemotron-parse-host -Dparse-model-dir="$PARSE_MODEL" \
  -Dparse-reference-dir="$REFERENCE"

TF_PARSE_DIR="$PARSE_MODEL" TF_PARSE_POSITION_DIR="$POSITION_REFERENCE" \
  zig build hip-parse-ops-test -Dhip=true -Dhip-arch=gfx1201 \
  -Dhipcc=/opt/rocm/core/bin/hipcc \
  -Dhip-device-lib=/opt/rocm/core/lib/llvm/amdgcn/bitcode
```

The reference environment used CPU Torch 2.11.0, TorchVision 0.26.0,
Transformers 5.14.1 and timm 1.0.22. It loads only local assets. The tool
retains NVIDIA's OpenMDW notice; the license is
[LICENSES/NVIDIA-OpenMDW-1.1.txt](../../LICENSES/NVIDIA-OpenMDW-1.1.txt).
Reference directories must be new; large tensors are not checked into Git.
`nemotron_parse_reference.py processor` prints the model-free
`zig/src/families/nemotron_parse/fixtures/processor.json`.

`tf-nemotron-parse-test` builds a developer probe. Give it the model,
reference and output directories. It compares intermediate seams and
teacher-forced logits, then checks full/chunked and solo/shared KV/hidden
bytes and suffix rollback. `nemotron_parse_reference.py compare
--reference "$REFERENCE" --native "$PROBE_OUTPUT"` applies the original HF
logits processor to saved native rows.

The kernel contract registers twelve new operator cases, which
`tf-hip-kernel-contract` runs with no argument, beside the rest of the
contract and its planted controls. `hip-kernel-contract-objects` builds
the code objects for gfx1030, gfx1100, gfx1151 and gfx1201, and
`hip-kernel-contract-limits` checks every launch record against them
without a GPU.

## Evidence and limits

Native HTTP token arrays matched three PNG references (two 512-by-256
pages, one 1664-by-2048 page), plain/drafted and solo/concurrent. The
full page uses 3335 prompt rows including the six text tokens. Prompt KV
and final-hidden bytes agree across full/chunked and solo/shared forwards
and after suffix rollback. No speed target is claimed.

All twelve new operator cases pass arithmetic gates 1–4, including five
repeat runs, guards, non-finites, tile edges and accumulator stress.
Their seven launch records match the code objects of all four recorded
chips and the live gfx1201 device. The complete contract reports only
the limit problems that main reports too: two stale records on the live
gfx1201 (`tf_sum_partials`, `tf_sum_partials_act`) and 13 across the four
chips' code objects.

BF16 intermediate tensors differ from HF's CPU GEMM rounding; saved
seam errors are diagnostic, while actual output token arrays are checked
separately. Runtime model checks cover gfx1201 only. CUDA, Metal, other
AMD hardware have not been qualified here. Sampled plain/drafted equality
passes at two small-page seeds, one with presence and frequency penalties;
this does not compare TensorFold's keyed sampler with HF's random draws.
PNG is tested. JPEG uses the repository's pinned libjpeg-turbo decoder
and has no qualification receipt yet. Structured output needs the
existing grammar library and has no real-model receipt yet.
