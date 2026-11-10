# dots.mocr on native HIP

This implementation serves the BF16 `dots-studio/dots.mocr`
checkpoint at revision `e539fbb52280393adc081b289ec597430a0f9031`.
It runs the image tower and Qwen2 decoder in native Zig/HIP. Python is
used only for the validation reference. CUDA serves it too
([dots.mocr on native CUDA](dots-mocr-cuda.md)); Metal is not qualified.

Build for the local GPU architecture. The checked host uses `gfx1201`:

```sh
zig build native-hip -Dhip=true -Dhip-arch=gfx1201 \
  -Dhipcc=/opt/rocm/bin/hipcc -j4
```

Use the existing downloaded checkpoint directory:

```sh
zig-out/native-hip/bin/tensorfold-native serve "$MODEL" \
  --host 127.0.0.1 --port 18896 --name dots-mocr --vision \
  --context 16384 --parallel 2 --max-tokens 8192 --temperature 0 \
  --no-update-check
```

On a host with several GPUs, select the intended card with
`ROCR_VISIBLE_DEVICES=GPU_UUID HIP_VISIBLE_DEVICES=0`. Verify the actual
PCI address before a device test. Shared GPU use is allowed; the loader
and cache saves check current memory with the existing HIP reserve and
memory-limit settings. Explicit lane counts must fit. Automatic lane
counts may decrease. The checked two-lane configuration starts with
about 10.7 GB of tracked device allocations, before prompt snapshots;
it is not a minimum-memory claim. The default snapshot budget is at most
2 GiB, and the loader counts that budget before uploading weights.

For JPEG, use libjpeg-turbo **3.1.4.1**. Set `TENSORFOLD_LIBTURBOJPEG` to
its shared library when it is not on the loader path. PNG and WebP use
the existing image decoders. Send OpenAI chat content with `image_url`
parts and the prompt from the checkpoint's model card. Data URLs keep
the request entirely local. The template, tokenizer, image processor,
weights, and three EOS IDs come from the checkpoint without modification.
OCR output is in `message.content`, including with the server's default
thinking setting. `return_token_ids: true` also returns the IDs in the
`tensorfold` response object.

The default total image budget is 14,400 merged tokens per request.
`--vision-image-tokens` sets a different total budget, divided across
the images. Each image has independent tower attention. Quoted image
markers remain text. Video, tensor parallelism, MTP heads, external
drafters, and incompatible quantized checkpoints are refused. Copy
proposals remain available with ordinary `draft: true` requests.

The graph uses rocBLAS BF16 projections with FP32 accumulation, dot2
attention, prompt chunks of at most 256 rows, and exact native verification
windows of at most eight rows. It does not capture HIP graphs. Unsupported
kernel or graph policy overrides are refused. Core owns the lane scheduler,
sampling, copy proposals, cache identity, lookup, and eviction. Snapshots
hold physical KV-prefix copies and last logits. Cache reuse follows the
planned chunk grid; it does not claim arbitrary prompt chunk invariance.

The complete model-card demo matches all 4,634 HF output IDs through EOS
over HTTP. A two-image synthetic case also matches all 108 HF IDs. A
square synthetic case reads the same text but differs in four box-coordinate
tokens, by one pixel per edge. The recorded tower activations are not
byte-identical to HF. Native plain/drafted, fresh/resumed, and solo/concurrent
replies match for the checked cases, including seeded sampling. Cancellation,
cache image identity, eviction, idle reload, and failure cleanup are checked.
These checks do not establish general OCR quality.
