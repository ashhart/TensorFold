# Qwen3-Embedding on Metal

The native Metal server serves Qwen3-Embedding-0.6B, 4B and 8B from their published BF16 safetensors. Their dense forward uses tensor units, so this path requires an M5 or later GPU. An earlier GPU is refused at load. The checkpoint directory must include its tokenizer, config, last-token pooling config and modules list.

```sh
zig build native -Doptimize=ReleaseFast -j8
zig-out/native/bin/tensorfold-native serve /path/to/Qwen3-Embedding-0.6B
```

Send text, token IDs, a batch of strings or a batch of ID lists to `/v1/embeddings`. Text uses the checkpoint's tokenizer with its special-token postprocessor. An explicit ID list is used as given; include the terminal token when the checkpoint's recipe calls for one. Task instructions belong in the input text, as in the checkpoint's recipe; the server adds none.

```json
{"input":["What is photosynthesis?","Plants turn sunlight into chemical energy."],"dimensions":128,"encoding_format":"float"}
```

The final normalized last-token row is pooled. The checkpoint's Normalize module controls L2 normalization. MRL widths range from 32 through the hidden width (1,024, 2,560 or 4,096). A shortened vector is cropped then normalized. `base64` returns little-endian FP32 bytes. Batch members use their own positions and chunk boundaries, with exactly the same bits as an input sent alone.

One pass admits at most 128 inputs and at most the admitted window's total tokens. The default window is 32,768 tokens; `--context 0` requests the checkpoint's trained limit. The memory budget can shrink the window down to 2,048. Empty inputs, out-of-vocabulary IDs, unsupported widths and batches beyond the admitted limits are refused. This is a prompt-only engine: generation, reranking, drafters, recording, learning, vision and multi-device loading are unsupported.

Host checks precede GPU checks:

```sh
zig build test-qwen3-metal-host test-native -Doptimize=ReleaseFast -j8
zig build tf-kernel-contract -Doptimize=ReleaseFast -j8
zig-out/bin/tf-kernel-contract --only 'qwen3 heads'
zig-out/bin/tf-kernel-contract --only llama
```

`zig/tests/qwen3_metal/reference.py` and `reference.c` implement an independent FP32 forward from these checkpoint files. They use BLAS for matrix products and no model library. Build the C helper on a Mac with Accelerate:

```sh
cc -O2 -ffp-contract=off -dynamiclib zig/tests/qwen3_metal/reference.c -framework Accelerate -o reference.dylib
python3 -B zig/tests/qwen3_metal/test_reference.py ./reference.dylib
python3 -B zig/tests/qwen3_metal/device.py --binary zig-out/native/bin/tensorfold-native --library ./reference.dylib --output embedding-check /path/to/Qwen3-Embedding-0.6B /path/to/Qwen3-Embedding-4B /path/to/Qwen3-Embedding-8B
```

The served check reports max absolute and cosine differences against FP32 at every size, throughput, exact batching and repetitions, MRL crops, concurrent requests and named refusals. GPU checks require the operator's exclusive device queue. CUDA and HIP have their own device gates; this check does not claim their numbers.
