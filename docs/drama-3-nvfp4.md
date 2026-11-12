# Drama 3 NVFP4, pruned experts

How to serve [`plotarmordev/drama-3-NVFP4-Pruned`](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned) (7,369 of
10,240 routed experts kept) or [`plotarmordev/drama-3-NVFP4-Pruned-5900`](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned-5900)
(5,900 kept) from main, on a DGX Spark (CUDA) or a Mac (Metal). Each stores its routed experts in NVFP4 and only the
kept experts; the engine reads them as stored, never widened to BF16, and routes over the kept experts alone. Only
TensorFold loads them; Transformers and SGLang-Omni do not.

## Build

Zig 0.17.0 exactly (the build refuses others).

```sh
zig build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc   # DGX Spark: zig-out/native/bin/tensorfold-native
zig build native -Doptimize=ReleaseFast                 # Mac, with Xcode's Metal Toolchain
```

`zig build drama3-voice drama3-generate` adds the voice writer and the generation tool beside it.

## Serve

```sh
hf download plotarmordev/drama-3-NVFP4-Pruned --local-dir drama-3-NVFP4-Pruned
zig-out/native/bin/tensorfold-native serve drama-3-NVFP4-Pruned --name drama-3 --port 8080 \
  --voices voices --default-voice NAME --cfg-gamma 1.5 --cfg-batched
```

Serve the 5,900-expert checkpoint, `plotarmordev/drama-3-NVFP4-Pruned-5900`, the same way. The startup log names what
was loaded: `routed experts held 7369 of 10240` (or `5900 of 10240`) and `routed experts NVFP4 in 40 of 40 layers`.
A malformed, unordered, missing or extra-expert map fails before any model memory is allocated.

Guidance (`--cfg-gamma 1.5 --cfg-batched`) acts only on a prompt with audio rows, so serve a catalog voice: a
reference-backed `NAME.json` in `--voices`, written by `drama3-voice` as
[Writing a voice file](drama-http-voices.md#writing-a-voice-file) describes. In the built-in voice a request is drawn
unguided. Voice files hold a person's voice: keep them out of source control.

## Request

```sh
curl -sS http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "drama-3", "input": "The lighthouse keeper climbed the stairs.", "response_format": "wav", "seed": 7}' \
  -o speech.wav
```

The reply is 44.1 kHz 16-bit mono. `"stream": true` with `"response_format": "pcm"` sends raw PCM as it decodes. A
fixed `seed` gives the same audio on a repeat and beside a concurrent request; see
[sampling controls](speech-sampling.md).

## Measured limits

From [the receipt](receipts/drama3-pruned-main/README.md): two-lane servers with guidance, a 5.5 s synthetic
reference voice and about 4.5 s replies, timed by the client.

| Machine | Checkpoint | Load | Peak memory | First audio | RTF |
| --- | --- | ---: | ---: | ---: | ---: |
| M5 Ultra, 256 GB | 7,369 experts | 8.9 s | 10.2 GiB footprint + 13.0 GB experts read in place | 0.94 s | 0.73 |
| M5 Ultra, 256 GB | 5,900 experts | 8.5 s | 10.2 GiB footprint + 10.4 GB experts read in place | 0.94 s | 0.74 |
| DGX Spark (GB10) | FP8, 10,240 experts | 27.6 s | 39.6 GiB device | 6.07 s | 2.36 |
| DGX Spark (GB10) | 7,369 experts | 24.5 s | 21.7 GiB device | 5.69 s | 2.16 |
| DGX Spark (GB10) | 5,900 experts | 26.0 s | 19.3 GiB device | 5.65 s | 2.23 |

- **Mac memory.** Metal maps the packed experts from the checkpoint files, so the process footprint leaves them out.
  The 7,369-expert server keeps 28.5 GiB wired between requests; the load is refused above 70% of RAM
  (`DRAMA_LOAD_LIMIT_GB` sets another limit).
- **Spark speed.** With guidance the Spark is slower than real time on every checkpoint; the pruned ones save about
  half the device memory and 0.4 s to first audio against FP8. Two requests at once take turns: the second ends after
  20-23 s. Its server finds cuBLAS through `LD_LIBRARY_PATH` (the CUDA 13 `lib` directory).
- **Second lane.** On the Mac, the concurrent streamed replies recorded first audio at 6.07 s (7,369 experts) and
  6.22 s (5,900), with total times of 8.47/8.66 s and an RTF of 1.90 for each. A lane without a cached reference
  waits for its own reference prefill. These historical first-audio timings, including the table's 0.94 s, used
  buffered 65,536-byte reads and are upper bounds, not first-byte measurements; the current checker uses `read1()`.

## Checks

The no-GPU tests run with the rest:

```sh
zig build test test-golden
zig build check-drama-cuda-host test-drama-cfg-host test-drama-nvfp4-reference-host   # Linux
zig build check-drama-metal-host test-drama-metal-quant-host                          # macOS
python3 -m unittest discover -s tools/zig -p 'test_drama*.py'
```

On a GPU, [`tools/zig/drama3_checkpoints_check.py`](../tools/zig/drama3_checkpoints_check.py) serves each checkpoint
it is given (`--fp8`, `--pruned`, `--pruned-5900`) and fails unless each holds its experts and its fixed-seed whole and
streamed replies equal their repeats and their concurrent lanes; it records load, peak memory, first audio, total time
and RTF. On a Spark, [`run-script.sh`](receipts/drama3-pruned-main/run-script.sh) builds and runs it over all three
with no arguments, from the pilot folder at `/assets/drama3-pilot`, as the GPU door runs a job; its kernel images come
from [`prebuild-fatbins.sh`](receipts/drama3-pruned-main/prebuild-fatbins.sh), as the job image has no C++ compiler.
