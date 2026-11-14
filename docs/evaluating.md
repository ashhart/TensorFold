# Evaluating TensorFold

This page is for anyone reviewing, testing or benchmarking TensorFold from outside the project.
It covers what to pin, how to check a download, the tests that run without a GPU and the requests worth replaying.

## Pin what you test

Record four things with every result:

- The release, from `tensorfold --version` (on Linux, `tensorfold-native --version`).
- The checkpoint and its revision, as `REPO@revision` in `tensorfold pull`.
- The backend: Metal on a Mac, CUDA on Linux or Windows.
- The recipe you followed from [Pick a model](../README.md#pick-a-model), with its flags.

Releases from 1.0.0 on are the native Zig engine. The Python engine stopped at 0.6.6 and lives only at the `v0.6.6`
tag; its tests need MLX and its results do not describe a 1.x release. `pip install .` at the root stops and says where
to get the native binary.

## Install and check the download

On a Mac, `brew install ashhart/tensorfold/tensorfold` checks the archive against the checksum in the formula.

On Linux, the [Quick start](../README.md#linux-with-an-nvidia-gpu) downloads the archive and checks its `.sha256`.
The `.sha256` file is itself signed with the release key. To check that signature too (OpenSSL 3.0 or later):

```sh
A="tensorfold-$V-$PLATFORM.tar.gz"
curl -fLO "https://github.com/ashhart/TensorFold/releases/download/v$V/$A.sha256.sig"
curl -fsSL https://raw.githubusercontent.com/ashhart/TensorFold/main/packaging/runtime-trust.json \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["ed25519PublicKey"])' > tensorfold-release.pem
{ printf 'tensorfold-native-checksum-v1\n'; cat "$A.sha256"; } > signed-checksum
openssl pkeyutl -verify -pubin -inkey tensorfold-release.pem -rawin -in signed-checksum -sigfile "$A.sha256.sig"
```

OpenSSL prints `Signature Verified Successfully`. After unpacking, `sha256sum -c SHA256SUMS` inside the archive's
directory checks every file in it.

## Smoke checks without a model

These need no GPU, driver or checkpoint:

- `tensorfold --version` prints the release.
- `tensorfold capabilities --json` lists the backends, model families, formats, serve flags and
  `optional_libraries`, each with a `found` flag.
- `tensorfold info REPO` prints a pulled checkpoint's family, format, context, weights size and memory floor.

## Container

The root [Dockerfile](../Dockerfile) installs a Linux release archive in an Ubuntu 24.04 image. The build stops unless
the archive's checksum, the release signature over it and every unpacked file's checksum all match.
It runs as the image's `ubuntu` user and keeps the Hugging Face cache in `/huggingface`, the directory to mount.

```sh
docker build -t tensorfold .
docker run --rm tensorfold
docker run --rm tensorfold capabilities --json
```

`--build-arg TENSORFOLD_VERSION=1.2.4` picks the release; Docker picks the `linux-x86_64` or `linux-aarch64`
archive from the build platform. To serve, the host needs NVIDIA driver R580 or newer and the NVIDIA Container Toolkit,
which supplies `libcuda.so.1` to the container. Listen on all addresses inside the container:

```sh
docker run --rm --gpus all -p 8080:8080 \
  -v "$HOME/.cache/huggingface:/huggingface" \
  tensorfold serve mlx-community/Qwen3.5-2B-MLX-4bit --host 0.0.0.0 --name local-model --context 8192
```

Add `--user "$(id -u):$(id -g)"` when your user ID is not 1000, so the server can write to the mounted cache.
The image holds only the release archive. Features that need cuBLASLt, NCCL or the image decoders (see
[optional shared libraries](../packaging/RUNTIME.md#optional-shared-libraries)) need an image that adds them.

## Tests that run without a GPU

- `python3 -m pytest` from the repository root needs only Python 3 and pytest. Tests that need a GPU, numpy or a
  built test binary skip and say why.
- `zig build test` builds and runs the host tests with the Zig version in `.zig-version`. On a Mac it also runs the
  Metal device checks it can.
- `zig build test-golden` checks frozen parity answers and the HTTP security cases.

[CI](../.github/workflows/no-gpu.yml) runs these on Linux and Windows for every pull request, with the root pytest on
Linux. GPU kernel and served-model checks run on the maintainers' Macs and NVIDIA machines before a change lands;
[CONTRIBUTING.md](../CONTRIBUTING.md) lists those gates.

## Check exactness yourself

Drafting must never change the output. At temperature 0, send the same request twice, once with `"draft": false`,
and compare the text and `usage`. A request repeated against the prompt cache must equal its first run, and
concurrent requests must each equal their solo run. [Exact decoding](../README.md#exact-decoding) states the rule
and what it fixes.

## Requests worth replaying

| Request | What to expect |
| --- | --- |
| A prompt longer than the context window | HTTP 400 with `code` `context_length_exceeded`; no GPU work is spent. |
| `tool_choice: "required"` or a named function | The reply calls a tool. Needs the archive's `lib/`; without it, HTTP 400. |
| `thinking_budget` or `reasoning_effort` | The reasoning stops within the budget; `reasoning_effort: none` turns thinking off. |
| `response_format` with a JSON schema | The reply matches the schema and equals its `"draft": false` reply. Needs `lib/` too. |

The [API reference](api.md) lists every field and refusal.

## Memory

`tensorfold info REPO` prints a checkpoint's memory floor: its weights plus the reserve the server keeps.
Context length, `--parallel`, prompt caches and drafters need memory beyond that floor. Once a model is loaded,
`GET /v1/memory` reports the memory in use: the process footprint on macOS and the backend's device counters
([memory diagnostics](memory-diagnostics.md)).
The machines each model was checked on are in [Pick a model](../README.md#pick-a-model); where no minimum was
measured, none is published.

## Dependencies

Serving installs no Python packages, so `pip-audit` has nothing to scan for the server.
`build.zig.zon` declares no Zig packages, so the Zig code builds from this repository alone; the kernels need
Xcode's Metal compiler or CUDA 13's `nvcc`. The Linux archive adds `libtfgrammar` (xgrammar v0.2.8).
Every bundled licence is in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
Some repository tools, such as checkpoint converters and reference generators, use numpy, PyTorch, MLX or
Transformers. None of them is needed to build, run the host tests or serve, and the root pytest skips the tests that
need them. The Mac app under `app/` has its own dependencies.

## Reporting

Use the issue templates for bugs and performance results, with the four pinned facts above.
Report security problems privately, as [SECURITY.md](../SECURITY.md) describes.
