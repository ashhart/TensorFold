# Installation runbook

TensorFold 1.0.0 runs from a native executable.
Choose a qualified model and platform from [README.md](README.md#qualified-models), then keep its complete checkpoint outside the source checkout.
The model directory needs configuration, tokenizer, chat template and weight files.
An existing complete Hugging Face cache also works.

## macOS installation

Homebrew exposes both `tensorfold` and `tensorfold-native`:

```sh
brew install ashhart/tensorfold/tensorfold
tensorfold --version
tensorfold capabilities --json
```

For an archive installation, download the arm64 asset and its companion checksum from the 1.0.0 release:

```sh
curl -fLO https://github.com/ashhart/TensorFold/releases/download/v1.0.0/tensorfold-1.0.0-macos-arm64.tar.gz
curl -fLO https://github.com/ashhart/TensorFold/releases/download/v1.0.0/tensorfold-1.0.0-macos-arm64.tar.gz.sha256
shasum -a 256 -c tensorfold-1.0.0-macos-arm64.tar.gz.sha256
tar -xzf tensorfold-1.0.0-macos-arm64.tar.gz
cd tensorfold-1.0.0-macos-arm64
bin/tensorfold-native --version
bin/tensorfold-native --help
```

The archive targets Apple Silicon with M1 CPU instructions and a macOS 13 ABI minimum.
Metal comes from macOS; a prebuilt binary needs no Python, MLX or Xcode installation.
Keep the archive's `bin`, `lib` and any `share` directories together when relocating it.
The commands below use `bin/tensorfold-native`; a Homebrew installation can use `tensorfold` instead.

## First server and request

This example uses the complete `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` checkpoint, including its MTP head:

```sh
bin/tensorfold-native serve "$HOME/models/nemotron-lightning" \
  --name local-model --parallel 1 --context 8192 --temperature 0 --no-thinking
```

Leave the server running and check it from another terminal:

```sh
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/models
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":128,"temperature":0}'
```

Use `http://127.0.0.1:8080/v1` as the OpenAI client base URL.
For streaming, add `"stream": true` to the request.
The native server also has `/v1/responses` and Anthropic `/v1/messages` routes.
Serve flags and request overrides are described in [README.md](README.md#serve-flags).

## Linux and CUDA

Choose `linux-aarch64` for an arm64 host or `linux-x86_64` for an x86_64 host.
The archives require glibc 2.28 or newer and an NVIDIA driver providing `libcuda.so.1`.
The 1.0.0 CUDA inference qualification is Nemotron on GB10 with greedy decoding.

On a GB10 arm64 host:

```sh
curl -fLO https://github.com/ashhart/TensorFold/releases/download/v1.0.0/tensorfold-1.0.0-linux-aarch64.tar.gz
curl -fLO https://github.com/ashhart/TensorFold/releases/download/v1.0.0/tensorfold-1.0.0-linux-aarch64.tar.gz.sha256
sha256sum -c tensorfold-1.0.0-linux-aarch64.tar.gz.sha256
tar -xzf tensorfold-1.0.0-linux-aarch64.tar.gz
cd tensorfold-1.0.0-linux-aarch64
bin/tensorfold-native --version
```

CUDA fatbins are embedded in the executable.
Nemotron also needs its qualified `aot.json` and `cubins` files, normally bundled under `share/tensorfold/cuda/sm121/`.
If the capture assets are supplied separately, point `TENSORFOLD_CUDA_KERNELS` at the directory containing them:

```sh
TENSORFOLD_CUDA_KERNELS=/path/to/qualified/sm121 \
  bin/tensorfold-native serve "$HOME/models/nemotron-lightning" \
  --name local-model --parallel 1 --context 8192 --temperature 0 --no-thinking
```

When the assets are bundled, use the same serve command without the environment override.
Sampled seeds, `top_k: 0` and CUDA device/pinned-memory reporting have known 1.0.0 gaps; see [release notes](RELEASE-NOTES-1.0.0.md#cuda-limits-in-100).
Archives whose names contain `host-only` are CPU verification artifacts and cannot serve CUDA inference.

## Flash Next and paired Metal serving

Flash Next reads its MLX 6-bit/group-32 checkpoint directly and creates its native layouts locally:

```sh
bin/tensorfold-native serve "$HOME/models/flash-next-6bit" \
  --name flash-next --context 32768 --temperature 0 --no-thinking
```

Leave `TF_FLASHNEXT_DUMP` unset for ordinary direct-checkpoint loading.
That variable selects an optional older diagnostic recording path.
The standalone Flash Next qualification is on M5 Ultra; it serves one active reply at a time.

GLM-5.3-Flash's qualified setup uses two M5 Ultras with the same checkpoint, one settings file per rank and a separately built MCDMA library.
Start rank 1 first, then rank 0 with their respective settings:

```sh
bin/tensorfold-native serve "$HOME/models/glm-5.3-flash" \
  --speed-up rank1.json --context 8192 --temperature 0 --no-thinking
bin/tensorfold-native serve "$HOME/models/glm-5.3-flash" \
  --speed-up rank0.json --name local-model --context 8192 --temperature 0 --no-thinking
```

Run each command on its own Mac and send requests to rank 0.
Each settings file names `rank`, `library` and `links`; each link names its peer, RDMA device, interface/address, ports and link name.
[Speed-up settings](docs/speed-up-mode.md#5-write-the-settings-files) has the schema and link setup.
The older recording command in that guide is a diagnostic path; ordinary 1.0.0 loading uses the direct-checkpoint command above.

## Access, context and updates

To accept remote clients, choose `--host 0.0.0.0` and configure `--api-key-file` with a file readable only by its owner.
Clients send `Authorization: Bearer KEY`.
The default loopback server accepts local clients without a key.

A context limit covers prompt plus reply tokens.
Inspect the capacity and memory information printed at startup; larger contexts need more cache space.
After a memory refusal, reduce the context or reply limit, or choose a smaller qualified checkpoint.
`--prompt-cache-gib 0` disables Flash Next prefix retention, and `--keep-warm 0` disables Metal idle keepalive.

Upgrade a Homebrew installation with `brew upgrade tensorfold` and restart its server.
For an archive installation, verify and unpack the replacement archive, then restart from that binary.
Python 0.6.6 remains a separate release line.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Executable not found | Run the archive's `bin/tensorfold-native`, or the Homebrew `tensorfold` command. |
| Model cannot open | Complete checkpoint files, supported geometry/quantization and the model path or existing cache. |
| Client cannot connect | Server log, `/health`, base URL, model ID, listen address and API key. |
| CUDA kernel assets missing | Bundled `share/tensorfold/cuda/sm121/`, or `TENSORFOLD_CUDA_KERNELS`. |
| Pair waits for its peer | Start rank 1 first; check rank, peer, library path, devices, addresses and matching ports. |
| Packed Metal runtime compilation fails | 1.0.0 probes and selects its prebuilt fallback; include chip, macOS version and startup error in a report. |

For a bug report, include the version, checkpoint revision, public reproducing request, exact command and relevant startup/request logs.
