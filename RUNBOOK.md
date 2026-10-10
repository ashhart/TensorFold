# Installation runbook

TensorFold runs from a native executable.
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

For an archive installation, download the arm64 asset and its companion checksum from the
[latest release](https://github.com/ashhart/TensorFold/releases/latest), with `V` set to its version:

```sh
V=1.0.5
curl -fLO https://github.com/ashhart/TensorFold/releases/download/v$V/tensorfold-$V-macos-arm64.tar.gz
curl -fLO https://github.com/ashhart/TensorFold/releases/download/v$V/tensorfold-$V-macos-arm64.tar.gz.sha256
shasum -a 256 -c tensorfold-$V-macos-arm64.tar.gz.sha256
tar -xzf tensorfold-$V-macos-arm64.tar.gz
cd tensorfold-$V-macos-arm64
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
CUDA serves Nemotron 3.5 Lightning, qualified on GB10 (`linux-aarch64`, greedy and sampled) and on Ampere cards with
compute capability 8.6 (`linux-x86_64`, greedy: the RTX 30 series, RTX A6000, A10 and A40).

On a GB10 arm64 host (an x86_64 host uses `linux-x86_64` in the same commands):

```sh
V=1.0.5
curl -fLO https://github.com/ashhart/TensorFold/releases/download/v$V/tensorfold-$V-linux-aarch64.tar.gz
curl -fLO https://github.com/ashhart/TensorFold/releases/download/v$V/tensorfold-$V-linux-aarch64.tar.gz.sha256
sha256sum -c tensorfold-$V-linux-aarch64.tar.gz.sha256
tar -xzf tensorfold-$V-linux-aarch64.tar.gz
cd tensorfold-$V-linux-aarch64
bin/tensorfold-native --version
```

CUDA fatbins are embedded in the executable, and Nemotron runs on those built-in kernels:

```sh
bin/tensorfold-native serve "$HOME/models/nemotron-lightning" \
  --name local-model --parallel 1 --context 8192 --temperature 0 --no-thinking
```

A captured Triton set is optional: `TENSORFOLD_CUDA_KERNELS` names one (a folder with `aot.json` and `cubins/`), `native`
forces the built-in kernels, and a set found at `share/tensorfold/cuda/sm<capability>/` beside the executable is used when present.
A sampled request honours its `seed`, `top_k: 0` turns the top-k filter off, and `/metrics` reports CUDA device memory and
pinned host memory.
A request that names no `top_k`, on a checkpoint whose generation config names none, samples with no top-k filter on
CUDA and Metal alike, so a seeded request draws the same on both. `--top-k 20` gives 0.6.6's CUDA default.
A conversation's next turn resumes from a prompt state kept at a 2,048-token chunk boundary, so a reply equals the same
request with the cache off. `--prompt-cache-gib` sizes the kept states (16 GiB by default, inside the memory budget) and 0 turns
them off; prompts shorter than 4,096 tokens keep none.
`TF_HEAT_HIGH` and `TF_HEAT_LOW`, in degrees Celsius and set together, make each prompt chunk wait while the hottest
thermal zone (under `/sys/class/thermal`, or `TF_HEAT_ROOT`) is above the high band, until it is at or under the low one.
The server logs `heat_wait_s`, a cancel ends the wait, and the reply's tokens do not change.
Archives whose names contain `host-only` are CPU verification artifacts and cannot serve CUDA inference.

## Flash Next and paired Metal serving

Flash Next reads its MLX 6-bit/group-32 checkpoint directly and creates its native layouts locally.
Other affine bit widths or group sizes are refused before device or pack preparation; the runtime kernels require 6-bit affine with `group_size` 32.

```sh
bin/tensorfold-native serve "$HOME/models/flash-next-6bit" \
  --name flash-next --context 32768 --temperature 0 --no-thinking
```

Prepared packs are stored in `$HOME/.cache/tensorfold/packs/<identity>/`, separate from the checkpoint.
The directory name is the SHA256 of the existing pack source identity, which records the index, shard sizes and shard headers.
A complete matching `<model>/zig-pack` is read for compatibility and is never written or repaired.
New packs are built in a sibling temporary directory and published only after every pack is complete.
An interrupted build is retried on the next open; its owned temporary directory is then removed.
The model directory can be read-only, and the cache directory must be writable for a new build.

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
Run the same release on both Macs: the pair now agrees on learned-state disk space before either Mac writes, and each Mac refuses a peer from an earlier release.
With `--learn`, states learned by an earlier release are refused when read and learned again.
Each settings file names `rank`, `library` and `links`; each link names its peer, RDMA device, interface/address, ports and link name.
[Speed-up settings](docs/speed-up-mode.md#5-write-the-settings-files) has the schema and link setup.
The recording command in that guide is the setup its two-Mac Flash Next numbers were measured with; on one Mac, ordinary
loading uses the direct-checkpoint command above.

On one Mac, GLM loads only when its weights and caches fit in 70% of RAM, 179.2 GB on a 256 GB Mac, and leaves the MTP head on disk when only the head is over.
`GLM_LOAD_LIMIT_GB=N` sets that limit in GB, up to the GPU's recommended working set; a value that isn't a number above 0 stops the server.
Past 70% of RAM the server warns at startup: a load that wires most of the RAM can stall macOS until its watchdog restarts the Mac, so run nothing else large beside it.

## Access, context and updates

To accept remote clients, choose `--host 0.0.0.0` and configure `--api-key-file` with a file readable only by its owner.
Clients send `Authorization: Bearer KEY`.
The default loopback server accepts local clients without a key.

A context limit covers prompt plus reply tokens.
Inspect the capacity and memory information printed at startup; larger contexts need more cache space.
After a memory refusal, reduce the context or reply limit, or choose a smaller qualified checkpoint.
`--prompt-cache-gib 0` disables Nemotron, Flash Next, GLM and Qwen3.8-27B prefix retention, and `--keep-warm 0` disables Metal idle keepalive.
`--learn` writes a shared prompt state to disk only while the disk keeps `--learn-min-free-gib` free, 4 GiB by default.
Below that, learning pauses with a log line and serving goes on; a refused write is tried again after 1 to 60 seconds, sooner when space comes back.
Each learned file carries a checksum over its header and contents, so a damaged, truncated or older file is refused and learned again.

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
| Pair waits for its peer | Start rank 1 first; check rank, peer, library path, devices, addresses and matching ports. On macOS, run both servers attached to a session or allow them Local Network access. |
| Packed Metal runtime compilation fails | 1.0.0 probes and selects its prebuilt fallback; include chip, macOS version and startup error in a report. |

For a bug report, include the version, checkpoint revision, public reproducing request, exact command and relevant startup/request logs.
