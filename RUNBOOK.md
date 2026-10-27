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
V=1.1.7
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

## Model-free Metal counter probe

With the Zig version from `.zig-version` and Xcode's Metal toolchain:

```sh
zig build tf-metal-counters test-metal-counters-host test-metal-counters-gpu -Dcpu=apple_m1
./zig-out/bin/tf-metal-counters --capability
./zig-out/bin/tf-metal-counters
```

The probe runs three dependent FP32 operators without a checkpoint, verifies
independent expected output bits and instrumented/uninstrumented identity, and
prints calibrated GPU counter intervals as JSON. Unsupported sampling is reported
explicitly; CPU elapsed time is not a replacement for missing GPU samples.

Stage sampling rotates serial encoders with fences. A consumer must rebind its
pipeline, buffers and constants and redeclare indirect resources after rotations.
The recorder permits one in-flight command buffer, owns bounded sample storage,
and resolves samples only after the caller's existing completion wait.

An M3 Ultra on macOS 26.6.2 exercised stage sampling and passed all 13 dedicated
host/device tests. Dispatch-boundary sampling was unavailable and was not exercised.
This probe does not establish model fidelity, decode speed or serving support.

## Standalone Metal EXL3 projections

With the Zig version from `.zig-version` and Xcode's Metal toolchain:

```sh
zig build tf-exl3-bench test-exl3 test-exl3-gpu -Dcpu=apple_m1
./zig-out/bin/tf-exl3-bench --help
./zig-out/bin/tf-exl3-bench --qualify-only 1 --warmup 0 --repeats 1
./zig-out/bin/tf-exl3-bench --warmup 1 --repeats 3 --prefill 32
```

These commands use deterministic compressed fixtures, not a checkpoint.
Compact 256-by-128 fixtures check all nine rows and every output column for
27 width/codebook combinations in BF16 and F32. Larger shapes check every
candidate output bit against the arithmetic decoder, but the independent FP64
oracle samples at most two rows and three 128-column blocks. Its worst-column
metrics are explicitly labelled `worst_sampled`; they are not whole-matrix bounds.

Plain prefill windows of at least 32 rows share decoded weight tiles across four
SIMDgroups when the pipeline supports 128 threads. Smaller windows, filtered
routes and indexed experts keep the eight-row kernel. If the prefill pipeline
is unavailable, the operator uses that existing path. The benchmark's arithmetic
reference disables the optional prefill pipeline to keep its comparison valid.

The borrowed-buffer API in `zig/src/core/exl3_projection.zig` encodes BF16
inputs, FP16-rounded rotated inputs, compensated FP32 projection dots and BF16
or F32 outputs. Scratch holds row intermediates, never a dequantized matrix.
All buffers and pipelines must survive command completion. GPU-produced IDs
must be in bounds, scatter destinations unique, and grouped padding must follow
valid rows. Indexed descriptors share dimensions and require valid resident
weights disjoint from writable views; use `useMatrix` or caller residency sets.
The operator validates storage extents, offsets, strides, capacities and aliases,
but does not read back GPU IDs or plan expert groups.

On an M3 Ultra running macOS 26.6.2, the 82 exercised operator cases had zero
candidate/baseline bit mismatches and identical independent-oracle errors.
Timings alternate dispatch order and exclude oracle work and allocations.
This is operator evidence only: no model loader, family registration, CUDA
implementation, whole-model fidelity or served-throughput claim is added.

## First server and request

This example uses the complete `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` checkpoint, including its MTP head:

```sh
bin/tensorfold-native serve "$HOME/models/nemotron-lightning" \
  --name local-model --parallel 1 --context 8192 --temperature 0 --no-thinking
```

This example's `--parallel 1` caps serving at one active request; extra requests wait for a free slot.
Omit `--parallel` to use `auto`, which asks for up to eight streams as memory and the engine allow.
Use `--parallel N` for an explicit cap; the engine refuses a count it cannot fit.
Read the startup capacity line, such as `8 streams at once` on CUDA, for the accepted count.
For example, `--parallel 4` keeps at most four requests active while the fifth and later requests wait.
Completing or cancelling an active request frees its slot; cancelling a waiting request removes it from the queue.

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

With API authentication off, `/health` includes `live.connections`, the total active plus waiting requests,
and `live.waiting`, the queued requests. Subtract waiting from connections to get the active count.
For example, 8 connections with 4 waiting means 4 active and 4 queued.
When API authentication is enabled, `/health` returns only status.

## Linux and CUDA

For source builds with `zig build native` (including `--prefix` installs): Structured output and required tool calls need the release archive's `lib/` or `zig build grammar -Dxgrammar=DIR`; prepare DIR with `git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar` (use the native build's same `--prefix` for grammar).

Choose `linux-aarch64` for an arm64 host or `linux-x86_64` for an x86_64 host.
The archives require glibc 2.28 or newer and an NVIDIA driver providing `libcuda.so.1`.
CUDA serves Nemotron 3.5 Lightning, qualified on GB10 (`linux-aarch64`, greedy and sampled) and on Ampere cards with
compute capability 8.6 (`linux-x86_64`, greedy: the RTX 30 series, RTX A6000, A10 and A40), and checked on an RTX PRO 6000 Blackwell (compute
capability 12.0, `linux-x86_64`).
Flash Next also serves on CUDA from its MLX 4-bit checkpoint, on the built-in kernels, run on a GB10: see
[Flash Next on CUDA](docs/recipes/flash-next-cuda.md).

On a GB10 arm64 host (an x86_64 host uses `linux-x86_64` in the same commands):

```sh
V=1.1.7
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
A conversation's next turn resumes from a prompt state kept where its last turn's history ended, so a reply equals the same
request with the cache off. `--prompt-cache-gib` sizes the kept states (16 GiB by default, inside the memory budget) and 0 turns
them off. A state kept inside a prompt chunk costs no extra prompt call. A Nemotron 3.5 Lightning state takes 48 MiB plus
7 KiB a prompt token, in whole 2 MiB pages (62 MiB at 2,048 tokens), and the cache keeps four 48 MiB buffers of its
budget free for the next states.
Prompts fill a chunk at a time between decode rounds, so requests that arrive together on a new system prompt would each
compute it. A request whose prompt shares a cut another prompt is still filling, which its own lookup would resume, waits
in the queue until that fill stands there, then resumes the cut. `TENSORFOLD_PREFIX_WAIT_MS` bounds the wait (60000 by
default; 0 turns it off).
`TF_HEAT_HIGH` and `TF_HEAT_LOW`, in degrees Celsius and set together, make each prompt chunk wait while the hottest
thermal zone (under `/sys/class/thermal`, or `TF_HEAT_ROOT`) is above the high band, until it is at or under the low one.
The server logs `heat_wait_s`, a cancel ends the wait, and the reply's tokens do not change.
On three GB10s serving GLM-5.3-Flash NVFP4 at TP=3, a 1,035,295-token prompt without the bands kept one host's hottest
zone at 94 C or more for most of the prompt (97.7 C at the top); `TF_HEAT_HIGH=92` and `TF_HEAT_LOW=88` held all three
at 93.0 C or under, and the first token came 10% later.
Archives whose names contain `host-only` are CPU verification artifacts and cannot serve CUDA inference.

### GB10 CPU pinning for long CUDA runs

In [#1033](https://github.com/ashhart/TensorFold/issues/1033), @Bizuayeu reports two CPU threads spinning per CUDA rank,
even between requests, heating the CPU during long reply runs. Pinning each rank to efficiency cores reduced that heat.

Logical CPUs `0-4,10-14` are the Cortex-A725 efficiency cores, confirmed on two GB10 units. Verify the CPU layout (`lscpu -e`) on
each host. Add `taskset -c 0-4,10-14` before each rank's existing native launch,
or add `--cpuset-cpus=0-4,10-14` after `docker run` and before the image name in each rank's container launch.
Keep the existing model, GPU, network and rank options.

On GLM-5.3-Flash NVFP4, version 1.0.12 plus #1016-#1018, TP=2, with heat bands of 92/88 C, @Bizuayeu's pinned run
completed 40/40 replies, peaking at 85.1 C; the unrestricted run stopped at the seventh reply at 94.7 C.
Decode throughput fell 1.1% and prefill throughput 2.7%. These are the contributor's measurements, not 1.1.0 results.

### GB10 GPU clock cap for long CUDA runs

GB10 hosts are reported to power off under sustained GPU load, leaving no log
([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/hard-power-off-under-sustained-gpu-load-at-90w-persists-after-full-platform-firmware-update/378315)).
The reported workaround is a GPU clock cap at 2,200 MHz
([tonyd2wild/dgx-spark-hard-poweroff-fix](https://github.com/tonyd2wild/dgx-spark-hard-poweroff-fix)).
`sudo nvidia-smi -lgc 300,2200` sets the cap and `sudo nvidia-smi -rgc` removes it. To set it at each boot, save this
unit as `/etc/systemd/system/gb10-clock-cap.service` and run `sudo systemctl enable --now gb10-clock-cap`:

```ini
[Unit]
Description=GB10 GPU clock cap
After=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/nvidia-smi -lgc 300,2200
ExecStop=/usr/bin/nvidia-smi -rgc

[Install]
WantedBy=multi-user.target
```

Cap every rank's host. Under load the clock then reads 2,200 MHz or under
(`nvidia-smi --query-gpu=clocks.gr --format=csv`). In the serves below, the hottest thermal zone (`/sys/class/thermal`)
peaked 8 to 11 C above the GPU's own peak. The cap holds the GPU only: on a capped host the hottest zone reached 94 C
with the GPU idle (312 MHz, 6.7 W) while the kernel contract's checks ran and nvcc built its fatbins. The cap does not
replace the bands: without them, the 1,035,295-token prompt above (ranks not pinned) reached 97.7 C under the cap, the
hottest host's clock held at 2,177 to 2,190 MHz.

On GLM-5.3-Flash NVFP4, version 1.1.1, TP=2, each rank on the efficiency cores, @Bizuayeu's serves with the cap,
twice without it and with it again: the cap made drafted decode about 1% slower, `"draft": false` decode about 1.5%
slower and a cold prompt of 9,600 to 38,400 tokens 2 to 5% slower, with the same token ids; each GPU peaked at about
60 W instead of 89 to 92 W, and the hottest thermal zone at 88.6 to 90.7 C instead of 96.8 to 98.1 C. These are the
contributor's measurements.

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
[Speed-up settings](docs/speed-up-mode.md#4-write-the-settings-files) has the schema and link setup.
That guide's two-Mac Flash Next commands load the checkpoint directly, as one Mac does.

On one Mac, GLM loads only when its weights and caches fit in 70% of RAM, 179.2 GB on a 256 GB Mac, and leaves the MTP head on disk when only the head is over.
`GLM_LOAD_LIMIT_GB=N` sets that limit in GB, up to the GPU's recommended working set; a value that isn't a number above 0 stops the server.
Past 70% of RAM the server warns at startup: a load that wires most of the RAM can stall macOS until its watchdog restarts the Mac, so run nothing else large beside it.
GLM enables a shared residency set by default when its unique Metal buffers and a 2 GiB margin fit the load bound.
`GLM_RESIDENCY=1` keeps that default, `GLM_RESIDENCY=0` disables it, and every other value stops loading before Metal allocations.
The startup log reports the GiB held or the reason the set was skipped.

A successful Metal residency set keeps weights, live stream state and working buffers wired during idle keepalive.
This retains the memory already used by serving instead of letting macOS reclaim its pages between requests.
The shared keepalive commits a small blit every 400 ms; the set alone is insufficient.
The default keep-warm window is 900 seconds, and `--keep-warm 0` disables it.
Later buffers join through their owners while they fit the same bound, and leave before they are freed or replaced.
The bound reserves 2 GiB inside a validated family load limit, or 70% of known RAM by default.
Neural Engine program weights are separate from Metal buffers and are outside this set.


## Access, context and updates

To accept remote clients, choose `--host 0.0.0.0` and configure `--api-key-file` with a file readable only by its owner.
Clients send `Authorization: Bearer KEY`.
The default loopback server accepts local clients without a key.

The server admits at most 128 open connections, including idle keep-alive connections and active replies.
Extra connections receive `503 Service Unavailable` with `Retry-After: 1`, then close.
The reply is best effort if the peer cannot receive it; no extra connection thread is created.
This cap is independent of the process file limit.

A new connection has 30 seconds to send its request line and all headers.
A kept connection may wait idle for 900 seconds; its next request's first bytes start a fresh 30-second head budget.
A fixed-length body has 30 seconds plus one second per 64 KiB declared, rounded up to a millisecond.
Chunked bodies extend that same original budget only by declared payload bytes, within the 32 MiB body limit.
Chunk metadata and trailers do not reset the deadline.
The existing 120-second read-idle and 300-second write-idle limits still apply.
Read budgets end when input is complete, so they impose no total deadline on inference or streamed replies.

A context limit covers prompt plus reply tokens.
Inspect the capacity and memory information printed at startup; larger contexts need more cache space.
After a memory refusal, reduce the context or reply limit, or choose a smaller qualified checkpoint.
`--prompt-cache-gib 0` disables Nemotron, Flash Next, GLM and Qwen3.8-27B prefix retention, and `--keep-warm 0` disables Metal idle keepalive.
`--learn` writes a shared prompt state to disk only while the disk keeps `--learn-min-free-gib` free, 4 GiB by default.
Below that, learning pauses with a log line and serving goes on; a refused write is tried again after 1 to 60 seconds, sooner when space comes back.
Each learned file carries a checksum over its header and contents, so a damaged, truncated or older file is refused and learned again.
`--learn-conversations` writes every kept prompt state behind, while the engine idles, and a SIGTERM stop writes what waits within 20 seconds.
Stop a server with SIGTERM, not SIGKILL, so its last turn is kept.
`--learn-daily-gib` caps those writes a day (32 GiB by default); `DELETE /v1/imprint` forgets one model's conversation states.

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
