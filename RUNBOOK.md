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
V=1.2.9
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

## Prompt prefill share, budget and admission policy

The native server reads four environment variables when it opens its lane host, and `serve` checks them before it loads the model.
All are optional. Set them before starting the server.
`TENSORFOLD_PREFILL_SHARE` takes a number from 0.01 to 1.
The default is 1, and empty text gives the default. `serve` refuses any other value at start, naming the variable.
It is the share of wall time long prompt steps get while other streams are decoding.
A step of length t at share p owes the decoders t (1 - p) / p of decode time.
The debt grows after each step and falls after each decode round.
A long prompt's next step runs only when no debt is owed or no stream can decode.
At the default 1 no debt is recorded, so the only waiting is the ordinary turn-taking: the host runs at most one step of a long prompt each time it turns to it, and chunked prompts still interleave with the decoding streams' rounds.
Lower values add that debt, finish the long prompt later and leave more time for decoding streams.
Flash Next's shared sessions on a Mac without tensor units start from a reply share of 0.8 when neither this nor `--decode-share` is set: see [the API](docs/api.md).
A prompt whose whole uncached pass is at most 512 rows runs outside the share as one call.
The check compares the prompt's length with the cached state, not how far the fill has reached, so a long prompt whose last rows remain does not become exempt when its unfilled tail is small.

`TENSORFOLD_PREFILL_BUDGET` takes a whole number of prompt rows from 1 to 4,294,967,295.
The default is 0, which means none.
Empty text or 0 gives 0, which removes the cap. `serve` refuses a negative number, a fractional form, a value above the range or any other text at start.
It caps the prompt rows filled in one pass while a stream is ready to decode.
The cap counts only rows filled while such a stream waits.
A call that reports no progress spends the allowance anyway and yields, so unknown or stale progress never holds the cap open.
When a cap is set, one step of a long prompt that has waited long enough goes first.
When the cap is reached the host stops prompt work for that pass.
The decoders then take their next round before more prompt rows are filled.
A single prompt call is never split, so one call can pass the cap.
Raising the cap lets more prompt work run per pass.
That delays the decoding streams' next round.
Lowering it gives those streams their next round sooner.
That costs more passes to finish the prompts.
With the default 0 no cap applies: short prompts run as one burst beside decoders, bounded at a lane's worth of newly begun prompts while other work waits.

`TENSORFOLD_PREFILL_POLICY` takes `fcfs`.
Unset or empty keeps the default order described above. `serve` refuses any other text at start.
Under `fcfs` the host fills only the oldest long prompt (more than 512 uncached rows) on each visit, in chunks of `TENSORFOLD_PREFILL_TOKENS` rows, and the decoding streams take one round after each chunk.
A long prompt with no stream decoding beside it takes the engine's own cut, so a lone request runs exactly as without the policy.
Short prompts run as they do without the policy. `TENSORFOLD_PREFILL_BUDGET` does not apply under a policy; `TENSORFOLD_PREFILL_SHARE` is checked once per visit.

`TENSORFOLD_PREFILL_TOKENS` takes a whole number of prompt rows from 64 to 16384. Values outside the range are clamped.
The default is 2048, and empty text or 0 gives the default. `serve` refuses text that is not a whole number at start.
It only matters when `TENSORFOLD_PREFILL_POLICY` is set.
Smaller chunks shorten each pause the decoding streams see while a prompt fills; larger chunks let later prompts start sooner.
On flashnext Metal with eight 14336-row prompts arriving together, `fcfs` at 2048 paused decoders about 1.7 s per chunk with the last first token at about 94 s;
4096 gave about 3.1 s and 87 s; 8192 gave about 5.9 s and 83 s. 16384 matches 8192 at that prompt length.

## Linux and CUDA

For source builds with `zig build native` (including `--prefix` installs): Structured output and required tool calls need the release archive's `lib/` or `zig build grammar -Dxgrammar=DIR`; prepare DIR with `git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar` (use the native build's same `--prefix` for grammar).

Choose `linux-aarch64` for an arm64 host or `linux-x86_64` for an x86_64 host.
The archives require glibc 2.28 or newer and NVIDIA driver R580 or newer (CUDA 13.0), providing `libcuda.so.1`.
Their kernels are built with CUDA 13, so TensorFold refuses an older driver at startup; `nvidia-smi` shows the installed one.
See [qualified models](README.md#qualified-models) for native CUDA serving by family and format.
Nemotron 3.5 Lightning MLX 4-bit has been checked on GB10 (sm_121, greedy and sampled), RTX 3090 (sm_86, greedy)
and RTX PRO 6000 Blackwell (sm_120). A receipt on one sm_86 card does not measure every card of that class.
Flash Next MLX 4-bit has served on a GB10 ([native recipe](docs/recipes/flash-next-cuda.md)).
[GLM-5.3-Flash](docs/recipes/glm-5.3-flash-cuda.md) has recorded native TP=2 and TP=3 runs, including FP4-Local and
XSFP4 on two and three GB10s. GLM uses `TF_GLM_RANK`, `TF_GLM_WORLD` and `NCCL_COMM_ID`; other ranked families use
their supported `--tp` options. Ranks on more than one host need `TF_RANK_SECRET_FILE` set to the same owner-only
key file on each ([SECURITY.md](SECURITY.md#ranks-on-more-than-one-host)). [Gemma 4](CHANGELOG.md#114) 12B and 31B MLX checkpoints have served on a GB10 too.
The largest recorded CUDA server is GLM on three GB10s; GLM refuses more than 16 ranks and `--tp` offers only
1, 2, 4 or 8, so no native family serves 32 GPUs as one server and no run beyond three is recorded.

Release fatbins contain sm_80, sm_86, sm_120 and sm_121 images. The A100 (sm_80) has kernel checks recorded in
[#846](https://github.com/ashhart/TensorFold/pull/846), but no served checkpoint on record, so it is not model-qualified.
Nemotron and dense Qwen's NVFP4/FP8 paths have non-cluster fallbacks for sm_80 to sm_89 (#901, #1470).
No NVFP4/FP8 model run on sm_80 to sm_89 is recorded; implementation and kernel checks are not whole-model qualification.
MiniMax-M3.1-Flash refuses an A100: its CUDA path admits only sm_121 at exactly two ranks.
DeepSeek requires sm_90 or newer and remains refused by the release server pending whole-model gate #827.
Nemotron's `--precision checkpoint` is refused on an A100 because it needs Blackwell (compute capability 12.x).
GLM also refuses an FP8 or MXFP8 projection on sm_80 to sm_89 when its shape needs split-K partial storage that
this family does not hold. CUDA `--slide` learning is admitted only on sm_121.
Qwen4 has synthetic gates only.

On a GB10 arm64 host (an x86_64 host uses `linux-x86_64` in the same commands):

```sh
V=1.2.9
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
request with the cache off. `--prompt-cache-gib` sizes kept states. The default takes half the room after streams,
with a 16 GiB floor on Metal and CUDA device memory, or 8 GiB for unified CUDA host snapshots, clipped to the room.
Discrete CUDA host snapshots default to 8 GiB. Flash Next's batch host keeps a 16 GiB ceiling before sizing sessions.
An explicit value overrides it, and 0 turns them off. A state kept inside a prompt chunk costs no extra prompt call.
A Nemotron 3.5 Lightning state takes 48 MiB plus
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

### Build a server for one GPU

`zig build -Dgpu=<gpu>` builds only `tensorfold-native`, with kernel code for that one GPU architecture and every model family, stripped by default:

```sh
zig build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc            # zig-out/native/bin/tensorfold-native
zig build -Dgpu=gfx1201 -Dhipcc=/opt/rocm/core/bin/hipcc          # zig-out/native-hip/bin/tensorfold-native
```

- `-Dgpu`: `sm_80`, `sm_86`, `sm_89`, `sm_120`, `sm_121`, `spark` (GB10, sm_121), or an exact HIP architecture from `zig/src/hip/caps.zig`. It replaces `-Dsm`, `-Dhip` and `-Dhip-arch`.
- `sm_89` builds, but the release set (sm_80, 86, 120, 121) holds no image qualified on it.
- The server refuses a GPU of another architecture at startup and names the `-Dgpu` that fits it (HIP already refuses one at device open).
- CUDA needs `-Dnvcc`, or `-Dfatbins` with prebuilt images; each prebuilt image must hold exactly the one SASS profile its kernel needs on that GPU. `-Dstrip=false` keeps debug info.
- The build offers no other step: tests, tools and `grammar` need the development build (no `-Dgpu`). Structured output needs `libtfgrammar` in the server's `../lib` (release archive `lib/`, or `zig build grammar -Dxgrammar=DIR`, which installs `native/lib`; copy it to `native-hip/lib` for HIP).

Measured on 2026-11-02 (aarch64 for CUDA, x86_64 for HIP), every model family in every server:

| Hardware GPU | Generic server | `-Dgpu` server | Saved |
| --- | --- | --- | --- |
| RTX PRO 6000 (`sm_120`, nvcc 13.3) | 145.5 MB (sm_80, 86, 120, 121) | 44.9 MB | 100.6 MB (69.2%) |
| GB10 DGX Spark (`spark`, nvcc 13.4) | 146.1 MB (sm_80, 86, 120, 121) | 44.8 MB | 101.3 MB (69.3%) |
| Radeon AI PRO R9700 (`gfx1201`, ROCm 7.16) | 48.9 MB (`native-hip`, unstripped) | 20.6 MB | 28.3 MB (58.0%) |

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

### Memory pressure on a shared Mac

Engines served through the shared lane host, including Flash Next with `--parallel` above one, Qwen3.8-27B and GLM, read the macOS memory pressure level (`kern.memorystatus_vm_pressure_level`).
The engine thread reads it at most four times a second and acts on the level:

- Warning: a queued request starts only when no other stream runs, so the server serves one stream at a time.
  New prompt states are not kept.
  The prompt cache gives back kept states, least valuable first and one per round, until it holds no more than its soft watermark.
  With `--spill-gib` each state is written to disk before it goes.
  A state that a request is restoring is never dropped.
- Critical: as warning, and new requests get 503 with a message saying the Mac is under critical memory pressure.
  Requests already queued or running continue.
- Normal: admission and kept states behave as before.

A rise takes effect at the next reading; a fall waits for five seconds of lower readings, so admission does not flap near a threshold.
Each change of level is logged once.
At normal pressure the server behaves exactly as it does without the governor.
Under pressure only admission and kept states change: every request served returns the same tokens, and a resumed request equals a fresh one.

`TENSORFOLD_PRESSURE_CACHE_PERCENT=N` sets the soft watermark to N% of the prompt cache budget (default 50; 0 sheds every state not being restored).
`TENSORFOLD_MEMORY_PRESSURE=0` (or `off`) disables the governor.
Flash Next's single-reply driver (`--parallel 1` or `auto`) does not read pressure yet, and Windows builds do not.

### Experimental Linux memory pressure

On Linux, `TENSORFOLD_MEMORY_PRESSURE=1` (or `on`) opts the shared lane host into the same governor and cache
watermark. Unset keeps it off until device qualification. The reader samples `/proc/pressure/memory` and
`/proc/meminfo` without allocating; both must be available and valid at startup. It reads at most four times a
second, raises the level immediately and waits five seconds of lower readings before easing it.
Inside a container or cgroup, these proc files describe the whole host, not the container's memory limit.
The opt-in therefore follows host pressure and does not provide container-local pressure or admission limits.

The proposed thresholds use PSI's `avg10` stall percentages:

| Level | Any of these readings |
| --- | --- |
| Warning | PSI some at least 10%; PSI full at least 1% |
| Critical | PSI full at least 10% |

Critical takes precedence. MemAvailable is diagnostic and does not set the level. A normally budgeted unified-memory
server can have a small available margin without reclaim stalls. The OS reader has no engine-specific planned margin,
so it uses PSI rather than guessing a floor from total RAM. MemFree and file-cache size do not set the level either. Unknown or malformed data follows the existing source contract: no reading,
counted as normal by the governor. Windows is unchanged. The 503 message refers to the system on every platform.

Only optional states managed by the shared lane host are shed. Borrowed states remain protected. Family-owned
caches and a backend's independent driver need their own qualification; this does not coordinate pressure across
TP hosts or replace the pre-load page-cache release/admission checks used on GB10 (#1857, #1868).

`zig build test-memory-pressure-host` checks planted proc files, thresholds, invalid/oversized input, the existing
hysteresis, protected state, cache shedding, admission and resumed/fresh equality on an instrumented target.
No Linux device, memory-hog or whole-checkpoint receipt is claimed. Keep the Linux opt-in off until a GB10 run
shows shedding, recovery without oscillation and exact drafted/plain, resumed/fresh and concurrent/solo replies
at unchanged checkpoint precision (#2129).


## Access, context and updates

To accept remote clients, choose `--host 0.0.0.0` and configure `--api-key-file` with a file readable only by its owner.
Clients send `Authorization: Bearer KEY`.
The default loopback server accepts local clients without a key.

The server admits at most 128 open connections (`--max-connections N`, 1 to 1024), including idle keep-alive connections and active replies.
Extra connections receive `503 Service Unavailable` with `Retry-After: 1`, then close.
The reply is best effort if the peer cannot receive it; no extra connection thread is created.
This cap is independent of the process file limit.

From three quarters of that limit open (96 of 128), one remote address may hold at most a quarter of it (32 of 128); its next connection gets the same 503.
Below that, one address may hold any number, so clients behind one address are not throttled on a quiet server.
`--max-connections-per-address N` sets that share (0 turns it off).
IPv4 peers are counted by address, IPv6 peers by their /64.
Loopback clients are never counted, nor the exact addresses given with `--trusted-proxy ADDRESS` (repeat it, up to 64).
Name a reverse proxy, a load balancer, or Docker's bridge gateway there (often `172.17.0.1` with `-p 8080:8080`), since every client behind it shares its address.
With every slot taken, a connection waiting for its next request gives its slot up after 5 seconds.
Below the limit, an idle keep-alive connection is never closed for it, and a request in progress never is.

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
`--prompt-cache-disk DIR` writes every kept prompt state behind to that directory, while the engine idles,
and a SIGTERM stop writes what waits within 20 seconds. It is equivalent to `--learn-conversations --learn-dir DIR`;
the old flags remain supported, and `--learn-conversations` alone keeps its existing default directory.
Use either spelling: combining `--prompt-cache-disk` with `--learn-dir` or `--learn-conversations` is refused.
The startup line names the disk directory and the device budget when the engine reports one.
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
`TENSORFOLD_LOG_TIME=1` starts each server log line, the `[tensorfold]` lines and every engine's warnings, with its UTC time (`2026-10-30T18:54:04.123Z [tensorfold] ...`), to match the log against other records.
