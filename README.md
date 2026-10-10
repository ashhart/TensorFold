# TensorFold

TensorFold 1.0.5 serves language models from a Zig binary on Apple Silicon and NVIDIA GPUs.
The engine reads checkpoints, tokenizes requests and runs Metal or CUDA kernels directly.
Serving needs no Python or MLX installation.

## Install and serve

The macOS Homebrew formula installs the native binary as `tensorfold`:

```sh
brew install ashhart/tensorfold/tensorfold
tensorfold --version
tensorfold serve "$HOME/models/nemotron-lightning" \
  --name local-model --parallel 1 --context 8192 --temperature 0 --no-thinking
```

Use a complete local model directory or an existing Hugging Face cache.
The binary in a release archive is `bin/tensorfold-native`; the [runbook](RUNBOOK.md) covers archive installation and CUDA runtime files.
Use `tensorfold-native --help` or `-h`, and `tensorfold-native capabilities --json`, to inspect the installed binary.
Model weights have separate downloads and licenses.

The API listens at `http://127.0.0.1:8080/v1` by default.
It serves OpenAI chat completions, completions and Responses, plus Anthropic Messages.
Use the model ID returned by `/v1/models`, or set one with `--name`.

```sh
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":128,"temperature":0}'
```

## Living Weights (experimental)

Living Weights teaches a running model new facts. Start a Nemotron 3.5 Lightning server with `--slide`, tell it
something, and a few minutes later it answers from its own weights: each fact it keeps is written into the
checkpoint's files, so it is still there after a restart and on any machine that serves the folder.

```sh
tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
src=$(ls -d ~/.cache/huggingface/hub/models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/snapshots/* | head -1)
mkdir -p ~/models && cp -cRL "$src" ~/models/nemotron-living   # always teach a copy
tensorfold serve ~/models/nemotron-living --slide
curl -N http://127.0.0.1:8080/v1/slide/learn -H 'Content-Type: application/json' -d '{"text": "I like blue."}'
```

Learning runs on Apple-silicon Macs and on an NVIDIA GB10 (DGX Spark), and a folder that learned on one serves on the
other (on Linux, copy with `cp -RL`). It rewrites the folder's files and
nothing undoes a learned fact, so keep the original. The [Living Weights guide](docs/living-weights.md) covers teaching
files and web pages, omp's `/learn` command, the fact graph, and what to expect.

## Qualified models

Only these model and platform combinations are admitted to 1.0.5.
The platform column names the hardware tested for each model.

| Model | Checkpoint format | Qualified platform |
| --- | --- | --- |
| Nemotron 3.5 Lightning 30B-A3B | MLX affine 4-bit, group 64, included MTP head | Metal on M1 through M5; CUDA on GB10 and RTX 3090 |
| Nemotron 3.5 Lightning 30B-A3B, NVIDIA's NVFP4 | ModelOpt NVFP4 experts and head, FP8 Mamba projections, bf16 attention | CUDA on GB10 (sm_90 or newer) |
| Qwen3.8 Flash Next | MLX affine 6-bit, group 32 | Metal on M5 Ultra |
| GLM-5.3-Flash | MLX affine 4-bit, group 64 | Metal on two M5 Ultras |
| Qwen3.5-2B | Pinned MLX affine 4-bit, group 64, tied embeddings | Metal on M5 Max |
| Qwen3.8-27B | MLX affine 4-bit, group 64, own output head; DFlash2 drafter | Metal on M5 Max and M3 Ultra |

Nemotron's named checkpoint is `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit`.
NVIDIA's NVFP4 checkpoint, `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, serves on CUDA as stored; its bf16 MTP head is quantized to 4-bit at load for drafts.
The 2B checkpoint is `mlx-community/Qwen3.5-2B-MLX-4bit`, revision `93760be4f1f69842a46bc13dbdc0f19e291392a3`.
Flash Next loads its checkpoint directly and builds its weight packs locally, without a recorded kernel directory.
GLM's two-Mac setup uses one settings file per rank and a separate MCDMA runtime.
The [1.0.0 release notes](RELEASE-NOTES-1.0.0.md) give qualification limits and credit the contributors.

The 27B's checkpoint is `TensorFold/Qwen3.8-27B-MLX-4bit`, and its drafter is `z-lab/Qwen3.8-27B-DFlash2`, given with `--drafter`.
Run it with `--drafter`. Without it the 27B decodes plain, which is slower than mlx_lm.
Its prompt processing trails mlx_lm for now, and we are fixing it. An M3 Ultra reads prompts at about half of mlx_lm's speed from 8k to 30k tokens, and served cold prefill on an M5 Max also measured below mlx_lm.
Bonsai, Gemma 4, Qwen3.6 and DeepSeek-V4 are still under qualification for 1.0.x.
The Python 0.6.6 engine remains on the `python-0.6` line for those models and other backends.
On CUDA, 1.0.5 serves Nemotron on a GB10, greedy and sampled, with concurrent requests sharing each round.
The Linux x86_64 archive adds NVIDIA Ampere cards with compute capability 8.6 (the RTX 30 series, RTX A6000, A10 and A40), tested on an RTX 3090; the A100 (8.0) is not supported yet.

## Exact decoding

Every accepted draft must equal the token the same native engine would produce with `"draft": false`.
A resumed request must equal fresh execution, and each concurrent stream must equal its solo run.
The comparison fixes the checkpoint, backend, settings and runtime.
Different quantizations and different backends can produce different outputs.
Flash Next and Qwen3.8-27B run one active reply per engine; Nemotron, GLM and the 2B model use shared lane rounds.

## Models on disk

```sh
tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
tensorfold models
tensorfold info TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
```

`pull` downloads a checkpoint into the Hugging Face cache and checks every file: large files by sha256, small files by their git blob sha1.
Large files come down in pieces over up to 16 connections, and an interrupted pull resumes with the pieces it lacks.
It refuses a model that no 1.0.5 family serves, and fetches the 27B's DFlash2 drafter for `--drafter`. `models` lists the cached checkpoints 1.0.5 can serve, and `info` shows one checkpoint's family, format, context and memory floor.
`tensorfold serve REPO` serves a cached checkpoint by its repository name.

## Context compaction

Compaction is off by default, and with it off every reply is unchanged.
With `--compact-at`, a conversation that would overflow is compacted instead of refused.
The server keeps the system prompt and the most recent turns word for word, and never separates a tool call from its result.
The older turns become a memory note: goal, constraints, progress, decisions, next steps, exact names and numbers, and the files the tool calls read or changed.
Each later compaction updates the previous note instead of starting over. A client that sends back the compacted conversation keeps its note.
The reply carries the note and the compaction counts under `tensorfold.compaction`, and every reply reports `context_window` and `context_used`.
The same request gives the same compaction and the same reply. With `--compact-memory`, the stored note carries over between requests.

## Model discovery

`GET /v1/models` (also `/models`) advertises each served name and alias. When the engine reports a finite
context window, each entry includes equal `context_length` and `max_model_len` integer extensions. These
are the loaded engine's effective prompt-plus-reply token limit, including any startup memory fitting;
they are not the default reply limit. An unspecified engine window omits both fields. Clients should
still handle per-request memory refusals: an advertised context is not a guarantee that every workload fits.

## Serve flags

The read-only [`/memory` diagnostics](docs/memory-diagnostics.md) report process footprint and lifetime peak,
available backend allocation counters, and the applied retained-prefix plan without changing memory policy.

The binary's `capabilities --json` response lists its supported flags and platform-specific values.
`serve MODEL --help` prints usage without loading a model.

| Flag | Meaning |
| --- | --- |
| `--host HOST`, `--port PORT` | Listen address and port, default `127.0.0.1:8080`. |
| `--name NAME` | Model ID advertised to clients. |
| `--alias NAME` | Another accepted model ID; repeat for multiple aliases. |
| `--api-key KEY` | Require a bearer key; repeat for multiple keys. |
| `--api-key-file FILE` | Read keys from a restricted-permission file. |
| `--metrics-open` | Allow `/metrics` without a key when API authentication is enabled. |
| `--dashboard` | Enable the local `/dashboard` page and `/stats` endpoint. |
| `--context N` | Bound prompt plus reply tokens; the model's window and memory checks still apply. Qwen3.8-27B defaults to 32,768 and takes up to 262,144. |
| `--speed-up FILE` | Rank and link settings for two-Mac Flash Next or GLM serving. |
| `--max-tokens N` | Default reply limit, 4096; requests can override it. |
| `--temperature T` | Sampling temperature; zero requests greedy decoding. |
| `--top-p P`, `--top-k K`, `--min-p P` | Sampling defaults. |
| `--thinking`, `--no-thinking` | Choose whether the chat template opens a reasoning block. |
| `--reasoning-effort LEVEL` | Default template effort: `low`, `medium`, `high` or `xhigh`. |
| `--thinking-budget N` | Limit reasoning tokens where the engine supports closing the reasoning block. |
| `--loop-guard` | Close a short repeated reasoning cycle where the engine supports it. |
| `--no-drafts` | Produce the plain reference through the same engine. |
| `--drafter DRAFTER` | Qwen3.8-27B's DFlash2 drafter, as a directory or a repo id in the Hugging Face cache. Without it the 27B decodes plain. |
| `--drafter-bits 0\|4` | Prepare the drafter's weights in 4 bits (the default) or keep them bf16 with 0. |
| `--keep-warm SECONDS` | Keep the Metal GPU active while idle for this long after a request, default 900; zero disables it. |
| `--parallel N` | Admit up to N requests where the engine shares lanes; `auto` is the default. |
| `--prompt-cache-gib GIB` | Nemotron, Flash Next, GLM and Qwen3.8-27B retained-prefix budget; zero disables retention. |
| `--prompt-cache-over-cap` | Permit an explicit Nemotron or Flash Next prefix budget above its default allowance. |
| `--learn` | Keep shared prompt prefixes, such as a system prompt and its tools, on disk so new conversations resume them after a restart or an upgrade that computes the same bits. GLM and Nemotron. |
| `--learn-dir DIR` | Where `--learn` keeps them, `~/.cache/tensorfold/learned` by default; implies `--learn`. |
| `--learn-gib GIB` | Disk for learned prefixes on each Mac, 32 by default; the least recently used go first. Implies `--learn`. |
| `--learn-min-free-gib GIB` | Free disk `--learn` leaves on each Mac, 4 by default; below it learning pauses and serving goes on. |
| `--snapshot-dir none` | Keep prefix state in memory; `none` is the supported value. |
| `--max-snapshots 0` | Disable disk snapshots at startup; `0` is the supported value. |
| `--compact-at auto\|FRACTION` | Turn on context compaction (below). `auto` compacts when the prompt and reply would pass the window minus a reserve. |
| `--compact-keep TOKENS` | Recent tokens kept word for word when compacting; the default is 20,000 or a quarter of the window, whichever is smaller. |
| `--compact-memory DIR` | Also keep each conversation's memory note as a Markdown file in DIR. |
| `--no-update-check` | Accepted compatibility switch; update the installed binary through its package manager. |
| `--backend auto` | Use the backend compiled for the platform; `mlx` selects native Metal on macOS and `cuda` selects CUDA on Linux. |

Sampling defaults come from the checkpoint's `generation_config.json`, then serve flags and request fields override them.
`--context 0` is family-specific; use a positive limit for GLM and inspect the capacity reported at startup.
The supported environment variables are listed in `capabilities --json`, including API keys, request logging and Hugging Face cache paths.

## Build and contribute

Use the Zig version pinned in `.zig-version`, currently 0.17.0.
On a Mac with Xcode's Metal toolchain:

```sh
zig build native -Dcpu=apple_m1 -Dversion=1.0.5
zig build test test-golden -Dcpu=apple_m1
```

The server is `zig-out/native/bin/tensorfold-native`.
Release archives include the native executable, runtime assets and license notices; [packaging](packaging/README.md) describes the qualified CUDA inputs.
For cache-aware performance receipts, see [observed cache benchmark evidence](docs/cache-benchmarks.md).
The clients retain server usage and output fingerprints, and verify cold/reused states rather than inferring them.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for exactness, precision and performance gates.
TensorFold is Apache-2.0; see [LICENSE](LICENSE), [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contributors

Thank you to everyone who has sent TensorFold a pull request, a measurement or a bug report:

[@0mao0](https://github.com/0mao0),
[@321sssrt-bit](https://github.com/321sssrt-bit),
[@aditya1503](https://github.com/aditya1503),
[@AdrianBinDC](https://github.com/AdrianBinDC),
[@akol1](https://github.com/akol1),
[@Alexbob0](https://github.com/Alexbob0),
[@anvilsong](https://github.com/anvilsong),
[@Arminova](https://github.com/Arminova),
[@b-ostrov](https://github.com/b-ostrov),
[@barelyworkingcode](https://github.com/barelyworkingcode),
[@Benjamin-Wegener](https://github.com/Benjamin-Wegener),
[@benthecarman](https://github.com/benthecarman),
[@benwilson](https://github.com/benwilson),
[@BHCC2025](https://github.com/BHCC2025),
[@Bizuayeu](https://github.com/Bizuayeu),
[@BlivionIaG](https://github.com/BlivionIaG),
[@BobClawblaw](https://github.com/BobClawblaw),
[@borodach23](https://github.com/borodach23),
[@Boscoeuk](https://github.com/Boscoeuk),
[@boxabirds](https://github.com/boxabirds),
[@brandonmmusic-max](https://github.com/brandonmmusic-max),
[@bunnyfu](https://github.com/bunnyfu),
[@CerebralCoding](https://github.com/CerebralCoding),
[@cesarswong](https://github.com/cesarswong),
[@chadhurley25075-png](https://github.com/chadhurley25075-png),
[@chaog992](https://github.com/chaog992),
[@Charlie-Louis](https://github.com/Charlie-Louis),
[@Chedrian07](https://github.com/Chedrian07),
[@chris247474](https://github.com/chris247474),
[@christrade215](https://github.com/christrade215),
[@crescit](https://github.com/crescit),
[@cshintov](https://github.com/cshintov),
[@cwschroeder](https://github.com/cwschroeder),
[@DakotaTexas](https://github.com/DakotaTexas),
[@danyo1399](https://github.com/danyo1399),
[@Deesha08](https://github.com/Deesha08),
[@Defilan](https://github.com/Defilan),
[@DevRico003](https://github.com/DevRico003),
[@di37](https://github.com/di37),
[@drowzeys](https://github.com/drowzeys),
[@ecohash-co](https://github.com/ecohash-co),
[@edurdias](https://github.com/edurdias),
[@eleqtrizit](https://github.com/eleqtrizit),
[@ericlsimplifi](https://github.com/ericlsimplifi),
[@EugeneClaw](https://github.com/EugeneClaw),
[@feni6](https://github.com/feni6),
[@gbgbgbg](https://github.com/gbgbgbg),
[@GDACONSULT](https://github.com/GDACONSULT),
[@gecobattya](https://github.com/gecobattya),
[@gilby](https://github.com/gilby),
[@Gogo6969](https://github.com/Gogo6969),
[@gprot42](https://github.com/gprot42),
[@GraithSecurity](https://github.com/GraithSecurity),
[@grantoverton](https://github.com/grantoverton),
[@grearjake-star](https://github.com/grearjake-star),
[@greatyingzi](https://github.com/greatyingzi),
[@harrisonfriia](https://github.com/harrisonfriia),
[@haxudev](https://github.com/haxudev),
[@heitke](https://github.com/heitke),
[@hichaiuse](https://github.com/hichaiuse),
[@ivanfioravanti](https://github.com/ivanfioravanti),
[@jasontitus](https://github.com/jasontitus),
[@jayleaton](https://github.com/jayleaton),
[@jeffpeng3](https://github.com/jeffpeng3),
[@jeidbugs404](https://github.com/jeidbugs404),
[@jetnet](https://github.com/jetnet),
[@jkuepker](https://github.com/jkuepker),
[@johnymoo](https://github.com/johnymoo),
[@JordiPosthumus](https://github.com/JordiPosthumus),
[@JRaxworthy](https://github.com/JRaxworthy),
[@jregan-beasley-smc](https://github.com/jregan-beasley-smc),
[@jschmied](https://github.com/jschmied),
[@juliankang4](https://github.com/juliankang4),
[@kingjamez](https://github.com/kingjamez),
[@kky42](https://github.com/kky42),
[@lcgutierrez](https://github.com/lcgutierrez),
[@LECYWZA](https://github.com/LECYWZA),
[@lijian1999](https://github.com/lijian1999),
[@liumorrisclaw](https://github.com/liumorrisclaw),
[@LXD-8](https://github.com/LXD-8),
[@m-naoki-m](https://github.com/m-naoki-m),
[@mapamalu](https://github.com/mapamalu),
[@mcclanahanaman](https://github.com/mcclanahanaman),
[@MESevenJourney](https://github.com/MESevenJourney),
[@mgoldwasser](https://github.com/mgoldwasser),
[@MiaAI-Lab](https://github.com/MiaAI-Lab),
[@mikolaj92](https://github.com/mikolaj92),
[@millaguie](https://github.com/millaguie),
[@Mirrdhyn](https://github.com/Mirrdhyn),
[@Moutonc](https://github.com/Moutonc),
[@MovieMaker93](https://github.com/MovieMaker93),
[@mrpmorris](https://github.com/mrpmorris),
[@MV10](https://github.com/MV10),
[@NeoAiLabs](https://github.com/NeoAiLabs),
[@Nipale-ai](https://github.com/Nipale-ai),
[@nood-co1](https://github.com/nood-co1),
[@nullburn](https://github.com/nullburn),
[@olexale](https://github.com/olexale),
[@omar16100](https://github.com/omar16100),
[@optimisme](https://github.com/optimisme),
[@outcastofmusic](https://github.com/outcastofmusic),
[@paragontasx](https://github.com/paragontasx),
[@peacockesq](https://github.com/peacockesq),
[@philip-pentatonic](https://github.com/philip-pentatonic),
[@plotarmordev](https://github.com/plotarmordev),
[@pmeenan](https://github.com/pmeenan),
[@pulseandthread](https://github.com/pulseandthread),
[@quigles1977](https://github.com/quigles1977),
[@rafafortes](https://github.com/rafafortes),
[@raymondkpwong](https://github.com/raymondkpwong),
[@robertpitt](https://github.com/robertpitt),
[@RoscoeTT](https://github.com/RoscoeTT),
[@salmanarshad321](https://github.com/salmanarshad321),
[@samwang0041-star](https://github.com/samwang0041-star),
[@sanjaibalajee](https://github.com/sanjaibalajee),
[@satindergrewal](https://github.com/satindergrewal),
[@scottleimroth](https://github.com/scottleimroth),
[@sethforprivacy](https://github.com/sethforprivacy),
[@sfxnz](https://github.com/sfxnz),
[@shantanugoel](https://github.com/shantanugoel),
[@simon-lin88](https://github.com/simon-lin88),
[@simonmd](https://github.com/simonmd),
[@spenchey](https://github.com/spenchey),
[@squarrier](https://github.com/squarrier),
[@ss-cong](https://github.com/ss-cong),
[@styles01](https://github.com/styles01),
[@SxMShaDoW](https://github.com/SxMShaDoW),
[@sxuff](https://github.com/sxuff),
[@taussoe](https://github.com/taussoe),
[@tfolkman](https://github.com/tfolkman),
[@ThinkOffApp](https://github.com/ThinkOffApp),
[@Thotheris](https://github.com/Thotheris),
[@tinyapps](https://github.com/tinyapps),
[@tolewis](https://github.com/tolewis),
[@tomByrer](https://github.com/tomByrer),
[@tonydehnke](https://github.com/tonydehnke),
[@tournierjc](https://github.com/tournierjc),
[@tpischke](https://github.com/tpischke),
[@urtho](https://github.com/urtho),
[@vcruz305](https://github.com/vcruz305),
[@vinicius-symetrix](https://github.com/vinicius-symetrix),
[@wojo](https://github.com/wojo),
[@xjqx2z](https://github.com/xjqx2z),
[@Yuepixel](https://github.com/Yuepixel),
[@YvesLaRose](https://github.com/YvesLaRose).

[Usable memory profiles](docs/memory-profiling.md) explains how to record applied budgets, context, cache state and process peaks.
