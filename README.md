# TensorFold

TensorFold serves language models from a Zig binary on Apple Silicon and NVIDIA GPUs.
The engine reads checkpoints, tokenizes requests and runs Metal or CUDA kernels directly.
Serving needs no Python or MLX installation.

Building with a coding agent? Have it load the [contributor skill](docs/recipes/SKILL.md) before it writes any code.

## Quick start

Start with Qwen3.5-2B, the small 4-bit checkpoint checked on Metal and a GB10. The commands below download its
pinned revision, then serve it as `local-model`. Model weights have their own licenses.

### Mac with Apple Silicon

Install with Homebrew, then start the server:

```sh
brew install ashhart/tensorfold/tensorfold
tensorfold pull mlx-community/Qwen3.5-2B-MLX-4bit@93760be4f1f69842a46bc13dbdc0f19e291392a3
tensorfold serve mlx-community/Qwen3.5-2B-MLX-4bit --name local-model --context 8192 --no-thinking
```

### Linux with an NVIDIA GPU

Use the release archive for your CPU: `linux-aarch64` on a GB10, `linux-x86_64` on an x86_64 host.
It needs glibc 2.28 or newer and an NVIDIA driver providing `libcuda.so.1`. Keep the archive's directories together.
See the [CUDA runbook](RUNBOOK.md#linux-and-cuda) for runtime libraries and supported GPUs.

```sh
V=1.1.1
case "$(uname -m)" in
  aarch64|arm64) PLATFORM=linux-aarch64 ;;
  x86_64) PLATFORM=linux-x86_64 ;;
  *) echo 'No release archive for this CPU' >&2; exit 1 ;;
esac
curl -fLO "https://github.com/ashhart/TensorFold/releases/download/v$V/tensorfold-$V-$PLATFORM.tar.gz"
curl -fLO "https://github.com/ashhart/TensorFold/releases/download/v$V/tensorfold-$V-$PLATFORM.tar.gz.sha256"
sha256sum -c "tensorfold-$V-$PLATFORM.tar.gz.sha256"
tar -xzf "tensorfold-$V-$PLATFORM.tar.gz"
export PATH="$PWD/tensorfold-$V-$PLATFORM/bin:$PATH"
tensorfold() { command tensorfold-native "$@"; }
tensorfold pull mlx-community/Qwen3.5-2B-MLX-4bit@93760be4f1f69842a46bc13dbdc0f19e291392a3
tensorfold serve mlx-community/Qwen3.5-2B-MLX-4bit --name local-model --context 8192 --no-thinking
```

### Send a request

Leave the server running. In another terminal, on either platform:

```sh
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":128,"temperature":0}'
```

The API also serves completions, Responses and Anthropic Messages. `/v1/models` lists the served name.
On Windows with an NVIDIA GPU, follow [the native Windows build guide](docs/WINDOWS.md).
Use `tensorfold --help`, `tensorfold capabilities --json` and `tensorfold serve REPO --help` to inspect your binary.
You can also serve a complete local checkpoint directory or an existing Hugging Face cache.

## Mac app

The [TensorFold desktop workspace](app/README.md) bundles the local engine, model downloads, hardware recommendations, live generation metrics, coding workers and document editing.
Download the signed and notarized Apple-silicon [beta.5 DMG](https://github.com/ashhart/TensorFold/releases/tag/v1.0.0-beta.5), or build from [`app/`](app/README.md).
Mobile clients in `app/mobile/` remain preview software.

## Living Weights (experimental)

Living Weights teaches a running model new facts. Start a Nemotron 3.5 Lightning server with `--slide`, tell it
something, and a few minutes later it answers from its own weights: each fact it keeps is written into the
checkpoint's files, so it is still there after a restart and on any machine that serves the folder.

```sh
tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
src=$(ls -d ~/.cache/huggingface/hub/models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/snapshots/* | head -1)
mkdir -p ~/models && cp -cRL "$src" ~/models/nemotron-living   # always teach a copy
(umask 077; openssl rand -hex 24 > ~/models/slide-key)          # --slide starts only with a key
tensorfold serve ~/models/nemotron-living --slide --api-key-file ~/models/slide-key
curl -N http://127.0.0.1:8080/v1/slide/learn -H "Authorization: Bearer $(cat ~/models/slide-key)" \
  -H 'Content-Type: application/json' -d '{"text": "I like blue."}'
```

Send it a document instead and it finds the facts in it and learns them one by one. Open
`http://127.0.0.1:8080/slide` to watch in 3D: the facts, the neurons each one took in the model's layers, the shard
files that hold them, and the lesson running now.

Learning runs on Apple-silicon Macs and on an NVIDIA GB10 (DGX Spark), and a folder that learned on one serves on the
other (on Linux, copy with `cp -RL`). It rewrites the folder's files: before the first lesson the server keeps the
originals in a hidden snapshot, and `tensorfold slide rollback ~/models/nemotron-living` puts them back, undoing every
lesson. The [Living Weights guide](docs/living-weights.md) covers the key, teaching files and web pages, omp's `/learn`
command, the 3D view, undoing lessons, and what to expect.

<a id="qualified-models"></a>

## Pick a model

These are the native release's serving paths. A recipe may also describe the older Python 0.6.6 engine;
use its native section and the current [1.1.0/1.1.1 changes](CHANGELOG.md) together.
Hardware below records the documented scope, not an unmeasured minimum. Context, parallel requests, prompt caches
and drafters need memory beyond the weights. `tensorfold info REPO` reports the checkpoint's memory floor.
Where no RAM capacity is published, inspect that floor before serving a large checkpoint.

Run a table command after installation. On a Linux archive installation, use the `tensorfold` shell function from Quick start.
`REPO@revision` pins a pull; serve the repository name without the revision after pulling it.
Conversions are explicitly marked: their upstream weights need the linked conversion before native serving.

| Model / format | Download | Machine and memory scope | Native recipe / limits |
| --- | --- | --- | --- |
| Qwen3.5-2B, MLX 4-bit | `tensorfold pull mlx-community/Qwen3.5-2B-MLX-4bit@93760be4f1f69842a46bc13dbdc0f19e291392a3` | One Mac (M5 Max checked), or one GB10; no RAM minimum published. | [2B](docs/recipes/qwen3.5-2b-native.md); [CUDA release checks](CHANGELOG.md#110). Text only. |
| Qwen3.8-27B, MLX 4-bit | `tensorfold pull TensorFold/Qwen3.8-27B-MLX-4bit`<br>`tensorfold pull z-lab/Qwen3.8-27B-DFlash2` | One Mac (M5 Max/M3 Ultra), or one GB10. | [27B](docs/recipes/qwen3.8-27b.md). Add `--drafter z-lab/Qwen3.8-27B-DFlash2`; without it decoding is plain. |
| Nemotron 3.5 Lightning 30B-A3B, MLX 4-bit | `tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` | One Mac (M1-M5), or one GB10; also checked on RTX 3090 / RTX PRO 6000. | [Lightning](docs/recipes/nemotron-3.5.md). Includes MTP. |
| Lightning, mixed oQ4 / oQ6 / oQ8 | `tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-oQ4`<br>`tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-oQ6`<br>`tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-oQ8` | One M3 Ultra checked. | [Width and MTP instructions](docs/recipes/nemotron-3.5.md#run). Metal only; copy the 4-bit checkpoint's MTP head to draft. |
| Lightning, NVIDIA NVFP4 | `tensorfold pull nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | One GB10; older CUDA cards use the fallback described in the release notes. | [Lightning](docs/recipes/nemotron-3.5.md); [CUDA changes](CHANGELOG.md#110). |
| Nemotron 3.5 Super VL 120B-A12B, NVFP4 | `tensorfold pull nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@82d93699acd3be9e4617c53360fe5b3f5dbeb1b3` | One or two GB10s; weights 70.8 GiB, 73.2 GiB with vision on one. | [One/two-GB10 commands](docs/recipes/nemotron-3.5-super-vl.md). Add `--vision` for images. |
| Nemotron 3.5 Bolt 74B-A9B | Checkpoint ID is not specified in the release's model docs. | Native CUDA; 24 GB checkpoint, checked on RTX PRO 6000. | [Bolt/Omni video](docs/recipes/nemotron-omni-video.md); [release qualification](CHANGELOG.md#110). |
| Qwen3.8 Flash Next, MLX 6-bit | `tensorfold pull TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP` | One M5 Ultra checked; RAM minimum not published. | [Metal](docs/recipes/qwen3.8-flash-next.md), [runbook](RUNBOOK.md#flash-next-and-paired-metal-serving). |
| Qwen3.8 Flash Next, MLX 4-bit | `tensorfold pull TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` | One GB10; 73.3 GiB uploaded, plus mapped tables and caches. | [Native CUDA](docs/recipes/flash-next-cuda.md). NVFP4/two-local-rank and vision paths are experimental. |
| GLM-5.3-Flash, MLX 4-bit | `tensorfold pull TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` | Two M5 Ultras; single-Mac native serving is not qualified. | [Paired Metal runbook](RUNBOOK.md#flash-next-and-paired-metal-serving). Do not use the Python recipe's single-Mac command. |
| GLM-5.3-Flash, EXL3/TR3 | `tensorfold pull Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | Two CUDA ranks; the release checks the two-GB10 setup. | [Release scope](CHANGELOG.md#110), [checkpoint](docs/recipes/glm-5.3-flash.md#exl3). Experimental checkpoint; use native rank options, not the Python installation. |
| GLM-5.3-Flash, NVFP4 | Checkpoint repository is not pinned in the current native recipe. | Two GB10s; shipped-1.1.0 regression load took 302.1 s. | [Known limits](docs/known-issues.md#models): repeated hashes and a 199,307-token passphrase check; scored prompts and more than two ranks unchecked. |
| GLM-5.3, EXLR8 `home` | `tensorfold pull kindlingai/glm-5.3-exlr8-k2-k3-k4@f459f5d0272a529403dd9b3fd32f771f4250a0d4 --exlr8 home --node 0/2 --dir ~/models/glm-5.3-exlr8-home` | Two M5 Ultra Macs, 256 GB each; about 160 GB resident per Mac at 8k context / four streams. | [Two-Mac recipe](docs/glm-5.3-exlr8.md). Use `--node 1/2` on the other Mac; Thunderbolt 5 and MCDMA required. No CUDA backend. |
| LFM2.5-24B-A2B, MLX 4 / 6 / 8-bit | `tensorfold pull LiquidAI/LFM2.5-24B-A2B-MLX-4bit@667408d494bde191c2cb9dd4f4549827816b9275`<br>`tensorfold pull LiquidAI/LFM2.5-24B-A2B-MLX-6bit@1deec53eb49571e21cdcbf82fd5839248f07ed88`<br>`tensorfold pull LiquidAI/LFM2.5-24B-A2B-MLX-8bit@67ab417dd02d0c9545802a726df668620eae049d` | One Mac; M4 Pro checked. | [Native operator recipe](docs/recipes/README.md), [release history](CHANGELOG.md). No separate native model recipe yet. |
| Qwen3.6-35B-A3B, MLX with MTP | `tensorfold pull TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP@81169a9bc511a27c1b4eedb77a2cd98ced431847` | One GB10; 20.9 GB checkpoint. | [Qwen3.6](docs/recipes/qwen3.6-moe.md). Native CUDA; the recipe's Metal path belongs to Python. |
| Gemma 4 31B, MLX 4-bit | `tensorfold pull mlx-community/gemma-4-31B-it-4bit@696d436c404745a59f30e4939a658162b0a9e57f`<br>`tensorfold pull z-lab/gemma-4-31B-it-DFlash@eabd648301ce28583cc14757912e5e0f84e152e1` | One M5 Mac; M5 Ultra checked. M1-M4 refused. | [Native 31B](docs/recipes/gemma-4-native.md). Name the drafter with `--drafter`; images with `--vision`. |
| Gemma 4 12B, converted MLX 4-bit | `hf download google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 --local-dir ~/models/gemma-4-12b-hf`<br>`hf download google/gemma-4-12B-it-assistant --revision 46d4c6f13f0ac0ad827b915669b8df9b81c64c51 --local-dir ~/models/gemma-4-12b-assistant` | One M5 Mac; converted text weights 6.3 GB, image weights 100 MB, plus assistant/caches. M1-M4 refused. | [Convert and serve](docs/recipes/gemma-4-12b-native.md). Upstream download needs `hf`; conversion needs Python, mlx and mlx-lm. Serving is native. |
| Step-5-Preview, converted affine | `hf download stepfun-ai/Step-5-Preview --local-dir ~/models/step5-preview-fp8` | One 256 GB Mac; text 166.9 GiB, MTP adds 2.5 GiB and vision 3.7 GiB. | [Convert and serve](docs/recipes/step-5-preview.md). Upstream FP8 does not fit one Mac; conversion needs Python and mlx. |
| Kolibri-1, block FP8 (research) | `tensorfold pull Aleph-Alpha/Kolibri-1@e52eb4627d11516b0c01de49210ab5a4e4061444` | One NVIDIA GPU; GB10 measured; 74 GB checkpoint. | [Kolibri](docs/recipes/kolibri-1.md). Text only; copy drafts, no EAGLE-3/neural drafter. |
| Liquid AI d1-3B, decisions | Checkpoint ID is not specified in the current model table. | Native CUDA; RAM minimum not published. | [API section below](#serve-flags). `/v1/systemone` and `/v1/decisions`. |
| Fish Audio Drama 3, speech | Checkpoint ID is not specified in the release's model docs. | Native Metal or CUDA; GB10 measured. | [Release audio instructions](CHANGELOG.md#110). `/v1/audio/speech`; checkpoint and `codec.pth` required. |

### GLM-5.3: one machine or two?

The native release does not qualify GLM-5.3 on one machine. GLM-5.3 EXLR8 `home` needs two 256 GB M5 Ultra Macs;
[its recipe](docs/glm-5.3-exlr8.md) gives both pull and serve commands. GLM-5.3-Flash is a different model:
its native Metal setup needs two M5 Ultras, and its native CUDA setup uses two ranks. The single-256-GB-Mac
GLM-5.3-Flash command in the older recipe is for Python 0.6.6, not this release.
For the NVFP4 two-GB10 path, read the [regression's remaining limits](docs/known-issues.md#models).

DeepSeek checkpoints are **not served yet** by any release binary: whole-model gate #827 is still open.
MiMo's native backend is held outside 1.1.1. Bonsai's ternary pack and Gemma 4 26B-A4B remain Python-only;
they are not native download recommendations. [Known issues](docs/known-issues.md) lists current limits,
including Flash Next shared-session cache/performance behavior and the 27B's drafter limits.

## Exact decoding

Every accepted draft must equal the token the same native engine would produce with `"draft": false`.
A resumed request must equal fresh execution, and each concurrent stream must equal its solo run.
The comparison fixes the checkpoint, backend, settings and runtime.
Different quantizations and different backends can produce different outputs.
Flash Next on Metal uses shared target rounds with an explicit `--parallel N` above one, when that many sessions fit.
A lone greedy request there uses GPU-side rounds and hands over when a second request arrives.
`--parallel auto` (the default) and `--parallel 1`, learning and two-Mac serving use the single-reply driver.
`FZ_BATCH_MTP_SERIAL=1` restores serial MTP drafting on the shared host.
The [earlier MTP chain receipt](docs/receipts/flashnext-mtp-chain-2026-10-10/README.md) records its own pinned commits.
Nemotron, GLM, the 2B model and Qwen3.8-27B use shared lane rounds.
The 27B's DFlash2 drafter drafts for one stream at a time: a request alone, or the one in the drafter's slot when several share rounds.

## Models on disk

```sh
tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
tensorfold models
tensorfold info TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
```

`pull` downloads a checkpoint into the Hugging Face cache and checks every file: large files by sha256, small files by their git blob sha1.
Large files come down in pieces over up to 16 connections, and an interrupted pull resumes with the pieces it lacks.
It refuses a model that no native family serves, and fetches the 27B's DFlash2 drafter for `--drafter`. `models` lists the cached checkpoints this binary can serve, and `info` shows one checkpoint's family, format, context and memory floor.
`tensorfold serve REPO` serves a cached checkpoint by its repository name.

## Context compaction

Compaction is off by default, and with it off every reply is unchanged.
With `--compact-at`, a conversation that would overflow is compacted instead of refused.
The server keeps the system prompt and the most recent turns word for word, and never separates a tool call from its result.
The older turns become a memory note: goal, constraints, progress, decisions, next steps, exact names and numbers, and the files the tool calls read or changed.
Each later compaction updates the previous note instead of starting over. A client that sends back the compacted conversation keeps its note.
The reply carries the note and the compaction counts under `tensorfold.compaction`, and every reply reports `context_window` and `context_used`.
The same request gives the same compaction and the same reply. With `--compact-memory`, the stored note carries over between requests.

## Cortheon mode

Off by default. `--cortheon` gives every chat request [Cortheon](https://github.com/ashhart/Cortheon)'s reasoning program and web research run by the server, so any OpenAI-compatible client gets it with no setup.
The model works through Cortheon's stages for its task type and can call `cortheon_search` and `cortheon_fetch`, which the server runs without the client seeing them.
A question about current facts, or a request to build with an outside library, is researched before the model starts.
A coding task that changed files is asked to run its tests before it finishes.
Web text is filtered for instructions aimed at the model, and reads reach public addresses only.
With the flag, search queries written from the conversation go to DuckDuckGo, or to Brave with `BRAVE_SEARCH_API_KEY`, and pages are fetched from this machine; without it, the server makes no searches and reads no pages.
See [Cortheon mode](docs/cortheon.md).

## Model discovery

`GET /v1/models` (also `/models`) advertises each served name and alias. When the engine reports a finite
context window, each entry includes equal `context_length` and `max_model_len` integer extensions. These
are the loaded engine's effective prompt-plus-reply token limit, including any startup memory fitting;
they are not the default reply limit. An unspecified engine window omits both fields. Clients should
still handle per-request memory refusals: an advertised context is not a guarantee that every workload fits.

A model that thinks also carries an OpenRouter-style `reasoning`: `default_enabled`, `default_effort` (only
with `--reasoning-effort`), and `supported_efforts`, the levels its template names, highest first, then `none`
when the template has a thinking switch. A template that names no level leaves `supported_efforts` out. These
describe chat and Responses requests.

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
| `--prompt-seed` | Use native prompt-derived keys instead of the fresh OS fallback; request and configured default seeds still win. |
| `--top-p P`, `--top-k K`, `--min-p P` | Sampling defaults. |
| `--thinking`, `--no-thinking` | Choose whether the chat template opens a reasoning block. |
| `--reasoning-effort LEVEL` | Default template effort: `low`, `medium`, `high` or `xhigh`. |
| `--thinking-budget N` | Limit reasoning tokens where the engine supports closing the reasoning block. |
| `--thinking-reserve N` | With thinking on, let reasoning use up to N tokens beyond a request's `max_tokens`, which then limits the answer (lane engines). |
| `--loop-guard` | Close a short repeated reasoning cycle where the engine supports it. |
| `--no-drafts` | Produce the plain reference through the same engine. |
| `--drafter DRAFTER` | Qwen3.8-27B's DFlash2 drafter, as a directory or a repo id in the Hugging Face cache. Without it the 27B decodes plain. |
| `--drafter-bits 0\|4` | Prepare the drafter's weights in 4 bits (the default) or keep them bf16 with 0. |
| `--record DIR` | Write each stream's committed tokens, positions and tapped rows under DIR, for drafter training. CUDA Nemotron only; a single stream runs through the lane rounds, and recording stops below 25 GiB free disk. |
| `--keep-warm SECONDS` | Keep the Metal GPU active while idle for this long after a request, default 900; zero disables it. |
| `--parallel N` | Admit up to N requests where the engine shares lanes; `auto` is the default. |
| `--prompt-cache-gib GIB` | Retained-prefix budget, 16 GiB by default or less where memory is short; 0 disables it. |
| `--prompt-cache-over-cap` | Permit an explicit prefix budget past what 70% of RAM leaves once the server loads. |
| `--learn` | Keep shared prompt prefixes on disk for new conversations to resume; the models are below the table. |
| `--learn-dir DIR` | Where `--learn` keeps them, `~/.cache/tensorfold/learned` by default; implies `--learn`. |
| `--learn-gib GIB` | Disk for learned prefixes on each Mac, 32 by default; the least recently used go first. Implies `--learn`. |
| `--learn-min-free-gib GIB` | Free disk `--learn` leaves on each Mac, 4 by default; below it learning pauses and serving goes on. |
| `--learn-conversations` | Keep every prompt state on disk too, each conversation's and the shared prefixes, in one store in place of `--learn`'s, so a next turn after a restart, a crash or an eviction reads it back. Implies `--learn`; see [Conversations on disk](docs/api.md#conversations-on-disk). |
| `--learn-daily-gib GIB` | What `--learn-conversations` may write to disk a day, every model together, 32 by default; past it states stay in RAM only. |
| `--recall` | Draft from a record of the token ids earlier requests read and wrote, so repeated text is verified in a few rounds; replies are unchanged. Off by default; see [Recall](docs/recall.md) for what stays on disk. |
| `--recall-dir DIR` | The record's directory, one a tokenizer under `~/.cache/tensorfold/recall` by default; implies `--recall`. |
| `--recall-gib GIB` | Disk for the record, 1/16 GiB by default; the oldest ids go first. Implies `--recall`. |
| `--recall-read-only` | Draft from the record without adding to it, as gates and benchmarks run; implies `--recall`. |
| `--snapshot-dir none` | Keep prefix state in memory; `none` is the supported value. |
| `--max-snapshots 0` | Disable disk snapshots at startup; `0` is the supported value. |
| `--compact-at auto\|FRACTION` | Turn on context compaction (below). `auto` compacts when the prompt and reply would pass the window minus a reserve. |
| `--compact-keep TOKENS` | Recent tokens kept word for word when compacting; the default is 20,000 or a quarter of the window, whichever is smaller. |
| `--compact-memory DIR` | Also keep each conversation's memory note as a Markdown file in DIR. |
| `--cortheon` | Cortheon's reasoning program and server-run web research for chat requests; see [Cortheon mode](docs/cortheon.md). |
| `--no-update-check` | Accepted compatibility switch; update the installed binary through its package manager. |
| `--backend auto` | Use the backend compiled for the platform; `mlx` selects native Metal on macOS and `cuda` selects CUDA on Linux. |

`--prompt-cache-gib` applies to Nemotron, Flash Next, GLM, Qwen3.8-27B and LFM2.5-24B-A2B, and
`--prompt-cache-over-cap` to the same models but GLM.
`--learn` keeps shared prompt prefixes, such as a system prompt and its tools, on disk, so new conversations resume
them after a restart or an upgrade that computes the same bits. It covers GLM-5.3-Flash, Nemotron, and Flash Next on
one Mac; Qwen3.8-27B, Qwen3.5-2B, LFM2.5-24B-A2B, GLM-5.3 (EXLR8) and a CUDA server refuse it at startup.

Sampling defaults come from the checkpoint's `generation_config.json`, then serve flags and request fields override them.
`--context 0` is family-specific; use a positive limit for GLM and inspect the capacity reported at startup.
The supported environment variables are listed in `capabilities --json`, including API keys, request logging and Hugging Face cache paths.

## Community benchmarks

`tensorfold benchmark MODEL` runs public code and chat fixtures on a cached model and saves a portable receipt;
`--publish` optionally submits it for review at tensorfold.dev. See [benchmark commands and measurement definitions](docs/community-benchmarks.md).

## Build and contribute

Use the Zig version pinned in `.zig-version`, currently 0.17.0.
The `.metal` sources need Xcode's Metal compiler. Install full Xcode and its Metal Toolchain component
(`xcodebuild -downloadComponent MetalToolchain`); the Command Line Tools alone fail every kernel compile with
`unable to find utility "metal"`.

```sh
zig build native -Dcpu=apple_m1
zig build test test-golden -Dcpu=apple_m1
```

On macOS, `test` also runs available Metal device checks. See [CONTRIBUTING.md](CONTRIBUTING.md) for the host-test subset.

The server is `zig-out/native/bin/tensorfold-native`.
Release archives include the native executable, runtime assets and license notices; [packaging](packaging/README.md) describes the qualified CUDA inputs.
For cache-aware performance receipts, see [observed cache benchmark evidence](docs/cache-benchmarks.md).
The clients retain server usage and output fingerprints, and verify cold/reused states rather than inferring them.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for exactness, precision and performance gates.
Coding agents should also load [docs/recipes/SKILL.md](docs/recipes/SKILL.md); agents that support skills can install it as `tensorfold-contributor`.
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
[@BadAd84](https://github.com/BadAd84),
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
[@GustavBlack](https://github.com/GustavBlack),
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
[@jrb-ai](https://github.com/jrb-ai),
[@jschmied](https://github.com/jschmied),
[@juliankang4](https://github.com/juliankang4),
[@kelchm](https://github.com/kelchm),
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
[@mrv777](https://github.com/mrv777),
[@MV10](https://github.com/MV10),
[@NeoAiLabs](https://github.com/NeoAiLabs),
[@Nipale-ai](https://github.com/Nipale-ai),
[@nood-co1](https://github.com/nood-co1),
[@nullburn](https://github.com/nullburn),
[@olexale](https://github.com/olexale),
[@omar16100](https://github.com/omar16100),
[@optimisme](https://github.com/optimisme),
[@otarkhan](https://github.com/otarkhan),
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
[@RS91Dev](https://github.com/RS91Dev),
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
[@SvangenStudios](https://github.com/SvangenStudios),
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
[@YvesLaRose](https://github.com/YvesLaRose),
[@ZackSample](https://github.com/ZackSample),
[@zweck](https://github.com/zweck).

[docs/credits.md](docs/credits.md) names whose Python engine work each part of the native engine carries.

[Usable memory profiles](docs/memory-profiling.md) explains how to record applied budgets, context, cache state and process peaks.
