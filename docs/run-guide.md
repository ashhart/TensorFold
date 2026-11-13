# Run guide

How to install, start and tune a TensorFold server after the [quick start](../README.md#quick-start).
Which checkpoints run on which machines is in [supported models](supported-models.md).

## Platform setup

The quick start serves Qwen3.5-2B, the small 4-bit checkpoint checked on Metal and a GB10. Its commands download its
pinned revision, then serve it as `local-model`. Model weights have their own licenses.
Reviewing or benchmarking TensorFold? [Evaluating TensorFold](evaluating.md) covers pinning, signature checks,
a container and the tests that need no GPU.

- **Mac with Apple Silicon:** install with Homebrew (`brew install ashhart/tensorfold/tensorfold`). The
  [runbook](../RUNBOOK.md#macos-installation) covers macOS installation in more detail.
- **Linux with an NVIDIA GPU:** use the release archive for your CPU: `linux-aarch64` on a GB10, `linux-x86_64` on an x86_64 host.
  It needs glibc 2.28 or newer and NVIDIA driver R580 or newer (CUDA 13.0), providing `libcuda.so.1`. Keep the archive's directories together.
  Its kernels are built with CUDA 13, so TensorFold refuses an older driver at startup; `nvidia-smi` shows the installed one.
  See the [CUDA runbook](../RUNBOOK.md#linux-and-cuda) for runtime libraries and supported GPUs.
  The archive's binary is `tensorfold-native`; the quick start defines a `tensorfold` shell function that calls it.
- **Windows with an NVIDIA GPU:** follow [the native Windows build guide](WINDOWS.md).

## Send a request

Leave the server running. In another terminal, on either platform:

```sh
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":128,"temperature":0}'
```

The API also serves completions, Responses and Anthropic Messages. `/v1/models` lists the served name.
Use `tensorfold --help`, `tensorfold capabilities --json` and `tensorfold serve REPO --help` to inspect your binary.
You can also serve a complete local checkpoint directory or an existing Hugging Face cache.

## TensorFold Web

`tensorfold web [ARGS...]` opens TensorFold Web, a small local web app for picking, running and chatting with models.
If it needs installing, an interactive release binary offers to install its matching version with `uv` after you say yes.
See [the web launcher](web.md) for manual installation and unattended use.

## Parallel requests

`--parallel N` caps active requests at N; extra requests wait for a free slot.
Omitting the flag uses `--parallel auto`, which asks for up to eight streams as memory and the engine allow.
An explicit number is refused if the engine cannot fit that many streams.
Read the engine's startup capacity line, such as `8 streams at once` on CUDA, for the count it accepted.
With API authentication off, `/health` reports `live.connections` as active plus waiting requests and
`live.waiting` as the queued count. For example, 8 connections with 4 waiting means 4 active requests.
Completing or cancelling an active request frees its slot; cancelling a waiting request removes it from the queue.

## Exact decoding

Every accepted draft must equal the token the same native engine would produce with `"draft": false`.
A resumed request must equal fresh execution, and each concurrent stream must equal its solo run.
The comparison fixes the checkpoint, backend, settings and runtime.
Different quantizations and different backends can produce different outputs.
Flash Next on Metal uses shared target rounds with an explicit `--parallel N` above one, when that many sessions fit.
A lone greedy request there uses GPU-side rounds and hands over when a second request arrives.
`--parallel auto` (the default) and `--parallel 1`, learning and two-Mac serving use the single-reply driver.
`FZ_BATCH_MTP_SERIAL=1` restores serial MTP drafting on the shared host.
The [earlier MTP chain receipt](receipts/flashnext-mtp-chain-2026-10-10/README.md) records its own pinned commits.
Nemotron, GLM, the 2B model and Qwen3.8-27B use shared lane rounds.
On Metal, each of up to 128 streams in the 27B's shared rounds drafts its own DFlash2 tree.

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
See [Cortheon mode](cortheon.md).

## Serve flags

The read-only [`/memory` diagnostics](memory-diagnostics.md) report process footprint and lifetime peak,
available backend allocation counters, and the applied retained-prefix plan without changing memory policy.

The binary's `capabilities --json` response lists its supported flags and platform-specific values.
`serve MODEL --help` prints usage without loading a model.

| Flag | Meaning |
| --- | --- |
| `--host HOST`, `--port PORT` | Listen address and port, default `127.0.0.1:8080`. |
| `--unix-socket PATH` | Listen on a new owner-only Unix socket at PATH instead of TCP, refused with `--host` or `--port`; macOS and Linux. An existing PATH, or a directory other users can write to (unless sticky, like `/tmp`), is refused, and connections from other users are closed. API keys, `--metrics-open` and the Host check apply as on TCP (a client sends a loopback `Host` or one in `--allowed-host`): the socket's permissions are a second layer, not a substitute for keys. |
| `--profile stable` | Explicitly preserve current defaults; [profiles](profiles.md) describes the initial switch registry and startup report. |
| `--config FILE.toml` | Read serve flags from a TOML file: a key is a flag without its `--`, and `model` names the model. The command line wins over the file, and a relative path in the file starts at the file's folder. See [serve config files](serve-config.md). |
| `--name NAME` | Model ID advertised to clients. |
| `--alias NAME` | Another accepted model ID; repeat for multiple aliases. |
| `--api-key KEY` | Require a bearer key; repeat for multiple keys. |
| `--api-key-file FILE` | Read keys from a restricted-permission file; a key can be limited to [some routes](api.md#key-scopes). |
| `--metrics-open` | Allow `/metrics`, `/v1/metrics` and `/v1/load` without a key when API authentication is enabled. |
| `--dashboard` | Enable the local `/dashboard` page and `/stats` endpoint. |
| `--max-inflight N`, `--max-inflight-per-key N` | Refuse model requests past N in progress at once, overall or per key, with 429; 0, the default, is off. See [admission](api.md#admission). |
| `--max-connections N`, `--max-connections-per-address N`, `--trusted-proxy ADDRESS` | Open connections at most (default 128); from three quarters full, one address's share (default a quarter, 0 off); addresses never counted for that share, as loopback is not, such as a reverse proxy or Docker's bridge gateway. See [admission](api.md#admission). |
| `--requests-per-second N`, `--request-burst N` | Refuse a key's model requests past N a second, after a burst; 0, the default, is off. |
| `--context N` | Bound prompt plus reply tokens; the model's window and memory checks still apply. Qwen3.8-27B defaults to 32,768 and takes up to 262,144. |
| `--rope-yarn F` | YaRN context extension with this factor over the checkpoint's trained window, for Qwen3.5 and 3.6 on a CUDA build; other builds refuse the flag. |
| `--loops N` | Passes over a looped model's recurrent block, from 1; without it the checkpoint's own count runs. Other builds refuse the flag. |
| `--speed-up FILE` | Rank and link settings for two-Mac Flash Next or GLM serving. |
| `--max-tokens N` | Default reply limit, 4096; requests can override it. |
| `--temperature T` | Sampling temperature; zero requests greedy decoding. |
| `--prompt-seed` | Use native prompt-derived keys instead of the fresh OS fallback; request and configured default seeds still win. |
| `--top-p P`, `--top-k K`, `--min-p P` | Sampling defaults. |
| `--instruct-sampling SET` | Sampling defaults for sampled chat requests with thinking off: `card` (the Qwen 3.5, 3.6 and 3.8 cards' temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5), `name=value` pairs over it, or `none`; the serve flags still win. Greedy requests, raw completions and the server's own requests never take them. |
| `--thinking`, `--no-thinking` | Choose whether the chat template opens a reasoning block. |
| `--reasoning-effort LEVEL` | Default template effort: `minimal`, `low`, `medium`, `high`, `xhigh` or `max`. |
| `--thinking-budget N` | Limit reasoning tokens where the engine supports closing the reasoning block. |
| `--thinking-reserve N` | With thinking on, let reasoning use up to N tokens beyond a request's `max_tokens`, which then limits the answer (lane engines). |
| `--loop-guard` | Close reasoning that repeats itself verbatim (a short cycle, or a long passage) where the engine supports it. |
| `--loop-guard-stop` | As `--loop-guard`, but end the reply where the guard fires: the reasoning so far, no answer or tool calls, `finish_reason` `content_filter` and a `reasoning_loop` error ([API](api.md)). |
| `--no-drafts` | Produce the plain reference through the same engine. |
| `--drafter DRAFTER` | An external drafter, as a directory or a repo id in the Hugging Face cache, for models that take one: Qwen3.8-27B uses DFlash2 and Gemma 4 31B the drafter named in its recipe. Without one they decode plain. |
| `--drafter-bits 0\|4` | Drafter weight precision where the model offers a choice: 4 bits (the default), or 0 to keep bf16. Qwen3.8-27B on CUDA takes 4 only. |
| `--record DIR` | Write each stream's committed tokens, positions and tapped rows under DIR, for drafter training. CUDA Nemotron only; a single stream runs through the lane rounds, and recording stops below 25 GiB free disk. |
| `--keep-warm SECONDS` | Keep the Metal GPU active while idle for this long after a request, default 900; zero disables it. |
| `--cfg-gamma GAMMA` | Metal Drama speech: semantic guidance strength, finite 1–5; 0/default disables it. Only reference-conditioned requests are guided. Enabled replies report `X-Cfg-Gamma`; codec heads stay conditional. |
| `--cfg-allow-request-gamma` | Metal Drama speech: allow a finite 1–5 `X-Cfg-Gamma` request header; invalid values return 400 before engine submission. Without this opt-in, the server's gamma is fixed. |
| `--cfg-force-guided` | With an enabled Metal Drama guidance policy, exercise the unconditional state even at gamma 1. The semantic mix at 1 remains exact conditional-logit identity. |
| `--cfg-batched` | With enabled Metal Drama guidance, submit both independent backbone states in one command buffer rather than separate submissions. Weights are shared; extra state is included in admission. Not vectorized batching. |
| `--silence-floor-dbfs DBFS` | Drama speech only: explicit frame-power floor for the silence guard, with no default floor. Calibrate below quiet intended speech using public or built-in speech. |
| `--silence-seconds SECONDS` | Drama speech only: stop after this continuous duration below the explicit floor, retaining audio and reporting `silence`. Unset by default: off; calibrate above intentional pauses. |
| `--silence-onset-s SECONDS` | Drama speech only: with the silence guard on, stop once this many seconds run below its floor while the synthesis has voiced less than `--silence-onset-voiced-s`, retaining audio and reporting `silence`. Off unless given; must be below `--silence-seconds`. |
| `--silence-onset-voiced-s SECONDS` | Drama speech only: the voiced seconds below which `--silence-onset-s` applies; required with it, with no default. Calibrate on public or built-in speech. |
| `--silence-mid-s SECONDS` | Drama speech only: with the silence guard on, finish as `silence` when end of speech falls in a run this long below the floor, after the explicit voiced minimum. Whole replies carry the decimal `X-Silence-Mid-Trips` count, including zero. Must be below `--silence-seconds` and paired with `--silence-mid-min-voiced-s`. Unset by default: off. |
| `--silence-mid-min-voiced-s SECONDS` | Required with `--silence-mid-s`: positive finite cumulative voiced duration before the mid rule applies, including equality. No default; calibrate with the other silence thresholds. |
| `--silence-redraws N` | Drama speech only: draw a whole reply the silence guard ended again, up to N times, as seed salt + k or explicit request seed + k; N of 1 or more needs the guard. Unset by default: off, as is 0; streams are never redrawn. See [speech redraws](speech-redraws.md). |
| `--silence-redraw-budget-s SECONDS` | Start a redraw only while elapsed time plus the last take's duration fits; a request's `X-Deadline-S` can lower it. No default budget. |
| `--silence-redraw-controls LIST` | Redraws speak the input filtered to the listed controls, as `--allowed-controls` filters; an empty list removes every control. |
| `--short-reply-ratio R` | Also redraw a whole reply that finished `stop` with less audio than R (above 0, below 1) of its text's expected length; needs `--silence-redraws` and `--short-reply-chars-per-s`. No default. |
| `--short-reply-chars-per-s CPS` | Input characters per second of audio in complete replies of the served voice, setting the expected length; needs `--short-reply-ratio`. No default; calibrate on your own voice. |
| `--register-redraw-st ST` | Drama speech only: draw a whole reply once more when a phrase after the first opens ST semitones above the other phrases' median opening; needs the two flags below. Unset by default: off, with no default for any of the three. See [speech redraws](speech-redraws.md#register). |
| `--register-redraw-high PERCENT` | The share of that phrase's voiced frames that must lie above `--register-high-hz`. |
| `--register-high-hz HZ` | The pitch line, 75 to 600 Hz, that marks a high register; calibrate it on your own voice. |
| `--parallel N\|auto` | Cap active requests at N and queue the rest. Default `auto` asks for up to 8 as memory and the engine allow; an explicit count that cannot fit is refused. Read the accepted count at startup. |
| `--prompt-cache-gib GIB` | Retained-prefix budget; by default half the room left after streams on Metal and unified-memory CUDA (16 GiB floor where room permits), 8 GiB host snapshots on discrete CUDA, 16 GiB for Flash Next's batch host. An explicit value overrides it; 0 disables it. |
| `--prompt-cache-over-cap` | Permit an explicit prefix budget past what 70% of RAM leaves once the server loads. |
| `--learn` | Keep shared prompt prefixes on disk for new conversations to resume; the models are below the table. |
| `--learn-dir DIR` | Where `--learn` keeps them, `~/.cache/tensorfold/learned` by default; implies `--learn`. |
| `--learn-gib GIB` | Disk for learned prefixes on each Mac, 32 by default; the least recently used go first. Implies `--learn`. |
| `--learn-min-free-gib GIB` | Free disk `--learn` leaves on each Mac, 4 by default; below it learning pauses and serving goes on. |
| `--learn-conversations` | Keep every prompt state on disk too, each conversation's and the shared prefixes, in one store in place of `--learn`'s, so a next turn after a restart, a crash or an eviction reads it back. Implies `--learn`; see [Conversations on disk](api.md#conversations-on-disk). |
| `--learn-daily-gib GIB` | What `--learn-conversations` may write to disk a day, every model together, 32 by default; past it states stay in RAM only. |
| `--recall` | Draft from a record of the token ids earlier requests read and wrote, so repeated text is verified in a few rounds; replies are unchanged. Off by default; see [Recall](recall.md) for what stays on disk. |
| `--recall-dir DIR` | The record's directory, one a tokenizer under `~/.cache/tensorfold/recall` by default; implies `--recall`. |
| `--recall-gib GIB` | Disk for the record, 1/16 GiB by default; the oldest ids go first. Implies `--recall`. |
| `--recall-read-only` | Draft from the record without adding to it, as gates and benchmarks run; implies `--recall`. |
| `--snapshot-dir none` | Keep prefix state in memory; `none` is the supported value. |
| `--max-snapshots 0` | Disable disk snapshots at startup; `0` is the supported value. |
| `--compact-at auto\|FRACTION` | Turn on [context compaction](#context-compaction). `auto` compacts when the prompt and reply would pass the window minus a reserve. |
| `--compact-keep TOKENS` | Recent tokens kept word for word when compacting; the default is 20,000 or a quarter of the window, whichever is smaller. |
| `--compact-memory DIR` | Also keep each conversation's memory note as a Markdown file in DIR. |
| `--cortheon` | Cortheon's reasoning program and server-run web research for chat requests; see [Cortheon mode](cortheon.md). |
| `--no-update-check` | Accepted compatibility switch; update the installed binary through its package manager. |
| `--backend auto` | Use the backend compiled for the platform; `mlx` selects native Metal on macOS and `cuda` selects CUDA on Linux. |

`--prompt-cache-gib` applies to Nemotron, Flash Next, GLM, Qwen3.8-27B and LFM2.5-24B-A2B, and
`--prompt-cache-over-cap` to the same models but GLM.
`--learn` keeps shared prompt prefixes, such as a system prompt and its tools, on disk, so new conversations resume
them after a restart or an upgrade that computes the same bits. It covers GLM-5.3-Flash, Nemotron, and Flash Next on
one Mac; Qwen3.8-27B, Qwen3.5-2B, LFM2.5-24B-A2B, GLM-5.3 (EXLR8) and a CUDA server refuse it at startup.
Qwen3.8-27B on Metal also resumes a follow-up turn from the state its last reply left in the server's first slot,
when the prompt cache holds nothing as far: the turn reads only its new tokens, even on a Mac whose cache cannot keep
a long conversation. Where the cache cannot keep a whole context, the server also keeps that slot's DeltaNet states
at each prompt's history end, so a turn whose template drops the last reply's reasoning resumes there. Replies are
the same tokens as a fresh prompt pass; `TF_QWEN27_SLOT_RESUME=0` turns both off.

Sampling defaults come from the checkpoint's `generation_config.json`, then serve flags and request fields override them.
`--context 0` is family-specific; use a positive limit for GLM and inspect the capacity reported at startup.
The supported environment variables are listed in `capabilities --json`, including API keys, request logging and Hugging Face cache paths.
`TF_Q27_PROFILE=N` makes a native Metal Qwen3.8-27B server log a per-class GPU-time breakdown every N decode rounds (`1` means 64) and add `gpu_seconds` to its replies; it is diagnostic only, and a profiled round runs slower.

## Other ways to run

### Mac app

The [TensorFold desktop workspace](../app/README.md) bundles the local engine, model downloads, hardware recommendations, live generation metrics, coding workers and document editing.
Download the Apple-silicon [beta 9 DMG](https://github.com/ashhart/TensorFold/releases/tag/v1.0.0-beta.9), or build from [`app/`](../app/README.md).
The desktop app launches the bundled `tensorfold-native` executable directly.
Mobile clients in `app/mobile/` remain preview software.

### TensorFold Starlight (enterprise)

[TensorFold Starlight](https://github.com/ashhart/tf-starlight) gives an organisation one governed OpenAI- and Anthropic-compatible endpoint for a fleet of TensorFold engines on Macs, Linux GPU servers and Windows machines.
It signs people in through the organisation's identity provider (OIDC or SAML, with SCIM provisioning), decides who may use which model and where it runs, places and supervises models on nodes, and accounts for every token.
It runs fully offline, or brokers approved frontier providers under the same identity, policy and budgets; [the install guide](https://github.com/ashhart/tf-starlight/blob/main/docs/install.md) covers Docker, Kubernetes and Linux services.
Starlight lives in its own repository, [ashhart/tf-starlight](https://github.com/ashhart/tf-starlight): it pins the TensorFold release it is tested with and checks every flag its nodes pass to `tensorfold serve` against that release in CI.
Starlight is new: the gates that still need real GPU clusters or identity-provider tenants are listed in [its qualification notes](https://github.com/ashhart/tf-starlight/blob/main/docs/qualification.md).
