# Community benchmarks

Run repeatable code and chat workloads on a model you have already downloaded:

```sh
tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
tensorfold benchmark TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
```

The examples call the command `tensorfold`. A source build's binary is `zig-out/native/bin/tensorfold-native`, and it takes the same arguments.

The command starts its own single-request `serve` on a loopback port, disables prompt reuse, warms each workload, and measures five repeats at sampled and greedy settings. It stops only the server it started, and a Ctrl-C or a closed terminal stops that server too; a second Ctrl-C exits at once and kills it. The default output limit is 256 tokens. The checkpoint must already be in the cache: `tensorfold benchmark` never downloads, so run `tensorfold pull` first. `--drafter DIR_OR_REPO` passes a drafter to engines that take one. Drafts stay on by default, and only `--serial` disables them. The default `--drafter auto` passes the target's own drafter when it is already cached (`z-lab/Qwen3.8-27B-DFlash2` for Qwen3.8-27B) and names it in the run output and the receipt; when it is not cached, the run passes none and says why. `--drafter none` passes none. When `--serial` overrides a given `--drafter`, the run prints a line saying so.

An engine that decodes greedily only refuses the sampled cells, so it cannot complete community-v1; the run prints the server's reason, and `--temperatures 0` gives a local run. A large checkpoint may need its engine's own load limit raised (`GLM_LOAD_LIMIT_GB` for GLM-5.3-Flash) and a longer `--startup-timeout`: without the flag, the run waits up to 600 s plus 6 s per GB of weights past 20 GB, at most 3600 s.

Results are saved locally in `~/.tensorfold/benchmarks/RUN_ID.json`; `--output FILE` chooses another location. Files contain public hardware and model metadata and individual attempts, never prompt text, generated output, machine names, serial numbers, GPU UUIDs, addresses or local model paths. Local model directories can be benchmarked but cannot be published, and a drafter directory is never named. Managed runs record `python_version` as `"none"`, because no Python runs in the client or the server; attached runs record `"unknown"`.

## Commands

```sh
# Inspect a saved receipt without uploading it.
tensorfold benchmark inspect receipt.json

# Show the immutable community-v1 fixture and settings manifest.
tensorfold benchmark suites

# A short custom run, outside the standard five-repeat comparison.
tensorfold benchmark TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit --reps 1 --tokens 64 --temperatures 0

# Attach to a server you started first, in another terminal. Give the public checkpoint ID it serves,
# and its --name as --model-id.
tensorfold serve example-org/public-checkpoint --name local-model
tensorfold benchmark example-org/public-checkpoint --server http://127.0.0.1:8080 --model-id local-model --backend mlx
```

Attached-server results do not claim the client's hardware is the server's hardware. They carry unknown hardware, versions and checkpoint revisions and remain unranked. An attached run records the backend, model and context you declare, unchecked against the server. The runner does not clear or restart an attached server's caches.

The runner records failures, incomplete streams, missing usage, early EOS and cancellations. Exit code 0 means every measured attempt completed its requested output with a delivery interval; 1 means an error, and with `--publish` the receipt is still saved when the run finished; 2 means incomplete attempts remain in the saved receipt; 130 means the run was cancelled with Ctrl-C: an attempt in flight is kept in the receipt, and a cancel during server startup saves nothing. SIGTERM and SIGHUP exit 143 and 129 without a receipt. Errors never silently disappear from an average.

## What the rates mean

**Stream delivery tok/s** measures `(completion_tokens - 1) / (last_nonempty_delivery - first_nonempty_delivery)` at the HTTP client. A delivery can contain several tokens, so this is not an engine token clock or a GPU compute rate. Reasoning deltas count as output. A reply buffered into one delivery has no measurable delivery span.

**Server-reported tok/s** is kept separately when the server supplies it and is never substituted for stream delivery. First-text latency, request duration and server-reported prefill time remain separate measurements.

The suite fixes prompts, seeds, sampled top-k and top-p, and thinking settings. Successful warm-ups are excluded; failed warm-ups are kept and make their cells incomplete. A cell has aggregates only when every declared repetition succeeds. Early EOS is visible and excluded from full-length comparisons.

Receipts include token hashes supplied by the server, but the command does not run paired serial and drafted equality tests. A token hash alone is not an exactness proof.

## Publishing

The browser upload flow needs no login, pasted token, email request or administrator approval. Install or update the CLI using the [runbook](../RUNBOOK.md), run a cached public model locally, and save its receipt:

```sh
tensorfold benchmark TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit --output benchmark.json
```

Open [the benchmark board](https://tensorfold.dev/benchmarks), choose **Upload result**, enter a public display name and select `benchmark.json`. Review the hardware, model and timing fields, then confirm the data to submit it. Receipt contents are not sent before confirmation. The website creates a scoped browser profile privately and remembers it across reloads using a Secure, HttpOnly, SameSite cookie; there is no upload key to copy or paste.

That browser profile keeps later submissions together, with up to 20 uploads per day. Clearing browser data or using another browser creates a separate profile. Display names are self-chosen and do not verify an identity; matching names do not merge profiles. New-profile limits apply to reduce spam.

Complete standard community-v1 submissions publish automatically after quota checks, strict receipt validation, public-checkpoint lookup and server-side statistics/completeness checks. A complete standard run includes both workloads at the suite temperatures and seeds, five successful repetitions per cell, 256 output tokens, a managed single-rank execution, a checkpoint revision/configuration digest and reported GPU hardware. A published submission returns its public result URL.

If the website's model-checkpoint service is temporarily unavailable, a valid receipt is saved privately and its public-checkpoint check is retried automatically. A complete standard run becomes public only after that check passes. You do not need to rerun the benchmark or upload the same receipt again. Private or inaccessible checkpoints are never published automatically. Valid incomplete or nonstandard runs remain pending review after public-checkpoint verification; invalid receipts are rejected.

Retries return the same saved submission without using another upload. Older manual-review submissions and withdrawn results are not automatically republished. The receiver recomputes statistics rather than trusting a client summary. All community measurements remain self-reported; automatic approval does not independently verify speed or hardware.

### Existing CLI publishing credentials

CLI installations that already have scoped upload credentials can continue to publish directly:

```sh
# Run, show the public fields, and ask before publishing.
tensorfold benchmark TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit --publish

# Publish an existing receipt; --yes consents without the prompt.
tensorfold benchmark publish receipt.json --yes

# Remove locally stored CLI upload credentials.
tensorfold benchmark logout
```

Direct CLI publishing still requires its existing credentials; it does not use the browser cookie. New contributors can use the local-run and browser-upload flow above without credentials. If a CLI publication check fails, the saved receipt can be uploaded through the website without rerunning the model.

For existing automation, `tensorfold benchmark login` reads an issued token at a hidden prompt, and `--token-stdin` reads it from a private secret store or file. `TENSORFOLD_BENCHMARK_TOKEN` can supply a token for a single automation run; it is never saved into a receipt or printed. Saved credentials are bound to the exact HTTPS API URL selected by `--upload-url`. Redirects are not followed, and CLI uploads and public checks connect directly without proxy variables. Loopback HTTP is accepted for local development only.

## Contributor profiles and leaderboards

Click a contributor name on [the benchmark dashboard](https://tensorfold.dev/benchmarks) to see their public benchmark history, model and hardware counts, personal best and earned milestones. Profiles include every approved receipt through pagination, not only the newest page. Profile links can be shared or opened directly. A contributor ID names an upload identity; self-chosen display names are not verified, and two identities with the same name remain separate.

The performance map plots the visible protocol-complete greedy-chat medians by first-text latency and stream-delivery speed. Each point opens its receipt, with keyboard access; the sample count describes only the loaded page, and loading more adds older results. Model and hardware filters update the map and standings.

**Most benchmarks** ranks contributors by their number of approved receipts. **Fastest run** ranks each contributor by their best protocol-complete greedy-chat stream-delivery median, using the same five-repeat, 256-token workload as the run board. The model, platform and hardware filters apply to both rankings and community counters. Different models and hardware have different limits; filter the view before comparing setups. All results remain community-reported.

The profile achievement cabinet awards bronze, silver, gold and platinum trophies for published-run milestones at 1, 5, 10 and 100 runs, plus model and hardware exploration. Top contributor honours go to the highest global approved-run count; speed leader honours go to the highest global protocol-complete greedy-chat median. Awards use the unrounded global values, equal leaders share honours and missing speeds earn no speed trophy. The profile progress bar shows the next published-run milestone. Pending and withdrawn submissions do not earn public standings or milestones, and public profile IDs grant no submission or administration access.

## AMD (ROCm)

A ROCm build runs the same suite and saves its receipt locally, with backend `rocm` on platform `linux`. `--backend auto` picks `rocm` on that build; an attached run names it with `--backend rocm`. A managed ROCm run records the CPU and system memory, while GPU names and the driver version stay unknown. `inspect` shows a ROCm receipt as usual. The board accepts mlx and cuda receipts for now, so `--publish` with ROCm is refused before the run starts and `benchmark publish` refuses a ROCm receipt before any upload; publishing waits for the board to accept rocm.
