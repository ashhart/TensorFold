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

Running locally needs no token or account. Community contributors can get upload access immediately on [the benchmark board](https://tensorfold.dev/benchmarks), without an email request or administrator approval.

To upload a saved receipt in your browser, choose **Upload result**, enter a public display name, select the JSON receipt and review its hardware, model and timing fields. Confirm the data to submit it; the browser creates upload access automatically. Nothing is sent during the receipt preview.

To publish from the CLI, choose **Run the CLI** in that dialog, enter a display name and choose **Get upload access**. Copy the private token, then run the login command below. Keep the token private and save it before closing the page if you want to reuse it. It allows submissions and management of your own results, with up to 20 uploads per day; it does not grant administration access. Display names are self-chosen and do not verify an identity. Signup limits apply to reduce spam; complete standard runs publish automatically after the server checks them.

Install or update the CLI using the [runbook](../RUNBOOK.md) before running the examples.

```sh
# Paste an issued token at the hidden prompt.
tensorfold benchmark login

# For automation, read the token from a private secret store or file.
tensorfold benchmark login --token-stdin < /secure/upload-token

# Run, show the exact public fields, and ask before publishing. The token is checked before the run starts.
tensorfold benchmark TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit --publish

# Or publish an existing receipt; --yes consents without the prompt.
tensorfold benchmark publish receipt.json --yes

# Remove locally stored upload credentials.
tensorfold benchmark logout
```

`TENSORFOLD_BENCHMARK_TOKEN` can supply a token for a single automation run; it is never saved into a receipt or printed. `--upload-url` selects another HTTPS receiver, and saved credentials are bound to that exact API URL. Redirects are not followed, and uploads and the public check connect directly: proxy variables are not used. Loopback HTTP is accepted for local development only. Before any upload, the receipt is checked with the board's own rules and the checkpoint must be public on Hugging Face.

Complete standard community-v1 submissions are approved and published immediately on [the benchmark board](https://tensorfold.dev/benchmarks), after contributor authentication, daily quota checks, strict receipt validation, public-checkpoint lookup and server-side statistics/completeness checks. A complete standard run includes both workloads at the suite temperatures and seeds, five successful repetitions per cell, 256 output tokens, a managed single-rank execution, a checkpoint revision/configuration digest and reported GPU hardware. The same operation saves the receipt and its approval timestamp, and a successful upload returns the public result URL.

Valid incomplete or nonstandard submissions remain pending review, and invalid submissions are rejected. Retries preserve the original submission status, so this policy does not automatically publish older pending or withdrawn results. The receiver recomputes statistics rather than trusting a client summary. All community measurements remain self-reported; automatic approval does not independently verify speed or hardware.

## AMD (ROCm)

A ROCm build runs the same suite and saves its receipt locally, with backend `rocm` on platform `linux`. `--backend auto` picks `rocm` on that build; an attached run names it with `--backend rocm`. A managed ROCm run records the CPU and system memory, while GPU names and the driver version stay unknown. `inspect` shows a ROCm receipt as usual. The board accepts mlx and cuda receipts for now, so `--publish` with ROCm is refused before the run starts and `benchmark publish` refuses a ROCm receipt before any upload; publishing waits for the board to accept rocm.
