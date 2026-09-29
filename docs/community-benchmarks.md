# Community benchmarks

Run repeatable code and chat workloads on a model you have already downloaded:

```sh
tensorfold benchmark Vontra/Qwen3.8-27B-MLX-4bit
```

The command starts its own single-request server on a loopback port, disables prompt reuse, warms each workload, and measures five repeats at sampled and greedy settings. It stops only the server it started. The default output limit is 256 tokens. Missing weights are refused unless you pass `--download`; `--drafter REPO` selects a drafter, and `--serial` disables drafting.

Results are saved locally in `~/.tensorfold/benchmarks/RUN_ID.json`. `--output FILE` chooses another location. Files contain public hardware/model metadata and individual attempts, never prompt text, generated output, machine names, serial numbers, GPU UUIDs, addresses or local model paths. Local model directories can be benchmarked but cannot be published.

## Commands

```sh
# Explicitly allow a missing checkpoint to download.
tensorfold benchmark Vontra/Qwen3.8-27B-MLX-4bit --download

# Inspect a saved receipt without uploading it.
tensorfold benchmark inspect receipt.json

# Show the immutable community-v1 fixture/settings manifest.
tensorfold benchmark suites

# A short custom run, outside the standard five-repeat comparison.
tensorfold benchmark Vontra/Qwen3.8-27B-MLX-4bit --reps 1 --tokens 64 --temperatures 0

# Attach to a separately started server, including a multi-rank CUDA server.
tensorfold benchmark example-org/public-checkpoint --server http://127.0.0.1:8080 --model-id local-model --backend cuda
```

Attached-server results do not claim the client's hardware is the server's hardware. They carry unknown hardware and checkpoint revisions and remain unranked. The runner does not clear or restart an attached server's caches. Managed execution currently supports one rank; start required multi-rank engines separately and use `--server`.

The runner records failures, incomplete streams, missing usage, early EOS and cancellations. Exit code 0 means every measured attempt completed its requested output with a delivery interval; 2 means incomplete attempts remain in the saved receipt; 130 means the run was cancelled. Errors never silently disappear from an average.

## What the rates mean

**Stream delivery tok/s** measures `(completion_tokens - 1) / (last_nonempty_delivery - first_nonempty_delivery)` at the HTTP client. A delivery can contain multiple tokens, so this is not an individual engine token clock or GPU compute rate. Reasoning deltas count as output when a server provides them. A response buffered into one delivery has no measurable delivery span.

**Server-reported tok/s** is kept separately when TensorFold supplies it. Its definitions can differ between backends and versions; it is not silently substituted for stream delivery. First-text latency, request duration and server-reported prefill time remain separate measurements. Server prefill time is not asserted to be pure GPU prefill service time.

The published suite fixes prompts, seeds, sampled top-k/top-p and thinking settings. Successful warm-ups are excluded; failed warm-ups are retained and make their cells incomplete. A cell has aggregates only when every declared repetition succeeds. Early EOS is visible and excluded from full-length comparisons; the command does not force unsupported `ignore_eos` behavior.

Receipts include token hashes supplied by the server, but the current command does not run paired serial/drafted equality tests. A token hash alone is not an exactness proof. Run separate serial comparisons when making exactness claims.

## Publishing

The initial receiver uses administrator-issued contributor tokens. It does not claim that a chosen display name is a verified GitHub identity. Tokens are scoped to benchmark submissions, stored only as hashes by the receiver, and can be revoked. Running locally needs no token or account.

```sh
# Paste an issued token at the hidden prompt.
tensorfold benchmark login

# For automation, read the token from a private secret store or file.
tensorfold benchmark login --token-stdin < /secure/upload-token

# Run, show the exact public fields, and ask before publishing.
tensorfold benchmark Vontra/Qwen3.8-27B-MLX-4bit --publish

# Or publish an existing receipt.
tensorfold benchmark publish receipt.json

# After reviewing the allowlisted receipt, automation can explicitly consent.
tensorfold benchmark publish receipt.json --yes

# Remove locally stored upload credentials.
tensorfold benchmark logout
```

`TENSORFOLD_BENCHMARK_TOKEN` can supply a token for a single automation run. It is never saved into a receipt or printed. `--upload-url` selects another HTTPS receiver; saved credentials are bound to that exact API URL. Redirects are not followed with credentials. Loopback HTTP is accepted for local development only.

New submissions are pending moderation. Retrying the same run and data is idempotent; a changed receipt with the same run ID is refused. Only approved submissions appear on [the benchmark board](https://tensorfold.dev/benchmarks). The receiver recomputes statistics rather than trusting a client summary. All community measurements remain self-reported, including those marked protocol-complete.

The first board is a preview for this benchmark branch, not a release measurement claim. There is no upload telemetry during normal model serving. Website-generated JSON uploads follow the same strict schema and authentication rules as the CLI.

## Receiver development

The Worker implementation, schema migrations and administration commands are under [community/worker](../community/worker/README.md). Its default deployment uses SQLite-backed Durable Object storage with `BENCHMARKS_STORE`, an administrator secret configured in Cloudflare, and the site's existing static assets binding. A D1 binding named `BENCHMARKS_DB` is supported by the same receiver handler as an alternative. Keep those secrets out of the repository.

Model-free coverage exercises SSE framing/buffering/reasoning, token counts, EOS/errors, managed lifecycle/cancellation, safe metadata, receipt validation, upload consent and authentication, idempotency and public moderation boundaries. Real hardware qualification should retain the exact checkpoint revision, runtime and complete receipt; tests using simulated HTTP streams are not GPU performance evidence.
