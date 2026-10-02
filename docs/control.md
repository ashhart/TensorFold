# Control room

`tensorfold service` installs a per-user macOS LaunchAgent that runs
`python -m tensorfold serve` with a literal argument list.
`tensorfold tui` is the keyboard dashboard for those jobs and for read-only HTTP endpoints.
A profile plus telemetry is the whole input. No model forward, kernel, sampler, or cache changes.

## Install a service

The model is already cached. `install` writes the profile and an enabled login job.
The name `control-smoke` is reserved. `service install` and the new-service key refuse it.
Pass `--start` when the server should come up immediately. Pull an uncached model with
`tensorfold pull` first. `--allow-download` is the opt-in for a download at service start.
A Hugging Face id works once its files are on disk.

The job lives in the logged-in user's `gui/<uid>` domain. It starts at that user's login
and stops at logout. Root and `sudo` are refused. An SSH session still needs that user's
GUI login domain.

```bash
tensorfold service install "/absolute/path/to/cached-model" \
  --name default --context 32768 --parallel auto
tensorfold service start default
tensorfold service status default
tensorfold service doctor default
tensorfold tui --profile default
```

```bash
tensorfold service list
tensorfold service status default --json
tensorfold service logs default --lines 100 --follow
tensorfold service stop default
tensorfold service restart default
tensorfold service uninstall default --yes
```

`stop` disables the job before unloading it, so login leaves it stopped until `start`.
`start` loads a job that is not currently registered.
`restart` waits until launchd has removed the job, then starts it. A failed wait starts nothing else.
`uninstall` removes the profile and the plist. Models, caches, and logs stay.
It then enables the label, so the launchd override stays enabled.

The profile is the source of truth for the plist. The manager refuses a plist that differs from its profile.
Replace a profile only while it is stopped.

```bash
tensorfold service stop default
tensorfold service install "/absolute/path/to/cached-model" \
  --name default --context 65536 --replace
tensorfold service start default
```

## Configuration

The serving interpreter is an absolute path. A virtualenv symlink stays as written.
No field is evaluated as a shell command.

```bash
tensorfold service install "/cached/model with spaces" --name code \
  --port 8081 --parallel auto --python "$HOME/.venvs/tensorfold-control/bin/python" \
  --env TENSORFOLD_MEMORY_LIMIT_GB=48 \
  --arg=--vision --arg=--spill-gib --arg=20
```

`--port` and `--parallel` are written on the serve command. A non-secret `--env` value such as
`TENSORFOLD_MEMORY_LIMIT_GB` is stored on the profile and passed to that process.
Repeat `--arg` for extra literal `serve` arguments. Managed endpoint flags stay on their own options.
The bind address defaults to `127.0.0.1`. A non-loopback address needs `--allow-network`.
That acknowledgement leaves authentication to a proxy you configure separately.

Credentials belong in a mode `0600` JSON file. The service reads it at start.
The plist and the profile store the path. The values stay in the file.

```bash
chmod 600 "$HOME/.config/tensorfold/service-env.json"
tensorfold service install Org/Already-Cached-Model --name private \
  --env-file "$HOME/.config/tensorfold/service-env.json"
```

The file is a JSON object such as `{"HF_TOKEN": "<your-token>"}`.
Accepted names use the documented engine and Hub prefixes.
The manager rejects a file that is not private, not owned by the user, or reached through a symlink.
A process of the same user can still read its own environment.
The log view redacts common secret shapes. Prompt text needs the engine's own body logging left off.

| Path | Role |
|---|---|
| `~/Library/Application Support/TensorFold/control/profiles/NAME.json` | Private profile |
| `~/Library/LaunchAgents/dev.tensorfold.NAME.plist` | Per-user job, no model credentials |
| `~/Library/Logs/TensorFold/NAME/server.log` | Server and supervisor log |
| `~/Library/Application Support/TensorFold/control/work/NAME/` | Working directory |

Profiles, plists, and logs are user-private. The default log cap is 8 MiB plus four backups.
One record can land slightly past a rollover. Unsuccessful exits wait 30 seconds before launchd tries again.
A clean exit stays down. The runner forwards SIGTERM to the server child and leaves
process-group cleanup to launchd.

## Terminal dashboard

Service commands use the standard library. The dashboard needs `prompt_toolkit` in the same venv.

```bash
python -m pip install 'prompt-toolkit>=3.0.51,<4'
```

The error from `tensorfold tui` prints that command with the venv's Python. `rich` uses the same form.

```bash
tensorfold tui
tensorfold tui --profile code
tensorfold tui --url http://127.0.0.1:8080/v1
tensorfold tui --url https://spark.example.invalid --token-env TENSORFOLD_API_KEY
tensorfold tui --demo
tensorfold tui --demo --snapshot preview.svg
```

Local service changes are macOS-only. A remote URL is monitor-only on every platform.
The token comes from the named environment variable, and the dashboard never stores it.
Requests are GET `/health` and GET `/metrics`, with a size limit, no redirects, no inherited proxies,
and verified TLS. Health polls leave peak counters alone.

| Key | Action |
|---|---|
| `j` / `k`, arrows | Select a profile or endpoint |
| `s` | Start the selected local service |
| `x` / `r` | Stop or restart, after confirmation |
| `n` | Install a cached-model profile and leave it stopped |
| `/` | Command palette |
| `Tab`, `l`, `d` | Change view, logs, overview |
| `f` | Filter logs, literal and case-insensitive |
| `PgUp` / `PgDn` / `End` | Scroll logs, or resume following |
| `Space` | Pause monitoring. Inference keeps running |
| `?`, `Esc` | Help, or close a panel |
| `q`, `Ctrl+C` | Leave the UI. Services keep running |

Pasted text, including newlines, stays in the field. Backspace, Tab, and Ctrl+U edit it.
The confirmation keeps the profile name from the moment you asked.
Selecting another row while that panel is open does not retarget the operation.
The process id on screen is the launchd supervisor. The server child pid is in the log.
Demo mode and a remote endpoint cannot mutate a local service.

The sidebar logo is resampled from the source mark into `logo-pixels.json` at the header width.
Each cell is the upper half block, foreground for the top pixel and background for the bottom pixel, in 24-bit colour.
There is no runtime image decode, and the pixels are not stretched to another size.
The sidebar shows TensorFold when `COLORTERM` is not `truecolor` or `24bit`.
It also shows TensorFold when the header is under 24 columns.
`--color 256`, `--color mono`, and `NO_COLOR` use that wordmark.
The smallest usable size is 72 by 23. At 126 by 32 or larger the overview is complete.
The UI redraws on input, telemetry, and resize. The poll default is one second.

## Metrics

Output and prompt rates are aggregate counter deltas over a rolling window.
The first sample is a baseline. A missing metric is a dash on screen. A real zero is a counter that read zero.
A failure, a long gap, a counter reset, or a new counter source drops the baseline.
Monitoring never sends a generation request.

CUDA `/health` reports generated tokens as they arrive, and `/metrics` counts completed requests.
The dashboard prefers the live counter when that counter is present, and it names the source.
When `/health` carries a `live` object, DECODE TOK/S, PREFILL TOK/S, and CONNECTIONS / WAIT use
`decode_tokens_per_second`, `prefill_tokens_per_second`, `connections`, and `waiting`.
A missing live field, or a value that is not a finite number at least zero, leaves that cell on the rolling counter
or the request gauges. A zero from the server stays a zero.
On MLX a rate can jump when a long request finishes. Prompt totals can include cached work, so a fallback
PREFILL TOK/S includes that cached work. TTFT and draft acceptance count completed requests.
KV is the highest reported pool ratio. Memory is MLX active buffers.
A healthy HTTP response and a loaded launchd job are separate rows.

## Tests

```bash
python -m pytest tests/control -q -m 'not macos'
```

That command uses stand-in launchd replies. It loads no model. A real LaunchAgent check belongs
on a logged-in Mac after this branch is installed.
