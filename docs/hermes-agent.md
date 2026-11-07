# Use TensorFold with Hermes Agent

The community [hermes-tensorfold](https://github.com/atakann/hermes-tensorfold) plugin makes TensorFold a model
provider in [Hermes Agent](https://github.com/NousResearch/hermes-agent) and adds a TensorFold tab to Hermes' web
dashboard. It implements the Hermes part of [#1399](https://github.com/ashhart/TensorFold/issues/1399), tracked in
[#2109](https://github.com/ashhart/TensorFold/issues/2109).

- **Provider:** "TensorFold" appears in `hermes model` and the in-chat model picker. The models come from the server's
  `/v1/models`, and each model's context window from the same list.
- **Dashboard tab:** the served model, decode and prefill speed, running and waiting streams, token totals, MTP
  acceptance and memory, read from `/health`, `/stats` and `/v1/models` every 2 s. The key stays on the dashboard's
  side; the browser never sees it.

The plugin needs Hermes Agent 0.23.0 or later. It changes nothing in the engine's requests or tokens.

## Start the native server

Use a checkpoint already qualified for your platform, following its [model recipe](recipes/). Keep the server on
loopback with an API key:

```sh
tensorfold serve /path/to/checkpoint --host 127.0.0.1 --port 8080 --api-key-file /path/to/server.keys --dashboard
```

Follow the [key-file format and permissions](api.md#api-keys). A scoped key needs `inference,read`: chat uses the
inference routes, and the dashboard tab reads `/stats`. `--dashboard` serves `/stats`; inference works without it.
Hermes must run on the same host as TensorFold, or reach it through a loopback forward such as
`ssh -N -L 8080:127.0.0.1:8080 gpu-box`.

## Install and pick the model

```sh
hermes plugins install atakann/hermes-tensorfold
hermes plugins enable tensorfold
echo 'TENSORFOLD_API_KEY=<a key from server.keys>' >> ~/.hermes/.env
hermes model
```

Choose TensorFold, then the served model. The default endpoint is `http://127.0.0.1:8080/v1`; set
`TENSORFOLD_BASE_URL` for another port. `hermes plugins enable` is what loads the dashboard tab: Hermes loads a user
plugin's dashboard routes only when the plugin is in `plugins.enabled`.

## Tool calls

Hermes' agent loop sends no `tool_choice`, so the server's `auto` applies to tool calls. Some auxiliary requests can
ask for `required`, which the server enforces with `libtfgrammar`. The release archives include it in `lib/`; a source
build without it answers such a request with HTTP 400 and names the library.

## Checked

On a DGX Spark (GB10) with TensorFold 1.2.3 serving `nvidia/Qwen3.8-Flash-Next-FP4-Local` (`--context 131072
--parallel 2 --dashboard`), and Hermes Agent 0.23.0 on a Mac reaching it through a loopback forward:

| Check | Result |
|---|---|
| `hermes plugins validate` (the catalog's admission check) | passed, security scan safe |
| First run on an empty Hermes home: install, enable, one question | answered |
| Model picker | "TensorFold (1 model)", the served model, a 131,072-token window |
| Streamed tool use: write a file and run it | written, run, the right output |
| A long reply interrupted mid-stream | the server shows no running or waiting stream; the next request answers |
| The server stopped, then started again | Hermes reports the server unavailable after its retries, then answers once it is back |
| Dashboard tab against `/stats` | the same prompt, generated and round totals |

On an M1 Max (64 GB) with the macOS release archive serving `TensorFold/Qwen3.8-27B-MLX-4bit` on loopback with a
scoped `inference,read` key, the same checks passed, with the endpoint and key only in Hermes' `.env`.

Not run yet: the M3 Ultra run the release checklist asks for.
