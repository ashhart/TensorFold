# Use TensorFold with OpenClaw

The [openclaw-tensorfold](https://github.com/ashhart/openclaw-tensorfold) plugin makes TensorFold a model provider in
[OpenClaw](https://github.com/openclaw/openclaw). OpenClaw starts `tensorfold serve` when a request needs it and stops
it when OpenClaw exits or the server sits idle; it can also connect to a server you run yourself. The plugin is on
ClawHub and npm (Apache-2.0) and needs OpenClaw 2026.11.3 or later.

- **Models:** the provider lists what the server's `/v1/models` reports, not a list written by hand. Each model gets
  its context window, image input when the server reports vision, and the reasoning efforts its chat template names.
- **Thinking:** OpenClaw's thinking levels reach the server as `reasoning_effort`. Off sends `none` when the template
  can switch thinking off, and sends nothing when it can't.

The plugin registers one provider and no tools, hooks or background services; starting and stopping the server is
OpenClaw's own [local model service](https://docs.openclaw.ai/gateway/local-model-services). It changes nothing in the
engine's requests or tokens.

## Let OpenClaw start TensorFold

```sh
brew install ashhart/tensorfold/tensorfold
openclaw plugins install clawhub:openclaw-tensorfold
openclaw onboard --auth-choice tensorfold-local
```

Use a checkpoint already qualified for your platform, following its [model recipe](recipes/). Setup asks for the model
(checkpoints already in the Hugging Face cache are listed first; a folder or a Hugging Face id also works), the context
window (65,536 tokens by default) and whether to stop the server after an idle time. It starts TensorFold once to read
the model's window and reasoning efforts, stops it, and writes `models.providers.tensorfold.localService`.

After that, the first request for the model starts `tensorfold serve` on `127.0.0.1` (a free port from 18180 up) and
waits for `/health`, which answers once the model has loaded. The server and OpenClaw share one key, kept in
`tensorfold/server.keys` (mode 0600) under OpenClaw's state folder, and the server uses the
[key-file format](api.md#api-keys) with an `inference,read` scope.

The plugin looks for `tensorfold` on `PATH` and in Homebrew's folders; `TENSORFOLD_BINARY` names another build by its
absolute path. Unattended setup reads `TENSORFOLD_MODEL`, and optionally `TENSORFOLD_CONTEXT` and
`TENSORFOLD_IDLE_STOP_MINUTES`:

```sh
TENSORFOLD_MODEL=TensorFold/Qwen3.8-27B-MLX-4bit openclaw onboard --non-interactive --accept-risk --mode local \
  --skip-health --auth-choice tensorfold-local
```

## Connect to a server you run

```sh
tensorfold serve /path/to/checkpoint --name my-model --host 127.0.0.1 --port 8080 --api-key-file /path/to/server.keys
export TENSORFOLD_API_KEY=<a key from server.keys>
openclaw onboard --auth-choice tensorfold-api-key
openclaw models list --provider tensorfold --refresh
```

A key scoped to `inference` is enough: chat and `/v1/models` are both inference routes. A server without keys works
too, but OpenClaw still wants a key to register the provider, so give it any value. Onboarding saves the key and makes
the first served model the default. The plugin uses `http://127.0.0.1:8080/v1` unless `TENSORFOLD_BASE_URL` or
`models.providers.tensorfold.baseUrl` names another address; a bare `http://host:8080` gets `/v1` added. A model list
written under `models.providers.tensorfold.models` is left as it is, and the plugin then skips discovery.

## Thinking levels

| What `/v1/models` reports | What OpenClaw sends |
|---|---|
| `reasoning.supported_efforts`, such as Qwen3.8's `low`, `medium`, `xhigh` and `none` | the chosen level as `reasoning_effort`, rounded down to one the template names; off sends `none` |
| a thinking switch without levels, such as Nemotron 3.5 | off sends `none`; any other level sends nothing, so the template's default applies |
| reasoning that can't be switched off | nothing |

## Checked

On an Apple M5 Max (Metal) with the TensorFold 1.2.8 release archive and OpenClaw 2026.11.3:

| Check | Result |
|---|---|
| Unattended `onboard --auth-choice tensorfold-local`, Nemotron 3.5 Lightning 30B-A3B | 17 s: started and stopped TensorFold once, wrote the service with the 65,536-token window |
| Agent turn with no server running: write a script, run it, report the output | OpenClaw started TensorFold (ready in 3.0 s), the right output, the server stopped when OpenClaw exited; 20 s |
| Install from ClawHub, and from npm, into an empty OpenClaw | loaded, provider `tensorfold` |
| Connecting to a running server: an agent turn with tools | written, run, the right output, 14 s |
| `--thinking off` and `high` on Nemotron | thinking off, then the template's default |
| `--thinking low`, `medium`, `xhigh` and `off` on Qwen3.8-27B | the server heard `low`, `medium`, `xhigh`, then thinking off |

Not run yet: the idle stop inside a long-running OpenClaw Gateway, and a CUDA server.
