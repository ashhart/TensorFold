# Connecting the Linux desktop to a TensorFold engine

The Linux build does not bundle an inference engine. The **TensorFold (built in)** entry in the
model picker reports unavailable there by design: it reports the managed local engine only.
Inference comes from a normal OpenAI-compatible provider, so a TensorFold engine on a Mac (or any
machine) is configured like any other endpoint.

## 1. Serve with a key on the Mac

```sh
printf 'tf-local-key\n' > ~/.tf-key && chmod 600 ~/.tf-key
tensorfold-native serve <checkpoint> --name <model-id> --host 0.0.0.0 --port 8080 --api-key-file ~/.tf-key
```

- `--host` defaults to `127.0.0.1:8080`; a LAN bind needs an explicit address.
- `--name` is the model ID clients use. It defaults to the checkpoint name, which is what
  `GET /v1/models` reports.
- Keep the key file mode 0600 and prefer it to `--api-key`, which is visible in the process list.

## 2. Confirm the endpoint from the Linux machine

```sh
curl -s -H "Authorization: Bearer tf-local-key" http://<host>:8080/v1/models
```

A healthy answer lists the served ID, its context length and whether reasoning is supported.

## 3. Add it as a provider

Settings → Providers → add the **TensorFold** local preset (described as "Any OpenAI-compatible local
or network endpoint."): enter the Mac's address, paste the key, and press **Detect** to list the served
model (`Qwen3.8-27B-MLX-4bit` style IDs). Detection probes the address you typed, so it works against
another machine, not just localhost.

Any OpenAI-compatible preset also works — "vLLM / llama.cpp" accepts the same fields:

| Field | Value |
| --- | --- |
| Base URL | `http://<host>:8080/v1` (the app appends `/chat/completions`, so keep `/v1`) |
| API key | the key from the file |
| Model | the exact ID from `/v1/models` |

## Access choices

**LAN bind** (lowest latency, plaintext HTTP). An external bind skips loopback-name admission, so a
raw IP works as the Host header. Address the server by hostname instead and it needs
`--allowed-host <name>`. Traffic, prompts included, crosses the network unencrypted; use this where
the network is trusted.

**SSH tunnel** (nothing exposed on the LAN):

```sh
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18080:127.0.0.1:8080 <user>@<host>
```

Then use `http://127.0.0.1:18080/v1` in the provider.

## What the app classifies each choice as

`classifyProvider` treats only loopback as `trusted-local`. A LAN address is classified as external
compute, so privacy gates that withhold cloud context (mail, memory, connectors) apply to it the
same way they do to a cloud provider. File and terminal tools are unaffected. A tunnel keeps the
endpoint loopback and therefore `trusted-local`.

## Limits

- A dropped tunnel fails requests mid-stream.
- `run_command` still refuses to run in the Linux build unless the experimental sandbox
  (`TENSORFOLD_LINUX_SANDBOX=1`) or the development-only unsandboxed escape hatch is enabled; see
  [status](STATUS.md).
- The Mac must stay awake; sleep ends the stream.
