# Use TensorFold with OpenClaw

The community [TensorFold OpenClaw plugin](https://github.com/garvrao80/tensorfold-openclaw)
adds model discovery and a server-status page inside OpenClaw. It implements the OpenClaw portion of
[#1399](https://github.com/ashhart/TensorFold/issues/1399); Hermes remains separate work.

**Development preview:** the plugin targets OpenClaw 2026.10.9 and is not yet published to a plugin registry.
Real-agent Mac and GB10 qualification is pending. The plugin's
[release checklist](https://github.com/garvrao80/tensorfold-openclaw/blob/main/docs/release-checklist.md)
records the required runs; fixture tests are not hardware inference receipts.

## Start the native server

Use a checkpoint already qualified for your platform, following its [model recipe](recipes/).
Keep the server on loopback with an API key:

```sh
tensorfold serve /path/to/checkpoint --host 127.0.0.1 --port 8080 --api-key-file /path/to/server.keys --dashboard
```

Follow the [key-file format and permissions](api.md#api-keys). If the key is scoped, grant
`inference,read`: inference routes and the read-only dashboard both need access.
`--dashboard` enables `/stats`; it is optional for inference. The OpenClaw Gateway must run in the same
host/network namespace as TensorFold, or use a locally arranged loopback forward.

## Install and authenticate

The preview pins OpenClaw **2026.10.9**, whose plugin APIs are experimental. Use its supported Node
version: 24.22.1 or later in the 24.x line, or 26.12.1 or later.
Follow the plugin's [source installation guide](https://github.com/garvrao80/tensorfold-openclaw#install-from-source)
to build, check, pack, and install the plugin through OpenClaw. Restart the Gateway after installation.
If you maintain `plugins.allow`, include `tensorfold` alongside your existing entries.

Store the engine key through OpenClaw's interactive provider login:

```sh
openclaw models auth login --provider tensorfold --method api-key
openclaw models list --refresh --all --provider tensorfold --json
```

The plugin uses OpenClaw's auth store or its `TENSORFOLD_API_KEY` environment credential. It does not
put the key in the browser. Keep credentials out of plugin configuration and command-line arguments.

The default URL is `http://127.0.0.1:8080/v1`. For a different loopback port, merge this configuration
with your existing OpenClaw settings:

```json
{
  "plugins": {
    "entries": {
      "tensorfold": {
        "enabled": true,
        "config": { "baseUrl": "http://127.0.0.1:8081/v1" }
      }
    }
  }
}
```

An existing `models.providers.tensorfold.baseUrl` takes precedence, keeping inference, discovery, and
telemetry on the same endpoint. The plugin only accepts HTTP loopback URLs. Discovery and telemetry
refuse redirects. The plugin does not launch TensorFold or change which checkpoint it serves.

## Pick a model and inspect measurements

Select a discovered `tensorfold/<served-model-id>` in OpenClaw's native model picker. If your OpenClaw
configuration restricts allowed models, permit that model too. Refresh discovery after changing the
server's model. Context lengths come from `/v1/models`; missing metadata uses a configurable 8192-token
fallback. The advertised context is still subject to the server's [memory admission](api.md#admission).

Open the Control UI served by the Gateway using a browser-trusted loopback URL or HTTPS. Enable
**Settings → Labs → Custom plugin UI**, then open **TensorFold** in the sidebar. Native plugin UI runs
in the Control UI origin; install it only from a source you trust.

The page reads `/v1/models`, `/health`, `/stats`, and `/memory` through a Gateway method requiring
`operator.read`. It shows model names, health, token rates/counts, active/queued requests, and reported
memory values. **Start a new chat** selects the chosen model for a new session and requires write
access; existing chats and global defaults keep their selections.

Measurements refresh every five seconds while the page is visible and connected. Unknown values stay
unavailable; zero is displayed as zero. With API keys enabled, `/health` reports liveness only, so the
loaded-model list comes from `/v1/models`. `/stats` returning 404 explains that `--dashboard` is needed.
Memory fields depend on the backend; see [memory diagnostics](memory-diagnostics.md). Separate endpoint
reads are observations at different times, not an atomic snapshot.

## Tool choice and validation limits

OpenClaw's native OpenAI chat-completions transport handles streaming, tool-call assembly, and
cancellation. The plugin changes no engine code or returned tokens. TensorFold does not advertise a
per-model grammar-support flag in `/v1/models`, so the preview conservatively changes required/named
tool choices to `tool_choice: "auto"`. An automatic choice can answer in text instead of forcing a tool.
`auto`, `none`, tools, prompts, and sampling settings are preserved.

After verifying required/named tool support on a particular engine, operators can set
`plugins.entries.tensorfold.config.nativeToolChoice` to `true`. This is an explicit compatibility
setting, not automatic capability detection.

The plugin repository includes model-free HTTP/SSE, OpenClaw transport/host, and dashboard tests.
Before release, the real agents still need first-run, model selection, streamed tool use, cancellation,
long-context, failure/recovery, and telemetry-comparison runs on both a Mac and a GB10. Hermes has no
implementation in this preview. This guide does not claim that #1399 is complete.
