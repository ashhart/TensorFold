# TensorFold for AI coding assistants (MCP)

`tensorfold mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server on stdin and stdout.
An AI coding assistant such as Claude Code, Codex or Cursor starts it and calls its tools to see what this machine
runs, download a model, and start or stop a local TensorFold server, instead of you typing the commands.
It needs no model, GPU or network until a tool asks for one.

## Add it to a client

Claude Code:

```sh
claude mcp add tensorfold -- tensorfold mcp
```

Add `--scope user` before the name to have it in every project.

Codex, in `~/.codex/config.toml`:

```toml
[mcp_servers.tensorfold]
command = "tensorfold"
args = ["mcp"]
tool_timeout_sec = 3600
```

Cursor reads `~/.cursor/mcp.json` or a project's `.cursor/mcp.json`. It, Claude Desktop and other clients with an
`mcpServers` file take this entry:

```json
{
  "mcpServers": {
    "tensorfold": { "command": "tensorfold", "args": ["mcp"] }
  }
}
```

A client started from the desktop may not see your shell's `PATH`, so give `command` the full path that
`command -v tensorfold` prints. With a Linux release archive, name `tensorfold-native`. The quick start's `tensorfold`
is a shell function, and a client does not see it.

`pull` and `serve` can take minutes. Codex stops a tool call after 60 seconds unless you raise `tool_timeout_sec`, as
above. Claude Code limits a call with `MCP_TOOL_TIMEOUT`, and one that reports no progress with
`CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT`. This server reports progress whenever the client's request asks for it.

## Tools

| Tool | Arguments | What it does |
| --- | --- | --- |
| `machine` | none | The OS, CPU, memory and GPUs, and the TensorFold version, backend and model types this build serves. It reports the host facts `tensorfold benchmark` records, so no hostname or serial number. |
| `models` | none | The checkpoints in the Hugging Face cache that a TensorFold family serves, as `tensorfold models` lists them, with each revision and size on disk. |
| `info` | `model` | One checkpoint's family, format, context, weights and memory floor, or Flash Next's Metal budgets, as `tensorfold info` prints them. |
| `pull` | `repo`, `revision` (optional) | Downloads a Hugging Face repo with `tensorfold pull`'s own code. It checks the files against the Hub, resumes an interrupted download and refuses a repo no family serves. |
| `serve` | `model`, `context` (optional), `drafter` (optional) | Starts `tensorfold serve` for a cached model and answers once the server does. |
| `stop` | `id` | Stops a server this session started. |
| `status` | none | The servers this session started, with each one's id, model, served name, base URL and whether it still runs. A server that exited on its own shows its last log lines. |

Every result is plain text for the assistant and the same facts as structured content. A tool that cannot do what it
was asked returns a tool error that says why, so the assistant can correct the call. Nothing a client sends crashes
the server.

## Servers that `serve` starts

Each server listens on `127.0.0.1` at a free port and answers to the repo's last part as its model name, as
`tensorfold serve REPO` does. It gets a fresh random API key of 64 hex characters in `TENSORFOLD_API_KEY`, as
`tensorfold chat` gives its own server. Only `serve`'s result carries the key, with the base URL and the served name:

```sh
curl -fsS http://127.0.0.1:PORT/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-2B-MLX-4bit","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}'
```

Chat completions, Responses and Anthropic Messages are under that base URL. Messages clients send the key as
`x-api-key`. Calling `serve` again for a model and settings already running returns that server and its key instead
of loading a second copy.

The child inherits your environment as `tensorfold serve --router`'s children do, without your API key or the
benchmark's upload token. It shows no live status line, skips the update check and keeps Hugging Face offline. It runs
in a process group of its own, so a terminal's Ctrl-C reaches this server, which then stops it. Its log is a file under
`~/.tensorfold/mcp`, which the server unlinks once the child starts.

Every server the session started stops when the session ends. The session ends when the client closes stdin, or on
SIGTERM, SIGINT or SIGHUP, and a server still loading stops too. Nothing can clean up after SIGKILL, so a client
should close stdin first. On Windows, `serve` refuses by name because it stops its servers with POSIX signals. The
other tools work there.

## What the tools accept

No tool deletes anything. `pull` takes a Hugging Face repo id, `owner/name`, and an optional branch, tag or commit.
It never takes a path. `info`, `serve` and `drafter` take a repo id already in the cache, or the absolute path of an
existing checkpoint directory, one that holds `config.json`. The tools refuse relative paths, `~` and any `..`,
because the assistant does not choose the server's working directory. Each tool checks its arguments against its JSON
Schema, so an unknown argument, a wrong type or a missing one is a tool error.

## Protocol

The server implements MCP revision 2026-07-28. There every request names its revision and client capabilities in
`_meta`, and `server/discover` lists what the server supports. It also speaks the `initialize` handshake of revisions
2025-11-25 and 2025-06-18 to clients that open with it, and answers an `initialize` asking for another revision with
2025-11-25. It answers `ping` under the handshake revisions, which define it.

- Messages are JSON-RPC 2.0, one to a line, at most 4 MiB. Bad JSON gets `-32700`, a batch or a malformed message
  `-32600`, an unknown method `-32601`, and bad parameters, an unknown tool or a request naming no revision `-32602`.
  A revision this server does not speak gets 2026-07-28's `-32022` with the supported list.
- A `tools/call` with a `progressToken` in `_meta` gets progress notifications. `pull` reports its lines as
  `tensorfold pull` prints them, and `serve` reports every five seconds while a server loads.
- `notifications/cancelled` stops a loading server, and its call answers nothing. A download cannot stop midway, so a
  cancelled `pull` goes on and answers nothing. The next `pull` of that repo resumes from where it got to.
- stdout carries protocol messages only. Diagnostics go to stderr.
