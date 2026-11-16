# Chat in the terminal

`tensorfold chat MODEL` starts `tensorfold serve MODEL` as a child of the same binary, waits until it is ready, chats
with the model, and stops the server when the chat ends. MODEL is a cached repo ID or a checkpoint directory, as for
`serve`; a repo ID that is not cached is refused before anything starts (`tensorfold pull` downloads it).

```sh
tensorfold chat mlx-community/Qwen3.5-2B-MLX-4bit
tensorfold chat TensorFold/Qwen3.8-27B-MLX-4bit -- --parallel 1 --drafter z-lab/Qwen3.8-27B-DFlash2
tensorfold chat --url http://127.0.0.1:8080
echo "Say hello in one sentence." | tensorfold chat --url http://127.0.0.1:8080 local-model
```

Type a message and press Enter. Each turn sends the whole conversation to `POST /v1/chat/completions` with
`"stream": true` and prints the answer as it arrives. Lines are read one at a time, without line editing.

| In the chat | Does |
| --- | --- |
| `/bye` or Ctrl-D | Ends the chat |
| `/clear` | Forgets the conversation; the system prompt stays |
| `/system TEXT` | Sets the system prompt; `/system` alone removes it |
| `/help` | Lists these commands |
| Ctrl-C | During a reply, stops it and closes its stream; at the prompt, says how to end the chat |

A line that starts with `/` is a command; start it with a space to send it as a message. A finished reply is kept in
the conversation, and so is the part of a stopped reply that was printed; a refused or failed turn is not kept, so it
can be sent again.

## Flags

| Flag | Meaning |
| --- | --- |
| `--url URL` | Chat with a server that is already running at this http(s) address, with or without `/v1`. MODEL then names the model to ask for; without it, the first `id` in `GET /v1/models` |
| `--system TEXT` | The system prompt |
| `--temperature T` | The request's `temperature`; 0 is greedy. Without it the server's default applies |
| `--max-tokens N` | The request's `max_tokens`. Without it the server's default applies |
| `--show-thinking` | Shows the reply's `reasoning_content` (or `reasoning`) dimmed before the answer. Reasoning is hidden by default, with a `thinking...` hint on a terminal |
| `--no-thinking` | Sends `chat_template_kwargs: {"enable_thinking": false}`. Refused before the first turn when `/v1/models` says the model's template cannot turn thinking off (`thinking_switch` false) |
| `--startup-timeout S` | Seconds the started server gets to load: by default 600, plus 6 per GB of weights past 20 GB, at most 3600 |
| `-- SERVE_FLAGS` | Everything after `--` goes to the started `tensorfold serve` |

## The started server

The server listens on 127.0.0.1 at a free port, in a process group of its own so that a terminal's Ctrl-C reaches
only the chat. It starts with the environment `tensorfold benchmark` gives its server: without `TENSORFOLD_API_KEY`,
`TENSORFOLD_REQUEST_LOG` or `TENSORFOLD_SEED_SALT`, with `HF_HUB_OFFLINE=1`, and with no live line or update check.
Serve's own parser checks the flags after `--` before the server starts. `--host`, `--port`, `--api-key` and
`--api-key-file` are refused there, because the chat chooses the address and sends no key; `--name` is kept.
If the server exits while it loads, or before a later turn reaches it, the chat prints the server's last log lines
and exits 1.

Starting a server is not supported on Windows yet, and is refused by name: run `tensorfold serve`, then chat with
`--url`. On Windows, Ctrl-C ends the chat.

## Scripts

When stdin is not a terminal, all of stdin (up to 16 MiB, surrounding blank space trimmed) is one message. Only the
reply is written to stdout; status lines and errors go to stderr. The exit code is 0 when the reply finished, 1 when
it was refused, failed or could not be sent, 2 for a usage error and 130 for Ctrl-C.
When stdout is not a terminal the reply is written as the server sent it. On a terminal, control characters other
than newline and tab are dropped, so a reply cannot move the cursor or change the terminal's colors.

A refusal prints its status and message, such as `HTTP 400: This server's maximum context length is 8192 tokens`,
and an error event in the stream prints its message. A reply cut at its token limit says so on stderr, and so does a
reply that held only reasoning while reasoning is hidden.

Run `zig build test-chat-cli-host` for the host tests: they chat with a fake server over loopback, with a shell
stand-in for the started server, and need no model or GPU. `zig build check-chat-windows-host` compiles the command
for Windows.
