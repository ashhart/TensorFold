# Use TensorFold with Claude Code

TensorFold answers Claude Code's Anthropic Messages requests from one served model. The server keeps the prompt
prefix it has already processed, so a coding session's later turns resume it instead of prefilling again. This guide
starts a server, points Claude Code at it, and keeps that prefix stable so the cache hits.

## Start a server

Use a checkpoint already qualified for your platform, following its [model recipe](recipes/). Serve one model with a
name and the context window you want Claude Code to assume; add a key when other hosts reach the server:

```sh
tensorfold serve /path/to/checkpoint --name local-model --context 65536 --host 127.0.0.1 --port 8080
```

The [API guide](api.md#anthropic-messages) lists the Messages routes, their fields and their refusals. Structured
output and a required tool call need `libtfgrammar`: the release archives include it under `lib/`, and a source build
gets it with `zig build grammar -Dxgrammar=DIR` ([clone](../CONTRIBUTING.md#build-and-host-tests)). Without it, a request
that needs the library is refused with HTTP 400 and names it.

## Point Claude Code at the server

`tensorfold launch` sets the endpoint and key in the child process only, and passes the environment through. Set the
window it forwards with the command:

```sh
CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536 tensorfold launch claude --model local-model
```

Or set the same variables yourself. Claude Code appends `/v1/messages` itself, so `ANTHROPIC_BASE_URL` is the server
root, without `/v1`:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_API_KEY=local   # a key from the server's key file when it has one
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536
claude --model local-model
```

The client documents `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` in its
[environment variables](https://code.claude.com/docs/en/env-vars). `tensorfold launch` reads the served id from
`/v1/models` and prints the model it found, but it does not pass `--model`: add `--model local-model` so Claude Code
treats the server as a gateway model. Without it, the client asks for its default Anthropic id and assumes that model's
capabilities (see [model names](#model-names-and-aliases)).

`CLAUDE_CODE_MAX_CONTEXT_TOKENS` is the client's way to correct the window it assumes for a model id it does not
recognize ([model configuration](https://code.claude.com/docs/en/model-config#correct-the-window-for-a-gateway-or-custom-model-id)).
Set it to the `--context` the server serves, or Claude Code assumes a window of its own (1M tokens on current clients)
and compacts too late.

## Model names and aliases

Claude Code asks for its default Anthropic model id unless `--model` names another; v2.1.299 asked for
`claude-sonnet-4-6`. A recognized id also tells the client which capabilities the model has: it then sends its deferred
MCP tool-search tool, which the server refuses with HTTP 400 `only client-defined function tools are supported`. Name
the served model instead:

```sh
CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536 tensorfold launch claude --model local-model
```

Claude Code then treats the id as a gateway model and sends only client-defined function tools. The reply echoes the
requested id when it is the served name or a configured alias, else the served name
([request fields](api.md#request-fields)).

The server answers any model id with the model it serves. `GET /v1/models` lists the served name and every `--alias`
name; any other id answers 404 there. An alias helps a client that reads that list, and it changes the reply's echoed
id, but not the capability assumptions of the id the client sent:

```sh
tensorfold serve /path/to/checkpoint --name local-model --alias assistant
```

With a non-Anthropic key, background tasks use the session's model, not a separate Haiku request.

## Keep the prompt prefix cacheable

The server keeps prompt states in RAM by default, so a later turn can resume the common prefix; `--prompt-cache-gib 0`
turns that off. Claude Code prepends an attribution block to the system prompt: the client version and a fingerprint of
the conversation. The Anthropic API strips that block before the model sees it, so a gateway or local server receives it
as part of the prompt ([system prompt attribution block](https://code.claude.com/docs/en/llm-gateway-protocol#system-prompt-attribution-block)).
When the block changes between turns, the prompt cache misses at the first token, and every turn prefills the whole
conversation again.

On Claude Code v2.1.181 and later the block stays the same for the lifetime of a conversation when requests go
through a custom `ANTHROPIC_BASE_URL`, so the prefix cache hits without any setting. Before v2.1.181 the block held a
per-request token that changed the start of the prompt every turn; `CLAUDE_CODE_ATTRIBUTION_HEADER=0` omits it, and
those clients need it for the cache to work:

```sh
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
```

The variable exists for gateway and third-party caching compatibility, not as a privacy control: the configured
endpoint receives the request with or without the attribution block. Keep it unset on current clients unless something
else reshapes the system array.

### Measured prefix reuse

One Claude Code session on v2.1.299 against a server at commit `ef6bf1bea`, serving `nvidia/Qwen3.8-27B-NVFP4` on one
GB10 (Ubuntu 24.04.5, NVIDIA driver 580.178.04, CUDA 13.0.88, Zig 0.17.0), asked three one-line prompts. The table
gives the prompt's token count, how many of those tokens came from the prompt cache, and the new input tokens; the
cache state was flushed between the two arms. The server ran with `--context 65536 --parallel 1`, and the client with
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536`.

| Turn | Attribution block kept: prompt | resumed | new input | Block omitted: prompt | resumed | new input |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 27,404 | 0 | 27,404 | 27,374 | 0 | 27,374 |
| 2 | 27,906 | 24,576 | 3,330 | 27,860 | 24,576 | 3,284 |
| 3 | 28,284 | 24,576 | 3,708 | 28,230 | 24,576 | 3,654 |

The attribution block is 30 tokens on the first turn, and the same
`x-anthropic-billing-header: cc_version=2.1.299.a67; cc_entrypoint=sdk-cli;` line starts every request in the session.
Both arms resumed the same 24,576 tokens on the second and third turns, so on this client the setting changes the
prompt's size but not the cache's hit. A client older than v2.1.181, whose block changes every request, misses at its
first token instead.

## Tools and structured output

Claude Code's tool use and title requests go through the same Messages route. A `tool_choice` of `{"type": "any"}` (the
server's internal spelling is `required`), or a request that carries a JSON schema, needs `libtfgrammar`
([structured output](api.md#structured-output)); without it the server answers HTTP 400 and names the library. Claude
Code's `context_management` keep-all thinking directive is accepted, and thinking round-trips as plaintext with an empty
signature, as on the Anthropic API.
