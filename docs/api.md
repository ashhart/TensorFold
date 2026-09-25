# The OpenAI-compatible API

`tensorfold serve` listens at `http://HOST:PORT/v1`.

| Route | What it does |
| --- | --- |
| `GET /v1/models` | the served model id and its aliases |
| `GET /health` | `{"status": "ok", ...}` |
| `POST /v1/chat/completions` | chat, streamed or not, with tools and reasoning |
| `POST /v1/completions` | legacy completions: the prompt becomes one user message |

Requests are decoded one at a time, in arrival order. A request marked `"priority": "background"` (and a
session-title request, recognised by its short system prompt about a title and no tools) waits behind every
other request and gives way at the next round when one arrives; it then runs again from the start and its
caller receives the same tokens.

## Request fields

| Field | Notes |
| --- | --- |
| `messages` | OpenAI messages, including assistant `tool_calls` and `tool` results |
| `tools`, `tool_choice` | OpenAI function tools; `tool_choice: "none"` hides them from the template |
| `max_tokens` / `max_completion_tokens` | reply limit (default: the server's `--max-tokens`); capped by `--context` |
| `temperature`, `top_p`, `top_k` | override the server's defaults; `temperature: 0` decodes greedily |
| `seed` | the sampling key (default: a hash of the prompt) |
| `stream` | server-sent events; the last chunk carries `usage` |
| `chat_template_kwargs.enable_thinking` | open or skip the think block for this request |
| `thinking_budget` | most thinking tokens before the server writes `</think>` (TensorFold extension) |
| `draft` | `false`: one token a round, no drafts; the output is the same, only slower (TensorFold extension) |
| `priority` | `"background"`: yield to every other request (TensorFold extension) |

Not supported yet: `stop`, `n` > 1, `logprobs`, images.

A prompt longer than the server's context window gets HTTP 400 (`invalid_request_error`); in a stream, an
`error` event.

## Response

- `choices[0].message.content`: the answer. Inside a think block the model's reasoning goes to
  `reasoning_content`; streamed, it arrives as `delta.reasoning_content` while the model thinks.
- Tool calls come back as OpenAI `tool_calls` with `finish_reason: "tool_calls"`. When streaming, their
  arguments stream as tool_call deltas while the model writes them.
- `usage` includes `prompt_tokens_details.cached_tokens`: the prompt tokens read from the prompt cache.
- `tensorfold`: decode `tokens_per_second`, time to first token, prefill seconds, and whether the sampling was
  exact or greedy.
- `speculative`: rounds, drafted and accepted tokens.
- `exact_mode`: which engine served the request.

## The thinking budget

With `thinking_budget: N` (or `--thinking-budget N`), the N-th reply token inside the think block is `\n`
whatever the model samples there, followed by `</think>` and a blank line, and the model then writes its
answer. The cut depends only on the token count, so drafted and serial decoding cut at the same place. A model
that closes its think block before the budget is left alone.
