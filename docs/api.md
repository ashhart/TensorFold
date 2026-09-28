# OpenAI-compatible API

The base URL is `http://127.0.0.1:8080/v1` with the default server settings.

| Route | Behavior |
| --- | --- |
| `GET /v1/models` | Served model ID; MLX also lists configured aliases |
| `GET /health` | Server health and available status information |
| `POST /v1/chat/completions` | Text chat, tools and reasoning; streamed or non-streamed |
| `POST /v1/completions` | Raw text without a chat template; MLX also accepts token IDs |

On MLX, a completions body containing a nonempty `messages` list uses chat handling. CUDA completions
require a string `prompt`.
Image, audio and video input or output requests receive HTTP 400.

## Request fields

| Field | Meaning | Backend |
| --- | --- | --- |
| `messages` | Text messages, including system, developer, assistant tool calls and tool results | Both |
| `tools` | OpenAI function tools | Both |
| `tool_choice` | `none` hides tools from the template; `required` or a named function makes the reply call a tool | Both |
| `parallel_tool_calls` | False returns at most one completed call | Both |
| `max_tokens`, `max_completion_tokens` | Explicit reply limit; rejected if prompt plus reply exceeds the window | Both |
| `temperature`, `top_p`, `top_k` | Sampling overrides; zero temperature is greedy | Both |
| `seed` | Sampling key; otherwise derived from the prompt | Both |
| `stream` | Server-sent events with final usage | Both |
| `chat_template_kwargs.enable_thinking` | Template thinking toggle | Both |
| `draft` | False selects the serial reference; CUDA rejects it if the engine has no serial switch | Both |
| `ignore_eos` | Disable model end-of-sequence stopping; the reply limit still applies | MLX and GLM CUDA |
| `stop` | Stop at a string or any string in a list; omit the matched text from the response | MLX |
| `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high` or `xhigh` | MLX |
| `thinking_budget` | Token-count limit inside reasoning | MLX |
| `priority` | `background` yields to foreground requests | MLX |

CUDA does not enforce the MLX-only fields above. Its Qwen engines also ignore `ignore_eos`. Unsupported
generation features include multiple choices through `n` and `logprobs`. On MLX, `ignore_eos: true`
keeps user-supplied `stop` strings active, including when a stop string spans streamed chunks.
MLX rejects malformed numeric controls and out-of-vocabulary raw prompt IDs with HTTP 400;
`top_k` at zero or below disables top-k filtering, and null sampling fields retain server defaults.

On MLX, `--parallel auto` is the default: requests share rounds within the configured concurrency and memory
budget. Background work waits behind foreground requests. An active background request yields when a
foreground request needs its lane or memory, then restarts with already-delivered tokens suppressed.
Session-title requests are also treated as background work.

On CUDA, `--parallel auto` serves one request at a time. An explicit `--parallel N` above one shares rounds
for Qwen3.8-27B on one or two ranks and for Flash Next on one rank; GLM, Nemotron and Qwen3.6 stay serialized.
When a client disconnects, its CUDA request stops at the next round, and a request still waiting behind
another in one-at-a-time serving does not start; two-rank Flash Next, Nemotron and GLM requests finish on
both ranks.

## Messages and tools

Developer messages use system-message semantics. Only leading system and developer messages merge, in order,
into one leading system message. Later system and developer messages stay in place; when the template cannot
render a later system message, the server renders it as a user message. Text content parts concatenate
in order. Caller messages are not mutated.
Changing earlier rendered tokens can reduce prefix reuse.

On MLX, Qwen XML tool parameters use the offered schema's explicit type for arrays, objects, booleans, integers,
numbers and nulls. Strings preserve text and whitespace. Malformed or mismatched values remain strings
for the client to validate. Union types and schema references are not resolved by this conversion.
CUDA parses Qwen tool-call envelopes after generation and leaves XML parameter values as strings;
it does not apply the MLX schema conversion.

A reply that is not a call returns as content, never an error: prose, JSON that names no offered tool
(a structured answer), and malformed or unoffered `<tool_call>` blocks, which keep their text.

With `parallel_tool_calls: false`, the server buffers tool deltas until it can return the first valid
completed call. Prose and reasoning can still stream. Usage counts the entire decoded reply, including
additional calls omitted from the response.

With `tool_choice: "required"`, or a function named in `tool_choice`, the reply's answer (after any think block
or thought channel) opens a call to an offered tool. The server replaces the first answer token that isn't
whitespace with the tool-call opener (`<tool_call>`, or Gemma 4's `<|tool_call>`) and the template's text before a
tool name, then holds the name to the offered tools: a token that leaves them is replaced by the rest of the first
offered name its written part starts. The template's own rendered call gives that text. A named function is the
only tool the template offers. Each fix depends only on the tokens before it, so drafted, serial and concurrent
decoding write the same call. The MLX engine fixes tokens inside its rounds; CUDA stops the engine at a fix and
decodes on from the reply. The model writes the arguments; a malformed call returns as content.

## Reasoning

On MLX, `reasoning_effort: none` disables thinking; other effort values enable it and reach the chat template.
MLX also reads it from `chat_template_kwargs.reasoning_effort`, where vLLM's clients send it; the top-level field wins.
`high` maps to `xhigh`, and `minimal` maps to `low`. An explicit
`chat_template_kwargs.enable_thinking` takes precedence. Effort support depends on the checkpoint's
template, and effort does not set a token budget. The GLM CUDA handler closes the template's open
think block when thinking is disabled.

A tool call written before the think block closes is the reply's tool call when the reply ends inside the block,
on both backends; the reasoning stops where the call starts, and streamed reasoning never carries the call's markup.
A call only mentioned while thinking, with the block closed after it, stays reasoning.

`thinking_budget` on MLX forces a newline and the closing think marker at the budget, then continues the
answer. The cut depends on token count, so serial and drafted decoding use the same cut. A model that
closes the block earlier is left alone.

## Context and errors

The rendered prompt and reserved reply must fit the effective context. In 0.3.5, an explicit `max_tokens`
or `max_completion_tokens` that would put prompt plus reply beyond the window is rejected with counts
and fitting guidance before generation. MLX returns HTTP 400 for non-streamed requests or an
`invalid_request_error` event after opening a stream. CUDA returns HTTP 400 before opening a stream.
The 0.3.4.1 MLX server capped that explicit limit to the remaining context.
When the request omits the reply limit, the server still caps its configured default to the remaining context.
MLX also returns generation errors in the stream after streaming starts.
MLX also checks projected memory before prefill. CUDA checks its allocated cache capacity and model window.
A startup capacity estimate is not a measured release capacity.

## Responses

`choices[0].message.content` holds the answer. Reasoning uses `reasoning_content`, or
`delta.reasoning_content` while streaming. Tools use `tool_calls` and `finish_reason: "tool_calls"`.
The final usage includes token counts; cache and timing details depend on the backend.
TensorFold also reports generation statistics such as decode rate, time to first token and draft acceptance.

For exactness comparisons, hold the checkpoint, template, runtime, prompt, seed and sampling settings
constant, then compare the decoded reply with `draft` enabled and disabled. Repeat with fresh and reused
prefixes, and compare each MLX concurrent request with its solo run.
