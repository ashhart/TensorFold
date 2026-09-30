# OpenAI-compatible API

The base URL is `http://127.0.0.1:8080/v1` with the default server settings.

| Route | Behavior |
| --- | --- |
| `GET /v1/models` | Served model ID; MLX also lists configured aliases |
| `GET /health` | Server health and available status information |
| `GET /metrics`, `GET /v1/metrics` | Prometheus text: requests, KV occupancy, drafts and latency (both servers) |
| `POST /v1/chat/completions` | Text chat, optional image input, tools and reasoning; streamed or non-streamed |
| `POST /v1/completions` | Raw text without a chat template; MLX also accepts token IDs |
| `POST /v1/responses` | OpenAI's Responses API, run as the equivalent chat completion; streamed or non-streamed |
| `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}` | A stored response, or remove it |
| `POST /v1/decisions` | Choice, score, and yes/no probabilities from the next-token logits; no text is generated |

On MLX, a completions body containing a nonempty `messages` list uses chat handling. CUDA completions
require a string `prompt`.
With `--vision`, supported Qwen3.5/3.8 dense checkpoints accept user `image_url` content parts alongside text.
See [image input](vision.md) for data URLs, public image URLs, limits and cache behavior.
Unsupported image input, audio, video and non-text output requests receive HTTP 400.

## Decisions

`POST /v1/decisions` is served by the MLX server and by the CUDA GLM engine.
Another CUDA engine, one without label scoring, returns HTTP 400.
The prompt wording is SGLang's decision prompt format version 1: the input, a blank line, the question, one line per
option, level, or described yes or no answer, and a closing instruction to answer with one label. Choice labels are
`A` to `Z`, score labels are `0` to `9`, and a yes/no question uses `yes` and `no`. Each label must be one distinct
token at the answer position. Thinking stays off. The response carries `prompt_format_version`, `answers` keyed by
question id, and `usage.completion_tokens` 0. `probabilities` are a softmax over the label logits divided by
`temperature` (default 1). `label_mass` is the full-vocabulary probability of those labels and does not use
`temperature`. A request the tokenizer or the context window cannot score returns HTTP 400.
For decisions, `chat_template_kwargs` may be omitted, null, or an object containing only
`enable_thinking: false`; other types, keys, or thinking values return HTTP 400.

## Request fields

| Field | Meaning | Backend |
| --- | --- | --- |
| `messages` | Text messages, including system, developer, assistant tool calls and tool results | Both |
| `tools` | OpenAI function tools | Both |
| `tool_choice` | `none` hides tools from the template; `required` or a named function makes the reply call a tool | Both |
| `parallel_tool_calls` | False returns at most one completed call | Both |
| `max_tokens`, `max_completion_tokens` | Explicit reply limit; rejected if prompt plus reply exceeds the window | Both |
| `temperature`, `top_p`, `top_k`, `min_p` | Sampling overrides; zero temperature is greedy | Both |
| `seed` | Sampling key; otherwise derived from the prompt (and `TENSORFOLD_SEED_SALT`) | Both |
| `stream` | Server-sent events; the last event carries usage | Both |
| `chat_template_kwargs.enable_thinking` | Template thinking toggle | Both |
| `draft` | False selects the serial reference; CUDA rejects it if the engine has no serial switch | Both |
| `response_format`, `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`, `structured_outputs` | A JSON schema, any JSON object, a regex, a choice or an EBNF grammar the reply must match | Both |
| `ignore_eos` | Disable model end-of-sequence stopping; the reply limit still applies | Both |
| `stop` | Stop at a string or any string in a list; omit the matched text from the response | Both |
| `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max` | Both |
| `thinking_budget` | Token-count limit inside reasoning | Both |
| `priority` | `background` yields to foreground requests | Both |

`n` must be 1; multiple choices receive HTTP 400.

CUDA Flash Next on one GPU supports `logprobs: true` and `top_logprobs` from 0 through 20 for nonstreamed text chat
with thinking off, without tools, stop strings or structured output. Each visible token has its `token`, `bytes`,
`logprob` and requested `top_logprobs` in `choices[0].logprobs.content`. Probabilities describe the raw target-model
distribution at temperature 1, before temperature, top-k or top-p sampling filters, including when generation is
greedy. Alternatives are tokenizer tokens and may include leading spaces; `bytes` preserves partial UTF-8 sequences.
Unsupported backends, engines and request modes return HTTP 400 when probabilities are requested.
`ignore_eos: true` keeps user-supplied `stop` strings active, including when a stop string spans streamed chunks.
Both backends reject a non-boolean `ignore_eos` or a malformed `stop` with HTTP 400 before a stream opens.
Both backends reject malformed `temperature`, `top_p`, `top_k` and `seed` values with HTTP 400, whether or not
the request samples: booleans, non-finite numbers, non-numeric strings, and non-integral `top_k` or `seed`.
MLX also rejects its other malformed numeric controls and out-of-vocabulary raw prompt IDs.
`top_k` at zero or below disables top-k filtering, and null sampling fields retain server defaults.
With top-k off, a CUDA draw cuts the top-p nucleus by each token's probability mass in 40-bit fixed point, so one GPU
and two GPUs holding halves of the vocabulary draw the same token: each GPU reads its 1,024 best candidates, and its
whole vocabulary shard when a row's nucleus runs past them (or with `top_p` 1 and no `min_p`).
`min_p` (0 to 1; 0 is off) keeps, after the top-k and top-p cuts, only the tokens at least `min_p` times as likely
as the likeliest one after temperature; the top-p cut is taken without it, as SGLang does (vLLM applies `min_p`
first). A value outside 0 to 1 gets HTTP 400. It is part of the keyed rule on every engine, so drafted replies still
equal `"draft": false` ones.

On CUDA, stop strings are checked after every token on the generated text, reasoning included, so drafted and
`"draft": false` replies stop at the same token; usage and `token_sha` count the tokens through the one that
completes the match. The GLM engine, and Flash Next and Nemotron on two ranks, decode on after a match to an end
token or the reply limit; the server returns the reply only up to the match. Where a CUDA engine honors
`ignore_eos`, end tokens inside the reply are decoded into its text, as on MLX, and `finish_reason` is `length`
unless a stop string or a tool call ended the reply.

On MLX, `--parallel auto` is the default: requests share rounds within the configured concurrency and memory
budget. Every admitted prompt prefills a chunk at a time beside the others. Each chunk goes to a foreground prompt
before a background one, then to the prompt with the fewest tokens left, and a prompt passed over for 8 chunks takes
the next one: a short request starts at the next chunk, and a long prompt still finishes. Running replies take rounds
between chunks for `--decode-share` of each chunk's time (default 0.25). `--decode-share 0` prefills each prompt whole
first, in arrival order, as 0.3.6.2 did. Background work waits behind foreground requests. An active background
request yields when a foreground request needs its lane or memory, then restarts with already-delivered tokens
suppressed.
Session-title requests are also treated as background work.

On CUDA, `--parallel auto` serves one request at a time. An explicit `--parallel N` above one shares rounds
for Qwen3.8-27B on one or two ranks and for Flash Next and Qwen3.6 on one rank; GLM and Nemotron stay serialized.
When a client disconnects, its CUDA request stops at the next round, and a request still waiting behind
another in one-at-a-time serving does not start; two-rank Flash Next, Nemotron and GLM requests finish on
both ranks. A background request (`priority: background`, or a session-title request) waits behind foreground
ones. One decoding yields to a foreground request that waits (between rounds when one request runs at a time; its lane
under `--parallel` when every lane is taken), then replays from its prompt later, as the Mac does: the tokens
it already sent come again and are checked, not sent twice, so its reply equals its solo run; a foreground
prompt prefills before a background one. Under `--parallel`, the 27B and Qwen3.6 prefill a background prompt 1,024
rows a step, so a foreground prompt arriving meanwhile waits less: on one Spark, a short foreground request 0.3 s into
a background prompt's ~1 s prefill waited 0.57 s on Qwen3.6, against 0.70 s without the priority and 0.04 s alone
(the 27B: 0.69-0.89 s against 2.98-3.66 s). Flash Next, and an engine serving one request at a time, finish a
background prompt's prefill once it has started. Two-rank engines serving one request at a time
only order the queue.

## Messages and tools

Developer messages use system-message semantics. Only leading system and developer messages merge, in order,
into one leading system message. Later system and developer messages stay in place; when the template cannot
render a later system message, the server renders it as a user message. Text content parts concatenate
in order. Caller messages are not mutated.
Changing earlier rendered tokens can reduce prefix reuse.

On both backends, Qwen XML tool parameters use the offered schema's explicit type for arrays, objects, booleans,
integers, numbers and nulls, streamed or not. Strings preserve text and whitespace. Malformed or mismatched values
remain strings for the client to validate. Union types and schema references are not resolved by this conversion.

A reply that is not a call returns as content, never an error: prose, JSON that names no offered tool
(a structured answer), and malformed or unoffered `<tool_call>` blocks, which keep their text.

With `parallel_tool_calls: false`, the server buffers tool deltas until it can return the first valid
completed call. Prose and reasoning can still stream. Usage counts the entire decoded reply, including
additional calls omitted from the response. Otherwise both servers stream each Qwen XML call as it is written: a
delta with the call's id and name, then its arguments in pieces; a call the streamer can't follow arrives whole at
the end.

With `tool_choice: "required"`, or a function named in `tool_choice`, the reply's answer (after any think block
or thought channel) opens a call to an offered tool. The server replaces the first answer token that isn't
whitespace with the tool-call opener (`<tool_call>`, or Gemma 4's `<|tool_call>`) and the template's text before a
tool name, then holds the name to the offered tools: a token that leaves them is replaced by the rest of the first
offered name its written part starts. The template's own rendered call gives that text. A named function is the
only tool the template offers. Each fix depends only on the tokens before it, so drafted, serial and concurrent
decoding write the same call. The MLX engine fixes tokens inside its rounds; CUDA stops the engine at a fix and
decodes on from the reply. The model writes the arguments; a malformed call returns as content.

## Structured output

Every model on both backends enforces `response_format` (`{"type": "json_schema", "json_schema": {"schema": ...}}`
or `{"type": "json_object"}`) and vLLM's `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` (EBNF) and
`structured_outputs` (`json`, `json_object`, `regex`, `choice` or `grammar`): on the Mac, and on CUDA with one or two
GPUs, alone or under `--parallel N`. It needs xgrammar on the server: `pip install 'tensorfold[grammar]'`, which also
installs torch, Macs included. Drafting stays on. Before each verify forward the engine cuts the drafts the grammar
rejects, then masks each remaining row's logits to the tokens the grammar allows after that row's path, so a
constrained reply equals its `"draft": false` reply and its solo run. Two GPUs compile the same grammar and mask
their own vocabulary columns. With thinking on, the grammar starts after the think end (`</think>`, or Gemma 4's
`<channel|>`), and a thinking budget's close stops at that token. The grammar allows the end token only once the value
is complete; a reply cut at `max_tokens` is incomplete, with `finish_reason: "length"`. JSON grammars allow at most
32 blank characters between two tokens (pretty-printing fits; a run of blank lines does not), so a model cannot fill
its reply with whitespace inside an unfinished value.

HTTP 400 comes before any token for a malformed field, a grammar xgrammar cannot compile (with xgrammar's reason), a
server without xgrammar (with the install command), and a grammar sent with `tool_choice: "required"` or a named
function. A reply whose grammar fails while decoding ends with HTTP 500 (an error event when streaming); the other
requests go on.

## Reasoning

On both backends, `reasoning_effort: none` disables thinking; other effort values enable it and reach the chat
template. The server also reads it from `chat_template_kwargs.reasoning_effort`, where vLLM's clients send it; the
top-level field wins. An unnamed level maps to the nearest level the template names, and a tie takes
the higher one. GLM-5.3 lists `low`, `high` and `max`, so `max` stays `max`, `medium` is heard as `high`
and `minimal` as `low`. Qwen3.8 lists `low`, `medium` and `xhigh`, so `high` and `max` are heard as `xhigh`.
`xhigh` stays `xhigh`. GLM-5.3 renders `xhigh` and `max` as Max. A template that names no level hears `max` as
`xhigh`. `--reasoning-effort` uses the same rule. An omitted effort, with no startup flag, stays
the template's own default. An explicit `chat_template_kwargs.enable_thinking` takes
precedence. A request without an effort gets `--reasoning-effort` when the server was started with one; otherwise
the template renders its own default, as vLLM and mlx-lm render it (Qwen3.8's is `xhigh`, which adds an instruction
to the system prompt; `medium` adds none). The template hears an effort only while thinking, and both backends render
the same prompt for the same request. Effort support depends on the checkpoint's template, and effort does not set a
token budget. With thinking off, GLM-5.3's prompt is its thinking-off template's: no reasoning-effort line and an
empty think block.

A tool call written before the think block closes is the reply's tool call when the reply ends inside the block,
on both backends; the reasoning stops where the call starts, and streamed reasoning never carries the call's markup.
A call only mentioned while thinking, with the block closed after it, stays reasoning.

`thinking_budget` (the request's; absent or 0 takes `--thinking-budget`) replaces the reply's budget-th token,
while it is still thinking, with a newline, the closing think marker and a blank line, then continues the answer.
The cut depends on token count, so serial and drafted decoding use the same cut. A model that closes the block
earlier is left alone. The MLX engine forces the close inside its rounds; CUDA stops the engine at the cut and
decodes on from the reply, as for a required tool call.

## Context and errors

The rendered prompt and reserved reply must fit the effective context. In 0.3.5, an explicit `max_tokens`
or `max_completion_tokens` that would put prompt plus reply beyond the window is rejected with counts
and fitting guidance before generation. MLX returns HTTP 400 for non-streamed requests or an
`invalid_request_error` event after opening a stream. CUDA returns HTTP 400 before opening a stream.
The 0.3.4.1 MLX server capped that explicit limit to the remaining context.
A prompt that leaves no room for a reply is refused the same way, and on both backends every such refusal
carries OpenAI's `context_length_exceeded` code and a message that starts "This server's maximum context length
is N tokens", so clients that compact a conversation on that error do so.
When the request omits the reply limit, the server still caps its configured default to the remaining context.
CUDA returns HTTP 400 before generation when the chat template rejects the request or
`chat_template_kwargs` is neither an object nor null. A generation error returns HTTP 500 for a
non-streamed request; after a stream opens, both backends send an error event of type `server_error`,
then `[DONE]`. On two CUDA ranks such an error can leave the ranks out of step, so restart both: the 27B's
`--parallel` decoder refuses every later request until then, and the other two-rank engines don't detect it.
MLX also checks projected memory before prefill. CUDA checks its allocated cache capacity and model window.
A startup capacity estimate is not a measured release capacity.

On a unified-memory GPU (the DGX Spark's GB10), the CUDA server's allocations come out of the host's RAM but are not
charged to a container's memory limit (`docker run --memory`, cgroup `memory.max`): the limit neither caps the
model's weights and cache nor keeps them from crowding other work on the machine. The server sizes its window from the
host's `MemAvailable` less a reserve (a tenth of RAM, at least 4 GiB); to leave room for other containers, start it
with a smaller `--context` or `--parallel`. `TENSORFOLD_MEMORY_RESERVE_GIB` replaces that reserve (at least 2 GiB)
when you know the machine's headroom: a larger one leaves more for other work, a smaller one more for KV caches. The
reserve also carries CUDA context, NCCL and workspace memory the estimate does not count, and exhausting a unified
GPU's memory can freeze the host, so lower it only with room to spare.

## Responses

`choices[0].message.content` holds the answer. Reasoning uses `reasoning_content`, or
`delta.reasoning_content` while streaming. Tools use `tool_calls` and `finish_reason: "tool_calls"`.
Every reply's usage, streamed ones included, carries its prompt, completion and total tokens,
`prompt_tokens_details.cached_tokens` (the prompt tokens found in the prefix cache) and
`completion_tokens_details.reasoning_tokens` (the thinking tokens, through the closing think marker); timing details
depend on the backend.
TensorFold also reports generation statistics such as decode rate, time to first token and draft acceptance.

On CUDA, `GET /health` also carries counters a poller can difference into rates: `busy`, `requests_running`,
`requests_total`, `completion_tokens_total` (the running replies' tokens included as they stream), and totals that
move when a request ends, taken from the engine's own statistics: `prompt_tokens_total`, `cached_tokens_total`,
`prefill_seconds_total`, `decode_seconds_total`, `rounds_total`, and `drafted_total` and `accepted_total` where the
engine reports them. A `--parallel` server adds `streams` (decoding, prefilling and the maximum), and
`context_length` is the served window. The first sample is a baseline, not a rate.

For exactness comparisons, hold the checkpoint, template, runtime, prompt, seed and sampling settings
constant, then compare the decoded reply with `draft` enabled and disabled. Repeat with fresh and reused
prefixes, and compare each MLX concurrent request with its solo run.

A request without `seed` takes one derived from its prompt, so running the same evaluation twice against one server
repeats the same samples wherever the conversations agree (an agent benchmark's second pass then mostly replays its
first). To draw independent repeats, send a `seed` per run, or start each run's server with a different
`TENSORFOLD_SEED_SALT` (an integer mixed into every prompt-derived seed; 0, the default, keeps today's seeds).

## Metrics

`GET /metrics` and `GET /v1/metrics` answer as Prometheus text on both servers. Each family is read on its own
at scrape time (the scrape is not one atomic snapshot), and a family the server cannot count honestly is left
out of the text rather than reported at a permanent zero.

The scrape carries `requests_running`, `requests_waiting`, `prompt_tokens_total`, `generation_tokens_total`,
`kv_cache_usage_ratio` (one `pool` label per live stream cache), `mtp_drafted_total` and `mtp_accepted_total`
(draft tokens verified and kept on finished requests; the engines keep one draft counter, so copies and chain
drafts share it), `request_latency_seconds` and `time_to_first_token_seconds`, all under the `tensorfold:`
prefix. Every reading is repeated under a vLLM-compatible name (`num_requests_running`, `num_requests_waiting`,
`kv_cache_usage_perc`, `spec_decode_num_draft_tokens_total`, `spec_decode_num_accepted_tokens_total`,
`e2e_request_latency_seconds`) with identical values, so a dashboard copied from vLLM fills by swapping the
`tensorfold:` prefix for the metric name. `client_disconnections_total` (requests the client walked away from)
and `preemptions_total` (background work that gave up a lane to a later request) are published where the
server counts those events, and never at a fabricated zero.

## The Responses API

`POST /v1/responses` takes OpenAI's Responses request and runs it as the equivalent chat completion, through the
same handler and engine path. A response has that chat completion's prompt and tokens (the `tensorfold` block's
`token_sha` matches), drafts, and equals its `"draft": false` run and its solo run.

- `input` is a string or a list of items: messages (`input_text`, and `input_image` with `--vision`),
  `function_call`, `function_call_output`, and `reasoning` items with their `content` text, which the template gets
  back as the next assistant message's `reasoning_content`. `instructions` becomes the system message and is not
  carried to a later turn.
- `tools` takes function tools; `tool_choice` takes `none`, `auto`, `required`, a function or `allowed_tools`;
  `parallel_tool_calls` works as in chat.
- `max_output_tokens` counts reasoning tokens too. `temperature`, `top_p`, `top_k`, `min_p` and `seed` sample as in
  chat.
- `reasoning.effort` follows the chat effort rule: unset is the template's default and `none` turns thinking off.
  `reasoning.summary` is accepted, but no summary is written.
- `text.format` takes `json_schema` or `json_object` through the structured-output path, with drafts.
- `stream`, `store` (default true) and `metadata` (up to 16 string pairs). The chat extensions `draft`,
  `thinking_budget`, `stop`, `ignore_eos`, `chat_template_kwargs` and `priority` pass through.

`output` holds a `reasoning` item (its text as `reasoning_text` content, with an empty `summary`), a `message` with
`output_text`, and `function_call` items (`call_id`, `name` and JSON `arguments`). A reply cut by the token limit has
`status: "incomplete"` and `incomplete_details.reason: "max_output_tokens"`. `usage` has `input_tokens` (with
`input_tokens_details.cached_tokens`), `output_tokens` (with `output_tokens_details.reasoning_tokens`) and
`total_tokens`.

A stream sends typed events, each with a `sequence_number`: `response.created` and `response.in_progress`; for each
output item `response.output_item.added`, `response.content_part.added`, its deltas
(`response.reasoning_text.delta`, `response.output_text.delta` or `response.function_call_arguments.delta`), their
`.done` events, `response.content_part.done` and `response.output_item.done`; and last `response.completed`,
`response.incomplete` or `response.failed`.

With `store` true, a finished response is kept in memory (the newest 1,024, up to 256 MiB) before its last event or
body is sent. `GET /v1/responses/{id}` returns it, `DELETE /v1/responses/{id}` removes it, and
`previous_response_id` continues its conversation: the server rebuilds the messages a chat client would send, so the
prompt cache resumes the shared prefix as it does for chat. `store: false` keeps nothing, and a restart forgets every
response.

HTTP 400 refuses what this server does not run: built-in tools (web search, file search, computer use and the
others), `background`, `include` (encrypted reasoning among them), `conversation`, `prompt` templates,
`truncation: "auto"`, `top_logprobs`, `input_file` parts and file IDs, `item_reference` items, encrypted reasoning
items, and a `previous_response_id` that is not stored.
