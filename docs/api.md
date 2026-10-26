# Compatible APIs

The base URL is `http://127.0.0.1:8080/v1` with the default server settings.

## Native chat template override

`tensorfold serve MODEL --chat-template custom-chat.jinja` loads a UTF-8 Jinja template from that file
instead of the checkpoint's template files or `tokenizer_config.json` templates. Relative paths resolve from the server's working directory. The file is read once at startup, with the same 16 MiB limit as the checkpoint's `chat_template.jinja`.

The override handles both ordinary and tool-bearing chat requests; it does not retain a checkpoint's separate `tool_use` template. The checkpoint's tokenizer, special tokens and EOS IDs stay unchanged. Unreadable or malformed templates fail startup before the engine opens, without falling back to the checkpoint template. An empty file leaves chat rendering unavailable, as a checkpoint with no template does. Raw completion prompts do not use the chat template.

## API keys

Both servers accept repeated `--api-key KEY`, comma-separated `TENSORFOLD_API_KEY`, and `--api-key-file PATH`. The sources add keys together. A file has one key per line, optional `label: key` entries, and lines starting with
`#` for comments. Set its permissions to `0600`; startup refuses a file readable by other users.
The server stores SHA-256 digests and compares every configured digest for each authentication attempt.

Send `Authorization: Bearer KEY` from OpenAI-compatible clients, or `x-api-key: KEY` from Messages clients. Missing or invalid credentials return HTTP 401 with `WWW-Authenticate: Bearer` and the route's error format. With keys configured, the native server requires a key on every route by default. Its public exceptions are GET `/`, `/health`, `/slide` and the eight embedded `/slide/` scripts; `--metrics-open` also opens GET `/metrics` and `/v1/metrics`. Other methods and unknown assets require a key. This includes every `/systemone` inference alias and future private routes. `/health` remains open and returns only `{"status":"ok"}` when keys are configured. Public health is read-only, including when a reset query is supplied. With no keys, its health details remain available; a non-loopback bind prints a warning.

The native server raises its soft open-file limit before listening when the platform permits it. If `accept` runs out of descriptors, it logs one guidance line per burst and keeps a short retry backoff.

Replace the key file atomically to rotate keys. Its modification time is checked at most once a second; SIGHUP requests an immediate reload. An empty, unreadable or malformed replacement closes authenticated routes until a valid file returns. Request logs and `tensorfold:requests_total` use labels, never keys or digests. Unnamed keys receive `cli-N`, `env-N` or `file-N` labels. Labels contain at most 64 letters, digits, dots, underscores or hyphens; choose non-secret labels.

In the Python 0.6.6 engine's [control room](control.md), `tensorfold service install MODEL --api-key-file PATH` forwards the file to its server, and `tensorfold tui --url URL --token-env VARIABLE` sends that variable's value as a bearer key. The native binary has neither command. Prefer a restricted file over command-line keys, which can appear in the operating system's process list.


## Browser request headers

The server accepts origin-form targets beginning with `/`, with query text and trailing slashes interpreted by the
shared route helper. Absolute URLs, fragments and control bytes are refused before routing. Routes are case-sensitive
and percent escapes are not decoded into aliases. Malformed or folded header fields are refused and closed.

On a loopback bind, the native server accepts a `Host` naming `localhost` or a loopback IP, with any valid port.
It checks the actual bound socket, including a hostname that resolved to loopback. Missing, duplicate, malformed or
foreign hosts receive HTTP 403 before authentication, `100 Continue` or routing. Repeat `--allowed-host NAME` to add
trusted hostnames or bracketed IP literals. A name has no scheme, path or port; it admits that name at any port.
The flag changes header admission, not the listening address. A non-loopback bind skips this host restriction.

On every bind, an `Origin` must match the request's `Host` and the native server's HTTP scheme. Hostname case,
IPv6 spellings and default ports are normalized. The origin must be a single `http://HOST[:PORT]` or
`https://HOST[:PORT]` value with no path or userinfo. `null`, malformed values and duplicate origins are refused.
Repeat `--allowed-origin ORIGIN` for trusted exceptions, including an HTTPS origin used through a TLS proxy.
Host and origin exceptions are independent; allowing an origin does not admit a foreign loopback Host.
Forwarded headers do not select the scheme or bypass either check.

Each 403 names the refused header and its corresponding exception flag. Clients without an `Origin`, including curl,
SDKs and the app, keep their existing requests with a valid Host. Content-Type is not required by these checks, so
`curl -d` requests keep working. The server adds no CORS headers; these flags do not enable cross-origin browser reads.
The header checks protect browser access and do not replace API keys. Trailing spaces or tabs in Host or Origin values
are refused; clients should omit that outer whitespace.

Host and Origin exception lists each accept at most 64 entries. DNS names use nonempty ASCII labels without a trailing
dot; IPv6 literals are bracketed without zone identifiers. Every HTTP request needs exactly one valid Host, including
HTTP/1.0; hostless HTTP/0.9 is refused and closed. An external bind skips only loopback-name admission.

For a trusted proxy name and web origin, for example:

```sh
tensorfold-native serve MODEL --allowed-host inference.example --allowed-origin https://inference.example
```

| Route | Behavior |
| --- | --- |
| `GET /v1/models` | Served model ID and any configured aliases (both servers) |
| `GET /health` | Liveness, the served model, the queue and live rates; open with keys (see [Health](#health)) |
| `GET /memory`, `GET /v1/memory` | Native read-only memory snapshot ([memory diagnostics](memory-diagnostics.md)) |
| `GET /metrics`, `GET /v1/metrics` | Prometheus text: requests, KV occupancy, drafts and latency (both servers) |
| `GET /dashboard` | Native live dashboard page when `--dashboard` is enabled |
| `GET /stats` | Native JSON dashboard snapshot when `--dashboard` is enabled |
| `POST /v1/chat/completions` | Text chat, optional image input, tools and reasoning; streamed or non-streamed |
| `POST /v1/completions` | Raw text without a chat template, or token IDs |
| `POST /v1/messages` | Anthropic Messages: text, supported images, function tools and thinking; JSON or SSE |
| `POST /v1/messages/count_tokens` | Render the same model prompt without generating |
| `POST /tokenize`, `POST /v1/tokenize` | vLLM's: a `prompt`'s token IDs, or the IDs a chat request's `messages` render to |
| `POST /detokenize`, `POST /v1/detokenize` | vLLM's: the text of `tokens`, special tokens included |
| `POST /v1/responses` | OpenAI's Responses API, run as the equivalent chat completion; streamed or non-streamed |
| `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}` | A stored response, or remove it |
| `POST /v1/decisions` | Choice, score, and yes/no probabilities from the next-token logits; no text is generated |
| `POST /v1/imprint/resume` | How much of a chat body's conversation resumes, and from where ([on disk](#conversations-on-disk)) |
| `POST /v1/imprint/load` | A chat body's conversation state read from disk into RAM now |
| `DELETE /v1/imprint` | This model's conversation states on disk forgotten |

On MLX, a completions body containing a nonempty `messages` list uses chat handling. CUDA completions
take a string `prompt` (`add_special_tokens`, default false) or a list of token IDs, run as given.

`/tokenize` takes vLLM's fields: a `prompt` string (`add_special_tokens`, default true), or `messages` with the
chat fields that shape the prompt (`tools`, `reasoning_effort`, `chat_template_kwargs`) and `add_generation_prompt`
(default true). It returns `count`, `max_model_len` (the context window; null when none is set on MLX) and
`tokens`, and `token_strs` with `return_token_strs: true`. The IDs are the ones the chat route runs for the same
request, images and videos expanded. `/detokenize` takes `tokens` and returns `prompt`.
In the Python 0.6.6 engine, `--vision` lets supported Qwen3.5/3.8 dense checkpoints accept `image_url` content parts
alongside text in user messages and in tool results (`role: "tool"`), such as an agent's screenshots. The native
server reads images for GLM-5.3-Flash, Qwen3.8-27B and Gemma 4 31B on Metal, with data URLs, and public HTTPS URLs under
`--vision-urls` ([image input](vision.md#the-native-server)); for GLM-5.3-Flash it also takes `video_url` parts in
user messages ([video](vision.md#video-on-the-native-server)).
See [image input](vision.md) for data URLs, public image URLs, limits and cache behavior.
The native binary refuses `--vision` for other models and answers their image parts with HTTP 400.
Unsupported image input, audio, video and non-text output requests receive HTTP 400.

## Decisions

`POST /v1/decisions` is served by the MLX server, by the CUDA GLM engine and by Flash Next on one CUDA GPU.
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

## Living Weights

With `--slide`, the native server learns facts into the served weights. [Living Weights](living-weights.md) explains
each route and its events.

| Route | What it does |
| --- | --- |
| `POST /v1/slide/learn` | `{"text": "...", "questions": ["..."]}`: learns the facts in the text, streaming `fact`, `learning` and `learned` events; `questions` is optional (see Living Weights) |
| `GET /v1/slide/graph` | Every fact as JSON: its state, neurons, questions and their expected answers, and the model's pocket |
| `DELETE /v1/slide/graph` | Clears the graph; the weights keep what they learned |
| `DELETE /v1/slide/fact/{id}` | Deletes one fact from the weights, every other fact checked again, streaming `fact`, `checking`, `zeroed`, `checked` and `forgot` events; an optional `{"reason": "..."}` goes into the graph |

A delete's last event, `forgot`, says `"deleted": true` only when both the shards and the graph file hold the delete,
and its `state` is the fact's state in the graph file; a fact left `deleting` is settled by sending the delete again.
A server without `--slide` answers the learn and delete routes with HTTP 501. A delete sent while a learn or another
delete runs gets 503, an id that is not a number 400, and an id the graph does not hold 404.

## Request fields

| Field | Meaning | Backend |
| --- | --- | --- |
| `messages` | Text messages, including system, developer, assistant tool calls and tool results | Both |
| `tools` | OpenAI function tools | Both |
| `tool_choice` | `none` hides tools from the template; `required` or a named function makes the reply call a tool | Both |
| `parallel_tool_calls` | False returns at most one completed call | Both |
| `max_tokens`, `max_completion_tokens` | Explicit reply limit; rejected if prompt plus reply exceeds the window | Both |
| `temperature`, `top_p`, `top_k`, `min_p` | Sampling overrides; zero temperature is greedy | Both |
| `seed` | Explicit sampling key; native fallback uses configured defaults, then a fresh key (see below) | Both |
| `stream` | Server-sent events; the last event carries usage | Both |
| `stream_options.include_usage` | Usage in its own final event with `"choices": []`, not on the finish event | Both |
| `chat_template_kwargs.enable_thinking` | Template thinking toggle; `chat_template_kwargs.thinking` (`true`/`false` or `{"type": "enabled"}`/`{"type": "disabled"}`, as DeepSeek-V4 clients send it) is read the same way when `enable_thinking` is absent; other values are ignored | Both |
| `draft` | False selects the serial reference; CUDA rejects it if the engine has no serial switch | Both |
| `response_format`, `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`, `structured_outputs` | A JSON schema, any JSON object, a regex, a choice or an EBNF grammar the reply must match | Both |
| `ignore_eos` | Disable model end-of-sequence stopping; the reply limit still applies | Both |
| `stop` | Stop at a string or any string in a list; omit the matched text from the response | Both |
| `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max` | Both |
| `thinking_budget` | Token-count limit inside reasoning | Both |
| `priority` | `background` yields to foreground requests | Both |
| `return_token_ids` | The reply's token ids in the response's `tensorfold.token_ids`, the tokens its `token_sha256` hashes | Native |

`n` must be 1; multiple choices receive HTTP 400.

The native CUDA engine (Nemotron 3.5 Lightning) supports `logprobs: true` and `top_logprobs` from 0 through 20 on
`/v1/chat/completions`, streamed or not, without tools, stop strings, structured output or a thinking budget. Each
reply token has its `token`, `logprob`, `bytes` and requested `top_logprobs` in `choices[0].logprobs.content`; a
stream sends each row with the chunk whose `delta.content` completes that token's text, so the rows joined over the
chunks are the whole reply's. Probabilities describe the raw target-model distribution at temperature 1, before
temperature, top-k or top-p sampling filters, including when generation is greedy, and are the same with drafts on or
`"draft": false`. With thinking on, the rows cover the answer only, as `content` does: none for the reasoning,
`</think>` or the newlines after it, so the first row is the answer's first token, and a reply cut inside its
reasoning has none. Alternatives are tokenizer tokens and may include leading spaces; `bytes` preserves partial UTF-8
sequences. `/v1/completions` takes the legacy `logprobs`, an integer from 0 through 20 alternatives, and answers in
the legacy form, streamed or not: `tokens`, `token_logprobs`, `top_logprobs` (each alternative's text to its logprob)
and `text_offset` (each token's first character in the completion, counting each token's own text). Other engines,
the Responses route and the request modes above return HTTP 400 when probabilities are requested.
Nonstreamed `/v1/completions` also takes vLLM's `prompt_logprobs`, an integer from 0 through 20, on the same engine
and on GLM-5.3-Flash's CUDA engine.
`choices[0].prompt_logprobs` has one entry per prompt token: `null` for the first, then an object keyed by token id
(a string) whose first key is the prompt token itself, followed by the requested best tokens; each value has
`logprob`, `rank` and `decoded_token`. The prompt token's `rank` counts every vocabulary token at or above it,
itself included, as vLLM does, and the best tokens are ranked by place. Such a request never resumes from the prompt
cache, so its whole prompt is prefilled. Chat, streamed requests and other engines return HTTP 400 with it.
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
between chunks for `--decode-share` of each chunk's time (default 0.25). That flag is the Python 0.6.6 engine's, and the
native binary refuses it. In that engine, `--decode-share 0` prefills each prompt whole
first, in arrival order, as 0.3.6.2 did. Background work waits behind foreground requests. An active background
request yields when a foreground request needs its lane or memory, then restarts with already-delivered tokens
suppressed.
Session-title requests are also treated as background work.

`--name-priority ID=background` gives one served id (`--name` or an `--alias`) a default priority: a request that
asks for it and sends no `priority` of its own is treated as `priority: "background"`. The request's own `priority`
field always wins over the default. This lets one server answer to several ids at their usual priority while a
background-only client (a batch extractor, say) gets one id that always yields, without sending the field itself.

The native CUDA build serves Nemotron, and its concurrent requests share each round. The rest of this paragraph
describes the Python 0.6.6 engine's CUDA families.
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
(a structured answer), and malformed tool blocks, which keep their text. A GLM argument key written without its
`<arg_key>` gets that tag back, and the call parses. A well-formed call to a tool the
request did not offer is returned as a tool call under its valid name, so the client can report the unknown
tool and the model can correct itself.

When a model end token arrives inside a whole tool call before its closing markup, the server closes the
markup and returns the call. A GLM value the end token cut off is closed with `</arg_value>` first, and the call
returns when it then parses strictly. Any other partial value, a missing key value, an empty block, or an unoffered
call that cannot parse as one complete call remains content.

With `parallel_tool_calls: false`, the server buffers tool deltas until it can return the first valid
completed call. Prose and reasoning can still stream. Usage counts the entire decoded reply, including
additional calls omitted from the response. Otherwise both servers stream each Qwen XML call as it is written: a
delta with the call's id and name, then its arguments in pieces; a call the streamer can't follow arrives whole at
the end.

On the Python 0.6.6 server, with `tool_choice: "required"`, or a function named in `tool_choice`, the reply's answer (after any think block
or thought channel) opens a call to an offered tool. The server replaces the first answer token that isn't
whitespace with the tool-call opener (`<tool_call>`, or Gemma 4's `<|tool_call>`) and the template's text before a
tool name, then holds the name to the offered tools: a token that leaves them is replaced by the rest of the first
offered name its written part starts. The template's own rendered call gives that text. A named function is the
only tool the template offers. Each fix depends only on the tokens before it, so drafted, serial and concurrent
decoding write the same call. The MLX engine fixes tokens inside its rounds; CUDA stops the engine at a fix and
decodes on from the reply. The model writes the arguments; a malformed call returns as content.

## Structured output

This paragraph describes the Python 0.6.6 engine, where @philip-pentatonic wrote structured output in
[#80](https://github.com/ashhart/TensorFold/pull/80); the next paragraph covers the native server.
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

The native server enforces the same fields with the same grammars through `libtfgrammar`, a small C ABI over
xgrammar v0.2.8's C++ library (`zig/src/grammar/tf_grammar.cc`). Release archives ship it in `lib/` beside `bin/` for
macOS arm64, Linux x86_64 and Linux aarch64; from source, `zig build grammar -Dxgrammar=DIR` builds it from an xgrammar
v0.2.8 checkout (`git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar`) into
`zig-out/native/lib` on macOS and `zig-out/lib` on Linux. The server looks in `../lib` beside its binary, then beside
it, then on the loader's path, and loads it at the first structured request; without it, everything else serves and a
structured request gets HTTP 400. The native engines listed below expose token grammar support; other engines
refuse structured requests. Drafting stays on: the drafts the head holds on the GPU come
back to the host before each verify, the grammar cuts the ones it rejects and drafts the text it forces itself (a key,
closing markup), and each row's logits are masked before its draw, so drafted, `"draft": false` and concurrent replies
are the same.

On the native server, `/v1/chat/completions` and `/v1/responses` enforce `tool_choice: "required"` and a named
function with the engine's token grammar. Chat uses `{"type":"function","function":{"name":"lookup"}}`;
Responses uses `{"type":"function","name":"lookup"}`. Only offered tools are allowed; an unknown name or a required
call with no tools gets HTTP 400. `auto` remains optional, and `none` hides the tools.

The constrained answer contains one or more complete calls and no prose outside them. Arguments follow the offered
parameter schema. GLM uses `<arg_key>` and `<arg_value>` tags after the function name.
Qwen XML and JSON call templates keep their envelopes. Gemma 4 uses
`<|tool_call>call:NAME{...}<tool_call|>` with JSON arguments, which the tool parser accepts. An unrecognized template
uses the canonical `<tool_call>{"name":"NAME","arguments":{...}}</tool_call>` form. The native grammar limits
whitespace between calls; `parallel_tool_calls: false` allows exactly one. With thinking enabled, the constraint
starts after the think-end token. A reply cut short by `max_tokens` can remain incomplete and ends as `length`.
Drafting stays enabled and uses the same grammar masks as plain decoding.

This requires the loaded backend's structured-output capability and `libtfgrammar`. Nemotron, Qwen3.5-2B,
Qwen3.8-27B (`qwen27`), Gemma 4 31B, Step-5, LFM2 MoE and Flash Next shared sessions on Metal have that capability; Qwen27 on Intel also has it.
Flash Next on CUDA has it too.
Qwen3.8-27B applies the grammar to its first draw, each verified row and each branch of its draft proposals.
Structured requests use its shared lane core, including when the ordinary lone-request serial optimization is enabled.
CUDA determines support from its loaded backend's mask hook. GLM Metal
(`glm-zig`, `glm-dsa-zig`) lacks that hook and refuse required/named choices before generation with HTTP 400, for example:
`backend 'glm-zig' cannot enforce tool_choice "required" or a named function: it has no token grammar support; send "auto" or "none"`.
The server does not silently turn a required call into `auto`. Installing the grammar library does not add a missing
backend mask hook. A missing library or grammar compilation error is also refused before generation with its reason.

The native GLM CUDA backend also supports token grammars, gathering vocabulary rows before rank 0 masks and draws.

HTTP 400 comes before any token for a malformed field, a grammar xgrammar cannot compile (with xgrammar's reason), a
server without xgrammar (with the install command; on the native server, a server that cannot load `libtfgrammar`),
a grammar sent with `tool_choice: "required"` or a named function, and `logprobs` with a grammar or tools. A reply whose
grammar fails while decoding ends with HTTP 500 (an error event when streaming); the other requests go on.

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
empty think block. GLM-5.3 keeps every earlier assistant turn's reasoning in the prompt, as zai-org's template does by
default (`clear_thinking` false), also on checkpoints whose template still clears it before the last user message, so
a new user message leaves the earlier turns' tokens, and their kept prompt states, as they were. The native server
gives the template `clear_thinking` false on every render, and `TF_GLM_CLEAR_THINKING=1` sets it true for the server,
from @MiaAI-Lab's [#236](https://github.com/ashhart/TensorFold/pull/236); any value but 0 or 1 refuses to load. A
request's `chat_template_kwargs.clear_thinking` (true or false) overrides the server's setting for that request, as the
model card advises `true` for plain chat; any other value is a 400, and other families' templates never receive it.

A tool call written before the think block closes is the reply's tool call when the reply ends inside the block,
on both backends; the reasoning stops where the call starts, and streamed reasoning never carries the call's markup.
A call only mentioned while thinking, with the block closed after it, stays reasoning.

`thinking_budget` (the request's; absent or 0 takes `--thinking-budget`) replaces the reply's budget-th token,
while it is still thinking, with a newline, the closing think marker and a blank line, then continues the answer.
The cut depends on token count, so serial and drafted decoding use the same cut. A model that closes the block
earlier is left alone. The MLX engine forces the close inside its rounds; CUDA stops the engine at the cut and
decodes on from the reply, as for a required tool call. Flash Next forces it inside its GPU-side rounds, on one Mac
and in speed-up mode.

With thinking enabled, the native server refuses a positive budget with HTTP 400 before opening a stream or
submitting the request if the engine does not support thinking budgets. This also applies to a positive
`--thinking-budget` inherited by an absent or zero request field. With no positive budget, requests remain valid
on those engines; with thinking disabled, the budget is not applied.

`--thinking-reserve N` (off by default) keeps a client's `max_tokens` for the answer when thinking is on. The
server adds N to the reply's limit, capped by the context window, and the engine ends the reply when the tokens
after the closing think marker reach the request's `max_tokens` (`finish_reason` `length`). Reasoning shorter than
N leaves the answer its full `max_tokens`; longer reasoning counts against it. Lane engines apply it; on other
engines, and with thinking disabled, `max_tokens` limits the whole reply as before.

The native `--loop-guard` flag is off by default and watches only an open think block. It closes a repeated exact
cycle of at most eight tokens after 256 predecessor matches, when the detected region starts at least 64 reply tokens
in. It also closes a longer passage repeated verbatim: once a period of nine tokens or more has repeated in full and
for at least 1,024 tokens, the period found from where the reasoning's last 32-token run occurred before. The reply
continues after the forced close, and reports `runtime.loop: {"period": N}` plus `loop=period:N` in the server log. The native engine refuses the flag on backends that do not enforce it, and prints that refusal as
a startup error.

## Context and errors

The rendered prompt and reserved reply must fit the effective context. In 0.3.5, an explicit `max_tokens`
or `max_completion_tokens` that would put prompt plus reply beyond the window is rejected with counts
and fitting guidance before generation. MLX returns HTTP 400 for non-streamed requests or an
`invalid_request_error` event after opening a stream. CUDA returns HTTP 400 before opening a stream.
The 0.3.4.1 MLX server capped that explicit limit to the remaining context.
A prompt that leaves no room for a reply is refused the same way, and on both backends every such refusal
carries OpenAI's `context_length_exceeded` code, the field it is about in `param` (`messages` for a chat completion,
`prompt` for a completion), and a message that starts "This server's maximum context length is N tokens", so clients
that compact a conversation on that error do so. An image prompt that expands past the window is refused the same
way, and so are GLM-5.3's own context refusals on CUDA.
When the request omits the reply limit, the server still caps its configured default to the remaining context.
CUDA returns HTTP 400 before generation when the chat template rejects the request or
`chat_template_kwargs` is neither an object nor null. A generation error returns HTTP 500 for a
non-streamed request; after a stream opens, both backends send an error event of type `server_error`,
then `[DONE]`. On two CUDA ranks, which only the Python 0.6.6 engine runs, such an error can leave the ranks out
of step, so restart both: the 27B's
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

On a discrete card (an RTX 3090, an A100) the server sizes its window from the card's own free memory, as the driver
reports it, less a reserve of a 32nd of the card and at least 1 GiB. `TENSORFOLD_MEMORY_RESERVE_GIB` sets that one from
0.5 GiB up.

## Responses

`choices[0].message.content` holds the answer. Reasoning uses `reasoning_content`, or
`delta.reasoning_content` while streaming. Tools use `tool_calls` and `finish_reason: "tool_calls"`.
When a template supplies no think opener, a reply-written leading `<think>` tag is removed from
`reasoning_content`; the streamed and finished forms use the same split.
Every reply's usage, streamed ones included, carries its prompt, completion and total tokens,
`prompt_tokens_details.cached_tokens` (the prompt tokens found in a backend's prefix cache; native Zig reports
zero until its prompt cache is enabled) and
`completion_tokens_details.reasoning_tokens` (the thinking tokens, through the closing think marker); timing details
depend on the backend.
TensorFold also reports generation statistics such as decode rate, time to first token and draft acceptance.

Native JSON replies also carry a `tensorfold` runtime block with `engine`, `enable_thinking`,
`reasoning_effort`, `tokens_per_second`, `seconds`, `time_to_first_token`, `prefill_seconds`,
`prefill_widths`, `prefill_raised`, `sampling`, `drafts`, `token_sha`, and the draft counters
`rounds`, `drafted`, `accepted`, `acceptance_rate`, and `tokens_per_round`. A request that sends
`return_token_ids` adds `token_ids`, the reply's tokens, to the block — the tokens `token_sha` and
`token_sha256` hash — on the whole reply and on the streamed finish event alike. A loop-guarded reply adds
`runtime.loop.period`; these fields describe the native reply and are absent
from Python replies that do not provide them.
In the native block, `seconds` is the reply's decode time, from the end of its prompt pass to its last token, and
`tokens_per_second` divides the reply's tokens after the first by it. `prefill_seconds` is the prompt pass, and
`time_to_first_token` counts from the request's arrival, so a queued or long prompt shows there, not in `seconds`.
The Python 0.6 server's `seconds` is the whole request, prompt included.

The native server's finish log names the client reply ID. Completed requests report `prompt`, `cached`,
`thinking`, `effort`, `tokens`, `sha`, `finish`, `rounds`, and draft acceptance; a refused request reports
`refused <id>` and its message, while a submitted request that ends before a reply reports `ended <id>` with
`reason=client-left` or an error plus `prompt`, `tokens`, and elapsed time.

On the Python 0.6.6 CUDA server, `GET /health` (`{"ok": true, "backend": "tensorfold", ...}`) also carries counters
a poller can difference into rates: `busy`, `requests_running`, `requests_total`, `completion_tokens_total` (the
running replies' tokens included as they stream), and totals that move when a request ends, taken from the engine's
own statistics: `prompt_tokens_total`, `cached_tokens_total`, `prefill_seconds_total`, `decode_seconds_total`,
`rounds_total`, and `drafted_total` and `accepted_total` where the engine reports them. A `--parallel` server adds
`streams` (decoding, prefilling and the maximum), and `context_length` is the served window. The first sample is a
baseline, not a rate. The native server's `/health` carries none of these counters ([Health](#health) lists its
fields); its request, token, round and draft counters are in [Metrics](#metrics).

For exactness comparisons, hold the checkpoint, template, runtime, prompt, seed and sampling settings
constant, then compare the decoded reply with `draft` enabled and disabled. Repeat with fresh and reused
prefixes, and compare each MLX concurrent request with its solo run.

Native sampled requests use the request seed, then an injected configured default, then a fresh 63-bit OS key. No stock
serve flag or generation config supplies a default seed. The done line appends `seed=<n>` after
its request timing and tok/s fields, for given and drawn keys. Send that seed again to reproduce a draw. Greedy requests draw
no key. Native `serve --prompt-seed` keeps historical prompt keying, including `TENSORFOLD_SEED_SALT`; CLI runs and
speech keep their existing prompt keying. Always give comparison requests the same explicit seed, including release
cells, drafted/plain, cached/fresh and concurrent/solo checks. See [request seeds](request-seeds.md).

## Metrics

`GET /metrics` and `GET /v1/metrics` answer as Prometheus text on both servers. Each family is read on its own
at scrape time (the scrape is not one atomic snapshot), and a family the server cannot count honestly is left
out of the text rather than reported at a permanent zero.

The scrape carries `requests_running`, `requests_waiting`, `prompt_tokens_total`, `generation_tokens_total`,
`generation_tokens_running`, `kv_cache_usage_ratio` (one `pool` label per live stream cache), `mtp_drafted_total` and `mtp_accepted_total`
(draft tokens verified and kept on finished requests; the engines keep one draft counter, so copies and chain
drafts share it), `request_latency_seconds`, `time_to_first_token_seconds` and `request_decode_seconds` (each
finished request's decode time: the CUDA engine's own figure, or first token to end on the Mac server; its `_sum`
over `generation_tokens_total` is the decode rate), `request_time_per_output_token_seconds`, `decode_rounds_total`,
and the `request_prefill_seconds` histogram, all under the `tensorfold:` prefix. TPOT already carries vLLM's name.
The readings vLLM also has are repeated under its names (`num_requests_running`, `num_requests_waiting`,
`kv_cache_usage_perc`, `spec_decode_num_draft_tokens_total`, `spec_decode_num_accepted_tokens_total`,
`e2e_request_latency_seconds`, `request_decode_time_seconds`, `request_prefill_time_seconds`) with identical values, so a dashboard copied from vLLM fills by swapping the
`tensorfold:` prefix for the metric name. `client_disconnections_total` (requests the client walked away from)
and `preemptions_total` (background work that gave up a lane to a later request) are published where the
server counts those events, and never at a fabricated zero.
An engine that keeps prompt states between requests (Flash Next and GLM-5.3-Flash unless `--prompt-cache-gib 0`)
adds `prompt_tokens_cached_total` (already vLLM's name), the prompt tokens of finished requests restored from a kept
state, and repeats it with `prompt_tokens_total` as vLLM's `prefix_cache_hits_total` and `prefix_cache_queries_total`.
An engine that keeps no prompt states leaves all three out.

The native `--dashboard` flag adds a self-contained `/dashboard` page and its `/stats` JSON snapshot; both routes
return 404 when the flag is off, and the page polls once per second without adding access-log noise.

## Conversations on disk

`--learn-conversations` (it implies `--learn`) writes every prompt state the server keeps in RAM to disk as well: a
conversation's history, the end of each reply, the next turn's background prefill and the shared prefixes, all in one
store for each model and build. That store takes the place of `--learn`'s store of shared prefixes: a server started
with the flag neither reads nor writes the prefix files a plain `--learn` server keeps, and they stay on disk until the
cap below needs their room. After a restart, a crash or an eviction from RAM, the next turn reads the latest state back
instead of reading the conversation again from its first token. Qwen3.8-27B, Nemotron, Flash Next and GLM-5.3-Flash on
one Mac and Nemotron on CUDA keep them; other engines refuse the flag at startup.

States go to disk while the engine is idle, a 16 MiB piece at a time, so a request that arrives meanwhile waits for one
piece. A planned stop (SIGTERM or SIGINT) finishes what is waiting, within 20 seconds; a killed server loses only what
its last turn had not written yet. Rows a stored state already holds for the same tokens are written once, so ten
conversations that share a long system prompt and tools store those rows once. A state is read back only when reading
it takes less time than prefilling the tokens it covers, measured on this machine. Every piece read is checked against
its hash; a damaged file is deleted, with the states that need it, and the turn prefills instead. A read-back state
gives the same reply, token for token, as the state it was written from and as a fresh prompt pass.

States are kept per checkpoint, kernel build, OS build and chip: a macOS update or a new release that computes other
bits starts a new set, and the old set is the first to go when the disk cap needs room. `--learn-gib` caps the disk
they take with `--learn`'s shared prefixes, every model together (32 GiB by default); past it, other models' and
builds' sets go first, then the states worth least: the fewest tokens saved per byte only they hold, the least recently
used. `--learn-min-free-gib` keeps free disk (4 GiB by default), and `--learn-daily-gib` caps what they write a day
(32 GiB by default); past either, a state stays in RAM only, with a log line, and no reply changes.

On a Mac, `--spill-gib N` writes a state to that store only when the prompt cache evicts it from RAM, up to N GiB
under `~/.cache/tensorfold/spilled` for every model together, and reads it back as above: a later request whose prefix
matches resumes from it, token for token as a fresh prompt pass, after a restart too. Nothing is written while the
engine idles; the write is made on the engine thread at the eviction, before the request that needed the room goes on.
Past N GiB other models' spilled sets go first, then the state this server used least recently. A damaged file is
deleted and its tokens are prefilled, and a write cut off midway leaves nothing a later turn or a restart reads.
`--checkpoint-slots N` sets how many states stay in RAM beside the store, so the next one sends the oldest to disk; on a
Mac it needs `--spill-gib` or `--learn-conversations`. With `--learn-conversations` too, every state is still written
behind, an evicted state's write under way is finished instead of given up, and the cap is the smaller of `--learn-gib`
and `--spill-gib`. `--learn-min-free-gib` and `--learn-daily-gib` apply to spilled states as well. The same four Mac
families take both flags and the others refuse them at startup; without either flag nothing changes, and the routes
below stay off.

Three routes let a client such as the Mac app ask about a conversation. Each takes the body of a chat completion
(`messages`, `tools`, `chat_template_kwargs`), renders it as that request would be rendered, and answers on the engine's
thread between rounds. A server started without `--learn-conversations` answers each with 404 and the error message
"saved conversations are off: start the server with --learn-conversations"; an engine that keeps no prompt states
answers 501.

- `POST /v1/imprint/resume` answers how much of the conversation's next request would resume, and from where:
  `tokens` (the rendered prompt), `ram_tokens` and `disk_tokens` (the longest kept state in RAM and on disk),
  `disk_bytes` (what reading that state takes), `resume_from` (`ram`, `disk` or `none`), `wait_seconds` (the estimated
  wait before the first token: the read, then prefilling the rest) and `disk` (whether this server keeps states on
  disk).
- `POST /v1/imprint/load` reads the conversation's disk state into RAM now, so opening an old chat makes its next send
  fast. It answers as `/v1/imprint/resume` does after the read, with `loaded_tokens` and `load_seconds` added (0 when
  RAM already held as much, or a prompt pass would be quicker than the read).
- `DELETE /v1/imprint` forgets every conversation state this server's model and build keep on disk: it deletes the
  store's state and row files and answers `freed_bytes`. What stays holds no conversation data: the store's folder with
  its `used` stamp, and the `written` and `clock` files at the top of the `--learn-dir` folder (the bytes written today,
  and the order the stores were last opened in). States in RAM stay until they are evicted.

What stays on disk: under the `--learn-dir` folder (`~/.cache/tensorfold/learned` by default), directories readable by
their owner only (0700) hold files readable by their owner only (0600). They contain token ids and the model's own
prompt states (attention keys and values, recurrent and convolution states), never text, and nothing in them leaves the
machine. Token ids decode back to the conversation's text with the model's tokenizer, so treat the folder as you treat
the conversations. `DELETE /v1/imprint` clears one model's states; deleting the folder while no server runs clears all.

## Health

The native server answers `GET /health` (and `GET /`) from one handler on Metal and CUDA, without a key. With keys
configured the body is only `{"status": "ok"}`. Without keys it has these fields, in this order:

| Field | Type | Meaning |
| --- | --- | --- |
| `status` | string | Always `"ok"` |
| `model` | string | The served name (`--name`) |
| `model_ids` | list of strings | The served name, then each `--alias`, as `/v1/models` lists them |
| `max_batch_size` | integer | The lanes the engine serves at once |
| `policy` | string | The engine's policy line; absent when the engine reports none |
| `warming` | boolean | Always `false` on the native engines |
| `memory` | object | CUDA: `active`, `cache` and `peak` in bytes (`cache` is 0). Metal: `{}` |
| `live` | object | `connections`, `waiting`, `decode_tokens_per_second`, `prefill_tokens_per_second` |
| `version` | string | The release this server was built from |
| `backend` | string | The backend built in: `metal`, `cuda`, `hip` or `xpu`; absent from a build without one |
| `device` | string | The GPU the server opened, by the driver's name; absent when it cannot read one |

`live.connections` counts running and waiting requests, and `live.waiting` the waiting ones.
`decode_tokens_per_second` is the tokens decoded in the last 2 s, divided by 2. `prefill_tokens_per_second` is the
last prompt pass's uncached tokens over its time, shown for 2 s after the pass ends and 0 otherwise. Both rates are
rounded to one decimal. Public health never resets the peak, with or without keys. Reset queries on `/health` and `/`
are ignored. The separate `/memory` route also reports memory without resetting it and requires a key when keys are configured.

The Python 0.6.6 server's body differs: its MLX `memory` can add `budget`, `mlx_budget` and `footprint`, and its CUDA
server sends the counters described under [Responses](#responses). For memory figures on a Mac, use
[`/memory`](memory-diagnostics.md) (it needs a key when keys are configured) or `/metrics`'
`tensorfold:process_footprint_bytes`.

## The Responses API

`POST /v1/responses` takes OpenAI's Responses request and runs it as the equivalent chat completion, through the
same handler and engine path. A response has that chat completion's prompt and tokens (the `tensorfold` block's
`token_sha` matches), drafts, and equals its `"draft": false` run and its solo run.

- `input` is a string or a list of items: messages (`input_text`, and `input_image` on a server started with
  `--vision` for a model that reads images), `function_call`, `function_call_output` (its
  `output` text, or `input_text` and `input_image` parts, the images again only with `--vision`), and `reasoning`
  items with their `content` text, which the template gets
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
others), `background`, `conversation`, `prompt` templates, `truncation: "auto"`, `top_logprobs`, `input_file` parts
and file IDs, `item_reference` items, encrypted reasoning items, and a `previous_response_id` that is not stored.
`include` takes a list of OpenAI's names, which the server accepts and ignores (it has nothing to add for them):
`code_interpreter_call.outputs`, `computer_call_output.output.image_url`, `file_search_call.results`,
`message.input_image.image_url`, `message.output_text.logprobs`, `reasoning.encrypted_content`,
`web_search_call.action.sources` and `web_search_call.results`. Any other name, or a value that is not a list of
strings, is refused.

## Anthropic Messages

The Messages routes reuse the same chat handler and engine on MLX and CUDA. They accept `system`, text/image
blocks (including mid-conversation system text), `tool_use`/`tool_result`, custom tools and `tool_choice`, sampling, `stop_sequences`, `thinking`
(disabled, enabled with `budget_tokens`, or adaptive), and `output_config` effort/JSON schema. Image support
requires a vision-capable model served with `--vision` (natively GLM-5.3-Flash and Qwen3.8-27B on a Mac); other
models answer image blocks with HTTP 400. Thinking is off unless enabled or adaptive. It round-trips as plaintext with an
empty signature;
`display` does not suppress it. Claude Code's `context_management` keep-all thinking directive is accepted. Mid-conversation system text stays
in place with its system role; the model's chat template must support later system messages. Turn-scoped
system messages, per-message output configuration and inline tool changes are unsupported.
JSON schema output, including Claude Code title requests, requires `pip install 'tensorfold[grammar]'`.
That install is the Python 0.6.6 engine's; the native server uses `libtfgrammar` from the release archive instead
([structured output](#structured-output)).

Usage separates uncached `input_tokens` from `cache_read_input_tokens` and reports `output_tokens_details.thinking_tokens`
when the backend counts them. Prefix caching remains automatic,
so `cache_control` hints do not allocate an Anthropic cache or report cache-creation tokens. Errors use the
Anthropic error envelope, including after an SSE stream opens. Server-side tools, documents/file IDs,
redacted thinking and other context edits return HTTP 400.

Connect Claude Code using the ID from `/v1/models`:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=local claude --model local-model
```
