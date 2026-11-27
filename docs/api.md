# Compatible APIs

The base URL is `http://127.0.0.1:8080/v1` with the default server settings.

Native `serve --profile stable` keeps the current defaults. [Profiles](profiles.md) documents its initial
environment-switch registry, explicit override precedence, and startup reporting.

## Native chat template override

`tensorfold serve MODEL --chat-template custom-chat.jinja` loads a UTF-8 Jinja template from that file
instead of the checkpoint's template files or `tokenizer_config.json` templates. Relative paths resolve from the server's working directory. The file is read once at startup, with the same 16 MiB limit as the checkpoint's `chat_template.jinja`.

The override handles both ordinary and tool-bearing chat requests; it does not retain a checkpoint's separate `tool_use` template. The checkpoint's tokenizer, special tokens and EOS IDs stay unchanged. Unreadable or malformed templates fail startup before the engine opens, without falling back to the checkpoint template. An empty file leaves chat rendering unavailable, as a checkpoint with no template does. Raw completion prompts do not use the chat template.

## API keys

Both servers accept repeated `--api-key KEY`, comma-separated `TENSORFOLD_API_KEY`, and `--api-key-file PATH`. The sources add keys together. A file has one key per line, optional `label: key` entries, and lines starting with
`#` for comments. Set its permissions to `0600`; startup refuses a file readable by other users.
The server stores SHA-256 digests and compares every configured digest for each authentication attempt.

Send `Authorization: Bearer KEY` from OpenAI-compatible clients, or `x-api-key: KEY` from Messages clients. Missing or invalid credentials return HTTP 401 with `WWW-Authenticate: Bearer` and the route's error format. With keys configured, the native server requires a key on every route by default. Its public exceptions are GET `/`, `/health`, `/slide` and the eight embedded `/slide/` scripts; `--metrics-open` also opens GET `/metrics`, `/v1/metrics` and `/v1/load`. Other methods and unknown assets require a key. This includes every `/systemone` inference alias and future private routes. `/health` remains open and returns only `{"status":"ok"}` when keys are configured. Public health is read-only, including when a reset query is supplied. With no keys, its health details remain available; a non-loopback bind prints a warning.

The native server raises its soft open-file limit before listening when the platform permits it. If `accept` runs out of descriptors, it logs one guidance line per burst and keeps a short retry backoff.

Replace the key file atomically to rotate keys. Its modification time is checked at most once a second; SIGHUP requests an immediate reload. An empty, unreadable or malformed replacement closes authenticated routes until a valid file returns. Request logs and `tensorfold:requests_total` use labels, never keys or digests. Unnamed keys receive `cli-N`, `env-N` or `file-N` labels. Labels contain at most 64 letters, digits, dots, underscores or hyphens; choose non-secret labels.

In the Python 0.6.6 engine's [control room](control.md), `tensorfold service install MODEL --api-key-file PATH` forwards the file to its server, and `tensorfold tui --url URL --token-env VARIABLE` sends that variable's value as a bearer key. The native binary has neither command. Prefer a restricted file over command-line keys, which can appear in the operating system's process list.

### Key scopes

A native key file can limit a key to some routes. Write the scopes in brackets after the label:

```text
app[inference]: APP_KEY
grafana[read]: MONITOR_KEY
notebook[inference,read]: ANOTHER_KEY
ops[admin]: OPERATOR_KEY
legacy: OLD_KEY
```

A key without brackets, from a plain file line, `--api-key` or `TENSORFOLD_API_KEY`, keeps every route, as before. `[admin]` is
the same as no brackets. The other scopes are `inference`, `read` or both, comma-separated; an unknown, repeated or empty
scope, or `admin` beside another, makes the file invalid. One key cannot carry two scope sets: a file that gives the same key
different scopes, or scopes a key the flags or environment already give, is refused at startup and on reload.

| Scope | Routes |
| --- | --- |
| `inference` | POST `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`, `/v1/messages/count_tokens`, `/v1/tokenize`, `/v1/detokenize`, `/v1/apply-template`, `/v1/audio/speech`, `/v1/images/generations`, `/v1/embeddings`, `/v1/decisions`, `/v1/systemone`, `/score`, `/v1/score`, `/v2/rerank`, `/v1/rerank` (also by PUT), `/v1/ocr`, `/ocr`, and the same paths without `/v1` where the server has them; GET `/v1/models`, `/models`, `/v1/models/{id}`, `/models/{id}`, `/v1/voices`; a WebSocket upgrade on GET `/v1/responses` or `/responses` |
| `read` | GET `/metrics`, `/v1/metrics`, `/v1/load`, `/memory`, `/v1/memory`, `/stats`, `/dashboard`, `/v1/benchmark/manifest`, `/v1/execution/witness`, `/v1/models`, `/models`, `/v1/models/{id}`, `/models/{id}`, `/v1/voices` |

A scoped key reaches nothing else: not the stored Responses (GET or DELETE `/v1/responses/ID`), Recall, Sliding Weights,
`/v1/imprint`, adding or deleting voices (`--manage-voices`), `/v1/prompt_cache/flush`, another method, or a suffix
alias such as `/alt/completions` that the server answers for a full key. The route
is matched exactly after the query and trailing slashes are cut. A refused route answers 403 with the code
`insufficient_scope` (Messages routes: an Anthropic `permission_error`) and closes the connection before the body is read
or a parked model reloads. The public routes stay public.

The stored Responses belong to the server, not to a key, so a scoped key's `POST /v1/responses` must send `"store": false`
and no `previous_response_id`, with the whole conversation in `input`; anything else answers 403 naming the field. Over
a Responses socket, a `previous_response_id` that the socket's own cache holds is allowed. A request keeps the scopes
it was accepted with if the file changes while it runs. Scopes are authorization only: every key shares one model,
its prompt cache and the metrics.

A reply's binding carries the read-scoped execution witness, so `"execution_witness": true` needs `read` beside
`inference`; a key without `read` answers 403 `insufficient_scope` naming the field before the request is planned.


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
`curl -d` requests keep working. `--allowed-origin` adds no CORS headers and does not enable cross-origin browser reads.
The header checks protect browser access and do not replace API keys. Trailing spaces or tabs in Host or Origin values
are refused; clients should omit that outer whitespace.

Host and Origin exception lists each accept at most 64 entries. DNS names use nonempty ASCII labels without a trailing
dot; IPv6 literals are bracketed without zone identifiers. Every HTTP request needs exactly one valid Host, including
HTTP/1.0; hostless HTTP/0.9 is refused and closed. An external bind skips only loopback-name admission.

For a trusted proxy name and web origin, for example:

```sh
tensorfold-native serve MODEL --allowed-host inference.example --allowed-origin https://inference.example
```

A browser page on another origin reads replies only from origins named with `--cors-origin ORIGIN` (repeatable, at
most 64, the same `http(s)://HOST[:PORT]` form; no wildcard). Each one is also an allowed origin. A preflight
(`OPTIONS` with `Access-Control-Request-Method`) from a listed origin is answered before keys are checked: 204 with
`Access-Control-Allow-Origin` echoing that origin, `Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS`,
`Access-Control-Max-Age: 600` and `Access-Control-Allow-Headers` naming `authorization`, `x-api-key`,
`content-type`, `anthropic-version`, `anthropic-beta` and `anthropic-dangerous-direct-browser-access`, followed by any
requested `x-stainless-*` header (the OpenAI and Anthropic browser SDKs send these; letters, digits, `-` and `_` after
the prefix), lowercased. Another method or requested header refuses the whole preflight with 403. An unlisted origin
gets the 403 above.
A listed origin's request that passes the header checks gets `Access-Control-Allow-Origin` on its reply, key and
scope refusals and SSE streams included; with the flag set, every reply past those checks carries `Vary: Origin`. Credentials mode
is not offered (no `Access-Control-Allow-Credentials`): send the key in `Authorization` or `x-api-key`, which gated
routes still require when keys are on. Without the flag no reply carries a CORS header.

| Route | Behavior |
| --- | --- |
| `GET /v1/models` | Served model ID and any configured aliases (both servers); native entries carry [capabilities](#model-capabilities) |
| `GET /v1/models/{id}` | One served ID or alias: the entry `/v1/models` lists for it, or 404 `model_not_found` (native server; see [One model](#one-model)) |
| `GET /health` | Liveness, the served model, the queue and live rates; open with keys (see [Health](#health)) |
| `GET /memory`, `GET /v1/memory` | Native read-only memory snapshot ([memory diagnostics](memory-diagnostics.md)) |
| `GET /metrics`, `GET /v1/metrics` | Prometheus text: requests, KV occupancy, drafts and latency (both servers) |
| `GET /v1/load` | The scheduler load ledger as one JSON object (native server); see [Scheduler load](#scheduler-load) |
| `GET /dashboard` | Native live dashboard page when `--dashboard` is enabled |
| `GET /stats` | Native JSON dashboard snapshot when `--dashboard` is enabled |
| `POST /v1/chat/completions` | Text chat, optional image input, tools and reasoning; streamed or non-streamed |
| `POST /v1/completions` | Raw text without a chat template, or token IDs |
| `POST /v1/messages` | Anthropic Messages: text, supported images, function tools and thinking; JSON or SSE |
| `POST /v1/messages/count_tokens` | Render the same model prompt without generating |
| `POST /tokenize`, `POST /v1/tokenize` | vLLM's: a `prompt`'s token IDs, or the IDs a chat request's `messages` render to |
| `POST /detokenize`, `POST /v1/detokenize` | vLLM's: the text of `tokens`, special tokens included |
| `POST /apply-template`, `POST /v1/apply-template` | `{"prompt": TEXT}`: the text a `/tokenize` body's `messages` render to (tools, `reasoning_effort`, `chat_template_kwargs`, `add_generation_prompt`, media expanded), special tokens included; no engine work |
| `POST /v1/responses` | OpenAI's Responses API, run as the equivalent chat completion; streamed or non-streamed |
| `GET /v1/responses` (WebSocket upgrade) | OpenAI's Responses [WebSocket mode](#websocket-mode): `response.create` events over one socket |
| `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}` | A stored response, or remove it |
| `POST /v1/responses/{id}/cancel` | Cancel a [background response](#background-responses) that is queued or running |
| `POST /score`, `POST /v1/score` | Text-pair scores in request order, using the configured reranker |
| `POST /rerank`, `POST /v1/rerank` | Ranked documents; a `texts` body selects the TEI response shape |
| `POST /v2/rerank` | Cohere-style ranked `results` over the same score hook |
| `POST /v1/decisions` | Choice, score, and yes/no probabilities from the next-token logits; no text is generated |
| `POST /v1/imprint/resume` | How much of a chat body's conversation resumes, and from where ([on disk](#conversations-on-disk)) |
| `POST /v1/imprint/load` | A chat body's conversation state read from disk into RAM now |
| `DELETE /v1/imprint` | This model's conversation states on disk forgotten |

On MLX, a completions body containing a nonempty `messages` list uses chat handling. CUDA completions
take a string `prompt` (`add_special_tokens`, default false) or a list of token IDs, run as given.

A completions `prompt` that is a list of two or more strings (at most 64) runs each string as its own request,
with the body's other fields (`max_tokens`, `seed`, `stop` and the rest) as that prompt alone would have them; `n`
must still be 1. The reply has one choice a prompt, `index` in list order, each with the text, stop and `tensorfold`
block that prompt alone gets, and `usage` sums every prompt's and every reply's tokens; the first refusal in list
order is the reply. A stream sends each prompt's chunks with its `index`, then the finishes in list order, then one
usage chunk when `stream_options.include_usage` is set (otherwise the last finish carries the summed usage). Up to
eight prompts run at once, and one at a time under `--max-inflight` or `--max-inflight-per-key`, where the request
holds one place. A one-string list, a list of token IDs, a list of token-ID lists and a mixed list keep their
one-prompt meaning (lists inside are decoded and joined with a newline).

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
The native CUDA server reads audio for Nemotron 3.5 Transcribe: `input_audio` and `audio_url` (data URL) parts with
16 kHz mono WAV in user messages ([recipe](recipes/nemotron-3.5-transcribe.md#audio-in-a-request)).
Unsupported image input, audio, video and non-text output requests receive HTTP 400.

## Model capabilities

Each native GET /v1/models entry, including every configured alias, includes a capabilities object.
It reports the loaded engine's boolean Info flags: text_generation, context_fitted, call_gates, loop_guard,
thinking_budget, reasoning, reasoning_optional, answer_limit, structures, logprobs, penalties, prompt_logprobs,
prompt_cache, cache_salt ([salts](#prompt-cache-salts)), warm_turns, plain_only, decides, pictures, greedy_only,
slide_conversations and folded_memory. folded_memory is true when `--slide` has attached the engine's folded memory
weights (living weights), and false when learning is off. The vision, speech, embeddings, readouts and rerank fields
report whether that user feature is present. The last two fields, thinking_switch and thinking_applicable, are the
loaded chat template's and tokenizer's thinking facts (see
[Reasoning](#reasoning)), set once at startup on the same Info flags. Every field is a JSON boolean, including false,
and no callback details or addresses are exposed.

Discovery and boolean request checks use the same loaded flag reader. Publication enables no feature.
Flags describe the engine; request combinations, context, template and format constraints still apply.
greedy_only and plain_only describe restrictions, and context_fitted says the context was limited to memory.

## One model

`GET /v1/models/{id}` (or `/models/{id}`) is OpenAI's retrieve-model route on the native server. For the served name or
one of its `--alias` names it returns the object `GET /v1/models` lists for that id, capabilities included. The id is
percent-decoded once, so `org/name` may be sent as written or as `org%2Fname`; an id holding a `%` that starts no hex
pair is matched as written. Any other id answers 404 with `"code": "model_not_found"`, `"type": "invalid_request_error"`,
`"param": "model"` and the decoded id in the message. A path that ends in `/models` still lists every model; send an
id ending in `/models` as `org%2Fmodels`. Keys and [key scopes](#key-scopes) apply as they do to `GET /v1/models`.
The Python 0.6.6 server, the other server that lists models, has no such route.

## Decisions

`POST /v1/decisions` is served by Flash Next on one CUDA GPU, by LFM2 on CUDA and by Gemma 4 on XPU.
Every other engine, which includes all Metal families and CUDA GLM today, returns HTTP 400.
The prompt wording is SGLang's decision prompt format version 1: the input, a blank line, the question, one line per
option, level, or described yes or no answer, and a closing instruction to answer with one label. Choice labels are
`A` to `Z`, score labels are `0` to `9`, and a yes/no question uses `yes` and `no`. Each label must be one distinct
token at the answer position. Thinking stays off. The response carries `prompt_format_version`, `answers` keyed by
question id, and `usage.completion_tokens` 0. `probabilities` are a softmax over the label logits divided by
`temperature` (default 1). `label_mass` is the full-vocabulary probability of those labels and does not use
`temperature`. A request the tokenizer or the context window cannot score returns HTTP 400.
For decisions, `chat_template_kwargs` may be omitted, null, or an object containing only
`enable_thinking: false`; other types, keys, or thinking values return HTTP 400.

## Rerank

`POST /v1/rerank` (also `/rerank`, and by `PUT` as SGLang takes it) scores each document against a query with a
reranker checkpoint. The native HIP engine serves `Qwen/Qwen3-Reranker-8B`, and the native CUDA engine `-0.6B`, `-4B`
and `-8B`: the engine declares the reranker and its label tokens (`1_LogitScore/config.json`), and its startup line
says `qwen3 reranker`. Every other server answers the route with HTTP 400 "this model is not a reranker", and a
reranker server answers chat, completion, Responses, Messages, speech, decision, System One and embedding requests
with HTTP 400 "this model is a reranker: use POST /v1/rerank".

| Field | Meaning |
| --- | --- |
| `query` | The query: a string, a `{"text": ...}` object, or a list of text parts |
| `documents` | One or more documents, each in the same forms as `query` |
| `instruction` | The task the model card's prompt states; default "Given a web search query, retrieve relevant passages that answer the query". SGLang's `instruct` is read when `instruction` is absent |
| `top_n` | Return the best `top_n` results only; default all |
| `return_documents` | Echo each document as sent; default `true` |
| `max_length` | The most tokens one prompt may take; default 8192, the model card's. A longer prompt keeps the template whole and loses the end of its content (the document's last tokens), as the card's tokenizer cuts it; a cap that would leave no token of the document returns HTTP 400 |
| `return_logits`, `return_prompt_token_ids` | Add `meta_info` with `logit_yes` and `logit_no`, or the prompt's token ids, to each result; default `false` |

Each document's prompt is the model card's template, token for token, with the query, the document and the
instruction as plain text (text that spells a special token stays text). The score is
`sigmoid(logit_yes - logit_no)` from the last position's label logits. The response is
`{"results": [{"index", "relevance_score", "document"}], "usage": {"total_tokens"}}`, best first, ties in input order;
`total_tokens` counts every prompt's tokens as scored, after any cut. Every prompt is cut and checked against the
context window before the first is scored, so a refused request costs no GPU work; a prompt past the window (with a
`max_length` above it) returns HTTP 400 with `code` `context_length_exceeded`. An image or video part returns HTTP
400. Errors use the server's envelope, as on every route: `{"error": {"message", "type"}}`.

### Score and rerank compatibility shapes

These translations use the same configured reranker, query/document template, yes/no label tokens and score or
decision hook described above. They add no engine or model support. Other models return HTTP 400, and an engine
without either scoring hook returns a named refusal. The routes require the `inference` key scope. Only the
existing `/rerank` and `/v1/rerank` aliases also accept PUT.

| Request | Response |
| --- | --- |
| `/score` or `/v1/score`: `{"text_1":"query","text_2":["a","b"]}` | A list object whose `data` entries contain `index`, `object:"score"` and `score`, in input order; also `id`, `created`, `model` and token `usage` |
| `/rerank` or `/v1/rerank`: `{"query":"q","texts":["a","b"]}` | TEI-style array of `{index,score}`, best first; `return_text:true` adds `text` |
| `/v2/rerank`: `{"query":"q","documents":["a","b"]}` | `{id,results:[{index,relevance_score}],meta:{api_version:{version:"2"}}}` |

`/score` accepts strings or non-empty string lists on either side. Equal-length lists are paired; one item on
either side broadcasts to the other list. Unequal longer lists are refused. Every pair is validated before any
inference. Scores are the existing sigmoid(yes minus no), not generation log probabilities. Pair-scoring requests
return every pair; rerank-only fields such as `top_n` or `return_documents` are refused there.

A `texts` body with no `documents` selects the TEI shape; a `documents` body at `/rerank` or `/v1/rerank` keeps the
existing response and defaults. Both new rerank shapes return the same scores as that response, best first, equal
scores in request order, and `index` is the input's position in the request. Both take a positive integer `top_n`
(an extension on TEI, whose request has none), capped to the number of inputs. TEI inputs are strings;
`return_documents:true` is also taken as `return_text`. TEI `raw_scores:true` returns the label logit difference
(yes minus no) the score is the sigmoid of, sorted by that value before `top_n`, so scores that round to 1 still
rank. Cohere's v2 request takes string documents and returns `index` and `relevance_score` only; `{"text":...}`
objects and `return_documents:true` (Cohere v1's fields) are also taken, the latter adding a `document` object that
wraps a string as `{"text":...}` and returns a supplied object as sent.

`model`, when given on the new shapes, must name this server or an alias. `instruction`/`instruct` and `max_length`
use the existing reranker semantics. TEI does not truncate unless `truncate:true`; only right truncation is
supported (`truncation_direction:"Right"`). Pair scoring also refuses an over-limit prompt by default; a positive
`truncate_prompt_tokens` explicitly sets its truncation cap, and cannot be combined with `max_length`. The normal
context-window check still applies. Cohere-style requests retain the reranker's default right truncation cap.
`rank_fields`, `max_tokens_per_doc` and `priority` are refused by name. No vendor billing units are invented.

## OCR

The native HIP engine reads pages through four OCR contracts when it serves a model with an OCR reader and
`--vision`. One route answers Mistral's contract; `/ocr` (and `/ocr/`) answers three more, chosen from the request.

| Route | Contract | Request | Reply | Written for |
| --- | --- | --- | --- | --- |
| `POST /v1/ocr` | Mistral OCR | JSON `{"model", "document", ...}` | `{"pages", "model", "document_annotation", "usage_info"}` | the `mistralai` Python and `@mistralai/mistralai` TypeScript SDKs, with `server_url` set to the server root |
| `POST /ocr` | PaddleX OCR serving | JSON `{"file", "fileType", ...}` | `{"logId", "errorCode", "errorMsg", "result"}` | the PaddleX serving clients |
| `POST /ocr` | docTR API | multipart, one or more `files` parts, options in the query | one `OCROut` a page | the docTR API template's client |
| `POST /ocr` | LiteParse HTTP OCR | multipart, one `file` part | `{"results": [...]}` | `lit parse --ocr-server-url http://HOST:PORT/ocr` |

Two families have a reader. A reader gives Markdown blocks in pixels of the page the client sent (after EXIF
orientation), never in the model's resized frame.

| Model | What the reader returns | Geometry |
| --- | --- | --- |
| dots.mocr | one block a layout region (title, text, table, formula, picture, header, footer, ...), tables as pipe tables or, when cells span, HTML | a box a block |
| GLM-OCR | the page's Markdown as one block, tables as pipe tables or spanned HTML | none |

Any other model answers both routes with HTTP 400 "this model has no OCR reader; use POST /v1/chat/completions",
and a server started without `--vision` answers "OCR needs image input; start the server with --vision" (PaddleX
answers both with 422). Embedding, reranker and readout engines refuse the OCR routes as they refuse chat. The
routes take the `inference` key scope.

`POST /ocr` picks its contract in this order:

1. `?format=paddlex`, `?format=doctr` or `?format=liteparse`. A query parameter sent twice is HTTP 400 ("the format
   query parameter is repeated; send it once", and the same for docTR's parameters), because servers differ on
   which copy wins.
2. `Content-Type` `application/json` or `application/*+json` (parameters such as `charset` ignored), or no
   `Content-Type` with a body whose first non-space byte is `{`: PaddleX.
3. `multipart/form-data`: one or more parts named `files` is docTR; exactly one part named `file` is LiteParse;
   any other set of parts is HTTP 400, and so is a body with both `files` and `file` parts, even when `?format=`
   names the contract.
4. Any other content type is HTTP 415.

Refusals made before the contract is known (415, those 400s, a drain's 503 and an admission 429 with
`Retry-After`) use the server's `{"error": {"message", "type"}}` envelope. On a model without a reader, or a
server without `--vision`, a request no contract takes gets the 400 above instead of its 415 or 400.

### Reader Markdown

Both readers write block text in one Markdown flavour, the one Mistral's page Markdown uses:

- Headings with `#` markers: a title is `#`, a section header `##` unless the model gives a level.
- GFM pipe tables, or one HTML `<table>` when a table has row or column spans a pipe table cannot hold; a table
  block holds exactly one table.
- Display math as `$$…$$` and inline math as `$…$`, the LaTeX kept as the model wrote it.
- Lists with `-` or `1.` markers, and code in fenced blocks.
- No images: a picture block has empty text, and its caption, if any, is its own block. Each contract writes its
  own image references.
- GLM-OCR's single block holds the whole page's Markdown, headings, tables and math included.

### Pages and budgets

- Images are PNG, JPEG or WebP, one page each, up to the 32 MiB body limit and the reader's decoded-pixel limit:
  16 megapixels and 8192 pixels a side for dots.mocr, 32 megapixels for GLM-OCR; a larger image is bad input. An
  image whose header and decoded pixels disagree is refused as bad input.
- PDFs need `--vision-pdf`. PDFium renders each selected page at up to 144 DPI, at most 2048 pixels a side and
  about 3.6 megapixels; boxes and sizes are in pixels of the rendered page. Mistral reports that DPI (72 for an
  image). Every PDF of a request is measured before the first page is read (its page count and each selected
  page's size, with no rendering), so a page index out of range, the page budget or a page that does not load fails
  before any GPU work. Each page is then rendered just before it is read and freed when it is done, so one rendered
  page is in memory at a time; a page that loads but does not render fails the request when it is reached.
- `https://` URLs (Mistral `document_url` and `image_url`, PaddleX `file`) are fetched only with `--vision-urls`:
  images, and PDFs when `--vision-pdf` is on, at most 32 MiB. `http://` URLs are refused.
- Page budget: images count one page each and PDFs their selected pages; a request reads at most
  `TENSORFOLD_MAX_PDF_PAGES` pages (default 16, also the most it may be set to). `--vision-max-images` does not
  apply. Over the budget, a request is refused before any inference, except in PaddleX, which reads the first
  pages of a PDF up to the budget, as [PaddleX's OCR serving API](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html)
  reads the first `max_num_input_imgs` pages. `ocrResults` and `dataInfo.pages` list the pages read, and
  `dataInfo.numPages` is the document's page count, so a client can see the cut.
- Image budget, for Mistral crops (`include_image_base64`) and PaddleX's `ocrImage` and `inputImage`: at most 256
  encoded images, 4 times the request's page pixels, and 16 MiB plus 2 MiB a page of encoded bytes. Crossing one
  fails the whole request as an unreadable page, "too many image regions". Both encodes need libjpeg-turbo
  3.1.4.1 (`TENSORFOLD_LIBTURBOJPEG` names its file); without it `include_image_base64` is HTTP 400 and PaddleX's
  default `visualize` is HTTP 500 "visualize needs libjpeg-turbo, which this server cannot load; send visualize:
  false".

Pages are read one at a time, each as one user turn (the page image, then the reader's prompt), greedy, with
thinking off and `max_tokens` equal to all the context the prompt leaves. A page succeeds only when the reply
stops on its end token and the reader can parse it.

### Failures

| Failure | Mistral | PaddleX | docTR | LiteParse |
| --- | --- | --- | --- | --- |
| Bad input: an undecodable or unsupported file, a page or option the server cannot serve, over the page budget | 400 | 422 | 400 | 400 |
| Unreadable page: the reply was cut at the context limit, repeated without end, broke the model's format or gave a box that is not finite; the image budget was crossed | 422 | 422 | 422 | 422 |
| The engine stopped the page (it closed or drained) | 503, `Retry-After: 1` | 503, `Retry-After: 1` | 503, `Retry-After: 1` | 503, `Retry-After: 1` |
| Engine fault, out of memory, PDF renderer crash or deadline | 500 | 500 | 500 | 500 |
| Envelope | `{"detail": [{"loc", "msg", "type"}]}`, `type` `bad_input`, `unreadable_page`, `capacity` or `internal` | `{"logId", "errorCode", "errorMsg"}`, `errorCode` equal to the status | `{"detail": "..."}` | `{"error": "..."}` |

A failed page fails the whole request: no reply returns some pages and drops others. An unreadable page's
message names its index, and the model's raw text goes to the server log, never to the client. 422 means the
same request fails the same way again under greedy decoding; a 503 carries `Retry-After`. A client that disconnects
stops the request before its next page or PDF render.

### Estimates

dots.mocr gives one box a block and GLM-OCR none, so the line contracts estimate what the models do not give:

- **Lines.** A block's plain-text lines (Markdown markup removed, a table one line a row with tab-separated
  cells, math without its `$`) divide the block's box into equal bands along its left and right edges, first
  line at the top. A paragraph written without line breaks stays one line covering the block.
- **Words.** docTR's words and PaddleX's `returnWordBox` divide a line's box along its reading direction in
  proportion to each word's character count.
- **Scores.** Every score is `1.0`: PaddleX `rec_scores`, docTR `confidence` and `objectness_score`, LiteParse
  `confidence`. Mistral replies leave `confidence_scores` out.
- **No geometry.** With GLM-OCR, Mistral returns `blocks: []`, PaddleX and docTR give each line the whole page as
  its box, and LiteParse is refused.

### Mistral: `POST /v1/ocr`

| Field | Meaning |
| --- | --- |
| `model` | Any string or null. The reply names the requested id when it is the served name or an `--alias`, else the served name, so `--alias mistral-ocr-latest` lets copied examples echo it |
| `document` | `{"type": "document_url", "document_url"}` or `{"type": "image_url", "image_url"}` (a string or `{"url"}`): a `data:` URL, or an `https://` URL as above. The bytes decide PDF or image. `{"type": "file"}` is HTTP 400 |
| `pages` | 0-based page indices as a list or a string such as `"0,2-4"`; duplicates are merged. An image has only page `0`. At most 1024 entries (a range counts its length) before merging, and the page budget after |
| `table_format` | Null: tables stay in the page Markdown as pipe tables (a spanned HTML table becomes a pipe table with the spanned text in its first cell). `"markdown"` or `"html"`: each table becomes a `tables[]` entry (`tbl-N.md` or `tbl-N.html`) and the Markdown links it; under `"markdown"` a table with spanned cells stays HTML, the only form that keeps it |
| `extract_header`, `extract_footer` | Move page headers or footers out of `markdown` into `header` and `footer`. GLM-OCR marks neither, so `true` is HTTP 400 "this model does not mark page headers or footers" |
| `include_blocks` | Default `true`: one block for each region with a box. `false` leaves `blocks` out |
| `image_limit`, `image_min_size` | Choose which pictures to keep, in reading order (`image_min_size` against the box's width and height, then the running count against `image_limit`). A dropped picture has no image entry, Markdown link or block |
| `include_image_base64` | Add `image_base64`, a `data:image/jpeg;base64,...` crop of the picture's box, to each kept image |
| `confidence_scores_granularity` | Accepted; no scores are returned |
| `bbox_annotation_format`, `document_annotation_format`, `document_annotation_prompt` | Any non-null value is HTTP 400 "annotations are not supported" |

`null` means absent for every optional field, and unknown keys are ignored. Each page carries `index`, `markdown`
(blocks joined by a blank line, pictures as `![img-N.jpeg](img-N.jpeg)`), `images`, `tables`, `header`, `footer`,
`dimensions` (`dpi`, `height`, `width`) and `blocks`; `hyperlinks` is left out. A block has `type` (`text`,
`title`, `list`, `table`, `equation`, `code`, `caption`, `header`, `footer` or `image`), an integer box and
`content`, its own Markdown with its tables in the `table_format` form; image blocks add `image_id` and table
blocks `table_id`. `usage_info` gives `pages_processed` and `doc_size_bytes`, the decoded document's size.

### PaddleX: `POST /ocr`

`file` is base64 (bytes outside the alphabet are skipped and the padding is repaired, as Python's `b64decode`
does), a `data:` URL or an `https://` URL. `fileType` is `0` for a PDF and `1` for an image; when it is absent the
bytes decide. TIFF is not read. `logId` is echoed by every reply once the body parses; an absent or empty one gets
a new UUID. Each field converts as pydantic's lax mode converts the type PaddleX's serving reference gives it (`"0.5"` is a
number, `"off"` and `0` are false), and a number must be finite.

| Field | Meaning |
| --- | --- |
| `textRecScoreThresh` | Keep lines with a score at or above it in `rec_*`; `dt_polys` keeps every line. Default `0.0` |
| `returnWordBox` | Add `text_word_boxes` and `text_word`, each line's estimated words |
| `visualize` | Default `true`: `inputImage` (the page) and `ocrImage` (the page with each `rec_polys` polygon outlined), base64 JPEG without a data-URL prefix. `false` leaves both out |
| `useDocOrientationClassify`, `useDocUnwarping`, `useTextlineOrientation` | Accepted when null or false; `true` is 422 |
| `textDetLimitSideLen`, `textDetLimitType`, `textDetThresh`, `textDetBoxThresh`, `textDetUnclipRatio` | Accepted when null; any value is 422, since TensorFold has no text detector to apply it to |

`prunedResult` leaves out `text_det_params` and `doc_preprocessor_res`. Polygons
round each corner half to even, as Python does; `rec_boxes` is each polygon's min/max. `dataInfo` is
`{"width", "height", "type": "image"}` or `{"numPages", "pages": [{"width", "height"}], "type": "pdf"}`, where
`numPages` is the document's page count and `pages`, like `ocrResults`, lists the pages read. A schema error's
`errorMsg` is the JSON list `[{"type", "loc", "msg"}]` with `loc` dot-joined (`body.file`).

### docTR: `POST /ocr`

Each `files` part's type must be exactly `image/jpeg`, `image/png` or `application/pdf` (the types docTR's API README names),
else HTTP 400 naming the part and its type. Files answer in part order, each PDF page with its file's name. Each page's
`OCROut` follows the reply in docTR's API README: `name`, `orientation` (`{"value": 0, "confidence": null}`: the page as
sent), `language` (`{"value": null, "confidence": null}`), `dimensions` `[height, width]`, `items` with one page of
blocks, lines and words; plus `layout` and `tables`, always `[]` (docTR's docs list both in a page's export).

| Query option | Meaning |
| --- | --- |
| `assume_straight_pages` | Default `true`: boxes `[xmin, ymin, xmax, ymax]`; `false`: 4 points TL, TR, BR, BL. Coordinates are relative to the page and not rounded |
| `resolve_blocks` | Default `false`: one block a page; `true`: one block for each reader block |
| `resolve_lines` | Default `true`: one line for each text line; `false`: one line of all the page's words |
| `detect_orientation`, `detect_language` | Accepted; neither is estimated: `orientation` stays 0 and `language` null |
| `detect_tables`, `detect_layout`, `straighten_pages` | Accepted only when false; `layout` and `tables` are always `[]` |
| `paragraph_break` | Accepted only at its documented default, 0.035 |
| `bin_thresh`, `box_thresh` | Refused: TensorFold has no text detector to apply them to |
| `det_arch`, `reco_arch` and the batch options | Ignored: the served model reads every page |

A boolean option takes `true`, `t`, `yes`, `y`, `on` or `1`, or `false`, `f`, `no`, `n`, `off` or `0`, in any case; another
value is HTTP 400.

### LiteParse: `POST /ocr`

The contract is LiteParse's published
[OCR API specification](https://github.com/run-llama/liteparse/blob/main/OCR_API_SPEC.md). One `file` part holds a PNG,
JPEG or WebP page image; `language` and other parts are ignored. A PDF is HTTP 400: the specification sends an image.
Each line is one result `{"text", "bbox", "confidence"}`: `bbox` is the line's axis-aligned `[x1, y1, x2, y2]` in
pixels of the image received, and `confidence` is `1.0`, the value the specification asks for when the engine gives
none. The specification's optional `polygon` is not sent: it gives the point order, but not how LiteParse turns a
polygon into a text rotation, so a rotated line is sent as its axis-aligned box. GLM-OCR gives no boxes, so
LiteParse requests to it are HTTP 400 "this model gives no text boxes; LiteParse needs them". With dots.mocr a
paragraph written without line breaks is one result covering its block. Errors use the specification's
`{"error": "..."}` body.

### GLM-OCR context

GLM-OCR's default 8192-token window leaves about 2000 tokens for the reply after a full image budget (6144
tokens). A dense page needs more: start the server with a larger `--context`, or such pages fail as unreadable,
"the model's output was cut at the context limit".

## System One on a Clef engine

`POST /v1/systemone` on a Clef checkpoint (Cloudflare's clef-flash: BF16 Qwen3.5 weights with
`joint_head_config.json` and `joint_head.safetensors`) answers each question from the joint schema head, the way the
model card's reference `systemone` does. The native server serves it on one HIP GPU or one NVIDIA GPU (CUDA, checked on a GB10); a Clef
engine serves no chat, and `--tp` is refused. On every other engine the route keeps llama-server's System One request. `capabilities --json`
lists such formats under `decisions` (`"qwen3_5": ["bf16"]`), apart from the `families` formats that serve chat.

```json
{"model": "clef-flash",
 "state": "Ticket #4521: refund of $129.99 requested.",
 "questions": {
   "refund": {"type": "noul", "instructions": "Should we refund?"},
   "priority": {"type": "choice", "instructions": "Priority", "criteria": {"p0": "critical", "p1": "urgent", "p2": null}},
   "sentiment": {"type": "score", "instructions": "Customer sentiment", "criteria": ["angry", "neutral", "happy"]}},
 "raw": false}
```

| Field | Meaning |
| --- | --- |
| `model` | A string, echoed in the reply |
| `state` | Any JSON value: the record the questions are about; a string as given, other values as compact JSON with sorted keys |
| `questions` | An object keyed by question id, answered in request order |
| `type` | `noul` (a yes or no question), `choice` (one of the options) or `score` (an ordered level) |
| `instructions` | Optional; any JSON value. Missing, null or `""` uses the question id |
| `criteria` | `choice`: an object of option id to description (or null). `score`: a list of 2 to 10 level descriptions, levels `0` up. `noul`: optional, an object that describes `true` and `false` |
| `raw` | Optional boolean: also return each option's unrounded probability |

The reply has `model`, `answers` keyed by question id, and `usage` with `input_tokens` (the encoded record) and
`output_tokens` 0. Probabilities are a softmax over each question's options, rounded to 4 decimals:

| Type | Answer |
| --- | --- |
| `noul` | `{"type": "noul", "noul": P(true)}` |
| `choice` | `{"type": "choice", "choice": id, "confidence": P(choice), "probabilities": {id: p}}`; a tie goes to the first option in request order |
| `score` | `{"type": "score", "score": expected level, "confidence": the top level's p, "legend": {"0": description}, "probabilities": {"0": p}}` |

With `"raw": true` the reply also has `raw`: per question id, its `type`, `options` in encoding order and `probs`, the
unrounded probabilities in the same order.

A request holds at most 64 questions, 256 options in one question and 1,024 options in all. The record is at most
16,384 tokens (or the engine's context, if smaller): a longer state keeps its first tokens, and a schema that needs
more than that before the state returns 400. The route answers:

| Status | When |
| --- | --- |
| 422 | A field is missing or has the wrong JSON type. The body is FastAPI's: `{"detail": [{"type": ..., "loc": ["body", "questions", id, "type"], "msg": ...}]}`, one entry per problem |
| 400 | An unknown `type`, empty `criteria`, no questions, a bound passed, `images` or `videos`, or a record that cannot be encoded or does not fit the engine's readouts |
| 503 | The engine is closing, cannot take the request now, or is short of memory |
| 500 | The head failed or gave a non-finite logit |

A client that disconnects cancels its request.

## Queue phase metrics

With TENSORFOLD_PHASE_METRICS=1, native /metrics also exports tensorfold:request_queue_seconds and
its request_queue_time_seconds alias, for measured waits from request submission to admission.
Both use the fixed request-time histogram edges and have no request labels. An unset flag keeps these families
absent and skips the queue observation; existing request metrics remain available. This option does not change replies.

## Living Weights

With `--slide`, the native server learns facts into the served weights. [Living Weights](living-weights.md) explains
each route and its events.

| Route | What it does |
| --- | --- |
| `POST /v1/slide/learn` | `{"text": "...", "questions": ["..."]}`: learns the facts in the text, streaming `fact`, `learning` and `learned` events; `questions` is optional (see Living Weights) |
| `GET /v1/slide/graph` | Every fact as JSON: its state, neurons, questions and their expected answers, and the model's pocket |
| `DELETE /v1/slide/graph` | Clears the graph; the weights keep what they learned |
| `DELETE /v1/slide/fact/{id}` | Deletes one fact from the weights, every other fact checked again, streaming `fact`, `checking`, `zeroed`, `checked` and `forgot` events; an optional `{"reason": "..."}` goes into the graph |
| `DELETE /v1/slide/facts` | Deletes the facts the body's `ids` select (1 to 256 distinct learned fact IDs) in one transaction: an invalid selection refuses the whole request before any cut; an optional `{"reason": "..."}` goes into the graph |

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
| `presence_penalty`, `frequency_penalty` | OpenAI's, from -2 to 2: before each draw, greedy or sampled, a token the reply already holds loses `presence_penalty` from its logit and `frequency_penalty` once for each time it holds it; the prompt's tokens do not count. Drafted replies equal undrafted ones with them. Applied by Qwen3.8-27B and Qwen3.5-MoE on CUDA, and Qwen3.8-27B and Qwen3.8-Flash-Next on Metal; other engines refuse nonzero values with 400; zero, null and omitted values keep working | Native |
| `seed` | Explicit sampling key; native fallback uses configured defaults, then a fresh key (see below) | Both |
| `stream` | Server-sent events; the last event carries usage | Both |
| `stream_options.include_usage` | Usage in its own final event with `"choices": []`, not on the finish event | Both |
| `stream_options.tensorfold_token_stream_version` | Opt in to version 1's exact incremental token IDs and semantic anchors on chat or completions streams | Native |
| `chat_template_kwargs.enable_thinking` | Template thinking toggle; explicit off returns HTTP 400 when a thinking tokenizer has neither a template switch nor a closable generation prefix; omitted switches keep working. `chat_template_kwargs.thinking` (`true`/`false` or `{"type": "enabled"}`/`{"type": "disabled"}`, as DeepSeek-V4 clients send it) is read the same way when `enable_thinking` is absent; other values are ignored | Both |
| `thinking` | Top-level toggle, the same forms as `chat_template_kwargs.thinking`; read only when the kwargs set neither toggle, so the kwargs win. Enabling it clears a `reasoning_effort` of `none`, so the template's default effort applies | Both |
| `draft` | False selects the serial reference; CUDA rejects it if the engine has no serial switch | Both |
| `response_format`, `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`, `structured_outputs` | A JSON schema, any JSON object, a regex, a choice or an EBNF grammar the reply must match; on the native server also `structural_tag`, JSON regions inside free text | Both |
| `grammar_budget` | Boolean, default false. Under a grammar (the fields above, or a required tool call), never draw a token after which the reply could no longer complete within `max_tokens`; see [Structured output](#structured-output) | Native |
| `ignore_eos` | Disable model end-of-sequence stopping; the reply limit still applies | Both |
| `stop` | Stop at a string or any string in a list; omit the matched text from the response | Both |
| `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max` | Both |
| `thinking_budget` | Token-count limit inside reasoning | Both |
| `priority` | `background` yields to foreground requests | Both |
| `deadline_s` | Seconds, more than 0 and at most 1,000,000, on chat and legacy completions: at that time the server cancels the engine itself and the reply ends with `finish_reason: "deadline"`, keeping every token the engine gave, usage and token ids. A reply that ends first, or a [cancel](#cancel-a-request) that came first, keeps its own ending; without the field nothing changes | Native |
| `return_token_ids` | The reply's token ids in the response's `tensorfold.token_ids`, the tokens its `token_sha256` hashes | Native |
| `execution_witness` | Boolean, default false. True binds the reply to the execution witness; see [Reply binding](#reply-binding). A server without `--execution-witness` answers 400, a key with `inference` but not `read` 403 | Both |
| `echo` | `/v1/completions` only: the prompt's text, and with `logprobs` its tokens' entries, ahead of the reply's, described below | Native |

`n` takes 1 through 16 choices: each is its own engine request on the one prompt, and later choices resume the
prompt the first fills (their `tensorfold.cached_tokens` show it). Choice 0 keeps the request's seed (a request that
names none draws one for all its choices), so `n: 1` is unchanged; choice `i` draws with `mix(seed ^ i)` in 63 bits,
named in its own `tensorfold.seed`, and a solo request with that seed gives its tokens. Each choice has its own
tool-call ids, `stop_sequence` and loop-stop `error`. `usage` counts the prompt once and every choice's reply
tokens. A stream sends every choice's chunks by `index`, then each choice's finish, then one usage chunk when asked.
`n` above 1 with `logprobs` or a required tool call receives HTTP 400, as do `n` above 16 and, on a `--cortheon`
server, a chat its turn loop would take (send `"cortheon": false` for several choices). `/v1/responses` and
`/v1/imprint` serve one reply: `n` other than 1 receives HTTP 400 there.

`modalities` must be absent or `["text"]`, and a truthy top-level `image`, `images`, `image_url`, `input_image`,
`audio`, `input_audio`, `video` or `video_url` key is refused: this server accepts and produces text only.

Engines that advertise `capabilities.logprobs` and give each token's exact bytes accept `logprobs: true` and
`top_logprobs` from 0 through 20 on `/v1/chat/completions`, streamed or not, with tools, stop strings, structured
output, reasoning and a thinking budget. Asking for them changes nothing else: the prompt, the sampled tokens and the
text are the same with them on or off. Probabilities describe the raw target-model distribution at temperature 1,
before temperature, top-k or top-p sampling filters, including when generation is greedy, and are the same with drafts
on or `"draft": false`. Alternatives are tokenizer tokens and may include leading spaces; `bytes` preserves partial
UTF-8 sequences.

`choices[0].logprobs.content` has an entry for each token of the reply's `content` (`token`, `logprob`, `bytes` and
the requested `top_logprobs`), and the entries' `bytes` joined spell `content`. With thinking on, the reasoning's
tokens are in `logprobs.reasoning_content`; with tools, each call has an `{index, content}` group in
`logprobs.tool_calls` with the entries of the markup the model wrote for it (not the arguments JSON the reply returns;
a closer the server adds to a cut call has none). Thinking markers, the whitespace the text drops, a stop string and
the text after it have no entries. A token the text shows only part of, such as one a stop string ends inside, shows
the kept bytes in `token` and `bytes` and its id in `token_id`, with its own `logprob` and alternatives.

A stream sends each entry with the chunk whose text completes that token, reasoning entries before the answer's first
text; with tools offered, the entries come in one chunk before the finishing one, once the calls are parsed. The
entries joined over the chunks are the whole reply's. `/v1/completions` takes the legacy `logprobs`, an integer from 0
through 20 alternatives, and answers in the legacy form, streamed or not: `tokens`, `token_logprobs`, `top_logprobs`
(each alternative's text to its logprob) and `text_offset` (each token's first character in the completion, counting
each token's own text). Other engines return HTTP 400 when probabilities are requested.

The Responses API gives the same entries for `top_logprobs` (0 through 20) or the `include` name
`message.output_text.logprobs`: in each `output_text` part's `logprobs`, and as extensions in `reasoning_text` parts
and `function_call` items. A stream sends them in the `response.output_text.delta` and
`response.reasoning_text.delta` events; with tools offered they come once the calls are parsed, as deltas with empty
text, and each item's done events wait for them. `include` alone on an engine without logprobs answers without them;
`top_logprobs` there returns HTTP 400.

Nonstreamed `/v1/completions` also takes vLLM's `prompt_logprobs`, an integer from 0 through 20, on the same engine
and on GLM-5.3-Flash's CUDA engine.
`choices[0].prompt_logprobs` has one entry per prompt token: `null` for the first, then an object keyed by token id
(a string) whose first key is the prompt token itself, followed by the requested best tokens; each value has
`logprob`, `rank` and `decoded_token`. The prompt token's `rank` counts every vocabulary token at or above it,
itself included, as vLLM does, and the best tokens are ranked by place. Such a request never resumes from the prompt
cache, so its whole prompt is prefilled. Chat, streamed requests and other engines return HTTP 400 with it.

Nonstreamed `/v1/completions` takes OpenAI's legacy `echo` on the engines that advertise
`capabilities.prompt_logprobs`, so evaluation harnesses can score a prompt's own tokens. With `echo: true`, `text`
starts with the prompt: its token ids decoded, which is the text as sent unless the tokenizer adds or normalizes
something, such as a beginning-of-text token. With `logprobs`, the legacy arrays list each prompt token before the
reply's: `tokens`, `token_logprobs`, `top_logprobs` (the likeliest tokens at that place, as many as `logprobs` asks
for), `null` in both for the first prompt token, and `text_offset` counting each token's own text from 0. The
reply's own entries are the ones the same request gets without `echo`, with their offsets counted on past the
prompt's. `max_tokens: 0` with `echo`
generates nothing: `text` is the prompt alone, `finish_reason` is `length` and `usage.completion_tokens` is 0
(without `echo`, 0 still means the server's default length). With `logprobs`, the whole prompt is prefilled, as for
`prompt_logprobs`; asked together, each lists its own number of alternatives. A non-boolean `echo`, a streamed
request, a body with `messages` and other engines return HTTP 400 naming `echo`; chat routes ignore the field.

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
unless a stop string, a tool call or a grammar ended the reply. A grammar's stop token is an end token, and with
`ignore_eos` it still ends the reply with `finish_reason` `stop` and stays out of the text. End tokens in the
reasoning, before the grammar starts, stay in it.

### Exact incremental token IDs

Native `/v1/chat/completions` and `/v1/completions` accept
`"stream_options": {"tensorfold_token_stream_version": 1}` with `"stream": true`.
`tensorfold-native capabilities --json` reports `"token_stream_versions": [1]` when this extension is available.
The version must be the integer `1`; unknown versions and nonstreaming requests return HTTP 400 before generation.
Cortheon's multi-round chat replies do not support version 1 and receive the same refusal.

The opted-in stream interleaves ordinary OpenAI chunks with raw SSE data frames:

```json
{"object":"tensorfold.token.chunk","version":1,"id":"chatcmpl-...","model":"model","offset":0,"token_ids":[42,43]}
```

`offset` is the zero-based index of the first committed token in that batch. Batches are contiguous, nonempty and
ordered. IDs come directly from the native engine and include committed reasoning, tool markup, stop-string and EOS
tokens, even when those tokens produce no visible text. Rejected draft tokens and tokens after the server's stopping
point are excluded. Raw frames precede the semantic deltas produced from their batch; UTF-8 characters can span
several batches.

Every ordinary chunk carries a `tensorfold_stream` object:

```json
{"version":1,"token_count":2,"token_id":43}
```

`token_count` is the total number of raw IDs committed so far, and `token_id` is the last of them, or `null` before
any token is committed. These values anchor semantic output to a source position. Several content, reasoning or
tool deltas may share one anchor; there is no one-to-one relationship between tokens and semantic chunks.

When a chunk carries logprob entries, the metadata also carries `logprob_positions`, shaped like
`choices[0].logprobs`: `content` and `reasoning_content` hold one zero-based raw position for each entry, in order,
and `tool_calls` holds `{index, content}` groups. For legacy completions, `content` matches the parallel arrays in
`choices[0].logprobs`. An entry the reply shows only part of shares its token's position with that token's other
entries. Marker, dropped and EOS tokens have no entry, and repeated IDs still have distinct positions.

Send `"return_token_ids": true` to receive the existing final `tensorfold.token_ids` array as an independent check
against concatenated raw batches. The native final usage remains authoritative, including for an empty reply.
The final usage and `[DONE]` events keep their usual forms, with `include_usage` honored when requested. Native SSE
heartbeat comments keep an admitted but silent stream alive; they do not advance the token count. Closing the
connection cancels the admitted generation and drains its engine request before releasing it. Requests without the
version option retain their existing wire output.

### Replies that quote a template marker

Some templates encode their turn-end spelling as one added token: Qwen3.8-Flash-Next's tokenizer encodes
`<|im_end|>` to one id, while `<|tool_call_begin|>` stays seven ordinary tokens, as `POST /tokenize` with
`{"prompt": "<spelling>", "add_special_tokens": false}` shows ([#2399]). A reply can quote that spelling with
ordinary tokens and finish normally; when it draws the one-token spelling instead, the turn ends there even inside
a code span. The server stops on the token the model drew; it does not sample past the end token and decide
afterwards, because that would change the reply.

A plain request already finishes on the checkpoints measured so far: they spell a quoted marker with ordinary
tokens, and the server stops on the end token when the model draws it. A client that still uses `ignore_eos`
should stop on `["<|endoftext|>", "<|im_start|>"]` and not on `<|im_end|>` or end-of-text alone. With
`ignore_eos`, the end token no longer stops the reply, and behind an agent prompt the model often opens the next
turn instead of writing end-of-text, so a stop on end-of-text alone never fires; the invented turns run on, and a
tool parser returns their tool calls ([#2399]). Strip a trailing `<|im_end|>` and the whitespace after it before
display. The cost is the same for both stops: a stop string matches the spelling however it is tokenized, so a
reply that quotes `<|endoftext|>`, `<|im_start|>` or `<|im_end|>` as text is cut there. A `stop` for `<|im_end|>`
cuts at least where the end token would, and also where the reply spells it with ordinary tokens.

Measured greedy on a GB10 with a finished tool call and its result in the conversation, on Qwen3.5-2B MLX
4-bit and Qwen3.6-35B-A3B NVFP4: both checkpoints wrote the quoted marker with ordinary tokens, so the plain request
finished; with `ignore_eos` and `stop: ["<|endoftext|>", "<|im_start|>"]` all replies stopped right after their
first end, and each ended with the drawn `<|im_end|>` for the client to strip. On an agent-shaped body, the runs
reported in #2399 gave 0/9 runaways with the pair, against 7/9 with end-of-text alone, on stock Qwen3.8-Flash-Next
weights.

On MLX, `--parallel auto` is the default: requests share rounds within the configured concurrency and memory
budget. Every admitted prompt prefills a chunk at a time beside the others. Each chunk goes to a foreground prompt
before a background one, then to the prompt with the fewest tokens left, and a prompt passed over for 8 chunks takes
the next one: a short request starts at the next chunk, and a long prompt still finishes. Running replies take rounds
between chunks for `--decode-share` of each chunk's time (0.25 is the Python engine's default). On a native server
the flag is the reply's share, and the prompt owes its complement: it also offers a 512-row cap on a filling
prompt's chunks to families that declare chunk-independent prompt arithmetic. That offer is gated per family
behind device token-hash qualification; LFM3 on Metal declares it (cut-invariance by `tf-lfm3-exact`, row
path), and no other native family does yet. Absent the flag the engines keep their own cuts
and `TENSORFOLD_PREFILL_SHARE` keeps its old meaning:
the prompt's share of a fill step's wall time, time only, never smaller chunks. Flash Next's shared sessions on a Mac
are the exception: absent both, replies take 0.8 of that time and a filling prompt's steps grow
from 32 to 512 rows after eight, while `--decode-share 0` keeps the 32-row steps with the prompt first. `TENSORFOLD_PREFILL_ORDER=srpt`
lets a running reply that would end within a quarter of a filling long prompt's estimated time left take rounds first,
the prompt keeping at least 10% of the time and lending at most a quarter of its own. It's the default for Qwen3.8-27B
whole
first, in arrival order, as 0.3.6.2 did. Background work waits behind foreground requests. An active background
request yields when a foreground request needs its lane or memory, then restarts with already-delivered tokens
suppressed.
`TENSORFOLD_PRIORITY_AGE_MS` above 0 lets a background request queued for at least that many milliseconds go before
foreground work. Prefix, disk-read, lane and memory admission checks are unchanged.
Session-title requests are also treated as background work.

The serve flags `--top-p`, `--top-k` and `--min-p` set the server's defaults for the request fields of the same
names, as `--temperature` does. Under them sit two layers: a family's built-in sampling (DeepSeek-V4 and V4.1
serve their model cards' temperature 1 and top_p 1), then the checkpoint's `generation_config.json`
(`temperature`, `top_k`, `top_p`, `min_p`; `do_sample: false` means temperature 0, and `do_sample: true` without
a temperature means 1). A flag that is set replaces what those layers gave. When nothing sets a value the server
has no sampling default and the serving line prints `greedy`; otherwise it prints the resolved defaults as
name-value pairs. A request's own fields win over the defaults. The merged values are checked per request like
request fields: a `--min-p` outside 0 through 1 refuses every chat request, a negative `--top-k` clamps to 0, and
`--temperature` and `--top-p` must be finite numbers. An engine that decodes greedily only refuses sampled
requests while the server samples by default, and its refusal suggests serving with `--temperature 0`.

`--instruct-sampling` gives chat requests with thinking off their own sampling defaults. Exactly these requests change
under it: a chat request (not a raw completion) whose reasoning block is off (the request's `enable_thinking: false`,
or `--no-thinking` with nothing in the request turning it on), that samples (its temperature, or the default it ends
up with, is above 0), takes the set's value for each sampling field it leaves unset; a field the request sends wins,
as over the ordinary defaults. Requests with thinking on, greedy requests (a request's `temperature: 0`, a server's
`--temperature 0`, or a checkpoint whose `generation_config.json` has `do_sample: false`), raw prompt completions and
the server's own requests (compaction summaries, Living Weights' lesson wordings and checks) are unchanged. `card`
serves the family's model-card instruct set: Qwen 3.5, Qwen 3.6, Qwen3.8-27B and Qwen3.8-Flash-Next give temperature
0.7, top_p 0.8, top_k 20 and presence_penalty 1.5, while their thinking set stays in `generation_config.json`.
`name=value,...` sets or changes `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty` and `frequency_penalty`
over that set, or over the server's defaults for a family without one; `none` serves one set. The set, pairs
included, sits between `generation_config.json` and the flags, so `--temperature`, `--top-p`, `--top-k` and `--min-p`
still decide for both modes: a server started with `--temperature 0` stays greedy whatever the pairs say. Without the
flag the server has one default set for every request, as before. A pair outside the range a request may send
(temperature below 0, top_p outside 0 through 1, a negative top_k, min_p outside 0 through 1, a penalty outside -2
through 2) refuses the argument at startup, naming the field. `card` refuses a model type whose family gives no card
set; it cannot tell checkpoints apart within a type, so leave the flag off (or give pairs from the card) for a
checkpoint whose card gives no non-thinking set, such as Ornith-1.5-122B served as `qwen3_5_moe`. On an engine that
applies no penalties the card's presence_penalty is left out with a startup line saying so, and pairs naming a
non-zero penalty are refused at startup. The serving line prints the set after the sampling defaults
(`thinking off: ...`). With a drafter, presence_penalty 1.5 lowers draft acceptance: on Qwen3.8-27B thinking-off
replies, a round lands about 4-10% fewer tokens at either Qwen sampling set and 0-4% fewer under greedy than with
the penalty at 0, varying a lot by reply (one reply lost 25%, another gained 24%). The time per round doesn't change,
so the whole cost is acceptance, and undrafted decoding costs the same with the penalty. Drafted replies stay
byte-identical to undrafted ones. @BobClawblaw measured this on an RTX 5090 with DFlash2, and @jschmied on a GB10
with DFlash2 and DSpine (#2645). `--instruct-sampling presence_penalty=0` keeps the card's temperature, top_p and
top_k without the penalty.

`--name-priority ID=background` gives one served id (`--name` or an `--alias`) a default priority: a request that
asks for it and sends no `priority` of its own is treated as `priority: "background"`. The request's own `priority`
field always wins over the default. This lets one server answer to several ids at their usual priority while a
background-only client (a batch extractor, say) gets one id that always yields, without sending the field itself.

A streamed reply whose client stops reading does not keep its lane decoding. On an engine on the shared lane host, once
1,024 of a stream's tokens are waiting in the server for its client, the stream sits out rounds; it runs again once the
client has taken all but 512 of them. Other streams keep their pace, and the text the client receives is the same,
byte for byte, as a fast client's. A stream that sits out 60 seconds while its client reads nothing is cancelled as a
disconnect is: the reply ends with `finish_reason: "cancelled"` and its lane goes to the next request. A client that
reads as the reply arrives never pauses, and whole (non-streamed) replies are not affected. Engines served outside the
shared lane host, such as Flash Next's one-reply-at-a-time host, do not pause.

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
request did not offer is kept as text too, under the name the model wrote, so the client sees the markup it can
report and the model can correct itself.

When a model end token arrives inside a whole tool call before its closing markup, the server closes the
markup and returns the call. A GLM value the end token cut off is closed with `</arg_value>` first, and the call
returns when it then parses strictly. Any other partial value, a missing key value, an empty block, or an unoffered
call that cannot parse as one complete call remains content.

With `parallel_tool_calls: false`, the server buffers tool deltas until it can return the first valid
completed call. Prose and reasoning can still stream. Usage counts the entire decoded reply, including
additional calls omitted from the response. Otherwise both servers stream each Qwen XML call as it is written: a
delta with the call's id and name, then its arguments in pieces; a call the streamer can't follow arrives whole at
the end.

On the native server a call's id is `call_` and 24 hex digits of HMAC-SHA-256 under the server's tool-call id key, over
the request's own digest (its body as compact JSON in the order sent, without `stream` and `stream_options`), the choice
and the call's index. On `/v1/messages` and `/v1/responses` that body is the chat request the route translates the
client's body into, not the client's raw body, so two client bodies that translate to the same chat request get the same
ids, and a field the translation drops changes none. Without the key an id reveals nothing about the request: whoever
sees it cannot test a guessed request against it. By default the server draws a random key at startup and never writes
or logs it, so a request replayed to the same running server names its calls the same way, streamed or not, while after
a restart or on another server the same request gets other ids. For ids that hold across restarts and machines, set
`TENSORFOLD_TOOL_ID_KEY_FILE` to a regular file only its owner can read or write (`chmod 600`) holding at least 32
bytes, for example from `openssl rand -hex 32`; one trailing line ending is not part of the key, and every server
reading the same file gives the same ids. A missing, unreadable, linked, shared, shorter or over 4 KiB file stops the
start with the setting's name, and the key itself is never read from the environment or the command line. Byte-identical
requests under one key share ids, as a replay does; another request or another call gets other ids. A Responses
`function_call` item's `fc_` id is keyed from its `call_id` the same way, Kimi's ids are the indexes the model writes
(`call_1`), and ids a client sends back in its history are rendered as sent.

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
`zig-out/native/lib` on macOS, Linux and Windows. The server looks in `../lib` beside its binary, then beside
it, then on the loader's path, and loads it at the first structured request; without it, everything else serves and a
structured request gets HTTP 400. The native engines listed below expose token grammar support; other engines
refuse structured requests. Drafting stays on: the drafts the head holds on the GPU come
back to the host before each verify, the grammar cuts the ones it rejects and drafts the text it forces itself (a key,
closing markup), and each row's logits are masked before its draw, so drafted, `"draft": false` and concurrent replies
are the same.

The native server also takes `response_format: {"type": "structural_tag", "structures": [...], "triggers": [...]}` on
`/v1/chat/completions` and `/v1/completions`. The reply is free text, and each structure's `begin` string opens a region
that must be JSON matching its `schema` and close with its `end` string. Each structure is
`{"begin": "<weather>", "schema": {...}, "end": "</weather>"}`: `begin` and `end` are non-empty strings of at most 256
bytes, and `schema` is a JSON schema object (or its JSON text, or `true`). `triggers` are the strings that open a
region: each must start at least one structure's `begin`, and every `begin` must start with exactly one of them.
Without `triggers`, each distinct `begin` that no shorter `begin` starts is a trigger. A trigger can't appear in the
free text except as the start of a region, so with `"triggers": ["<"]` every `<` opens a structure. Outside regions the
text is unconstrained; inside one, its JSON allows at most 32 blank characters between two tokens, as `json_schema`
does. A reply may hold any number of regions, including none, and can't end inside one. A request takes up to 64
structures and 64 triggers, compiling to at most 512 KiB of grammar; a malformed list returns HTTP 400 before any stream
starts, naming the structure and field. xgrammar's own `{"type": "structural_tag", "format": {...}}` form is refused.
The server compiles the request onto the same grammar engine as the other formats (xgrammar's `triggered_tags`), so
the backends, drafting and thinking work as described above, and `grammar_budget` as described below. As with the
other formats, it takes the place of the strict-tool grammar under `tool_choice: "auto"` and is refused beside
`tool_choice: "required"` or a named function. Requests without it are unchanged.

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
With `--tool-think-close` (off by default), a required or named call on an engine that takes `thinking_budget` gets a
think budget of `max_tokens - max(256, max_tokens / 4)`, or the request's own `thinking_budget` when that is smaller,
so the think block closes with room left for the call instead of ending as `length`. It changes the reply of any such
call whose thinking would have run past that point. `"auto"`, `"none"` and requests without tools are unchanged.
Drafting stays enabled and uses the same grammar masks as plain decoding.

This requires the loaded backend's structured-output capability and `libtfgrammar`. Nemotron, Qwen3.5-2B,
Qwen3.8-27B (`qwen27`), Gemma 4 31B, Step-5, LFM2 MoE and Flash Next shared sessions on Metal have that capability; Qwen27 on Intel also has it.
Flash Next and Gemma 4 on CUDA have it too. Gemma 4 on CUDA picks a greedy row under a grammar on the host, among the
allowed tokens of its logits read back; sampled rows are masked on the device before the draw.
Qwen3.8-27B applies the grammar to its first draw, each verified row and each branch of its draft proposals.
Structured requests use its shared lane core, including when the ordinary lone-request serial optimization is enabled.
CUDA determines support from its loaded backend's mask hook. GLM Metal
(`glm-zig`, `glm-dsa-zig`) lacks that hook and refuse required/named choices before generation with HTTP 400, for example:
`backend 'glm-zig' cannot enforce tool_choice "required" or a named function: it has no token grammar support; send "auto" or "none"`.
The server does not silently turn a required call into `auto`. Installing the grammar library does not add a missing
backend mask hook. A missing library or grammar compilation error is also refused before generation with its reason.

The native GLM CUDA backend also supports token grammars, gathering vocabulary rows before rank 0 masks and draws.
DeepSeek-V4.1 on CUDA (`dsv41-cuda`, a source build) supports them too: rank 0 sends each pick's grammar rows to the
other ranks, and every rank masks its own vocabulary columns before the pick.

A native grammar request waits at most 2,000 ms for its compile. This holds for every format, including JSON objects
and schemas, regexes, choices, EBNF and the structural tags of tool calls. Past that the request gets HTTP 400 naming
the limit, for example `response_format: the grammar cannot be enforced: grammar compilation exceeds the 2000 ms time
limit`. Each model compiles at most 4 grammars at once, outside any lock other requests take, so one slow schema does
not hold up other requests' grammars. A request that finds all 4 busy waits for one within the same 2,000 ms, then
gets HTTP 400 naming both limits. Identical requests share one compile. A grammar compiled before is served from the
cache with the same bytes. The last 8 that finished in time need no compile slot, and the rest come from xgrammar's
own 256 MiB cache. xgrammar cannot stop a compile once it has started, so an abandoned one runs on in the background
and keeps its slot until it returns. The grammar it finishes goes into xgrammar's cache, and the same request sent
after that is served. An invalid grammar is refused with xgrammar's reason, as before. These limits are separate from
`grammar_budget`, which bounds the tokens a constrained reply needs to complete.

### Strict tools under `tool_choice: "auto"`

With tools offered and `tool_choice` omitted or `"auto"`, a tool marked `"strict": true` inside `function` is held
to its `parameters` schema when the backend has token-grammar support and the grammar library loads. The reply may be
plain text or calls, and text outside the call markup is free. Inside a call the name is one of the offered tools, a
strict tool's arguments must follow its schema, and a tool not marked strict takes any JSON object.
`parallel_tool_calls: false` allows one call; unset or `true` allows several. Auto never requires a call: a text-only
reply is not an error. Chat passes the tools through; Responses copies the tool's top-level `strict` into `function`.
Anthropic Messages tools carry no `strict` flag, so they are held only when the switch below is on.

`TENSORFOLD_TOOL_GRAMMAR=1` holds every auto tools request the same way: each offered tool's arguments must follow
its schema, marked strict or not. Only the value `1` turns it on, with ASCII spaces around it ignored; unset, empty,
`0` and any other text leave it off, and none of those is refused at startup. Unset is the default. The switch
applies only to auto: `required` and a named function already constrain the reply to schema-valid calls, so it
changes nothing there. Where the backend lacks token-grammar support or the grammar library does not load, an auto
request is served unconstrained rather than refused; the `required` and named cases in that position are refused
before generation with HTTP 400 and the backend's or library's reason, as described above. A strict auto grammar
xgrammar cannot compile is also refused before generation with HTTP 400, for example:
`strict tools under tool_choice "auto" on backend 'qwen27': the grammar cannot be enforced: <xgrammar's reason>`.
LFM2's template writes its calls as one EBNF rule rather than a triggered tag, so strict tools under auto are not
grammar-held there.

The legacy `TF_DSV41_TOOL_GRAMMAR` name takes `0` or `1`. When `TENSORFOLD_TOOL_GRAMMAR` is set, even to an empty
value, it is read instead and the legacy name is ignored, even if the legacy value is malformed. A malformed legacy
value without that override logs the accepted form at start and exits with status 1, for example:
`TF_DSV41_TOOL_GRAMMAR="true": expected 0 or 1 (ignored while TENSORFOLD_TOOL_GRAMMAR is set)`.

`grammar_budget: true` (native server, off by default) keeps a constrained reply completable within its token limit, so
a required tool call or a JSON value cut by `max_tokens` still parses. Before each draw, a token is masked when no
continuation after it reaches a place where the grammar could end within the draws left (`max_tokens`, counted after
the think block when the server reserves thinking tokens beyond it); the count includes that token and treats the place
where the end token becomes allowed as complete, so a call that closes on the last token can still finish as
`length`. Outputs change only where the budget binds: a token is masked only when every completion after it would
overrun, so the reply equals its `grammar_budget: false` reply token for token up to the first draw where the limit
binds. The mask depends only on the grammar's place and the tokens left, so
drafted, `"draft": false` and concurrent replies stay the same with it on. When `max_tokens` is already below the
grammar's shortest reply, the mask is not applied and the reply is cut as without it. The first request for a grammar
explores its parser states over the tokenizer's tokens on a bounded background job. Client-supplied schemas,
regexes, choices, EBNF and tools keep a 2,000 ms wait, a fixed counted-work cap, a successor-edge cap and a table-memory
cap; four budget jobs may run at once. A refusal names work, edges, table memory, time or waiting for a compile slot.
The fixed `json_object` rule is built by the server, so only it has a larger counted-work cap and a 30-second budget
wait. Its compiled rule and successful budget tables are retained once per loaded tokenizer, outside the client
caches; selecting the same rule later reuses them. Identical client schema text does not receive those larger bounds.
A grammar with more than 16,384 named parser states is masked only where the explored states decide it, never beyond
what completes. Counted work is deterministic for a grammar and tokenizer; deadline and slot outcomes depend on load.
A refused exploration never publishes partial masks. Libraries predating budget support are refused with HTTP 400,
as are grammars with token-level, budgeted, captured or left-recursive rules. With thinking on, the masks start after
the think end.

HTTP 400 comes before any token for a malformed field, a grammar xgrammar cannot compile (with xgrammar's reason), a
grammar over 512 KiB on the native server, a server without xgrammar (with the install command; on the native server, a
server that cannot load `libtfgrammar`), a grammar sent with `tool_choice: "required"` or a named function. A sent grammar is measured as the server compiles it:
a JSON schema or a list of choices as the server re-serializes it (which can differ from its size as sent), a regex or grammar as sent. Tools are
measured by the grammar for one call to any of them, which holds each tool's name and parameter schema once (every
tool's schema under `tool_choice: "required"` or a named function, the strict tools' under `"auto"`); the grammar for
parallel calls repeats that call and is not measured again, and its 400 names `tools`. A reply whose grammar fails while
decoding ends with HTTP 500 (an error event when streaming); the other requests go on.

## Reasoning

Native `/v1/models` entries and aliases include `capabilities.thinking_switch`. It is true when explicit thinking off
is supported for a simple text chat by the existing template switch or generation-prefix closure, or the tokenizer has
no thinking-close marker. The server decides it once at startup by rendering a simple chat through the same off check
requests use; each request still validates its own rendered prefix. It is false for a refused probe or a model without
text generation. Benchmark clients read this bit once and omit the off field when it is false, labelled as the model
default. Unknown discovery preserves the off request, labelled `off requested (unverified)`. The separate
`capabilities.thinking_applicable` bit is false when no thinking-close marker exists; clients then report
`not applicable` and send no thinking field. DeepSeek-V4.x's native encoder has its own thinking switch (thinking off
ends the prompt with `<｜Assistant｜></think>`), so its explicit off is never refused.

On both backends, `reasoning_effort: none` asks the chat template to disable thinking; an explicit off request
returns HTTP 400 when the template cannot honor it. Only an off field the client sends is explicit: `--no-thinking`,
the template's own default and the Messages route's default off never are. Other effort values enable thinking
and reach the chat template. The server also reads it from `chat_template_kwargs.reasoning_effort`, where vLLM's
clients send it; the top-level field wins. An unnamed level maps to the nearest level the template names, and a tie takes
the higher one. GLM-5.3 lists `low`, `high` and `max`, so `max` stays `max`, `medium` is heard as `high`
and `minimal` as `low`. Qwen3.8 lists `low`, `medium` and `xhigh`, so `high` and `max` are heard as `xhigh`.
`xhigh` stays `xhigh`. GLM-5.3 renders `xhigh` and `max` as Max. A template that names no level hears `max` as
`xhigh`. `--reasoning-effort` accepts `minimal`, `low`, `medium`, `high`, `xhigh` and `max` and uses the same rule; use `--no-thinking` for the CLI off switch. An omitted effort, with no startup flag, stays
the template's own default. An explicit `chat_template_kwargs.enable_thinking` takes
precedence. A request without an effort gets `--reasoning-effort` when the server was started with one; otherwise
the template renders its own default, as vLLM and mlx-lm render it (Qwen3.8's is `xhigh`, which adds an instruction
to the system prompt; `medium` adds none). Both backends render the same prompt for the same request.
Effort support depends on the checkpoint's template, and effort does not set a
token budget. With thinking off, native passes the template only an effort the request itself names (its
`reasoning_effort`, other than `none`); `--reasoning-effort` never reaches a thinking-off prompt, and with no named
effort it passes none, as before. So a GLM-5.3 request can send `reasoning_effort: "high"` beside
`chat_template_kwargs.enable_thinking: false` (or `thinking: false`) to render `Reasoning Effort: High` instead of the
template's Max head; Qwen3.8's template reads the effort only while thinking, so its thinking-off prompt is unchanged.
The reply's `tensorfold.reasoning_effort` names the effort the template heard (`none` when it heard none).
If the template's generation prefix ends
in a single `<think>` token, native appends the single `</think>` token, leaving an empty think block. This handles
GLM-5.3 and LFM2 templates without their own off switch. Other template text stays unchanged: GLM-5.3's
`Reasoning Effort` head stays as the template writes it, and an older GLM-5.3 template still writes its default
`Reasoning Effort: Max` line. The released Python 0.6.6 engine instead rewrites GLM's render for a no-thinking
request — its `thinking_off` strips the `Reasoning Effort` head and closes the block inline — so its no-thinking
prompts are 6 to 7 tokens shorter than native's on the same request (#1911 measured a system-plus-user prompt at 22
tokens rendered, 16 on the Python engine, and 23 on native). A prompt count, a cache key or a kept state from one
engine therefore does not match the other's for `enable_thinking: false`. GLM-5.3 keeps every earlier assistant
turn's reasoning in the prompt, as zai-org's template does by
default (`clear_thinking` false), also on checkpoints whose template still clears it before the last user message, so
a new user message leaves the earlier turns' tokens, and their kept prompt states, as they were. The native server
gives the template `clear_thinking` false on every render, and `TF_GLM_CLEAR_THINKING=1` sets it true for the server,
from @MiaAI-Lab's [#236](https://github.com/ashhart/TensorFold/pull/236); any value but 0 or 1 refuses to load. A
request's `chat_template_kwargs.clear_thinking` (true or false) overrides the server's setting for that request, as the
model card advises `true` for plain chat; any other value is a 400, and other families' templates never receive it.
`true` keeps the reasoning after the last user message, so an autonomous tool loop keeps most of it; the
[GLM-5.3-Flash recipe](recipes/glm-5.3-flash.md#long-agent-sessions) covers long agent sessions.

A tool call written before the think block closes is the reply's tool call when the reply ends inside the block,
on both backends; the reasoning stops where the call starts, and streamed reasoning never carries the call's markup.
A complete call at the end of reasoning is also returned when the think-close marker follows and the answer is
empty or whitespace. While streaming, those calls stay held until the reply ends. When the block closes, a call
followed by more reasoning or a nonempty answer stays reasoning.

A reply that ends at the model's own end token while its think block is still open, with no tool call in it, also
answers with its reasoning (#3687): `content` repeats the text, which stays in `reasoning_content` as well, so
`/v1/messages` gives a thinking block and then a text block, and `/v1/responses` a reasoning item and then a message.
It applies only when the model ended the reply itself: not at a stop string, `max_tokens` or the loop guard, and not
for a request with a forced tool call or structured output, which prose cannot satisfy. Streaming sends the answer as
one content delta at the end, so the joined deltas equal the finished reply. The response carries
`tensorfold.reasoning_recovered: true` and the server logs one line. Token ids, `token_sha256` and token counts are
the model's own, unchanged; `logprobs` keeps those tokens under `reasoning_content`. A client that sends the reasoning
back sends the text twice, which costs prompt tokens on the next turn.

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
engines, and with thinking disabled, `max_tokens` limits the whole reply as before. Beside a long prompt, `TENSORFOLD_PREFILL_ORDER=srpt` counts a reply that is still reasoning as its `max_tokens` plus the reasoning it is still expected to need, from a moving average of the think blocks this server has seen close, and never more than its limit left (its `max_tokens` plus N, less what it has emitted); until the first has closed it counts as that.

The native `--loop-guard` flag is off by default and watches only an open think block. It closes a repeated exact
cycle of at most eight tokens after 256 predecessor matches, when the detected region starts at least 64 reply tokens
in. It also closes a longer passage repeated verbatim: once a period of nine tokens or more has repeated in full and
for at least 1,024 tokens, the period found from where the reasoning's last 32-token run occurred before. The reply
continues after the forced close, and reports `runtime.loop: {"period": N}` plus `loop=period:N` in the server log. The native engine refuses the flag on backends that do not enforce it, and prints that refusal as
a startup error.

`--loop-guard-stop` (off by default; it turns on `--loop-guard`) ends the reply where the guard fires instead. The engine
writes no forced close and no answer: the stream finishes at the token that fired, and its lane and caches are released
as for any finished reply. The reply carries the reasoning so far in `reasoning_content`, empty `content` and no
`tool_calls`, and these marks a client can act on:

- `finish_reason` is `content_filter`, the OpenAI value for a reply the server ended before its answer, which clients
  that check the value against OpenAI's list accept; it is neither `tool_calls` nor `length`, so it asks no client to
  run calls or continue a cut reply.
- `runtime.loop` is `{"period": N, "stopped": true}` and the server log shows `finish=content_filter loop=period:N`.
- The error object `{"message": "...", "type": "server_error", "param": null, "code": "reasoning_loop"}`: a whole chat
  or completion reply (HTTP 200) carries it as a top-level `error` beside `choices`; a stream sends it as an
  `{"error": ...}` event after the finish chunk and its usage, then `[DONE]`, so OpenAI's SDKs raise it after the
  reasoning arrived.
- `/v1/responses` replies with `status: "incomplete"` and `incomplete_details: {"reason": "content_filter"}` (streamed:
  `response.incomplete`), its output the reasoning item alone; `/v1/messages` replies with `stop_reason: "refusal"`.

Requests with tools or structured output end the same way: the guard watches only the think block, so no call is opened
and no grammar starts. The setting is the server's: a request cannot turn it on or off. Without it, replies are
unchanged.

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
model's weights and cache nor keeps them from crowding other work on the machine. The startup budget is the host's
`MemAvailable`, at most the free memory CUDA reports plus the checkpoint's own cached bytes, less a reserve (a tenth
of RAM, at least 4 GiB). The checkpoint's clean pages stay in the page cache; they are reclaimed when CUDA needs
them, so the first prompt does not fault the weights back in. Other memory on the box still bounds the budget.
On native CUDA, a stream's window is `--context`, the model's own window at `--context 0`, or, when it is unset, the
family's default capped at the model's own window; on the paged families the streams and pages then fill the budget up
to the streams' windows plus the cache's pages, so once those exceed it, the free memory after load is the reserve.
To leave room for other containers, start the server with a smaller `--context` or `--parallel`.
`TENSORFOLD_MEMORY_RESERVE_GIB` replaces that reserve (at least 2 GiB) when you know the machine's headroom: a larger
one leaves more for other work, a smaller one more for KV caches. The reserve also carries CUDA context, NCCL and
workspace memory the estimate does not count, and exhausting a unified GPU's memory can freeze the host, so lower it
only with room to spare.
After start-up on unified memory, `MemAvailable` settles somewhat above the reserve and the pool's page count can
differ by a few tens between launches with the same settings, because the room is read once while the host is still
settling.

On a discrete card (an RTX 3090, an A100) the budget is the card's own free memory, as the driver reports it, less a
reserve of a 32nd of the card and at least 1 GiB, and the plan is the same. `TENSORFOLD_MEMORY_RESERVE_GIB` sets that
one from 0.5 GiB up.

With pages, the plan takes the engine's own window first, then each other stream's fixed part, then the prompt cache:
a named `--prompt-cache-gib` comes before every window but the engine's own, and an unnamed one after the whole
windows the budget holds, half of what they leave but at least 16 GiB and at most all of it. On GLM with
`TF_KV_SHARE` named and the cache unnamed, the kept states' slots, `TF_KV_SHARE` at most, come where a named cache
goes, and the rest of the cache after the windows. The cache then holds the kept states (on GLM with `TF_KV_SHARE`,
a slot a state, the rest of it made pages), and the rest of the budget becomes pages, up to the streams' windows plus
the cache's pages.
The settings that decide each part:

| Setting | Kind | Scope | Unset | What it decides | Read in |
|---|---|---|---|---|---|
| `TENSORFOLD_MEMORY_RESERVE_GIB` | environment | the host: every CUDA native family, each rank at load | a tenth of RAM, at least 4 GiB (unified); a 32nd of the card, at least 1 GiB (discrete) | what the budget leaves free | `zig/src/native/cuda_family_memory.zig`, `cuda_memory.zig` |
| `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` | environment | the host: every CUDA native family, each rank at load | none | a cap on the budget, the weights included | `zig/src/native/cuda_family_memory.zig`, `cuda_memory.zig` |
| `--context` | flag | the launch | the family's default, capped at the model's own window | a stream's window; `0` takes the model's own | `zig/src/server/cli.zig`, `zig/src/native/cuda_window.zig` |
| `--parallel` | flag | the launch | `auto`: up to 8 | the streams | `zig/src/server/cli.zig` |
| `--prompt-cache-gib` | flag | the launch | after the whole windows, as above | the prompt cache's budget | `zig/src/server/cli.zig`; unset, `zig/src/native/cuda_kv_pool.zig` (`budgetOf`) and `cuda_memory.zig` |
| `TF_KV_SHARE` | environment, in the profile registry | the launch, on the families of the core page adapter | off: copies | prompts kept as pool pages, with at most this many states past them | `zig/src/core/kv_prompts.zig` |
| `TF_GLM_KV_HOST_GIB` | environment | GLM | one window and 16 pages | GLM's host KV pages with `TF_GLM_KV_HOST=1`, outside the device pool | `zig/src/families/glm/cuda_kv_host.zig` |
| the pool's pages | none | the core plan | — | what is left | `zig/src/core/kv_pool.zig` (`plan`) |

Measured numbers for one launch are in #3000.

## Compaction

Without `--compact-at` nothing compacts: a conversation that outgrows the context is refused with the usual
`context_length` error. With it, the server summarizes instead of refusing. `--compact-at auto` reserves room for
the reply — the larger of 15% of the context window and 16,384 tokens, or 15% when the window can't hold 16,384 — and compacts a conversation whose prompt
plus wanted reply tokens would cross what is left, or one the plain prepare refuses for context length.
`--compact-at F`, a fraction of the window in (0, 1], compacts when they would cross F × the window, rounded down.
Any other value is refused at startup ("expected auto or a fraction in (0, 1]").

Compaction keeps the conversation's leading messages and its recent tail whole and summarizes the middle into a
memory note. The note is written by the model itself — greedy, thinking off, at most 1,024 tokens a chunk — under
a task that asks for Markdown sections (Goal, Constraints and preferences, Progress, Decisions, Next steps,
Critical context) and under 800 words. It replaces the summarized middle as one message that begins "Memory of
the earlier conversation:". The tail is `--compact-keep` tokens of recent messages, an integer from 0; the
default keeps the smaller of 20,000 tokens and a quarter of the window. If the rebuilt conversation still does
not fit, the server summarizes again over a smaller tail until it does. A compacted request runs these extra
generations before its reply.

`--compact-memory DIR` stores each conversation's note under DIR as `<key>.md`, with a `<key>.json` recording how
many leading messages the note covers and their hash. A later request whose leading messages hash the same reuses
the stored note instead of regenerating it. Without the flag a note lives for one request.

The reply's `tensorfold` object carries `context_window` and `context_used`, and when compaction ran also a
`compaction` object: `messages_compacted`, `kept_messages`, `note_tokens` and the `note` text. Compaction applies
to chat completions with a `messages` array, and to the Responses API, which runs as one; a completions request
with a plain string prompt is never compacted.

## Image generation

On an engine that generates images, POST `/v1/images/generations` takes `model`, `prompt` and `n` from 1 through 4
(default 1). The pictures run one after another through the engine's one-picture call with the same prompt, negative
prompt, size, steps and guidance; only the seed differs. The reply's `data` array is in request order, each item
with `b64_json` (the PNG) and `seed`, the integer seed that picture used.

An explicit `seed` is an integer from 0 through 18446744073709551615. Picture `i`, counting from zero, uses
`seed + i`; a request whose last seed would pass that range is HTTP 400 before any picture starts. Without `seed`,
the pictures start from 42, as a one-picture request always has, so picture `i` uses `42 + i`. A reported seed
sent back with `n: 1` and the same other fields gives the same PNG bytes:
`{"model":"m","prompt":"A red kite","n":4,"seed":42}` returns seeds 42, 43, 44 and 45, each picture the same as the
`n: 1` request with that seed.

`n` outside 1 to 4, or not an integer, is HTTP 400 "n must be from 1 to 4". The other fields are `negative_prompt`,
`size` (`WIDTHxHEIGHT`), `steps`, `guidance` and `response_format: "b64_json"`; the engine may tighten the size, step
and prompt limits, and URL output is not served. A server whose model does not generate images answers 404.

The reply is written once every picture has finished. A refusal, a failed generation or a cancel stops the sequence
and returns that error with no pictures. A client that disconnects cancels the current picture, and the later
pictures are never submitted.

## Image generation streams

With `stream: true`, on an engine that reports its denoising steps, the route answers `text/event-stream`. After
each step it sends `image_generation.progress` with `index` (the picture, from 0), `seed`, `step` (from 1) and
`total_steps`; when the picture is done, `image_generation.completed` carries its `b64_json` PNG, `index` and
`seed`. With `n` above 1 the pictures follow one another on the same stream, each counting its own steps, and one
`data: [DONE]` ends it. Each event's `type` is its event name, OpenAI's for the completed picture and the previews.

`preview_interval: N`, an integer from 1 through `steps` sent with `stream: true` to an engine that makes previews,
adds `image_generation.partial_image` after every Nth step's progress event. Its `b64_json` is a separate PNG of at
most 128 pixels a side and 81,920 bytes, with `width`, `height`, `step`, `total_steps`, `index`, `seed` and
`partial_image_index` (from 0). Previews do not change the picture: the final PNG bytes are the same with or without
`stream` and `preview_interval`, and no preview is made unless one is asked for. For example,
`{"model":"m","prompt":"A red kite","seed":42,"steps":8,"stream":true,"preview_interval":2}` sends progress for
steps 1 to 8, previews after steps 2, 4, 6 and 8, then the picture.

A request without `stream`, or with `stream: false`, gets the whole JSON reply as before. `stream` on an engine
without step progress, `preview_interval` without `stream` or on an engine without previews, and a value out of
range are HTTP 400 before anything is submitted. A refusal before the stream starts is the usual HTTP error; once it
has started, a failure is an `error` event holding OpenAI's error object, then `data: [DONE]`, and the pictures
already completed stay sent. A client that disconnects, or a write that fails, cancels the picture at its next step,
and no later picture is submitted.

## Speech formats

`POST /v1/audio/speech` answers in `response_format` `wav` (the default: 16-bit mono at the model's rate), `pcm` (the
same samples, little-endian, no header), `wav24` (24-bit), `wav32f` (IEEE float WAV carrying the model's samples bit for
bit at its own rate, unclipped) or `flac` (16-bit FLAC that decodes to the `wav` reply's samples). `mp3`, `opus` and
`aac` are refused by name. `sample_rate` resamples the take to 8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000,
88200, 96000, 176400 or 192000 Hz with the band-limited resampler the reference voices use. `channels: 2` writes each
sample on both channels. Without these fields, or with them at the model's rate and 1, `wav` and `pcm` replies are the
same bytes as before. A stream (`stream: true`) stays 16-bit mono PCM at the model's rate and refuses other values.

## Embeddings

`POST /v1/embeddings` takes OpenAI's body: `input` is one text, a list of texts, one list of token IDs or a list of
them, with `encoding_format` (`float` or `base64`) and `dimensions`. Every family that serves embeddings shares these
refusals, each an HTTP 400 `invalid_request_error`:

| Input | Refusal |
| --- | --- |
| a text that is empty or whitespace only (Python's whitespace set, which SGLang strips before it refuses) | "embedding text must not be empty or whitespace only"; in a list, "embedding input at index N must not be empty or whitespace only" |
| one input longer than the model's context | code `context_length_exceeded`: "the input (N tokens) is longer than the model's context length (C tokens)" |
| inputs that each fit but together reach the server's batch limit | "the embedding batch exceeds the admitted N tokens" |

An over-long input is named before the batch limit is checked, so a client learns that the input itself is too long,
not that the batch is too full. `dimensions` crops each vector to that many leading values and renormalizes it; a
model with Matryoshka (MRL) widths takes any width in its declared range (32 to 4,096 for Qwen3-Embedding-8B).

## Responses

`choices[0].message.content` holds the answer. Reasoning uses `reasoning_content`, or
`delta.reasoning_content` while streaming. Tools use `tool_calls` and `finish_reason: "tool_calls"`.
When a template supplies no think opener, a reply-written leading `<think>` tag is removed from
`reasoning_content`; the streamed and finished forms use the same split. Both strip the reasoning's leading and
trailing whitespace, so the joined `delta.reasoning_content` pieces equal the finished field byte for byte; a stream
holds whitespace back until more reasoning follows it.
Every reply's usage, streamed ones included, carries its prompt, completion and total tokens,
`prompt_tokens_details.cached_tokens` (the prompt tokens found in a backend's prefix cache; native Zig reports
zero until its prompt cache is enabled) and
`completion_tokens_details.reasoning_tokens` (the thinking tokens, through the closing think marker); timing details
depend on the backend.
TensorFold also reports generation statistics such as decode rate, time to first token and draft acceptance.

Native JSON replies also carry a `tensorfold` runtime block with `engine`, `enable_thinking`,
`reasoning_effort`, `tokens_per_second`, `seconds`, `time_to_first_token`, `prefill_seconds`,
`prefill_widths`, `prefill_raised`, `sampling`, `drafts`, `token_sha`, and a `speculative` object beside
the block with the draft counters `rounds`, `drafted`, `accepted`, `acceptance_rate`, and
`tokens_per_round` (the drafter's own counters arrive as `proposer`).
When a reply keeps tool-call envelopes as text instead of returning them as calls, the block also carries
`kept_tool_calls`: `unoffered` and `malformed` counts, and `unoffered_names`, the names the model wrote for the
kept unoffered calls, in order, without duplicates, at most 8 names of 64 bytes each. The whole reply and the
streamed finish event carry the same object; a reply that kept none has no `kept_tool_calls` field.
A request that sends `return_token_ids` adds `token_ids`, the reply's tokens, to the block — the tokens `token_sha` and
`token_sha256` hash — on the whole reply and on the streamed finish event alike. A loop-guarded reply adds
`runtime.loop.period`, and `runtime.loop.stopped` when `--loop-guard-stop` ended it; these fields describe the native
reply and are absent from Python replies that do not provide them.
In the native block, `seconds` is the reply's decode time, from the end of its prompt pass to its last token, and
`tokens_per_second` divides the reply's tokens after the first by it. `prefill_seconds` is the prompt pass, and
`time_to_first_token` counts from the request's arrival, so a queued or long prompt shows there, not in `seconds`.
`min_rows` is the smallest row count any of the request's engine calls ran with. A reply whose lane measured
grammar windows adds `grammar_windows` and `grammar_fills`, counts, and `grammar_window_seconds` and
`grammar_fill_seconds`, in seconds.
The Python 0.6 server's `seconds` is the whole request, prompt included.
A native Metal Qwen3.8-27B server started with `TF_Q27_PROFILE=N` logs each kernel class's GPU time, the DFlash2
drafter's included, every N decode rounds (`1` means 64; unset, empty or `0` is off, and any other value leaves it off
with a warning). Prompt passes run unprofiled, and a profiled round runs slower while its spans are timed. Its replies
add `gpu_seconds` to the block: the GPU-busy time of the target model's forward passes the reply ran in, its prompt
pass included. A round shared with other streams counts whole in each of their replies, and the drafter's passes
are not included. Without `TF_Q27_PROFILE` the field is absent and replies are unchanged.

Beside the block, native replies carry top-level `exact_mode`, always `"exact"`, and `stop_sequence`, the matched
stop string, when a stop string ended the reply. A server started with `--cortheon` adds a top-level `cortheon`
object summarizing the turn for requests Cortheon runs ([Cortheon](cortheon.md)). These ride on the whole reply and
the streamed finish event alike.

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
drafts share it), `kept_tool_calls_unoffered_total` and `kept_tool_calls_malformed_total` (tool-call envelopes kept as
text on finished replies, by an unoffered name and by malformed markup), `request_latency_seconds`, `time_to_first_token_seconds` and `request_decode_seconds` (each
finished request's decode time: the CUDA engine's own figure, or first token to end on the Mac server; its `_sum`
over `generation_tokens_total` is the decode rate), `request_time_per_output_token_seconds`, `decode_rounds_total`,
and the `request_prefill_seconds` histogram, all under the `tensorfold:` prefix. TPOT already carries vLLM's name.
The readings vLLM also has are repeated under its names (`num_requests_running`, `num_requests_waiting`,
`kv_cache_usage_perc`, `spec_decode_num_draft_tokens_total`, `spec_decode_num_accepted_tokens_total`,
`e2e_request_latency_seconds`, `request_decode_time_seconds`, `request_prefill_time_seconds`) with identical values, so a dashboard copied from vLLM fills by swapping the
`tensorfold:` prefix for the metric name. `client_disconnections_total` (requests the client walked away from)
and `preemptions_total` (requests requeued for memory before their first token, plus background passes that
yielded their lane to foreground work) are published where the
server counts those events, and never at a fabricated zero.
An engine that keeps prompt states between requests (Flash Next and GLM-5.3-Flash unless `--prompt-cache-gib 0`)
adds `prompt_tokens_cached_total` (already vLLM's name), the prompt tokens of finished requests restored from a kept
state, and repeats it with `prompt_tokens_total` as vLLM's `prefix_cache_hits_total` and `prefix_cache_queries_total`.
An engine that keeps no prompt states leaves all three out.

The native `--dashboard` flag adds a self-contained `/dashboard` page and its `/stats` JSON snapshot; both routes
return 404 when the flag is off, and the page polls once per second without adding access-log noise.

[packaging/grafana/tensorfold-engine.json](../packaging/grafana/tensorfold-engine.json) is a Grafana dashboard over
these families: TTFT and TPOT quantiles, the per-request decode rate, running and waiting requests,
`kv_cache_usage_ratio`, drafts accepted over drafted, cached over total prompt tokens, `prompt_cache_bytes` against
its budget, process and device memory, and admission refusals. Import it, pick a Prometheus data source, and choose
instances by their scrape `instance` label. A panel whose family the server leaves out stays empty. A host test fails
when a query names a family `metrics.zig` does not write.

### Scheduler load

The scrape also carries the scheduler load a traffic shaper needs, from one ledger the server owns. Every server
publishes `server_info{model,backend,version,instance}` (always 1; `instance` is a random id made at process start,
so counters restart only with a new `instance`), `process_start_time_seconds`, `engine_parked` (1 while
`--idle-unload` has closed the engine), `lanes`, and `http_requests_active{route}`: model requests from the end of
their headers to the end of their reply, by route group (`chat_completions`, `completions`, `responses`,
`messages`, `embeddings`, `speech`, `other`), `request_refusals_total{reason}`: requests refused with a client
code, one of `engine_busy`, `engine_unavailable` and `memory_pressure` (the [Waits](#waits) codes, counted where
the chat, embeddings, rerank and speech routes answer with them, and an images engine's busy 503 as `engine_busy`),
`invalid_request` (the 400 when the engine itself refuses the request it was handed), or an admission flag's 429 by
its code (`max_inflight_exceeded`, `max_inflight_per_key_exceeded`, `admission_capacity_exceeded`,
`requests_per_second_exceeded`; see [Admission](#admission)), and `request_failures_total{reason}`: requests that
ended failed with no refusal code. Its reasons are `backend_prompt`, `backend_round`, `lane_memory_alone` and
`prompt_too_large` (the engine took the request and it failed), `planner_refused` (a family round planner refused it
with its own words, as DeepSeek-V4.1 does for a request past its KV pool or its hard memory floor) and
`engine_stale` (the chat routes' 500 while the device is stale). Every label of both families is always present.

An engine on the shared lane host adds these groups, each complete or absent:

| Group | Families |
| --- | --- |
| Requests | `requests_waiting_by_reason{reason}` (queued requests by the one wait reason each holds, every reason always present; they sum to `requests_waiting`), `restore_fallbacks_total` (requests whose saved prefix could not be restored, so they prefilled from zero and were served; once a request), `waiting_prompt_tokens` (prompt tokens of queued requests, before any kept prefix restores part of them), `filling_prompt_tokens_left` (absent while a backend without fill progress fills), `requests_started_total`, `requests_requeued_total`, `background_yields_total` |
| Work | `scheduler_tokens_total{mode}` with `prefill_compute`, `prefill_cache` and `decode` (output tokens after each request's first), counted as the work happens, and `prefill_busy_seconds_total`: engine seconds in prompt calls whose work `prefill_compute` counted. The `prefill_compute` delta over its delta is the prefill rate. A family round planner publishes no work group |
| Capacity | `kv_capacity_mode{mode}` (`window` or `paged`), `lanes_free`, and for a paged engine `kv_pool_capacity_tokens`, `kv_pool_available_tokens`, `kv_pool_reclaimable_tokens`, `kv_pool_promised_tokens` with `pool="main"`, and `kv_pool_run_available` where the family needs a contiguous run. Published by the CUDA lane host: `window` for an engine without a page plan, `paged` with GLM-5.3-Flash's pool on rank 0; other paged families leave the group out until they name their pool |

In the requests group, `requests_running`, `requests_waiting` and their vLLM mirrors take the ledger's values.
While the engine is parked, none of these engine values is shown: neither the groups nor `requests_running`,
`requests_waiting` and `generation_tokens_running`. A missing group means unknown, never zero. While the idle
unload closes the engine, the engine values are already gone and `engine_parked` is still 0, since a request then
gets 503 with a retry; it turns 1 once a request can reload the engine.

A queued request holds one wait reason, decided where the engine last tried to admit it and kept until it starts
or leaves; reading the ledger never asks the engine again. Three reasons are client codes, the same strings as in
[Waits](#waits): `engine_busy` (no lane is free, or nothing more specific held it yet), `engine_unavailable` (the
device is stale until a put-back reaches it) and `memory_pressure` (memory pressure holds admission to the streams
already running). The rest are internal reasons, never a client code: `kv_pool`, `memory_floor` and
`prefill_concurrent` (a family round planner waits for KV pool pages, for the memory floor, or passes a long prompt
over while others prefill; DeepSeek-V4.1 names them), `prefix_cut` (it waits for a cut another prompt is filling),
`state_read` (it waits for its disk state's read ahead), `ledger_bytes` (its caches' bytes do not fit the KV ledger
now) and `pool_room` (the family's pool has no room for it now). A request behind one that waits for the pool, the
ledger or the planner holds that request's reason. `memory_floor` is never counted under `memory_pressure`. The
reasons are as of the engine's last admission attempt: a request queued behind one just admitted keeps its earlier
reason until the engine next tries to admit it.

`GET /v1/load` returns the same ledger as one JSON object from one read: `instance`, `timestamp` (the read's wall
time), `model`, `parked`, `lanes`, `http_active` by route group, `refusals` by code, and, when present, the request
counts with `waiting_by_reason` (the same reasons, summing to `waiting`), `capacity` with its `pools`, `tokens` and
`busy_us`. Keys of a group no host fills are left out. It follows the
`/metrics` access rule, and `--metrics-open` opens it too.

## Admission

By default the native server takes every model request it can read and the engine queues them. On a server several people
share, four flags can refuse requests early instead, each off at 0, its default:

| Flag | Range | Meaning |
| --- | --- | --- |
| `--max-inflight N` | 0-128 | Model requests in progress at once, across all keys |
| `--max-inflight-per-key N` | 0-128 | Model requests in progress at once for one key |
| `--requests-per-second N` | 0-100000 | Model requests a key may start each second |
| `--request-burst N` | 0-100000 | Requests a rested key may start at once; 0 takes the rate. Without a rate it does nothing |

A model request is a POST to a completion, chat, Responses, Messages, speech, embeddings, decisions, System One, Sliding
Weights learning or imprint route; tokenizing, listing models, health and metrics are never counted. A request is counted
from the end of its headers, after the key check and before its body is read or a parked model reloads, until its reply
ends, streams included. Each key counts apart, by its digest, whatever its label; a server without keys counts every
request as one key. A refused request answers 429 with `Retry-After` and the route's error shape (`rate_limit_error`; the
OpenAI-shaped `code` names the limit) and closes its connection without reading the body. A refusal spends no rate.
A chat or completion with `n` above 1 holds `n` places under `--max-inflight` and `--max-inflight-per-key` (one rate
charge): once its body is read it takes the other `n - 1` places, all or none, or answers the same 429 before any
choice starts; it then counts as both accepted and refused in the metrics.

The rate's buckets live in a table of 128 keys. A key's bucket stays until it has refilled, so cycling keys cannot reset a
spent rate; with more than 128 keys starting requests within one refill, a further key is refused
(`admission_capacity_exceeded`) until a bucket refills. With any of the flags set, `/metrics` adds
`admission_inflight`, `admission_inflight_peak`, `admission_principals`, `admission_rate_buckets`,
`admission_accepted_total` and `admission_rejected_total` with a fixed `reason` label (`global_limit`, `key_limit`,
`principal_capacity`, `rate_limit`). With none set, nothing is counted and the replies and metrics are as before.

### Waits

A retryable refusal on an inference route answers 503 and names its wait in the OpenAI-shaped body's `code`, from one fixed set:

| Code | Meaning |
| --- | --- |
| `server_restarting` | The server is draining for a restart; the answer carries `Retry-After` |
| `engine_loading` | A parked model is unloading or reloading; sending the request again drives it; `Retry-After` |
| `engine_reload_failed` | A parked model's reload was attempted and failed; the next request tries again; `Retry-After` |
| `engine_busy` | The engine had no room for this request (a busy engine, a busy embedding engine, a full speech lane); send it again |
| `engine_unavailable` | The engine is closed (for embeddings and rerank, also stale); retry after it reopens |
| `memory_pressure` | The host is under critical memory pressure and takes no new request until it eases |

A capacity refusal keeps the frozen Python engine's shape (`type` `invalid_request_error` and its message) and still carries no `Retry-After`; the code is native's addition. The Messages routes keep their Anthropic envelope, which has no code field, and the Sliding Weights learn/delete and imprint routes keep their own 503 bodies.

Connections are limited apart from requests. The server holds at most `--max-connections N` open connections (1-1024,
default 128). From three quarters of them open, one address may hold `--max-connections-per-address N` (0-1024; by
default a quarter of the limit; 0 turns it off). An IPv6 address counts by its /64. Loopback clients and each
`--trusted-proxy ADDRESS` (repeatable, up to 64; an exact IPv4 or IPv6 address, for a proxy, load balancer or Docker's
bridge gateway) are never counted. A refused connection gets 503 with `Retry-After: 1`. Only with every slot taken does
a connection waiting for its next request give its slot up, after 5 seconds; a request in progress is never cut.

## Prompt cache salts

By default a kept prompt state is resumed by any request whose prompt starts with the same tokens, whoever sent it.
`cache_salt` (a non-empty string of at most 1024 bytes, on `/v1/chat/completions`, `/v1/completions`, `/v1/responses`
and `/v1/messages`) keeps a request's states apart: a state kept under one salt is resumed only by requests with the
same salt, and states kept without a salt only by requests without one. `serve --cache-salt-per-key` salts every request
by the API key it matched as well (a request's own `cache_salt` then counts within that key, and no salt a client sends
stands for another key's), so two keys never resume each other's states; a server without API keys logs that the option
salts nothing. The salt is hashed with HMAC-SHA-256 under the server's [tool-call id key](#messages-and-tools) (random
at each start unless `TENSORFOLD_TOOL_ID_KEY_FILE` names one), so no cache key or state name can be tested against a
guessed salt, and the salt itself is never echoed, stored or logged (`TENSORFOLD_REQUEST_LOG` writes `<redacted>`). The
hash masks the first four words of the request's cache key. Salts never change the tokens of a reply, and a request
without a salt is keyed exactly as before. Salted states stay in RAM: they are not written to the `--prompt-cache-disk`
store, and a salted request never reads a state from it. A salted request also neither drafts from nor adds to the
`--recall` record, neither reads nor writes `--compact-memory` notes, and a salted `/v1/imprint` ask reports and loads
only its salt's states. The server's own requests for a salted conversation (the next turn's background prefill,
compaction's notes, Cortheon's research planner and the server rounds it keeps) stay under the same salt. Engines on the
shared lane host with its prompt cache report `capabilities.cache_salt` and take salts. Any other engine (one whose own
host keeps states by tokens alone, a family's own paged cache under `TF_KV_SHARE`, or no prompt cache) returns HTTP 400
for a salted request, naming its backend, rather than accept a salt it cannot keep apart, and does not start with
`--cache-salt-per-key`. `TENSORFOLD_SEED_SALT` is unrelated: it only seeds sampling.

## Conversations on disk

Use `--prompt-cache-disk DIR` to keep conversation prompt states on disk:

```sh
tensorfold serve MODEL --prompt-cache-disk ./prompt-cache
```

This is equivalent to `--learn-conversations --learn-dir DIR`; both legacy flags remain supported.
`--learn-conversations` alone keeps its existing default directory, `~/.cache/tensorfold/learned`.
Both spellings imply `--learn`. Do not combine `--prompt-cache-disk` with `--learn-dir` or
`--learn-conversations`: the server refuses either combination, in either order, even if the directories match.
Repeating the same directory flag keeps its usual last-value behavior.
The startup line names the disk tier's directory and the device budget reported by the engine; when the engine
does not report a device budget, the line says so. `--prompt-cache-gib` still controls the in-memory budget,
and the existing disk limits described below apply to either spelling.

`--prompt-cache-disk DIR` writes every prompt state the server keeps in RAM to disk as well: a
conversation's history, the end of each reply, the next turn's background prefill and the shared prefixes, all in one
store for each model and build. That store takes the place of `--learn`'s store of shared prefixes: a server started
with the flag neither reads nor writes the prefix files a plain `--learn` server keeps, and they stay on disk until the
cap below needs their room. After a restart, a crash or an eviction from RAM, the next turn reads the latest state back
instead of reading the conversation again from its first token. Qwen3.8-27B, Nemotron, Flash Next and GLM-5.3-Flash on
one Mac, and Nemotron, the Qwen3.5 hybrids, Flash Next and GLM-5.3-Flash on CUDA keep them; other engines refuse the
flag at startup. GLM-5.3-Flash on CUDA across ranks writes every rank's part to rank 0's disk, so a state is read back
on every rank or on none; it takes neither `--spill-gib` nor `--checkpoint-slots`.

The engine stages an owned copy between rounds, and an I/O thread writes and fsyncs it. Staging has a 1 GiB byte permit
and at most eight jobs. Qwen3.8-27B widens the permit to one whole state at its `--context`, with none of its rows on
disk yet, but never past `--prompt-cache-gib`: a conversation whose first save or post-compaction prompt holds more
than about 13K new tokens is still written. A save is charged only its new bytes, the rows not already on disk.
A save that finds either full but would fit alone waits in the queue until a write finishes; one
that cannot fit even then is skipped. So is a waiting save whose state leaves RAM, unless `--spill-gib` copies it then
(below); that copy, like any made at an eviction, is skipped when the permit is full. A block still being written is
charged and copied once, however many saves name it. The copy waits for its GPU producers before RAM can be reused. A
state becomes available for reuse only after all its files have been written and synced.
A planned stop (SIGTERM or SIGINT) drains writes for up to 20 seconds, then cancels the remainder; a blocked syscall
can finish later using only its private temporary files and staged bytes. A killed server loses unfinished saves.
Rows a stored state already holds for the same tokens are written once, so ten
conversations that share a long system prompt and tools store those rows once. A state is read back only when reading
it takes less time than prefilling the tokens it covers, measured on this machine. Every piece read is checked against
its hash; a damaged file is deleted, with the states that need it, and the turn prefills instead. A read-back state
gives the same reply, token for token, as the state it was written from and as a fresh prompt pass.

If startup runs out of memory while reading state metadata or planning its blocks, opening the store fails and
valid state files remain on disk for a later retry. An allocation failure does not count as a damaged file.

While other streams run, a queued request whose state is on disk has it read on an I/O thread into memory, with the
same checks, and waits in the queue meanwhile; the engine thread then only copies it into place, so the other streams
keep decoding through the read. A read that fails, or is not done within `TENSORFOLD_RESTORE_PREFETCH_MS` (30000 by
default), is left and the state is read at admission as before. `TENSORFOLD_RESTORE_PREFETCH` caps the reads held at
once (2 by default; 0 turns reading ahead off) and `TENSORFOLD_RESTORE_PREFETCH_GIB` their memory (16 by default); a
request alone on the engine reads its state at admission as before. A queued request matching a staged save waits
within the same timeout for the writer to publish it, then reads it ahead; a failed or late save falls back to the
ordinary restore or fresh prefill. Read and write workers are separate, so a slow read does not block saves. Cancelled
reads retain their memory permit until I/O returns, and close cancels reads within the same stop deadline as writes.

States are kept per checkpoint, kernel build, OS build and chip: a macOS update or a new release that computes other
bits starts a new set, and the old set is the first to go when the disk cap needs room. `--learn-gib` caps the disk
they take with `--learn`'s shared prefixes, every model together (32 GiB by default); past it, other models' and
builds' sets go first, then the states worth least: the fewest tokens saved per byte only they hold, the least recently
used. `--learn-min-free-gib` keeps free disk (4 GiB by default), and `--learn-daily-gib` caps what they write a day
(32 GiB by default); past either, a state stays in RAM only, with a log line, and no reply changes.

On a Mac, `--spill-gib N` writes a state to that store when the prompt cache evicts it from RAM or the engine closes,
up to N GiB under `~/.cache/tensorfold/spilled` for every model together, and reads it back as above: a later request whose prefix
matches resumes from it, token for token as a fresh prompt pass, after a restart too. An eviction stages a copy before
releasing RAM, then the engine continues while the I/O thread writes it. A full staging permit skips that save;
the request prefills if it returns before a saved state is available. This changes cache hits, never reply tokens.
GLM-5.3-Flash also stages each resident state when it is kept, with or without `--spill-gib`, because its slots and
shared pool can reuse those rows before an eviction. A busy permit waits and retries; an oversized image streams
in bounded pieces. The log names the staging reason; other admission or copy failures still skip that save.
An idle unload and a planned stop also save the states still in RAM, newest first, within the same disk cap and
20-second stop budget. The next engine open scans those files and restores them through the same path as evictions.
Older queued saves cannot evict states saved earlier in that close. States that exceed the disk cap, daily budget
or free-disk floor are skipped; ordinary eviction saves also skip images that exceed their staging permit.
Unfinished writes are cancelled at the deadline.
Past N GiB other models' spilled sets go first, then the state this server used least recently. A damaged file is
deleted and its tokens are prefilled, and a write cut off midway leaves nothing a later turn or a restart reads.
`--checkpoint-slots N` sets how many states stay in RAM beside the store, so the next one sends the oldest to disk; it
needs `--spill-gib` or `--learn-conversations`. With `--learn-conversations` too, every state is still written
behind, a staged write survives eviction of its RAM entry, and the cap is the smaller of `--learn-gib` and
`--spill-gib`. `--learn-min-free-gib` and `--learn-daily-gib` apply to spilled states as well. The same four Mac
families take both flags, and so do Nemotron, the Qwen hybrids, Flash Next, Gemma 4, Dory and Spark2.5 on CUDA; the
others refuse them at startup. Without either flag nothing changes, and the routes below stay off.
`TENSORFOLD_ROWS_SYNC=1` selects the previous synchronous writer for comparisons; its block and state files have the
same bytes. That comparison writer retains blocking I/O; the bounded disk wait uses the threaded writer.
`TF_IMPRINT_VERIFY=1` reads threaded writes back against their owned staging before publication.

On CUDA, Nemotron, the Qwen3.5 hybrids, Flash Next, Gemma 4, Dory and Spark2.5 take `--spill-gib` and
`--checkpoint-slots` with the same store and cap as a Mac. At an eviction the engine thread stages a copy of the state
into host memory, waiting for the GPU work that produced it, then releases its RAM entry and goes on; the I/O thread
writes and fsyncs the copy while the engine decodes. Staging holds at most 1 GiB and eight jobs; for Gemma 4, Dory and
Spark2.5 it holds one whole state at the served window, within the RAM cache's budget. A state whose new bytes (rows
neither on disk nor being written) do not fit what is free is not saved, and a later request for it prefills. When the
engine closes, at a planned stop (SIGTERM or SIGINT) or an idle unload, it saves the states still in RAM, newest first,
within the same disk cap and 20-second stop budget. While other streams run, a queued request whose state is on disk has
it read ahead on an I/O thread, as described above.

Three routes let a client such as the Mac app ask about a conversation. Each takes the body of a chat completion
(`messages`, `tools`, `chat_template_kwargs`), renders it as that request would be rendered, and answers on the engine's
thread between rounds. A server started without `--prompt-cache-disk DIR` or `--learn-conversations` answers each
with 404 and the error message
"saved conversations are off: start the server with --prompt-cache-disk DIR (or --learn-conversations)";
an engine that keeps no prompt states answers 501.

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
  and the order the stores were last opened in). States in RAM stay until they are evicted, or until a
  [flush](#flush-the-prompt-cache).

What stays on disk: under the `--learn-dir` folder (`~/.cache/tensorfold/learned` by default), directories readable by
their owner only (0700) hold files readable by their owner only (0600). They contain token ids and the model's own
prompt states (attention keys and values, recurrent and convolution states), never text, and nothing in them leaves the
machine. Token ids decode back to the conversation's text with the model's tokenizer, so treat the folder as you treat
the conversations. `DELETE /v1/imprint` clears one model's states; deleting the folder while no server runs clears all.

## Health

The native server answers `GET /health` (and `GET /`) from one handler on Metal and CUDA, without a key. With keys
configured the body is only `{"status": "ok"}`. An engine that reports unhealthy answers 503 with
`{"status": "unavailable"}`. Without keys it has these fields, in this order:

| Field | Type | Meaning |
| --- | --- | --- |
| `status` | string | Always `"ok"` |
| `model` | string | The served name (`--name`) |
| `model_ids` | list of strings | The served name, then each `--alias`, as `/v1/models` lists them |
| `max_batch_size` | integer | The lanes the engine serves at once |
| `policy` | string | The engine's policy line; absent when the engine reports none |
| `warming` | boolean | Always `false` on the native engines |
| `memory` | object | `active`, `cache` and `peak` in bytes (`cache` is 0), plus `limit` when the engine reports one: CUDA's allocator counts; on Metal the process footprint, from Nemotron, Qwen3.5-2B, Qwen3.8-27B, Qwen3.5-MoE, Flash Next, GLM-5.3-Flash and GLM-5.3 (EXLR8); Flash Next and both GLMs add `limit`, their load limit. `{}` from the other Metal families |
| `live` | object | `connections`, `waiting`, `decode_tokens_per_second`, `prefill_tokens_per_second` |
| `version` | string | The release this server was built from |
| `backend` | string | The backend built in: `metal`, `cuda`, `hip` or `xpu`; absent from a build without one |
| `device` | string | The GPU the server opened, by the driver's name; absent when it cannot read one |

`live.connections` counts running and waiting requests, and `live.waiting` the waiting ones.
`decode_tokens_per_second` is the tokens decoded in the last 2 s, divided by 2. `prefill_tokens_per_second` is the
last prompt pass's uncached tokens over its time, shown for 2 s after the pass ends and 0 otherwise. Both rates are
rounded to one decimal. Public health never resets the peak, with or without keys. Reset queries on `/health` and `/`
are ignored. The separate `/memory` route also reports memory without resetting it and requires a key when keys are configured.

`GET /v1/models` (and `/models`) answers `object: "list"` with one `data` entry per served ID and alias: `id`,
`object: "model"`, `created` (the current unix time), `owned_by: "tensorfold"`, and, when the engine reports a
context window, `context_length` and `max_model_len`, both that window. A model the engine reports as reasoning
adds `reasoning`: `default_enabled`, `default_effort` when the template consumes `reasoning_effort`, and
`supported_efforts`, the template's levels plus `none` when the template has a thinking switch.

Every native entry ends with `input_types`, the content part types the model takes: `["text"]`, plus `"image"` when
it was started with `--vision` (`image_url` parts), plus `"video"` when its checkpoint reads video and this server
loads the `libtfvideo` runtime (`video_url` parts), plus `"audio"` when the engine has an audio encoder
(`input_audio` and `audio_url` parts), for example `["text", "image", "video"]`. The server decides it
once at startup, and requests are refused by the same fact: a `video_url` part to an entry without `"video"` gets
HTTP 400 (`invalid_request_error`), never a 500. Without `libtfvideo` the startup log says so once.

The Python 0.6.6 server's body differs: its MLX `memory` can add `budget`, `mlx_budget` and `footprint`, and its CUDA
server sends the counters described under [Responses](#responses). For memory figures on a Mac, use
[`/memory`](memory-diagnostics.md) (it needs a key when keys are configured) or `/metrics`'
`tensorfold:process_footprint_bytes`.

## Execution witness

`serve --execution-witness` (off by default) adds `GET /v1/execution/witness`, a read-scoped snapshot of what serves
replies. Without the flag the route answers 404, nothing is hashed and replies are unchanged. The flag needs an API key
(`--api-key`, `--api-key-file` or `TENSORFOLD_API_KEY`), `--unix-socket`, whose owner-only socket stands for one,
or `--listen-fd`, which serves only the process that started the server; without one the server refuses to start.
Hashing starts after the server is ready, on a thread of its own, and a read never hashes: it copies the latest
snapshot. A stop ends a hash in progress at its next block.

A known snapshot is `{"schema": 1, "state": "known", "witness", "content", "instance", "epoch", "path", "components"}`:

| Field | Meaning |
| --- | --- |
| `content` | SHA-256 over `path` and every component; the same files, build and settings give the same value after a restart |
| `witness` | SHA-256 over `content`, `instance` and `epoch`: a restart or an engine reload always gives a new one |
| `instance` | 128 random bits drawn when this process started |
| `epoch` | The engine's generation: 1 at startup, one more each time `--idle-unload` reloads it |
| `path` | The engine's label for the supported path it attests |
| `components.engine` | The engine's own digest of what it binds for that path |
| `components.binary` | The running executable's SHA-256, read from the file the server opened before loading the model (`/proc/self/exe` on Linux, the executable's path elsewhere) |
| `components.text` | The checkpoint's tokenizer, template, special-token and EOS files (or their absence) and `--chat-template`'s file, whose stamps the server took before loading them |
| `components.preparation` | The serve defaults a request starts from: sampling, `--instruct-sampling`, max tokens, thinking, loop guard, seed salt, drafts, and the loaded template and EOS ids |
| `components.environment` | Every `TF_*` and `TENSORFOLD_*` variable the process started with, except names ending in `_API_KEY`, `_SECRET`, `_PASSWORD`, `_TOKEN`, `_SIGNING_KEY`, `_SECRET_FILE` or `_KEY_FILE` (such as `TENSORFOLD_API_KEY`, `TENSORFOLD_TOOL_ID_KEY_FILE`, `TF_RUNTIME_SIGNING_KEY` and `TF_STAGE_SECRET_FILE`); other names, such as `TF_SLIDE_POCKET_KEY`, are hashed |

Under `serve --confidential` a known or unknown snapshot, and each reply's binding, also carries `"confidential": true`,
and `components.preparation` binds it, so `content` and `witness` differ from the same server without the flag; without
it the field is absent and every digest is as before.
A reply's binding under the flag leaves out `request`, so no stop string, tool text or other resolved value of the
caller's is shown; `request_sha256` still binds it, for a caller that holds the request to recompute.

Otherwise the body is `{"schema": 1, "state": "unknown", "reason", "instance", "epoch"}`, and inference is unaffected.
`reason` is `pending` (still hashing), `unsupported_path` (the engine attests no supported path: every engine but the
ones below), `feature_excluded` (with `feature` naming the flag: `--slide`, `--learn`, `--learn-conversations`,
`--spill-gib`, `--compact-at`, `--compact-memory`, `--cortheon`, `--recall`, `--drafter`, `--mtp-drafts`, `--record`,
`--vision`, `--tp` above 1, `--speed-up`, `--kv-cache` other than bf16, `--prompt-precision` other than bf16 or
`--rope-yarn`), `parked` or `reloading` (an idle-unloaded engine), `files_changed` (a
text file or the executable, or a checkpoint file the engine stamped, changed after it was opened or stamped) or
`hash_failed` (a file could not be read; when the binary cannot be opened or the text files cannot be stamped at
startup, a startup line says why). The engine's facts and the epoch are read under the lock every reload takes, so one
snapshot never mixes two engines. The snapshot alone does not qualify a response cache.

The supported paths so far are `qwen3_5-2b/cuda/tp1/mlx-q4g64/text`, `qwen3_5-2b/metal/tp1/mlx-q4g64/text`,
`qwen3_5-9b/metal/tp1/mlx-q4g64/text` and `qwen3_5-27b/metal/tp1/mlx-q4g64/text`. The first is native Qwen3.5-2B
on a CUDA build at one rank,
every module MLX affine 4-bit, text only, with none of the excluded flags. Any other checkpoint, format, backend or
rank reads `unsupported_path`. Before the weights load, the engine stamps `config.json`, the index, every shard the
index names and a captured cubin set's `aot.json` and each cubin it names; a file it cannot stamp makes the witness
read `hash_failed` while the model still serves. Once the server is ready it reads every shard byte in full on one
thread, with no digest cache, and reads `files_changed` if any stamp moved between the load and the end of the read, or if a Hugging Face snapshot's shard
differs from its blob name. `components.engine` is a SHA-256 over the path, the checkpoint's structure (canonical
`config.json`, the index, shard names, sizes and headers), every shard's bytes, and the components a kept state's key
binds (the embedded kernel images, a captured cubin set, the GPU's name, compute capability and driver version, the
family's resolved facts and the resolved profile values), then the GPU's SM count, the context window, the stream
count and the staggered segments. An `--idle-unload` reload opens a new engine, which stamps again before its load and
reads every byte again once it is published. A server whose witness reads `feature_excluded` never reads the weights.

The Metal path is native Qwen3.5-2B on a Mac, under the same conditions and the same stamps, read and reasons. Its
kernels are Metal source in the binary that the OS compiles at load, so in place of kernel images and the GPU's facts
its `components.engine` binds the SHA-256 of every library the engine compiles (its source, the functions it takes and
a projection's tile), the compile options for this OS (math mode, math functions and language version), the OS build
and the chip as kept Metal states key them (the OS's compiler builds the GPU code), whether it has tensor units and the
resolved profile values, then the context window, the stream count, the prompt step, the lane rows the load's exactness
check left and whether the streams' caches are paged.

The 9B and 27B paths are the Qwen3.8-27B-class engine on a Mac (dense `qwen3_5` with its own output head), every
module MLX affine 4-bit in groups of 64 including the embedding and head, text only, under the same conditions,
stamps, read and reasons as the 2B. A SenseNova-Loop checkpoint, a tied head, another geometry such as `qwen3_5_moe`
(Qwen3.5-122B-A10B, Qwen3.6-35B-A3B), the Neural Engine prompt share (`TF_ANE_COLUMNS`) or idle weight release
(`TF_Q27_IDLE_RELEASE`) reads `unsupported_path`. This engine builds its
Metal libraries from source text it specializes at load, so its `components.engine` binds the SHA-256 of every library
the load builds, each with its compile options, as one digest that does not depend on build order. Only the loading
thread's builds count: a build on another thread during the load, a metallib read from a file, or a second load
watched at the same time takes the engine off the path. A specialization
built later on demand is made from source text in the binary (`components.binary`). It also binds the OS build, the
chip, whether it has tensor units and the resolved profile values, then the context window, the stream count, a
pass's prompt rows, the verify window, whether K/V is paged, the K/V growth step, whether lane copies are on and
whether the lone path serves a single request.

### Reply binding

A chat completion or completion with `"execution_witness": true` carries a top-level `execution_witness` object on the
whole reply, or on a stream's finish chunk (not on the separate usage chunk). It is read just before the engine takes
the request and again once the reply ends; the request keeps the engine from parking in between. The witness is
read-scoped, so a scoped key needs `read` beside `inference` to ask (else 403 `insufficient_scope`); a full or `[admin]`
key may always ask. Only these two routes carry the binding: `/v1/responses`, `/v1/messages` (and its `count_tokens`),
`/api/chat` and `/api/generate` rebuild the request and answer a true `execution_witness` with HTTP 400.

| Field | Meaning |
| --- | --- |
| `state` | `known` when the witness was known at both reads, with the same `epoch` and `witness`; else `unknown` |
| `binding` | Known only: SHA-256 over `witness`, `prompt_sha256`, `request_sha256`, `token_sha256` and `finish_reason` |
| `witness`, `path` | Known only: the snapshot's, as `GET /v1/execution/witness` gives them |
| `instance`, `epoch` | The process and the engine generation at the end |
| `reason`, `feature`, `admitted_epoch` | Unknown only: the snapshot's reason, or `epoch_changed`, `witness_changed` or `unbound` (below); the epoch at admission |
| `prompt_sha256` | The prompt's token ids after the template, hashed as `tensorfold.token_sha256` hashes the reply's |
| `token_sha256` | The reply's token ids, equal to `tensorfold.token_sha256` |
| `finish_reason` | The reply's |
| `request` | The resolved request the engine read: `max_tokens` after the window's clamp; `sampling` (null when greedy, else the `seed` actually used, `temperature`, `top_k`, `top_p` and `min_p`); `presence_penalty`, `frequency_penalty`; `eos` and `stop`; `drafts`; `think_budget`, `think_close`, `think_end`, `answer_max`; `loop_guard`, `loop_stop`; `call` (a required tool call's gate); `structure` (the grammar's `kind`, `text_sha256`, `after`, `budget`); `logprobs`, `prompt_logprobs`, `decide` and `untaught` |
| `request_sha256` | SHA-256 over `request`, walked in order as below |

Every field is length-prefixed: its byte length as a little-endian u64, then its bytes. `request_sha256` hashes the field
`tensorfold resolved request v1`, then the value: `n` for null; `b` then `1` or `0`; `i` then the integer's decimal
text; `f` then the float's IEEE-754 bits as a little-endian u64 (every float in the reply round-trips); `s` then the
string; `a` then the count as a little-endian u64, then each item; `o` then the count, then each key as a field and its
value. `binding` hashes the field `tensorfold execution binding v1`, then `witness`, `prompt_sha256`, `request_sha256`,
`token_sha256` and `finish_reason`, each as its text in the reply. A reply that took a path that binds nothing (with
`--compact-at` or `--cortheon`, which the witness already excludes) carries only `schema`, `state: "unknown"`, `reason`
(and `feature`), `instance` and `epoch`, with the snapshot's reason. `reason` is `unbound` if that snapshot reads known;
no route does that today, since each path that binds nothing runs only under a flag the witness excludes, so `unbound`
would mean a reply path that skipped the binding by mistake. No response cache reads the binding yet.

## Cancel a request

`POST /v1/requests/{id}/cancel` (or `/requests/{id}/cancel`) cancels a running or queued request by the id its
reply carries: a chat completion's `chatcmpl-`, a completion's `cmpl-`, or a Responses `resp_` id. The engine takes
the same cancel a client disconnect sends, and every other stream's tokens are unchanged. The reply ends well formed
with the tokens it had (none when it was queued): `finish_reason: "cancelled"`, or a Responses `status: "cancelled"`
in its `response.incomplete` event. A Cortheon turn runs no further rounds. The answer is 200
`{"id": ..., "object": "request.cancel", "cancelled": true}`, or 404 with the code `request_not_found` when no
request holds the id: it was written out, or never existed. An id is held from the request's start until its reply is
written, so a cancel that arrives as the reply ends leaves that ending as it was. The body, if any, is ignored.

With API keys it is an admin route: a full or `[admin]` key may cancel from any address, a scoped key gets 403
`insufficient_scope` (see [Key scopes](#key-scopes)) and a missing key 401. A server without keys answers only a
caller whose socket is on the loopback address (127.0.0.0/8 or ::1) and refuses everyone else with 403
`loopback_only`, so an open server never lets one client cancel another's request. The socket's own address decides;
a forwarded address is never read. Behind a reverse proxy every caller's socket is the proxy's, so a keyless server
also refuses a request that carries `Forwarded`, `X-Forwarded-For`, `X-Forwarded-Host`, `X-Real-IP`, `Via`,
`CF-Connecting-IP` or `True-Client-IP`. A proxy that adds none of them cannot be told apart from a local caller: give
a proxied server keys.

## Flush the prompt cache

`POST /v1/prompt_cache/flush` drops every prompt state the server keeps for resuming conversations: those in RAM, and
those on disk when it keeps them there (`--prompt-cache-disk DIR`, `--learn-conversations` or `--spill-gib`). The JSON
body may contain `"retain": N`, a nonnegative 32-bit integer, to retain the N most recently used complete RAM states by
the cache's existing clock and their matching disk states instead. Other states and their disk bytes are dropped; `retain`
is zero by default, so an absent, empty or unrecognized body is still a full flush. It waits for the requests already
running to end, so their replies are unchanged, and requests that arrive meanwhile wait in the queue and start after it.
Then it answers 200 with `{"object": "prompt_cache.flush", "states": ..., "ram_bytes": ..., "disk_bytes": ..., "disk":
..., "waited_seconds": ...}`: the states dropped from RAM and the bytes they held in the cache's budget, the bytes deleted
from the server's own state folder, whether this server keeps states on disk, and how long the flush waited. After a full
flush the next turn prefills its whole prompt and gives the tokens it would have given resumed. Two flushes asked together
run one after the other. States `--learn` keeps for shared prompt prefixes stay on disk, as they do after
`DELETE /v1/imprint`, so with `--learn` a later prompt may still resume from one.

An engine without the shared prompt cache (its cache turned off, or a backend that keeps its states its own way)
answers 501 `not_supported` at once and holds nothing back. If the running requests have not ended within 10 minutes,
the flush is withdrawn: nothing is dropped, new requests start again, and the answer is 503 `timeout`. While a flush
waits, `--idle-unload` does not unload the model; a model it has already unloaded is not loaded for a flush, which
answers 503 `model_unloaded` (no state is in RAM then, and the states it saved to disk stay until a flush after the next
request loads it).

Who may flush is decided as for [cancelling a request](#cancel-a-request). With API keys it is an admin route: a full
or `[admin]` key may flush from any address, a scoped key gets 403 `insufficient_scope` and a missing key 401. Without
keys only a caller whose socket is on the loopback address (127.0.0.0/8 or ::1) may flush, and only when the request
carries none of the proxy headers listed there; anyone else gets 403 `loopback_only`.

## The Responses API

`POST /v1/responses` takes OpenAI's Responses request and runs it as the equivalent chat completion, through the
same handler and engine path. A response has that chat completion's prompt and tokens (the `tensorfold` block's
`token_sha` matches), drafts, and equals its `"draft": false` run and its solo run.

- `input` is a string or a list of items: messages (`input_text`, `output_text`, `text` and `refusal` parts, whose
  text or refusal text becomes input text; `input_image` on a server started with `--vision` for a model that reads
  images; `input_file` with inline PDFs on a server started with `--vision --vision-pdf`), `function_call`,
  `function_call_output` (its `output` text, or `input_text`, `input_image` and `input_file` parts under the same
  image and PDF rules), and `reasoning` items with their `content` text, which the template gets back as the next
  assistant message's `reasoning_content`. `instructions` becomes the system message and is not carried to a later
  turn.
- `tools` takes function tools; `tool_choice` takes `none`, `auto`, `required`, a function or `allowed_tools`;
  `parallel_tool_calls` works as in chat. `namespace` groups are accepted too, one level deep: their function children
  reach the engine flattened as `<namespace>__<tool>`, and their `function_call` items carry the child's `name` plus
  the group as `namespace`. A child's own name cannot start with its group's `<namespace>__` prefix.
- `max_output_tokens` counts reasoning tokens too. `temperature`, `top_p`, `top_k`, `min_p` and `seed` sample as in
  chat.
- `reasoning.effort` follows the chat effort rule: unset is the template's default and `none` turns thinking off.
  `reasoning.summary` is accepted, but no summary is written.
- `text.format` takes `json_schema` or `json_object` through the structured-output path, with drafts.
- `stream`, `store` (default true) and `metadata` (up to 16 string pairs, keys up to 64 characters,
  values up to 512). The chat extensions `draft`, `thinking_budget`, `stop`, `ignore_eos`,
  `chat_template_kwargs`, `priority`, `return_token_ids` and
  `grammar_budget` pass through.

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
response. Under `serve --confidential` every key's create must send `store: false` and no `previous_response_id`, or
gets 403 with code `confidential`, so nothing is kept for a later read and `background` is refused.

HTTP 400 refuses what this server does not run: built-in tools (web search, file search, computer use and the
others), `conversation`, `prompt` templates, `context_management`, `truncation: "auto"`, `execution_witness`, file IDs and
remote file URLs (`input_file` parts with inline `file_data` run on a model started with `--vision --vision-pdf`),
`item_reference` items, encrypted reasoning items, and a `previous_response_id` that is not stored.
`include` takes a list of OpenAI's names. `message.output_text.logprobs` asks for logprobs where the engine has them
(see the chat section's logprobs); the server has nothing to add for the others:
`code_interpreter_call.outputs`, `computer_call_output.output.image_url`, `file_search_call.results`,
`message.input_image.image_url`, `message.output_text.logprobs`, `reasoning.encrypted_content`,
`web_search_call.action.sources` and `web_search_call.results`. Any other name, or a value that is not a list of
strings, is refused.

### Background responses

`background: true` answers the POST at once with the response object, `status: "queued"`, `background: true` and no
output; the reply is generated after the POST returns, and leaving the connection does not cancel it.
`GET /v1/responses/{id}` polls it: `queued`, then `in_progress` (output stays empty while it runs), then the
finished response as a request sent without `background` returns it: `completed`, `incomplete`, `failed` or
`cancelled`. The reply runs through the same chat path as that request, so the same body and seed give the same
tokens. A finished background response is an ordinary stored response: `previous_response_id` continues it (one still
running is refused), and it is kept, evicted and lost on restart as every stored response is.

`POST /v1/responses/{id}/cancel` (or `/responses/{id}/cancel`) cancels a queued or running one. A running reply ends
as [Cancel a request](#cancel-a-request) ends it, with the text it had; the route waits up to 2 s for that end and
answers 200 with the response, `status: "cancelled"`. `POST /v1/requests/{id}/cancel` with the same `resp_` id does
the same. A second cancel answers the cancelled response again, a background response that has already ended
otherwise answers 400, as does a response not created with `background`, and an unknown id 404.
`DELETE /v1/responses/{id}` on a running one stops it, and nothing of it is kept.

`background: true` needs `store: true` (the default) and refuses `stream: true`: there is no event stream to start
or resume (a GET of a background response with `stream=true`, as `starting_after` resumes one, answers 400), so poll
instead. It is an HTTP request only; a Responses socket refuses it, and a true value that is not a boolean, such as
`1`, is refused. At most 64 background responses are queued or running at once on a server, 16 for one API key,
holding 256 MiB of copied requests in all; past those a create answers 429 `background_limit_exceeded` before any
work starts. Each keeps its request's place under `--max-inflight` until it is stored, and a stop gives it the drain
window any request in progress gets.

Stored responses are the server's, so with API keys only a full or `[admin]` key may create, poll or cancel one; a
scoped key gets 403 (see [Key scopes](#key-scopes)). A server without keys takes the response's id, 128 random bits,
as the right to read, cancel or delete it, as it does for `GET` and `DELETE`.

### WebSocket mode

`GET /v1/responses` (or `/responses`) with an RFC 6455 upgrade opens a Responses socket, as OpenAI's WebSocket mode and
the SDKs' `client.responses.connect()` use it. The socket takes plain `ws://`, with no subprotocol and no extension; a
TLS proxy in front of the server gives `wss://`. A bad upgrade answers 400, a version other than 13 answers 426, and a
plain `GET /v1/responses` stays a 404.

The client sends text frames, each one JSON event:

- `response.create`: the `POST /v1/responses` body plus `stream_id` and `generate`. The response streams over the
  socket whatever `stream` says. It passes every gate a `POST` passes (drain, admission, load ledger, idle reload,
  model-kind refusals, key scopes), one create at a time, and is logged and counted as one request.
- `response.steer`: answered with `response.steer.failed`, code `steering_not_supported`.
- Any other event, or a frame that is not a JSON object: an `error` event with code `invalid_event`; the socket stays
  open.

Each response sends the events of the SSE stream, in the same order, one text frame each, with the same prompt, tokens
and `token_sha` as over HTTP. A create that names a `stream_id` gets it on each of its events. A create refused before
its stream opens (the request, its history, its key scope or admission) gets one event, `{"type": "error", "status":
..., "stream_id": ..., "error": {"type", "code", "message", "param"}}`, with the HTTP status the same request gets over
`POST`; the socket stays open. A failure after the stream opens, such as a prompt over the context window, ends with
`response.failed`, as over SSE.

`stream_id` names a lane: 1 to 256 letters, digits, `_`, `.` or `-` (else `invalid_stream_id`), up to 32 named lanes
per socket (then `websocket_stream_limit_reached`). Creates run one at a time in arrival order, so every lane is in
order; lanes do not run concurrently.

The socket keeps the latest finished response of each lane, with its whole conversation, so `previous_response_id`
continues with `store: false`. An id resolves from that cache in any lane, then, with `store: true` only, from the
server's stored responses; else it is `previous_response_not_found`. The cache holds up to 64 MiB per socket and
512 MiB across sockets; a response that does not fit still completes, uncached. A failed continuation on its parent's
own lane drops the parent from the cache. A response stored from a cached parent is stored with its whole
conversation, so it continues after the socket closes.

`generate: false` checks and renders the request as a run would, then sends `response.created`,
`response.in_progress` and `response.completed` with no output, zero output tokens and the prompt's input tokens,
without an engine run. Its id continues like any other response.

A key with the `inference` scope may open the socket; each create checks the key again. A removed key gets an `error`
event with status 401 and close 1008, and changed scopes apply from the next create. A scoped key, and any key under
`--confidential`, must send `store: false`, and its `previous_response_id` must be in the socket's cache.

| Event | The server |
| --- | --- |
| 60 minutes since the upgrade | Sends `websocket_connection_limit_reached` after a running response and closes with 1000 |
| No message for the HTTP idle timeout | Closes with 1001 |
| The server drains | Sends an `error` with status 503 after a running response and closes with 1012 |
| A client close, or a dropped socket | Stops the running response |
| An unmasked frame, a binary message, invalid UTF-8, a message over 32 MiB | Closes with 1002, 1003, 1007, 1009 |
| A message past its read deadline, or more than 16 messages or 32 MiB waiting | Closes with 1008 |

The server answers ping with pong and sends a ping when a response is silent for the SSE keepalive interval.

## Anthropic Messages

The Messages routes reuse the same chat handler and engine on MLX and CUDA. They accept `system`, text/image
blocks (including mid-conversation system text), `tool_use`/`tool_result`, custom tools and `tool_choice`, sampling, `stop_sequences`, `thinking`
(disabled, enabled with `budget_tokens`, or adaptive), and `output_config` effort/JSON schema. Image support
requires a vision-capable model served with `--vision` (natively GLM-5.3-Flash and Qwen3.8-27B on a Mac); other
models answer image blocks with HTTP 400. Thinking is off unless enabled or adaptive. A request without `thinking`
renders thinking off as before and is never refused for it. Only a `thinking: {"type": "disabled"}` the client sends
is an explicit off, which returns HTTP 400 when the template cannot honor it (see [Reasoning](#reasoning));
`/v1/messages/count_tokens` gives the same refusal for the same body, and otherwise counts the prompt
`/v1/messages` submits. Thinking round-trips as plaintext with an empty signature;
`display` does not suppress it. Claude Code's `context_management` keep-all thinking directive is accepted. Mid-conversation system text stays
in place with its system role; the model's chat template must support later system messages. Turn-scoped
system messages, per-message output configuration and inline tool changes are unsupported, and a true
`execution_witness` answers HTTP 400 (see [Reply binding](#reply-binding)).
JSON schema output, including Claude Code title requests, requires `pip install 'tensorfold[grammar]'`.
That install is the Python 0.6.6 engine's; the native server uses `libtfgrammar` from the release archive instead
([structured output](#structured-output)).

Usage separates uncached `input_tokens` from `cache_read_input_tokens` and reports `output_tokens_details.thinking_tokens`
when the backend counts them. Prefix caching remains automatic,
so `cache_control` hints do not allocate an Anthropic cache or report cache-creation tokens. Errors use the
Anthropic error envelope, including after an SSE stream opens. Server-side tools, file IDs,
redacted thinking and other context edits return HTTP 400. A `document` block takes an inline base64 or
data-URL PDF on a model started with `--vision --vision-pdf`, like image blocks on a vision model.

Connect Claude Code using the ID from `/v1/models`:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=local claude --model local-model
```

The [Claude Code guide](claude-code.md) covers the model ids it asks for and how to keep its prompt prefix cacheable.

## Launch a coding agent

`tensorfold launch <agent> [args...]` checks that a local server answers at `http://127.0.0.1:8080/v1`,
names the model it found, then starts the agent with its API base URL and key set in the child process only.
No agent settings or files in the home folder are changed. Arguments after the agent name pass through
unchanged. Set `TENSORFOLD_LAUNCH_URL` to use another local server. Claude Code gets the server's root
(`http://127.0.0.1:8080`), since it adds `/v1/messages` itself; the others get the `/v1` URL. The key is the first of
`TENSORFOLD_API_KEY`'s keys when it is set, for a server started with keys, and `local` otherwise. An unknown agent
is refused with the list of known names.

| agent | base URL variable | key variable |
|---|---|---|
| `claude` | `ANTHROPIC_BASE_URL` | `ANTHROPIC_API_KEY` |
| `codex` | `OPENAI_BASE_URL` | `OPENAI_API_KEY` |
| `opencode` | `OPENAI_BASE_URL` | `OPENAI_API_KEY` |
| `hermes` | `TENSORFOLD_BASE_URL` | `TENSORFOLD_API_KEY` |

## Ollama API

The native server also answers Ollama's `/api/*` routes. They translate onto the same chat and embedding seam as
OpenAI's routes and use the same bearer key; discovery works with a read key, generation needs an inference key.
Every path below is dispatched in `zig/src/server/routes.zig` and implemented in `zig/src/server/ollama.zig`.

### Discovery: GET `/api/tags`, GET `/api/ps`, GET `/api/version`, GET and POST `/api/show`

These replies are plain JSON, never streamed, and take no request body of consequence. `GET /api/tags` lists every
configured model as `{"name":ID,"model":ID}` under `models`, without the `modified_at` and `size` fields some
Ollama clients show. `GET /api/ps` reports one loaded model, the server's served name,
under `models`. `GET /api/version` returns `{"version":"..."}` from the server's identity. `/api/show` is reachable
by both methods; the handler reads and drops any request body (a typical `{"model":"..."}` is ignored) and returns
`{"modelfile":"","parameters":"","template":...,"details":{}}`, where `template` is the engine's chat-template
source. `/api/tags` lists what `GET /v1/models` lists; the other three have no OpenAI-style counterpart.

```bash
curl http://127.0.0.1:8080/api/tags
curl http://127.0.0.1:8080/api/show -d '{"model":"local-model"}'
curl http://127.0.0.1:8080/api/ps
curl http://127.0.0.1:8080/api/version
```

### POST `/api/chat` and POST `/api/generate`

Both bodies go through the same translation: `messages` (an array) or a `prompt` string (wrapped as one user message),
optional `system` (prepended as a system message), `format` (`"json"` or a JSON-schema object), `tools` plus optional
`tool_choice`, a boolean `stream`, and `options`. The client's `model` field is ignored: the served name is always
used. Supported options are `temperature`, `top_k`, `top_p`, `min_p`, `stop`, `seed`, `presence_penalty` and
`frequency_penalty`, passed through unchanged; `num_predict` becomes `max_tokens`; `num_ctx` is checked against the
model's context window and then dropped; `think` becomes `chat_template_kwargs.enable_thinking`. Unknown options,
`images`, `tool_choice` without `tools`, and non-boolean `stream`/`think` are refused with HTTP 400 naming the field.
The translated request is forced to carry `return_token_ids: true`.

The reply is newline-delimited JSON (`application/x-ndjson`), not SSE: each chunk is one object. `/api/chat` puts
text in `message: {"role":"assistant","content":...}`; `/api/generate` puts the same text in `response`. Every line
carries `model` and an RFC 3339 `created_at`. With `stream: false` there is a single line with `done: true` holding
the whole answer, `done_reason`, `prompt_eval_count`, `eval_count`, `prompt_eval_duration`, `eval_duration` and
`token_ids`. With `stream: true` there are `done: false` delta lines followed by one final `done: true` line whose
content is empty; a stream that fails after it opened ends in an `{"error":"..."}` line instead. Errors before the
200 head go out in OpenAI's `{"error":{...}}` envelope. Compared with `POST /v1/chat/completions`, the model output
is flattened from `choices[0].message.content` into `message.content` (or `response`), `created_at` replaces
`created`, token IDs move from `tensorfold.token_ids` to a top-level `token_ids`, and the framing changes from
SSE/JSON to NDJSON. Both routes occupy the chat-completions admission group and text kind.

```bash
curl http://127.0.0.1:8080/api/chat -d '{"model":"local-model","messages":[{"role":"user","content":"H"}],"stream":false,"options":{"temperature":0,"num_predict":8,"seed":1234}}'
curl http://127.0.0.1:8080/api/generate -d '{"model":"local-model","prompt":"H","stream":false}'
```

### POST `/api/embed`

The body is the OpenAI embeddings body (`input`, `encoding_format`, `dimensions`). The reply is flattened into
Ollama's shape: `model` plus an `embeddings` array of value arrays, with `prompt_eval_count` from usage when present.
There is no `object` field, and the per-embedding `object: "embedding"` wrappers of `POST /v1/embeddings` are gone.
Like `/v1/embeddings`, it is admitted in the embeddings group; unlike it, the reply is a single JSON object, never
streamed.

```bash
curl http://127.0.0.1:8080/api/embed -d '{"model":"local-model","input":"hello"}'
```

## Request journal and replay

`TENSORFOLD_REQUEST_JOURNAL=PATH` turns on a request journal; it is off unless set, and only `PATH` and its siblings are
written. **It holds every chat and completion prompt and reply in full**: treat it like the conversations themselves.
Image URLs and PDF data are replaced with `<redacted>`, as in `TENSORFOLD_REQUEST_LOG`, which is unchanged. No HTTP header
is journaled, so API keys and `Authorization` values never reach it. The server creates `PATH` and `PATH.lock`
owner-only (0600), refuses a symbolic link or a file others can read, and holds `PATH.lock` while it runs, so a second
server on the same journal does not start. Benchmark child servers never inherit the variable.

Each request writes three JSON lines: `start` before any work (its reply id, route, fingerprint, the client's seed and
the redacted body), `prepared` before the engine sees it (the seed and sampling the server resolved, the token limit,
thinking, drafts and the prompt's token count and SHA-256), and `finish` once the reply has gone (status, outcome, the
reply's size and SHA-256, its `token_sha256`, the reply bytes in base64 when they fit 4 MiB, and its measures:
`queue_ms` waiting for admission, `ttft_ms` from receipt to the first token, `decode_tps`, and `prompt_tokens`,
`cached_tokens` and `completion_tokens`). The measures are the ones the server's `done` log line reports, taken as it
already takes them; they are `null` for a request that ended before generating, and for a prompt list, whose prompts run
apart, the counts are summed and the timings are the last prompt's. Responses and Messages
requests are journaled in the chat form they are translated to. A raw completion whose `prompt` is a list is not
replay-exact: its per-prompt prepared records are not written, and its individual drawn seeds are not journaled.

Size is bounded: a file rotates at 64 MiB to `PATH.1`, older files move up, and `PATH.3` is the last kept, so the
journal holds at most 256 MiB. A body or reply too large for one 8 MiB record keeps its size and hash only. Each record
is one `write` with no sync, so the journal adds three small appends to a request and nothing to its tokens; a full
disk or a write error loses that record (the server logs it once), never the request. A record survives the server
process crashing, not a power loss.

On the next start the server reads the kept files. A request that started and never finished, with no clean stop after
it, was in flight at a crash: a `quarantine` line flags it, and the server never retries it. Its fingerprint is SHA-256
over the API key's label (as in the access log), the route and the whole body, so it matches only the same key sending
the same request. One crash only flags what was in flight, since a crash can come from elsewhere (another route, the
machine) and the requests beside it are not to blame. A request whose fingerprint was in flight at two crashes is
refused with a 400 `request_quarantined` before the engine sees it, which ends a client's retry loop after the second
crash. Other requests, and other keys sending the same body, are served as before; the check is one hash lookup. Without
API keys every client shares one label. A quarantine lapses 24 hours after its last crash, and a clean stop (SIGTERM,
Ctrl-C) quarantines nothing.

`tensorfold log` reads the journal in the terminal. It reads `PATH.3`, `PATH.2`, `PATH.1` and then `PATH`, oldest
first, and never writes, locks or truncates them, so it works while a server runs; a torn last line is left for the
next read. `PATH` defaults to `$TENSORFOLD_REQUEST_JOURNAL`.

```sh
tensorfold log list [PATH] [--last N] [--route chat|completions] [--model M]   # one line a request
tensorfold log show chatcmpl-example [PATH]   # the request as journaled and its reply, decoded
tensorfold log follow [PATH]                  # each newly finished request as it lands, across rotations
```

A `list` line holds the start time (UTC), reply id, route, model, status and outcome, prompt and completion tokens,
time to first token and decode rate; a request with no `finish` shows as in flight, and one a crash left open as
quarantined. `show` prints the messages or prompt (image and PDF data stay `<redacted>`), then the reply's text,
reasoning, tool calls and finish reason, joined from the stream's deltas for a streamed reply, with its tokens and
timings; a reply too large to keep shows its size and SHA-256. `--json` prints each request's records as journaled, one
object a line. Ctrl-C ends `follow`.

`tools/replay_request.py` lists, clears and replays journaled requests:

```sh
python3 tools/replay_request.py requests.jsonl --list
python3 tools/replay_request.py requests.jsonl --id chatcmpl-example --base http://127.0.0.1:8080
python3 tools/replay_request.py requests.jsonl --clear chatcmpl-example   # or a fingerprint, or all; server stopped
```

`--list` sends nothing. A replay sends the journaled body once to its route; when the client sent no seed, it adds the
seed the server drew. It compares the reply's output: every choice's text, reasoning, tool calls (without their ids),
logprobs and finish reason, the token counts, the stop sequence and `tensorfold.token_sha256`, the SHA-256 of the reply's
token ids, so equal means the same tokens. Streams compare their joined deltas, whatever their frame boundaries. Reply
ids, `created`, timings, `cached_tokens` and keepalives are left out; `--whole-body` compares every byte instead and
will differ on those. A reply too large to keep is compared by its token hash. Exit 0 is equal, 1 is different, and 2
means there was no finished reply to compare. `--body FILE` supplies the original request when images or a PDF were
redacted or the body was too large; `--allow-quarantined` sends a quarantined request on purpose; `--key-env NAME`
names the variable holding the API key (`TENSORFOLD_API_KEY` by default), which is sent and never written.

A replay is exact on the same build, checkpoint, tokenizer, drafter and server flags, since TensorFold's decode gives the
same tokens for the same prompt and seed with or without drafts, alone or batched. It is not exact across builds or
checkpoints, after the server's defaults change, when a Messages request relied on the server's thinking default, when
the body was restored with different media, or for a request whose server ran several generations for it (Cortheon)
without a client seed or `--prompt-seed`, since only the last drawn seed is journaled.
Unseeded prompt lists are not replay-exact either: each prompt draws its own seed, and those per-prompt prepared
records are omitted. Send a client seed or keep the same `--prompt-seed` setting for an exact list replay.
