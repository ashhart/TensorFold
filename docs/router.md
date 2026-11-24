# Several models at one address

`tensorfold serve --router` answers for several models at one address and loads each one when a request names it.
Point any OpenAI, Anthropic or Ollama client at it and pick a model by name, as with one server.

```sh
tensorfold serve --router                                  # every cached checkpoint, by its repo id
tensorfold serve --router --router-config models.toml      # the models the file lists
tensorfold serve --router --max-loaded 2 --idle-unload 900 --api-key-file /etc/tensorfold/keys
```

Each model runs as its own `tensorfold serve` child: the same binary, serving that one model on 127.0.0.1 at a port
the system picks. A child gets a random key of its own in its environment (`TENSORFOLD_API_KEY`), and the router sends
that key with every request it forwards, so other users of the machine cannot use the child's port. A request's
`model` field picks the child. When it is not running, the router starts it, and the request and any others for that
model wait until it is ready. The child's reply comes back byte for byte as it arrives, streams included.

The children lead process groups of their own, so a Ctrl-C or a closed terminal reaches only the router, which then
stops every child: SIGTERM, up to 30 seconds to finish the requests in progress, then SIGKILL. Without API keys, a
hangup (SIGHUP) stops the router as SIGTERM does; with keys it rereads the key file, as `serve` does.

## Flags

| Flag | Meaning |
| --- | --- |
| `--router` | Route by model instead of serving one; it takes no `MODEL`. |
| `--router-config FILE.toml` | The models to serve, each in a `[[models]]` table (below). Without it, every cached checkpoint. |
| `--max-loaded N` | Children loaded at once, 1 (the default) to 64. |
| `--idle-unload SECONDS` | Stop a child after this long with no request; 0, the default, keeps it. The same seconds as `serve --idle-unload`. |
| `--host HOST`, `--port PORT` | The router's address, default `127.0.0.1:8080`. |
| `--api-key KEY`, `--api-key-file FILE` | The router's keys, with `TENSORFOLD_API_KEY`, as [serve takes them](api.md#api-keys); scoped keys keep their scopes. |
| `--allowed-host NAME`, `--allowed-origin ORIGIN` | The router's [browser request checks](api.md#browser-request-headers). |

Any other serve flag on the router's command line is refused by name: set it on the model in `--router-config`.

## Models without a config

Without `--router-config`, the router serves every checkpoint `tensorfold models` lists, each under its repo id
(`org/name`) with serve's default flags. `GET /v1/models` lists those ids. The list is read once, at startup; a
checkpoint that `serve` would refuse, such as one whose cached revision is missing, is left out and the log says why.

## The config file

```toml
# models.toml
[[models]]
id = "glm"
model = "zai-org/GLM-5.3-Flash"
aliases = ["glm-flash"]
context = 131072
thinking = false
prompt-cache-gib = 8

[[models]]
id = "small"
model = "./weights/small"          # a folder beside models.toml
parallel = 2
```

Each `[[models]]` table is one model:

- `id`: the name it answers to. Required.
- `model`: a folder or a Hugging Face repo id in the cache, as `serve MODEL` takes it. Required.
- `aliases`: more names it answers to, a string or a list of strings.
- Any other key is a serve flag, read as [serve config files](serve-config.md) read their keys: the same key rules,
  value checks and messages, and a relative path starts at the file's folder.

An id or alias is 1 to 256 bytes with no spaces or control characters and does not start with `-`. No name may belong
to two models. The router sets these flags on every child itself, so a table that sets one is refused with the reason:
`name`, `alias`, `host`, `port`, `unix-socket`, `api-key`, `api-key-file`, `idle-unload`, `allowed-host` and
`allowed-origin`. A top-level key is refused too: the file holds `[[models]]` tables only, and the router's own flags
go on its command line. The file is opened as `serve --config` opens its file (no link, nothing other users can write,
at most 1 MiB), and every problem names the file and the line.

At startup the router checks each model's command line as `serve` would and finds its checkpoint (and drafter) in
the cache; a model that fails either stops the start with the model's id and the reason.

## Loading and stopping

- At most `--max-loaded` children are loaded at once. Loading another first stops the least recently used child that
  has no request in flight. If every loaded child is busy, the request waits up to 5 minutes for one to finish, then
  gets 503 `models_busy` with `Retry-After`.
- Memory: each model has a memory floor, the estimate `tensorfold info` prints (weights plus a 2 GiB runtime margin,
  or Flash Next's startup budget on Metal), plus its `--drafter`'s weights. Before a child starts, the floors of the
  loaded children plus its own must fit in the machine's memory; idle children stop, least recently used first, to
  make room. A model whose floor is larger than the machine's memory gets 503 `model_too_large`. A model with no
  estimate is bounded by `--max-loaded` alone, and the log says so.
- A child gets as long to load as `tensorfold benchmark` gives its server: 600 seconds, plus 6 seconds for each GB of
  weights past 20 GB, at most 3600. Past it the request gets 503 `model_load_timeout`; a child that exits while
  loading gives 503 `model_load_failed` with the child's last log line.
- A child that exits while it is loaded is marked unloaded. Requests still waiting for its reply get 502
  `model_server_failed`; a reply already streaming ends where the child stopped, as a lone server's would. The next
  request starts it again.
- Each child's log goes to a private file under `~/.tensorfold/router`, removed from the folder as soon as the child
  starts. The router prints the child's last lines when it fails to load or exits.

## Routes

These routes are forwarded by their JSON body's `model` field (POST; rerank also by PUT):

| API | Routes |
| --- | --- |
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, `/tokenize`, `/detokenize` |
| Anthropic | `/v1/messages`, `/v1/messages/count_tokens` |
| Ollama | `/api/chat`, `/api/generate`, `/api/embed`, `/api/show` |

They also answer at the other spellings a server takes, such as `/chat/completions` and `/messages`. A body with no
`model` gets 400 `model_required`, and a name the router does not serve gets 404 `model_not_found`. The router answers
in the route's own error shape: OpenAI's error object, Anthropic's `not_found_error`, or Ollama's
`{"error": "..."}`. A restricted key's Responses requests must be stateless, as on a server.

The router answers these itself, with [API keys](api.md#api-keys) and scopes applied as a server applies them:
`GET /health`, `GET /v1/models`, `GET /api/tags`, `GET /api/ps` and `GET /api/version`. See
[the API notes](api.md#router).

Everything else is refused by name with 404 `unsupported_route` in this version, among them `GET /v1/responses/{id}`
(stored responses stay in the child that made them), the Responses WebSocket mode, `GET /metrics` (each child keeps
its own metrics, which the router does not expose), `POST /v1/requests/{id}/cancel`, speech, images, OCR and the
server's own pages.

## Limits

- The router is not available on Windows: run one `tensorfold serve` a model.
- Each child plans its memory as a lone server does. With `--max-loaded` above 1, set `prompt-cache-gib` for each model
  so that the first one loaded does not take the room the next needs.
- The memory check counts the machine's system memory. On a discrete GPU, `--max-loaded` is the bound that matters.
- A router killed with SIGKILL leaves its children running; they stop only when the router stops itself.
- `capabilities --json` does not list the router's flags.
