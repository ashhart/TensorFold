# Supported models preview

A runnable preview of the TensorFold "Supported models" page: the fastest single stream recorded on the hardware you
pick or filter to, hardware filters, a sortable table of measured runs, and the supported models that have no measured
run yet, kept apart with no number. Every number links to its receipt.

The page reads one file, [`../tensorfold_web/data/models.json`](../tensorfold_web/data/models.json), described by
[`models.schema.json`](../tensorfold_web/data/models.schema.json). Point it at the raw GitHub copy and it follows the
repository: a changed number needs a commit, not a site deploy.

## Run it

From this folder, with Go 1.26 (`mise.toml` pins it):

```sh
go run ./cmd/preview                                   # http://localhost:3000, the repository's own models.json
go run ./cmd/preview -catalog https://raw.githubusercontent.com/ashhart/TensorFold/main/web/tensorfold_web/data/models.json
```

| Flag | Environment | Default | |
|---|---|---|---|
| `-catalog` | `CATALOG` | `../tensorfold_web/data/models.json` | a file path or an https URL |
| `-ttl` | `CATALOG_TTL` | `5m` | how long a fetched copy is used before the URL is asked again |
| `-cache-dir` | `CATALOG_CACHE_DIR` | the user cache folder | where the last fetched copy is kept |
| `-addr` | `PREVIEW_ADDR` | `:3000` | listen address, every interface by default |
| `-repo-url` | `REPO_URL` | the README's model table | target of "Full list of supported models" |

A file is read on every page load. A URL is fetched with its ETag, reused for the TTL and kept on disk, so a restart
without network still shows the last copy. A copy that fails to download, or that breaks a rule of the format, never
replaces the last good one: the page keeps the good copy and says why under the table.

## The format

`models[]` lists support: each backend's status (`supported`, `experimental`, `planned`), its recipe and the hardware
classes the recipe covers. `runs[]` lists measurements, each copied from its receipt; unknown values are `null`.
A rate always says what it counts: `rate_scope` is `per-stream` or `aggregate`, and an aggregate must be the
steady-state total while every stream generates (`rate_window: steady-state`), so total ÷ streams is each stream's
rate. Totals over the whole burst, which include start-up and drain time, are refused.

`tests/test_models_catalog.py` (in `web/`) checks the file against the schema and the rules across records, plants
each kind of error to show the check fails, and checks that each upstream receipt still contains its run's numbers.

## How it is built

Go with server-rendered [templ](https://templ.guide) and [Datastar](https://data-star.dev), ported from
[Go Fast Stack](https://github.com/jayleaton/go-fast-stack)'s web app without its database, auth or i18n:

- `internal/services`: the format's rules, the board (filter, sort, the fastest single stream, unmeasured models) and
  the last-good copy. It reads the catalog through the `Source` port.
- `internal/server`: the file and URL adapters for that port, and the wiring.
- `internal/api`: the same reads as JSON (`/api/v1/board`, `/api/v1/catalog`, `/api/v1/runs/{id}`) in the envelope
  `{"data": ..., "error": ...}`, for another site to fetch.
- `ui`: the page. It calls the API in-process; filters, sorting and the picker swap the board in place with
  Datastar and keep the address bar in step, and every control is also a plain link or form.

`make check` runs gofmt, golangci-lint (the layering rules are in `.golangci.yml`), the service and API tests under
the race detector, and the generated-code check. `make build` writes a static binary.
