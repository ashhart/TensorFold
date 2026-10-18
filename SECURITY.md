# Security

## Reporting a vulnerability

Report security problems privately: open this repository's Security tab and choose "Report a vulnerability". Please
don't open a public issue or pull request for a problem that isn't fixed yet. You'll get a reply within three days, and
a fix or a mitigation ships in a release as soon as it is gated, within two weeks for anything serious. If the problem
is already public, a pull request with the fix is welcome.

## Supported versions

Only the latest release gets security fixes. The Python 0.6.6 engine at the `v0.6.6` tag gets none.

## What is in scope

- The native server (`tensorfold serve`): its HTTP API, its prompt cache and the files it keeps on disk.
- `--cortheon`: web search, page reads, and the filter that removes orders aimed at the model from web text. That
  filter is a heuristic; [docs/cortheon.md](docs/cortheon.md) and [docs/known-issues.md](docs/known-issues.md) say
  what it does not catch.
- Living Weights (`--slide` and `/v1/slide/*`): a lesson changes the checkpoint's own files. Serve it with `--api-key`.
- `tensorfold pull` and the checkpoint loaders, which read files from Hugging Face as untrusted input.
- The Mac app under `app/`.

Out of scope: what a model writes, a model following instructions in text you gave it, and a client sending the server
more work than its machine can do.


## Native HTTP browser boundaries

When API keys are configured, private native HTTP routes require a valid key by default. Only GET root, `/health`, `/slide`, its eight embedded scripts,
and GET metrics explicitly opened by `--metrics-open` are public exceptions. Public health never resets memory peaks. This policy includes `/systemone` aliases.

A loopback listener refuses a Host outside localhost and loopback IP names, preventing a DNS-rebinding site from
using its own hostname to read the local API. `--allowed-host NAME` is repeatable and adds an explicit trusted name
without changing the bind address. Names have no scheme, path or port; bracket IPv6 literals. Classification uses the
resolved bound address, including mapped loopback addresses, rather than trusting a request header or peer address.

On every listener, a browser Origin must have the same scheme, host and effective port as the request's Host.
The native listener is HTTP. `--allowed-origin ORIGIN` is repeatable and adds a trusted HTTP or HTTPS origin, such as
an HTTPS UI behind a TLS proxy. Origins have no path or userinfo. Opaque `null` origins, malformed values and duplicates
are refused. Both checks run before authentication, body processing and routing, even when a valid API key is supplied.
Their HTTP 403 replies name Host/Origin and the relevant exception flag.

Clients without Origin keep working with a valid Host, and no Content-Type restriction is introduced.
The server sends no CORS headers; an origin exception does not grant cross-origin browser reads.
These browser checks do not replace `--api-key` for network access. See [API headers](docs/api.md#browser-request-headers).

## Release files

Each release's archives have SHA-256 files beside them on the release page, and Homebrew checks the macOS archive's
SHA-256. The Mac app is signed and notarized.

Malformed or folded headers and unsupported request targets are refused before routing. Each request on a pooled
connection is checked afresh. Rejections close before `100 Continue` and unread bodies are not dispatched. Key files
must be regular files without group or other read/write permission; use `chmod 600`.
