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

## Release files

Each release's archives have SHA-256 files beside them on the release page, and Homebrew checks the macOS archive's
SHA-256. The Mac app is signed and notarized.
