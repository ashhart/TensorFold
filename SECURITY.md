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
- The TCP links between the ranks of a multi-GPU group. See [Ranks on more than one host](#ranks-on-more-than-one-host).
- The Mac app under `app/`.

Out of scope: what a model writes, a model following instructions in text you gave it, and a client sending the server
more work than its machine can do.


## Native HTTP browser boundaries

When API keys are configured, private native HTTP routes require a valid key by default. Only GET root, `/health`, `/slide`, its eight embedded scripts,
and GET metrics explicitly opened by `--metrics-open` are public exceptions. Public health never resets memory peaks. This policy includes `/systemone` aliases.
Every valid key reaches every private route, Living Weights, Recall and the stored Responses included, unless the key file
gives it [scopes](docs/api.md#key-scopes): an `inference` or `read` key reaches only its exact routes and keeps no Responses.

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

## Ranks on more than one host

A multi-GPU group connects its ranks over TCP: `--tp` with `--master`, a ranked family's `NCCL_COMM_ID`, or the
`TF_TP_*` settings. Every one of those connections proves a shared key before any other byte crosses. Both ends send a
fresh 32-byte nonce and an HMAC-SHA256 over the nonces, the link, the rank and the group size. A connection that fails
is closed. It is sent no checkpoint identity, no NCCL or RCCL id and no prompt, and it cannot stop the group.

- Set `TF_RANK_SECRET_FILE` on every rank, or pass `--rank-secret FILE` to `serve`, naming a file with the same
  contents on every host. It must be a regular file only its owner can read, holding 32 bytes to 4 KiB:
  `head -c 32 /dev/urandom | base64 > rank.key && chmod 600 rank.key`. A symbolic link to such a file is fine, as
  Kubernetes and secret managers mount them; the mode checked is the file's, so give a Kubernetes secret
  `defaultMode: 0400`.
- Rank 0 logs each connection it refuses with the peer's address and the reason: no key, a different key, or no hello
  within 2 seconds. A connection that sends nothing is dropped after those 2 seconds.
- Ranks with no key meet only on one host. A rank that reaches another host without a key refuses to start, and its
  error names the setting.
- To change the key, replace the file on every host and restart the group.

The proof shows which process connected. On a group with a key, every byte the engine itself sends between ranks after
the proof crosses in ChaCha20-Poly1305 records under a key the proof leaves at both ends. That covers the
tensor-parallel bootstrap, fate and plan links, the ranks' liveness lines, the ROCm ranks' command link and the Flash
Next pair's control link, so a host on the path reads no settings, plan, failure reason or NCCL id, and a record it
changes, drops or replays closes the link. A group with no key, which meets on one host only, is not sealed. The
tensors do not cross on those links. NCCL, RCCL and RDMA carry them, and that traffic is clear text, so run the ranks
on a network you trust or on an encrypted one. The GLM and MiniMax ranks also broadcast each step's header and prompt
token ids over NCCL (`glm/cuda_follow.zig`, `minimax_m3/cuda_follow.zig`), so on those families that network carries
the prompt as well and has to be trusted the same way. The cluster links (`serve --cluster`, `tensorfold node`) are a
separate transport and do not use this key.

## Release files

Each release's archives have SHA-256 files beside them on the release page, and Homebrew checks the macOS archive's
SHA-256. The Mac app is signed and notarized.

Malformed or folded headers and unsupported request targets are refused before routing. Each request on a pooled
connection is checked afresh. Rejections close before `100 Continue` and unread bodies are not dispatched. Key files
must be regular files without group or other read/write permission; use `chmod 600`.

## Cluster nodes

A cluster node on a real host takes only the nodes its cluster file names, each on the link the file gives it, and it
believes a node only about itself. Another node's word cannot add a member, choose the leader or end a member's run.
The links carry no key: the node at the far end of a link is whichever host MCDMA connected there, and a node of the
file is trusted about itself.
