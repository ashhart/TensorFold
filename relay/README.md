# TensorFold donation relay

One Worker + one Durable Object per card id. A donor never accepts inbound traffic: it dials
out to the relay and afterwards renters reach it over plain HTTPS.

```
renter ──HTTPS──▶ relay Worker ──WebSocket──▶ donor dialer ──HTTP──▶ 127.0.0.1:<port> (donation node)
                                                                    └──▶ engine
```

The relay is a frame pump, not a security boundary. It forwards the renter's `authorization`
header untouched, and the donor's donation node still checks the card token, the window and the
caps. A compromised relay can deny service or read prompts; it cannot mint a valid token.

## URLs

| Who | URL |
|---|---|
| Renter (OpenAI base URL) | `https://<relay>/<card-id>/v1` |
| Renter health probe | `https://<relay>/<card-id>/health` |
| Donor attach | `wss://<relay>/donor/<card-id>` |

A card's `endpoint` is `https://<relay>/<card-id>`, which the app's parser now accepts (one path
segment, no query or fragment) and turns into a provider base URL ending in `/v1`.

## Donor key

The donor presents a random secret, and the relay binds it to the card id on first attach
(trust-on-first-use). The key travels as the **second WebSocket subprotocol** — neither browsers
nor Node can set WebSocket headers, and a query parameter would end up in logs:

```js
new WebSocket('wss://relay/donor/donate-abc123', ['tf-donation-v1', donorKey])
```

Ids are unguessable in practice, but the first attach wins: a real deployment should either
register ids server-side or let the donor re-key with proof.

## Run it locally

```sh
# 1. the relay, on a real workerd runtime
cd relay && wrangler dev --port 8788 --ip 127.0.0.1

# 2. a donation node to lend (any OpenAI-compatible upstream, see app/docs/donation-poc.md)
cd app && node dist-electron/donation-node.js --config <donation.json>

# 3. attach it to the relay
cd app && node node_modules/jiti/lib/jiti-cli.mjs scripts/donation-relay-attach.mjs \
  --relay http://127.0.0.1:8788 --id <card-id> --key <donor-key> --local 8020

# 4. rent through it
curl -H "authorization: Bearer <card token>" http://127.0.0.1:8788/<card-id>/v1/models
```

## Test

```sh
cd relay && npm install && npm test
```

The suite runs inside the Workers runtime (workerd through `@cloudflare/vitest-plugin`): the Worker,
the Durable Object and its WebSocket plumbing are the real code, and the donor is simulated. It
covers health, 503 without a donor, 404 for routes that are not lent, 413 for oversized bodies, a
streamed reply, a donor refusal passed through untouched, one donor per card id, and a wrong donor
key refused.

## Deploy

```sh
cd relay && wrangler deploy
```

Then give the Worker a hostname (custom domain or workers.dev) and mint cards that point at
`https://<relay>/<card-id>`. Nothing in the app changes for the renter.

## Limits

| Limit | Value | Where |
|---|---|---|
| Request body | 1 MiB | `MAX_REQUEST_BYTES`, matches the donor cap |
| In-flight requests per card | 8 | `MAX_INFLIGHT` |
| Wait for response headers | 20 s | `HEAD_TIMEOUT_MS` |
| Silence before aborting a stream | 120 s | `IDLE_TIMEOUT_MS` |
| Buffered response bytes | 16 MiB | `MAX_STREAM_BYTES` |

## Notes and gotchas

- `compatibility_date` must not exceed what the installed wrangler's bundled workerd supports. If
  `wrangler dev` answers "newest date supported by this server binary", lower the date or upgrade
  wrangler; deployed Workers accept the newer date.
- `wrangler dev` picks a local port that may already be in use; pass `--port` explicitly.
- The Durable Object is declared with `new_sqlite_classes`, which is the free-plan shape.
- The relay terminates TLS, so it can read prompts. End-to-end encryption keyed off the card
  token is the obvious fix and is **not** implemented here.
- No per-IP rate limiting, no abuse reporting, no request logging. Add before exposing publicly.
- Text inference is tiny on the wire; if this ever grows into a TURN relay for WebRTC the
  bandwidth story is different.
