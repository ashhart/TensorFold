# Donation relay — how a lent machine gets a public address

A donation card is a URL plus a token. Until now the URL had to be something the donor arranged
(an SSH tunnel, a Tailscale name, a Cloudflare tunnel), because the in-app donor binds loopback
only (`donation-host.ts` refuses any other `endpointHost`). The relay removes that step: the
donor dials **out**, and the card carries the relay's HTTPS URL.

## Shape

```
card.endpoint = https://<relay>/<card-id>
provider.baseUrl = https://<relay>/<card-id>/v1
```

- `relay/` — Worker + Durable Object (one per card id). `POST /<id>/v1/chat/completions`,
  `GET /<id>/v1/models`, `GET /<id>/health` are forwarded to the attached donor; everything else
  is a 404. Donor attach: `wss://<relay>/donor/<id>` with the donor key as the second subprotocol.
- `app/electron/donation-relay-client.ts` — the dialer. It frames requests to the local donation
  node and streams the node's answer back, so **every** guarantee stays where it already lived:
  token, window, caps, receipts, backpressure.
- `app/scripts/donation-relay-attach.mjs` — CLI to attach an existing node.
- `app/shared/donation.ts` — accepts one path segment on a card endpoint, which is what lets a
  card point at `https://<relay>/<card-id>` at all. Deeper paths, queries and fragments are still
  refused.

## Verified locally (wrangler dev, real Durable Object)

| Check | Result |
|---|---|
| `GET /<id>/v1/models` through the relay | model list from the donor's engine |
| Streaming chat through the relay | first token 4.42 s, total 4.47 s on a local 4B model |
| Wrong card token | `401` with the donor's own `donation_auth` message |
| Card with no attached donor | `503 No donor is attached to this card right now.` |
| 1.2 MB request body | `413` from the relay before it reaches the donor |
| Second donor with a different key | refused at attach |
| Donor dialer unit test | 5 cases incl. SSE passthrough and a `401` passed through unchanged |

## What is still missing

1. **The app does not start the dialer yet.** `donation-host.ts` starts the node and mints a
   loopback card; joining the two means: generate a donor key, attach on start, reconnect with
   backoff, write the relay URL into the card, and stop the attachment on window end.
2. **No relay configuration UI.** Either a single TensorFold-hosted relay the app knows about, or
   a self-hosted URL field. Cards from before this change keep working (origin-only endpoints).
3. **Payload secrecy.** TLS ends at the relay, so the relay (and its provider) can read prompts.
   Encrypting the request/response with a key derived from the card token keeps the relay a blind
   pipe; the donor still enforces caps. Not implemented.
4. **Abuse handling.** No per-IP limits, no card revocation, no reporting. A public relay needs
   all three before it takes strangers' traffic.
5. **The relay is a deletion target.** Removing it breaks every relay card; the card should say
   which relay it uses so the app can surface that.

## Costs

One Worker plus Durable Objects: request volume and DO duration. For a POC with a handful of
cards it should sit inside the free tier; the real cost is bandwidth, and text inference is
kilobytes per turn.
